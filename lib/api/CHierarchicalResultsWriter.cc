/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CHierarchicalResultsWriter.h>

#include <core/CLogger.h>

#include <model/CHierarchicalResults.h>

#include <boost/optional.hpp>

namespace ml {
namespace api {

namespace {
using TOptionalDouble = boost::optional<double>;
using TOptionalUInt64 = boost::optional<uint64_t>;
using TDouble1Vec = core::CSmallVector<double, 1>;
const std::string COUNT_NAME("count");
const std::string EMPTY_STRING;
const CHierarchicalResultsWriter::TStr1Vec EMPTY_STRING_LIST;
}

CHierarchicalResultsWriter::SResults::SResults(
    bool isAllTimeResult,
    bool isOverallResult,
    const std::string& partitionFieldName,
    const std::string& partitionFieldValue,
    const std::string& overFieldName,
    const std::string& overFieldValue,
    const std::string& byFieldName,
    const std::string& byFieldValue,
    const std::string& correlatedByFieldValue,
    core_t::TTime bucketStartTime,
    const std::string& functionName,
    const std::string& functionDescription,
    const TDouble1Vec& functionValue,
    const TDouble1Vec& populationAverage,
    double rawAnomalyScore,
    double normalizedAnomalyScore,
    double probability,
    const TOptionalUInt64& currentRate,
    const std::string& metricValueField,
    const TStoredStringPtrStoredStringPtrPrDoublePrVec& influences,
    bool useNull,
    bool metric,
    int identifier,
    core_t::TTime bucketSpan)
    : s_ResultType(E_PopulationResult), s_IsAllTimeResult(isAllTimeResult),
      s_IsOverallResult(isOverallResult), s_UseNull(useNull),
      s_IsMetric(metric), s_PartitionFieldName(partitionFieldName),
      s_PartitionFieldValue(partitionFieldValue), s_ByFieldName(byFieldName),
      s_ByFieldValue(byFieldValue), s_CorrelatedByFieldValue(correlatedByFieldValue),
      s_OverFieldName(overFieldName), s_OverFieldValue(overFieldValue),
      s_MetricValueField(metricValueField), s_BucketStartTime(bucketStartTime),
      s_BucketSpan(bucketSpan), s_FunctionName(functionName),
      s_FunctionDescription(functionDescription),
      s_FunctionValue(functionValue), s_PopulationAverage(populationAverage),
      s_BaselineRate(0.0), s_CurrentRate(currentRate), s_BaselineMean{0.0},
      s_CurrentMean{0.0}, s_RawAnomalyScore(rawAnomalyScore),
      s_NormalizedAnomalyScore(normalizedAnomalyScore), s_Probability(probability),
      s_Influences(influences), s_Identifier(identifier) {
}

CHierarchicalResultsWriter::SResults::SResults(
    EResultType resultType,
    const std::string& partitionFieldName,
    const std::string& partitionFieldValue,
    const std::string& byFieldName,
    const std::string& byFieldValue,
    const std::string& correlatedByFieldValue,
    core_t::TTime bucketStartTime,
    const std::string& functionName,
    const std::string& functionDescription,
    const TOptionalDouble& baselineRate,
    const TOptionalUInt64& currentRate,
    const TDouble1Vec& baselineMean,
    const TDouble1Vec& currentMean,
    double rawAnomalyScore,
    double normalizedAnomalyScore,
    double probability,
    const std::string& metricValueField,
    const TStoredStringPtrStoredStringPtrPrDoublePrVec& influences,
    bool useNull,
    bool metric,
    int identifier,
    core_t::TTime bucketSpan,
    TStr1Vec scheduledEventDescriptions)
    : s_ResultType(resultType), s_IsAllTimeResult(false), s_IsOverallResult(true),
      s_UseNull(useNull), s_IsMetric(metric), s_PartitionFieldName(partitionFieldName),
      s_PartitionFieldValue(partitionFieldValue), s_ByFieldName(byFieldName),
      s_ByFieldValue(byFieldValue), s_CorrelatedByFieldValue(correlatedByFieldValue),
      // The simple count output is characterised by both 'by' and 'over' field names being 'count'
      // TODO: this could be done differently now, with changes to both this class and CJsonOutputWriter
      s_OverFieldName((byFieldName == COUNT_NAME) ? COUNT_NAME : EMPTY_STRING),
      s_OverFieldValue(EMPTY_STRING), s_MetricValueField(metricValueField),
      s_BucketStartTime(bucketStartTime), s_BucketSpan(bucketSpan),
      s_FunctionName(functionName), s_FunctionDescription(functionDescription),
      s_FunctionValue{0.0}, s_PopulationAverage{0.0}, s_BaselineRate(baselineRate),
      s_CurrentRate(currentRate), s_BaselineMean(baselineMean),
      s_CurrentMean(currentMean), s_RawAnomalyScore(rawAnomalyScore),
      s_NormalizedAnomalyScore(normalizedAnomalyScore), s_Probability(probability),
      s_Influences(influences), s_Identifier(identifier),
      s_ScheduledEventDescriptions(scheduledEventDescriptions) {
}

CHierarchicalResultsWriter::CHierarchicalResultsWriter(
    const model::CLimits& limits,
    const model::CAnomalyDetectorModelConfig& modelConfig,
    const TResultWriterFunc& resultWriterFunc,
    const TPivotWriterFunc& pivotWriterFunc)
    : m_Limits(limits), m_ModelConfig(modelConfig),
      m_ResultWriterFunc(resultWriterFunc), m_PivotWriterFunc(pivotWriterFunc),
      m_BucketTime(0) {
}

void CHierarchicalResultsWriter::visit(const model::CHierarchicalResults& results,
                                       const TNode& node,
                                       bool pivot) {
    if (pivot) {
        this->writePivotResult(results, node);
    } else {
        this->writePopulationResult(results, node);
        this->writeIndividualResult(results, node);
        this->writePartitionResult(results, node);
        this->writeSimpleCountResult(node);
    }
}

void CHierarchicalResultsWriter::writePopulationResult(const model::CHierarchicalResults& results,
                                                       const TNode& node) {
    if (this->isSimpleCount(node) || !this->isLeaf(node) || !this->isPopulation(node) ||
        !this->shouldWriteResult(m_Limits, results, node, false)) {
        return;
    }

    // We need to output an overall row, plus one for each of the top
    // attributes.

    // The attribute probabilities are returned in sorted order. This
    // is used to set the human readable description of the anomaly in
    // the GUI.
    const std::string& functionDescription =
        node.s_AnnotatedProbability.s_AttributeProbabilities.empty()
            ? EMPTY_STRING
            : model_t::outputFunctionName(
                  node.s_AnnotatedProbability.s_AttributeProbabilities[0].s_Feature);

    TOptionalDouble null;
    for (std::size_t i = 0;
         i < node.s_AnnotatedProbability.s_AttributeProbabilities.size(); ++i) {
        const model::SAttributeProbability& attributeProbability =
            node.s_AnnotatedProbability.s_AttributeProbabilities[i];

        // TODO - At present the display code can only cope with all the
        // attribute rows having the same output function name as the
        // overall row.  Additionally, the population metric models will
        // create attribute rows for min/max/mean even if only one of
        // these is anomalous.  We hack around both problems by making
        // two passes through the output data, first to decide which
        // single output function name to use and second to output only
        // the rows relating to the chosen output function name.
        // However, in the future we should fix both problems properly.
        // See bugs 396 and 415 in Bugzilla for more details. Similarly
        // it can't handle output from multiple different keys. This
        // needs to change at some point.
        model_t::EFeature feature = attributeProbability.s_Feature;
        if (functionDescription != model_t::outputFunctionName(feature)) {
            continue;
        }

        const std::string& attribute = *attributeProbability.s_Attribute;
        const TDouble1Vec& personAttributeValue = attributeProbability.s_CurrentBucketValue;
        if (personAttributeValue.empty()) {
            LOG_ERROR(<< "Failed to get current bucket value for " << attribute);
            continue;
        }

        const TDouble1Vec& attributeMean = attributeProbability.s_BaselineBucketMean;
        if (attributeMean.empty()) {
            LOG_ERROR(<< "Failed to get population mean for " << attribute);
            continue;
        }

        m_ResultWriterFunc(TResults(
            false,
            false, // not an overall result
            *node.s_Spec.s_PartitionFieldName,
            *node.s_Spec.s_PartitionFieldValue, *node.s_Spec.s_PersonFieldName,
            *node.s_Spec.s_PersonFieldValue, *node.s_Spec.s_ByFieldName,
            attribute, // attribute field value
            attributeProbability.s_CorrelatedAttributes.empty()
                ? EMPTY_STRING
                : *attributeProbability.s_CorrelatedAttributes[0],
            node.s_BucketStartTime, *node.s_Spec.s_FunctionName, functionDescription,
            personAttributeValue, attributeMean, node.s_RawAnomalyScore,
            node.s_NormalizedAnomalyScore, attributeProbability.s_Probability,
            node.s_AnnotatedProbability.s_CurrentBucketCount, *node.s_Spec.s_ValueFieldName,
            node.s_AnnotatedProbability.s_Influences, node.s_Spec.s_UseNull,
            model::function_t::isMetric(node.s_Spec.s_Function),
            node.s_Spec.s_Detector, node.s_BucketLength));
    }

    // Overall result for this person
    m_ResultWriterFunc(TResults(
        false,
        true, // this is an overall result
        *node.s_Spec.s_PartitionFieldName, *node.s_Spec.s_PartitionFieldValue,
        *node.s_Spec.s_PersonFieldName, *node.s_Spec.s_PersonFieldValue,
        *node.s_Spec.s_ByFieldName, EMPTY_STRING, EMPTY_STRING, node.s_BucketStartTime,
        *node.s_Spec.s_FunctionName, functionDescription, TDouble1Vec(1, 0.0), // no function value in overall result
        TDouble1Vec(1, 0.0), // no population average in overall result
        node.s_RawAnomalyScore, node.s_NormalizedAnomalyScore,
        node.probability(), node.s_AnnotatedProbability.s_CurrentBucketCount,
        *node.s_Spec.s_ValueFieldName, node.s_AnnotatedProbability.s_Influences,
        node.s_Spec.s_UseNull, model::function_t::isMetric(node.s_Spec.s_Function),
        node.s_Spec.s_Detector, node.s_BucketLength));

    // TODO - could also output "all time" results here
    // These would have the first argument to the SResults constructor
    // set to true, but would still need to store the current bucket
    // time (so that we could search for the most recent "all time"
    // results)
}

void CHierarchicalResultsWriter::writeIndividualResult(const model::CHierarchicalResults& results,
                                                       const TNode& node) {
    if (this->isSimpleCount(node) || !this->isLeaf(node) || this->isPopulation(node) ||
        !this->shouldWriteResult(m_Limits, results, node, false)) {
        return;
    }

    model_t::EFeature feature =
        node.s_AnnotatedProbability.s_AttributeProbabilities.empty()
            ? model_t::E_IndividualCountByBucketAndPerson
            : node.s_AnnotatedProbability.s_AttributeProbabilities[0].s_Feature;

    const model::SAttributeProbability& attributeProbability =
        node.s_AnnotatedProbability.s_AttributeProbabilities[0];

    m_ResultWriterFunc(TResults(
        E_Result, *node.s_Spec.s_PartitionFieldName, *node.s_Spec.s_PartitionFieldValue,
        *node.s_Spec.s_ByFieldName, *node.s_Spec.s_PersonFieldValue,
        attributeProbability.s_CorrelatedAttributes.empty()
            ? EMPTY_STRING
            : *attributeProbability.s_CorrelatedAttributes[0],
        node.s_BucketStartTime, *node.s_Spec.s_FunctionName,
        model_t::outputFunctionName(feature), node.s_AnnotatedProbability.s_BaselineBucketCount,
        node.s_AnnotatedProbability.s_CurrentBucketCount,
        attributeProbability.s_BaselineBucketMean, attributeProbability.s_CurrentBucketValue,
        node.s_RawAnomalyScore, node.s_NormalizedAnomalyScore, node.probability(),
        *node.s_Spec.s_ValueFieldName, node.s_AnnotatedProbability.s_Influences,
        node.s_Spec.s_UseNull, model::function_t::isMetric(node.s_Spec.s_Function),
        node.s_Spec.s_Detector, node.s_BucketLength, EMPTY_STRING_LIST));
}

void CHierarchicalResultsWriter::writePartitionResult(const model::CHierarchicalResults& results,
                                                      const TNode& node) {
    if (!m_ModelConfig.perPartitionNormalization() || this->isSimpleCount(node) ||
        this->isPopulation(node) || !this->isPartition(node) ||
        !this->shouldWriteResult(m_Limits, results, node, false)) {
        return;
    }

    model_t::EFeature feature =
        node.s_AnnotatedProbability.s_AttributeProbabilities.empty()
            ? model_t::E_IndividualCountByBucketAndPerson
            : node.s_AnnotatedProbability.s_AttributeProbabilities[0].s_Feature;

    TDouble1Vec emptyDoubleVec;

    m_ResultWriterFunc(TResults(
        E_PartitionResult, *node.s_Spec.s_PartitionFieldName,
        *node.s_Spec.s_PartitionFieldValue, *node.s_Spec.s_ByFieldName,
        *node.s_Spec.s_PersonFieldValue, EMPTY_STRING, node.s_BucketStartTime,
        *node.s_Spec.s_FunctionName, model_t::outputFunctionName(feature),
        node.s_AnnotatedProbability.s_BaselineBucketCount,
        node.s_AnnotatedProbability.s_CurrentBucketCount, emptyDoubleVec, emptyDoubleVec,
        node.s_RawAnomalyScore, node.s_NormalizedAnomalyScore, node.probability(),
        *node.s_Spec.s_ValueFieldName, node.s_AnnotatedProbability.s_Influences,
        node.s_Spec.s_UseNull, model::function_t::isMetric(node.s_Spec.s_Function),
        node.s_Spec.s_Detector, node.s_BucketLength, EMPTY_STRING_LIST));
}

void CHierarchicalResultsWriter::writePivotResult(const model::CHierarchicalResults& results,
                                                  const TNode& node) {
    if (this->isSimpleCount(node) ||
        !this->shouldWriteResult(m_Limits, results, node, true)) {
        return;
    }

    LOG_TRACE(<< "bucket start time " << m_BucketTime);
    if (!m_PivotWriterFunc(m_BucketTime, node, this->isRoot(node))) {
        LOG_ERROR(<< "Failed to write influencer result for " << node.s_Spec.print());
        return;
    }
}

void CHierarchicalResultsWriter::writeSimpleCountResult(const TNode& node) {
    if (!this->isSimpleCount(node)) {
        return;
    }

    m_BucketTime = node.s_BucketStartTime;

    TOptionalDouble baselineCount = node.s_AnnotatedProbability.s_BaselineBucketCount;
    TOptionalUInt64 currentCount = node.s_AnnotatedProbability.s_CurrentBucketCount;

    m_ResultWriterFunc(TResults(
        E_SimpleCountResult, *node.s_Spec.s_PartitionFieldName, *node.s_Spec.s_PartitionFieldValue,
        *node.s_Spec.s_ByFieldName, *node.s_Spec.s_PersonFieldValue, EMPTY_STRING,
        m_BucketTime, EMPTY_STRING, EMPTY_STRING, baselineCount, currentCount,
        baselineCount ? TDouble1Vec(1, *baselineCount) : TDouble1Vec(),
        currentCount ? TDouble1Vec(1, static_cast<double>(*currentCount)) : TDouble1Vec(),
        node.s_RawAnomalyScore, node.s_NormalizedAnomalyScore, node.probability(),
        *node.s_Spec.s_ValueFieldName, node.s_AnnotatedProbability.s_Influences,
        node.s_Spec.s_UseNull, model::function_t::isMetric(node.s_Spec.s_Function),
        node.s_Spec.s_Detector, node.s_BucketLength, node.s_Spec.s_ScheduledEventDescriptions));
}

void CHierarchicalResultsWriter::findParentProbabilities(const TNode& node,
                                                         double& personProbability,
                                                         double& partitionProbability) {
    // The idea is that if person doesn't exist then the person probability is
    // set to the leaf probability, and if partition doesn't exist then the
    // partition probability is set to the person probability (or if person
    // doesn't exist either then the leaf probability)

    personProbability = node.probability();
    partitionProbability = node.probability();

    for (const TNode* parent = node.s_Parent; parent != nullptr; parent = parent->s_Parent) {
        if (CHierarchicalResultsWriter::isPartition(*parent)) {
            partitionProbability = parent->probability();
            // This makes the assumption that partition will be higher than
            // person in the hierarchy
            break;
        }
        if (CHierarchicalResultsWriter::isPerson(*parent)) {
            personProbability = parent->probability();
            partitionProbability = parent->probability();
        }
    }
}
}
}
