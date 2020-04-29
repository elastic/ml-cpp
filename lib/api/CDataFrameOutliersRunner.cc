/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameOutliersRunner.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CProgramCounters.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <maths/COutliers.h>

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameAnalysisSpecification.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <iterator>
#include <string>

namespace ml {
namespace api {
namespace {
const CDataFrameAnalysisConfigReader& parameterReader() {
    static const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
        CDataFrameAnalysisConfigReader theReader;
        theReader.addParameter(CDataFrameOutliersRunner::STANDARDIZATION_ENABLED,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(CDataFrameOutliersRunner::N_NEIGHBORS,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(
            CDataFrameOutliersRunner::METHOD, CDataFrameAnalysisConfigReader::E_OptionalParameter,
            {{maths::COutliers::LOF, int{maths::COutliers::E_Lof}},
             {maths::COutliers::LDOF, int{maths::COutliers::E_Ldof}},
             {maths::COutliers::DISTANCE_KNN, int{maths::COutliers::E_DistancekNN}},
             {maths::COutliers::TOTAL_DISTANCE_KNN, int{maths::COutliers::E_TotalDistancekNN}}});
        theReader.addParameter(CDataFrameOutliersRunner::COMPUTE_FEATURE_INFLUENCE,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(CDataFrameOutliersRunner::FEATURE_INFLUENCE_THRESHOLD,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        theReader.addParameter(CDataFrameOutliersRunner::OUTLIER_FRACTION,
                               CDataFrameAnalysisConfigReader::E_OptionalParameter);
        return theReader;
    }()};
    return PARAMETER_READER;
}

// Output
const std::string OUTLIER_SCORE_FIELD_NAME{"outlier_score"};
const std::string FEATURE_INFLUENCE_FIELD_NAME_PREFIX{"feature_influence."};
}

CDataFrameOutliersRunner::CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec,
                                                   const CDataFrameAnalysisParameters& parameters)
    : CDataFrameOutliersRunner{spec} {

    m_StandardizationEnabled = parameters[STANDARDIZATION_ENABLED].fallback(true);
    m_NumberNeighbours = parameters[N_NEIGHBORS].fallback(std::size_t{0});
    m_Method = parameters[METHOD].fallback(maths::COutliers::E_Ensemble);
    m_ComputeFeatureInfluence = parameters[COMPUTE_FEATURE_INFLUENCE].fallback(true);
    m_FeatureInfluenceThreshold = parameters[FEATURE_INFLUENCE_THRESHOLD].fallback(0.1);
    m_OutlierFraction = parameters[OUTLIER_FRACTION].fallback(0.05);

    m_Instrumentation.featureInfluenceThreshold(m_FeatureInfluenceThreshold);
}

CDataFrameOutliersRunner::CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec)
    : CDataFrameAnalysisRunner{spec}, m_Method{static_cast<std::size_t>(
                                          maths::COutliers::E_Ensemble)},
      m_Instrumentation{spec.jobId()} {
}

std::size_t CDataFrameOutliersRunner::numberExtraColumns() const {
    return m_ComputeFeatureInfluence ? this->spec().numberColumns() + 1 : 1;
}

void CDataFrameOutliersRunner::writeOneRow(const core::CDataFrame& frame,
                                           const TRowRef& row,
                                           core::CRapidJsonConcurrentLineWriter& writer) const {
    std::size_t scoreColumn{row.numberColumns() - this->numberExtraColumns()};
    std::size_t beginFeatureScoreColumns{scoreColumn + 1};
    std::size_t numberFeatureScoreColumns{this->numberExtraColumns() - 1};
    writer.StartObject();
    writer.Key(OUTLIER_SCORE_FIELD_NAME);
    writer.Double(row[scoreColumn]);
    if (row[scoreColumn] > m_FeatureInfluenceThreshold) {
        for (std::size_t i = 0; i < numberFeatureScoreColumns; ++i) {
            writer.Key(FEATURE_INFLUENCE_FIELD_NAME_PREFIX + frame.columnNames()[i]);
            writer.Double(row[beginFeatureScoreColumns + i]);
        }
    }
    writer.EndObject();
}

bool CDataFrameOutliersRunner::validate(const core::CDataFrame& frame) const {
    if (frame.numberColumns() < 1) {
        HANDLE_FATAL(<< "Input error: analysis needs at least one feature")
        return false;
    }
    return true;
}

const CDataFrameAnalysisInstrumentation& CDataFrameOutliersRunner::instrumentation() const {
    return m_Instrumentation;
}

CDataFrameAnalysisInstrumentation& CDataFrameOutliersRunner::instrumentation() {
    return m_Instrumentation;
}

void CDataFrameOutliersRunner::runImpl(core::CDataFrame& frame) {

    core::CProgramCounters::counter(counter_t::E_DFONumberPartitions) =
        this->numberPartitions();
    core::CProgramCounters::counter(counter_t::E_DFOEstimatedPeakMemoryUsage) =
        this->estimateMemoryUsage(frame.numberRows(),
                                  frame.numberRows() / this->numberPartitions(),
                                  frame.numberColumns());

    maths::COutliers::SComputeParameters params{this->spec().numberThreads(),
                                                this->numberPartitions(),
                                                m_StandardizationEnabled,
                                                static_cast<maths::COutliers::EMethod>(m_Method),
                                                m_NumberNeighbours,
                                                m_ComputeFeatureInfluence,
                                                m_OutlierFraction};
    maths::COutliers::compute(params, frame, m_Instrumentation);
}

std::size_t
CDataFrameOutliersRunner::estimateBookkeepingMemoryUsage(std::size_t numberPartitions,
                                                         std::size_t totalNumberRows,
                                                         std::size_t partitionNumberRows,
                                                         std::size_t numberColumns) const {
    maths::COutliers::SComputeParameters params{this->spec().numberThreads(),
                                                numberPartitions,
                                                m_StandardizationEnabled,
                                                static_cast<maths::COutliers::EMethod>(m_Method),
                                                m_NumberNeighbours,
                                                m_ComputeFeatureInfluence,
                                                m_OutlierFraction};
    return maths::COutliers::estimateMemoryUsedByCompute(
        params, totalNumberRows, partitionNumberRows, numberColumns);
}

const std::string CDataFrameOutliersRunner::STANDARDIZATION_ENABLED{"standardization_enabled"};
const std::string CDataFrameOutliersRunner::N_NEIGHBORS{"n_neighbors"};
const std::string CDataFrameOutliersRunner::METHOD{"method"};
const std::string CDataFrameOutliersRunner::COMPUTE_FEATURE_INFLUENCE{"compute_feature_influence"};
const std::string CDataFrameOutliersRunner::FEATURE_INFLUENCE_THRESHOLD{"feature_influence_threshold"};
const std::string CDataFrameOutliersRunner::OUTLIER_FRACTION{"outlier_fraction"};

const std::string& CDataFrameOutliersRunnerFactory::name() const {
    return NAME;
}

CDataFrameOutliersRunnerFactory::TRunnerUPtr
CDataFrameOutliersRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec) const {
    return std::make_unique<CDataFrameOutliersRunner>(spec);
}

CDataFrameOutliersRunnerFactory::TRunnerUPtr
CDataFrameOutliersRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec,
                                          const rapidjson::Value& jsonParameters) const {
    auto parameters = parameterReader().read(jsonParameters);
    return std::make_unique<CDataFrameOutliersRunner>(spec, parameters);
}

const std::string CDataFrameOutliersRunnerFactory::NAME{"outlier_detection"};
}
}
