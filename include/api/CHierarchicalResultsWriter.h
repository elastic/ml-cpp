/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CHierarchicalResultsWriter_h
#define INCLUDED_ml_api_CHierarchicalResultsWriter_h

#include <core/CNonCopyable.h>
#include <core/CSmallVector.h>
#include <core/CStoredStringPtr.h>

#include <model/CAnomalyDetector.h>
#include <model/CHierarchicalResults.h>

#include <api/ImportExport.h>

#include <boost/optional.hpp>

#include <functional>

namespace ml {
namespace api {

//! \brief Writes out hierarchical results using a callback to write
//! individual results.
//!
//! DESCRIPTION:\n
//! Implements a hierarchical results object visitor which writes out
//! the results stored in a CHierarchicalResults object.
//!
//! For each node one or more CAnomalyDetector::SResults objects are
//! constructed and written by the callback supplied to the constructor.
class API_EXPORT CHierarchicalResultsWriter : public model::CHierarchicalResultsVisitor,
                                              private core::CNonCopyable {
public:
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TOptionalDouble = boost::optional<double>;
    using TOptionalUInt64 = boost::optional<uint64_t>;

    // Influencers
    using TStoredStringPtrVec = std::vector<core::CStoredStringPtr>;
    using TStoredStringPtrStoredStringPtrPr =
        std::pair<core::CStoredStringPtr, core::CStoredStringPtr>;
    using TStoredStringPtrStoredStringPtrPrDoublePr =
        std::pair<TStoredStringPtrStoredStringPtrPr, double>;
    using TStoredStringPtrStoredStringPtrPrDoublePrVec =
        std::vector<TStoredStringPtrStoredStringPtrPrDoublePr>;

    using TStr1Vec = core::CSmallVector<std::string, 1>;

public:
    enum EResultType { E_SimpleCountResult, E_PopulationResult, E_Result };
    //! Type which wraps up the results of anomaly detection.
    struct API_EXPORT SResults {
        //! Construct for population results
        SResults(bool isAllTimeResult,
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
                 core_t::TTime bucketSpan);

        //! Construct for other results
        SResults(EResultType resultType,
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
                 TStr1Vec scheduledEventDescriptions);

        EResultType s_ResultType;
        bool s_IsAllTimeResult;
        bool s_IsOverallResult;
        bool s_UseNull;
        bool s_IsMetric;
        const std::string& s_PartitionFieldName;
        const std::string& s_PartitionFieldValue;
        const std::string& s_ByFieldName;
        const std::string& s_ByFieldValue;
        const std::string& s_CorrelatedByFieldValue;
        const std::string& s_OverFieldName;
        const std::string& s_OverFieldValue;
        const std::string& s_MetricValueField;
        core_t::TTime s_BucketStartTime;
        core_t::TTime s_BucketSpan;
        const std::string& s_FunctionName;
        const std::string& s_FunctionDescription;
        TDouble1Vec s_FunctionValue;
        TDouble1Vec s_PopulationAverage;
        TOptionalDouble s_BaselineRate;
        TOptionalUInt64 s_CurrentRate;
        TDouble1Vec s_BaselineMean;
        TDouble1Vec s_CurrentMean;
        double s_RawAnomalyScore;
        double s_NormalizedAnomalyScore;
        double s_Probability;
        const TStoredStringPtrStoredStringPtrPrDoublePrVec& s_Influences;
        int s_Identifier;
        TStr1Vec s_ScheduledEventDescriptions;
    };

public:
    using TResults = SResults;
    using TResultWriterFunc = std::function<bool(TResults)>;
    using TPivotWriterFunc = std::function<bool(core_t::TTime, TNode, bool)>;

public:
    CHierarchicalResultsWriter(const model::CLimits& limits,
                               const TResultWriterFunc& resultWriter,
                               const TPivotWriterFunc& pivotsWriterFunc);

    //! Write \p node.
    virtual void visit(const model::CHierarchicalResults& results, const TNode& node, bool pivot);

private:
    //! Write out a population person result if \p node is a
    //! member a population.
    void writePopulationResult(const model::CHierarchicalResults& results, const TNode& node);

    //! Write out an individual person result if \p node is
    //! an individual time series result.
    void writeIndividualResult(const model::CHierarchicalResults& results, const TNode& node);

    //! Write out the pivot (influencer) result if \p node is a
    //! pivot.
    void writePivotResult(const model::CHierarchicalResults& results, const TNode& node);

    //! Write out a simple count result if \p node is simple
    //! count.
    void writeSimpleCountResult(const TNode& node);

    //! Given a leaf node, search upwards to find the most appropriate
    //! values for person and partition probability results.
    static void findParentProbabilities(const TNode& node,
                                        double& personProbability,
                                        double& partitionProbability);

private:
    //! The various limits.
    const model::CLimits& m_Limits;

    //! The results writer.
    TResultWriterFunc m_ResultWriterFunc;

    //! The influencers/pivots writer
    TPivotWriterFunc m_PivotWriterFunc;

    //! Remember the current bucket time for writing pivots
    core_t::TTime m_BucketTime;
};
}
}

#endif // INCLUDED_ml_api_CHierarchicalResultsWriter_h
