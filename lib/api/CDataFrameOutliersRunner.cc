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
// Configuration
const std::string STANDARDIZE_COLUMNS{"standardize_columns"};
const std::string N_NEIGHBORS{"n_neighbors"};
const std::string METHOD{"method"};
const std::string COMPUTE_FEATURE_INFLUENCE{"compute_feature_influence"};
const std::string FEATURE_INFLUENCE_THRESHOLD{"feature_influence_threshold"};
const std::string OUTLIER_FRACTION{"outlier_fraction"};

const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
    const std::string lof{"lof"};
    const std::string ldof{"ldof"};
    const std::string knn{"distance_kth_nn"};
    const std::string tnn{"distance_knn"};
    CDataFrameAnalysisConfigReader theReader;
    theReader.addParameter(STANDARDIZE_COLUMNS,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(N_NEIGHBORS, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(METHOD, CDataFrameAnalysisConfigReader::E_OptionalParameter,
                           {{lof, int{maths::COutliers::E_Lof}},
                            {ldof, int{maths::COutliers::E_Ldof}},
                            {knn, int{maths::COutliers::E_DistancekNN}},
                            {tnn, int{maths::COutliers::E_TotalDistancekNN}}});
    theReader.addParameter(COMPUTE_FEATURE_INFLUENCE,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(FEATURE_INFLUENCE_THRESHOLD,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(OUTLIER_FRACTION, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    return theReader;
}()};

// Output
const std::string OUTLIER_SCORE_FIELD_NAME{"outlier_score"};
const std::string FEATURE_INFLUENCE_FIELD_NAME_PREFIX{"feature_influence."};
}

CDataFrameOutliersRunner::CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec,
                                                   const rapidjson::Value& jsonParameters)
    : CDataFrameOutliersRunner{spec} {

    auto parameters = PARAMETER_READER.read(jsonParameters);
    m_StandardizeColumns = parameters[STANDARDIZE_COLUMNS].fallback(true);
    m_NumberNeighbours = parameters[N_NEIGHBORS].fallback(std::size_t{0});
    m_Method = parameters[METHOD].fallback(maths::COutliers::E_Ensemble);
    m_ComputeFeatureInfluence = parameters[COMPUTE_FEATURE_INFLUENCE].fallback(true);
    m_FeatureInfluenceThreshold = parameters[FEATURE_INFLUENCE_THRESHOLD].fallback(0.1);
    m_OutlierFraction = parameters[OUTLIER_FRACTION].fallback(0.05);
}

CDataFrameOutliersRunner::CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec)
    : CDataFrameAnalysisRunner{spec}, m_Method{static_cast<std::size_t>(
                                          maths::COutliers::E_Ensemble)} {
}

std::size_t CDataFrameOutliersRunner::numberExtraColumns() const {
    return m_ComputeFeatureInfluence ? this->spec().numberColumns() + 1 : 1;
}

void CDataFrameOutliersRunner::writeOneRow(const TStrVec& featureNames,
                                           TRowRef row,
                                           core::CRapidJsonConcurrentLineWriter& writer) const {
    std::size_t scoreColumn{row.numberColumns() - this->numberExtraColumns()};
    std::size_t beginFeatureScoreColumns{scoreColumn + 1};
    std::size_t numberFeatureScoreColumns{this->numberExtraColumns() - 1};
    writer.StartObject();
    writer.Key(OUTLIER_SCORE_FIELD_NAME);
    writer.Double(row[scoreColumn]);
    if (row[scoreColumn] > m_FeatureInfluenceThreshold) {
        for (std::size_t i = 0; i < numberFeatureScoreColumns; ++i) {
            writer.Key(FEATURE_INFLUENCE_FIELD_NAME_PREFIX + featureNames[i]);
            writer.Double(row[beginFeatureScoreColumns + i]);
        }
    }
    writer.EndObject();
}

void CDataFrameOutliersRunner::runImpl(const TStrVec&, core::CDataFrame& frame) {

    core::CProgramCounters::counter(counter_t::E_DFONumberPartitions) =
        this->numberPartitions();
    core::CProgramCounters::counter(counter_t::E_DFOEstimatedPeakMemoryUsage) =
        this->estimateMemoryUsage(frame.numberRows(),
                                  frame.numberRows() / this->numberPartitions(),
                                  frame.numberColumns());

    maths::COutliers::SComputeParameters params{this->spec().numberThreads(),
                                                this->numberPartitions(),
                                                m_StandardizeColumns,
                                                static_cast<maths::COutliers::EMethod>(m_Method),
                                                m_NumberNeighbours,
                                                m_ComputeFeatureInfluence,
                                                m_OutlierFraction};
    std::atomic<std::int64_t> memory{0};
    maths::COutliers::compute(
        params, frame, this->progressRecorder(), [&memory](std::int64_t delta) {
            std::int64_t memory_{memory.fetch_add(delta)};
            core::CProgramCounters::counter(counter_t::E_DFOPeakMemoryUsage).max(memory_);
        });
}

std::size_t
CDataFrameOutliersRunner::estimateBookkeepingMemoryUsage(std::size_t numberPartitions,
                                                         std::size_t totalNumberRows,
                                                         std::size_t partitionNumberRows,
                                                         std::size_t numberColumns) const {
    maths::COutliers::SComputeParameters params{this->spec().numberThreads(),
                                                numberPartitions,
                                                m_StandardizeColumns,
                                                static_cast<maths::COutliers::EMethod>(m_Method),
                                                m_NumberNeighbours,
                                                m_ComputeFeatureInfluence,
                                                m_OutlierFraction};
    return maths::COutliers::estimateMemoryUsedByCompute(
        params, totalNumberRows, partitionNumberRows, numberColumns);
}

const std::string& CDataFrameOutliersRunnerFactory::name() const {
    return NAME;
}

CDataFrameOutliersRunnerFactory::TRunnerUPtr
CDataFrameOutliersRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec) const {
    return std::make_unique<CDataFrameOutliersRunner>(spec);
}

CDataFrameOutliersRunnerFactory::TRunnerUPtr
CDataFrameOutliersRunnerFactory::makeImpl(const CDataFrameAnalysisSpecification& spec,
                                          const rapidjson::Value& params) const {
    return std::make_unique<CDataFrameOutliersRunner>(spec, params);
}

const std::string CDataFrameOutliersRunnerFactory::NAME{"outlier_detection"};
}
}
