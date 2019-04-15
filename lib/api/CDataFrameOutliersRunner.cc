/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameOutliersRunner.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
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
const char* const STANDARDIZE_COLUMNS{"standardize_columns"};
const char* const K{"k"};
const char* const METHOD{"method"};
const char* const COMPUTE_FEATURE_INFLUENCE{"compute_feature_influence"};
const char* const MINIMUM_SCORE_TO_WRITE_FEATURE_INFLUENCE{"minimum_score_to_write_feature_influence"};
const char* const OUTLIER_FRACTION{"outlier_fraction"};

const CDataFrameAnalysisConfigReader PARAMETER_READER{[] {
    const std::string lof{"lof"};
    const std::string ldof{"ldof"};
    const std::string knn{"knn"};
    const std::string tnn{"tnn"};
    CDataFrameAnalysisConfigReader theReader;
    theReader.addParameter(STANDARDIZE_COLUMNS,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(K, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(METHOD, CDataFrameAnalysisConfigReader::E_OptionalParameter,
                           {{lof, int{maths::COutliers::E_Lof}},
                            {ldof, int{maths::COutliers::E_Ldof}},
                            {knn, int{maths::COutliers::E_DistancekNN}},
                            {tnn, int{maths::COutliers::E_TotalDistancekNN}}});
    theReader.addParameter(COMPUTE_FEATURE_INFLUENCE,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(MINIMUM_SCORE_TO_WRITE_FEATURE_INFLUENCE,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(OUTLIER_FRACTION, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    return theReader;
}()};

// Output
const char* const OUTLIER_SCORE{"outlier_score"};
const char* const FEATURE_INFLUENCE_PREFIX{"feature_influence."};
}

CDataFrameOutliersRunner::CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec,
                                                   const rapidjson::Value& jsonParameters)
    : CDataFrameOutliersRunner{spec} {

    auto parameters = PARAMETER_READER.read(jsonParameters);
    m_StandardizeColumns = parameters[STANDARDIZE_COLUMNS].fallback(true);
    m_NumberNeighbours = parameters[K].fallback(std::size_t{0});
    m_Method = parameters[METHOD].fallback(maths::COutliers::E_Ensemble);
    m_ComputeFeatureInfluence = parameters[COMPUTE_FEATURE_INFLUENCE].fallback(true);
    m_MinimumScoreToWriteFeatureInfluence =
        parameters[MINIMUM_SCORE_TO_WRITE_FEATURE_INFLUENCE].fallback(0.1);
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
    writer.Key(OUTLIER_SCORE);
    writer.Double(row[scoreColumn]);
    if (row[scoreColumn] > m_MinimumScoreToWriteFeatureInfluence) {
        for (std::size_t i = 0; i < numberFeatureScoreColumns; ++i) {
            writer.Key(FEATURE_INFLUENCE_PREFIX + featureNames[i]);
            writer.Double(row[beginFeatureScoreColumns + i]);
        }
    }
    writer.EndObject();
}

void CDataFrameOutliersRunner::runImpl(core::CDataFrame& frame) {
    maths::COutliers::SComputeParameters params{this->spec().numberThreads(),
                                                this->numberPartitions(),
                                                m_StandardizeColumns,
                                                static_cast<maths::COutliers::EMethod>(m_Method),
                                                m_NumberNeighbours,
                                                m_ComputeFeatureInfluence,
                                                m_OutlierFraction};
    maths::COutliers::compute(params, frame, this->progressRecorder());
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

const char* CDataFrameOutliersRunnerFactory::name() const {
    return "outlier_detection";
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
}
}
