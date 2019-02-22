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

#include <api/CDataFrameAnalysisSpecification.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <boost/make_unique.hpp>

#include <algorithm>
#include <cstring>
#include <iterator>

namespace ml {
namespace api {
namespace {
// Configuration
const char* const NUMBER_NEIGHBOURS{"number_neighbours"};
const char* const METHOD{"method"};
const char* const LOF{"lof"};
const char* const LDOF{"ldof"};
const char* const DISTANCE_KTH_NN{"distance_kth_nn"};
const char* const DISTANCE_KNN{"distance_knn"};
const char* const VALID_MEMBER_NAMES[]{NUMBER_NEIGHBOURS, METHOD};

// Output
const char* const OUTLIER_SCORE{"outlier_score"};
const char* const FEATURE_INFLUENCE_PREFIX{"feature_influence."};

template<typename MEMBER>
bool isValidMember(const MEMBER& member) {
    return std::find_if(std::begin(VALID_MEMBER_NAMES),
                        std::end(VALID_MEMBER_NAMES), [&member](const char* name) {
                            return std::strcmp(name, member.name.GetString()) == 0;
                        }) != std::end(VALID_MEMBER_NAMES);
}

std::string toString(const rapidjson::Value& value) {
    rapidjson::StringBuffer valueAsString;
    rapidjson::Writer<rapidjson::StringBuffer> writer(valueAsString);
    value.Accept(writer);
    return valueAsString.GetString();
}
}

CDataFrameOutliersRunner::CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec,
                                                   const rapidjson::Value& params)
    : CDataFrameOutliersRunner{spec} {

    auto registerFailure = [&params](const char* name) {
        HANDLE_FATAL(<< "Input error: bad value '" << toString(params[name])
                     << "' for '" << name << "'.");
    };

    if (params.HasMember(NUMBER_NEIGHBOURS)) {
        if (params[NUMBER_NEIGHBOURS].IsUint() && params[NUMBER_NEIGHBOURS].GetUint() > 0) {
            m_NumberNeighbours.reset(params[NUMBER_NEIGHBOURS].GetUint());
        } else {
            registerFailure(NUMBER_NEIGHBOURS);
        }
    }

    if (params.HasMember(METHOD)) {
        if (std::strcmp(LOF, params[METHOD].GetString()) == 0) {
            m_Method = static_cast<std::size_t>(maths::COutliers::E_Lof);
        } else if (std::strcmp(LDOF, params[METHOD].GetString()) == 0) {
            m_Method = static_cast<std::size_t>(maths::COutliers::E_Ldof);
        } else if (std::strcmp(DISTANCE_KTH_NN, params[METHOD].GetString()) == 0) {
            m_Method = static_cast<std::size_t>(maths::COutliers::E_DistancekNN);
        } else if (std::strcmp(DISTANCE_KNN, params[METHOD].GetString()) == 0) {
            m_Method = static_cast<std::size_t>(maths::COutliers::E_TotalDistancekNN);
        } else {
            registerFailure(METHOD);
        }
    }

    // Check for any unrecognised fields; these might be typos.
    for (auto i = params.MemberBegin(); i != params.MemberEnd(); ++i) {
        if (isValidMember(*i) == false) {
            HANDLE_FATAL(<< "Input error: unexpected member '"
                         << i->name.GetString() << "'. Please report this problem.")
        }
    }
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
    if (row[scoreColumn] > m_WriteFeatureInfluenceMinimumScore) {
        for (std::size_t i = 0; i < numberFeatureScoreColumns; ++i) {
            writer.Key(FEATURE_INFLUENCE_PREFIX + featureNames[i]);
            writer.Double(row[beginFeatureScoreColumns + i]);
        }
    }
    writer.EndObject();
}

void CDataFrameOutliersRunner::runImpl(core::CDataFrame& frame) {
    maths::COutliers::SComputeParameters params{
        this->spec().numberThreads(), this->numberPartitions(),
        m_ComputeFeatureInfluence, m_ProbabilityOutlier};

    maths::COutliers::compute(params, frame, this->progressRecorder());
}

std::size_t
CDataFrameOutliersRunner::estimateBookkeepingMemoryUsage(std::size_t numberPartitions,
                                                         std::size_t totalNumberRows,
                                                         std::size_t partitionNumberRows,
                                                         std::size_t numberColumns) const {
    std::size_t result{0};
    switch (numberPartitions) {
    case 1:
        result = estimateMemoryUsage<maths::CMemoryMappedDenseVector<maths::CFloatStorage>>(
            totalNumberRows, partitionNumberRows, numberColumns);
        break;
    default:
        result = estimateMemoryUsage<maths::CDenseVector<maths::CFloatStorage>>(
            totalNumberRows, partitionNumberRows, numberColumns);
        break;
    }
    return result;
}

template<typename POINT>
std::size_t
CDataFrameOutliersRunner::estimateMemoryUsage(std::size_t totalNumberRows,
                                              std::size_t partitionNumberRows,
                                              std::size_t numberColumns) const {
    maths::COutliers::EMethod method{static_cast<maths::COutliers::EMethod>(m_Method)};
    return m_NumberNeighbours != boost::none
               ? maths::COutliers::estimateMemoryUsedByCompute<POINT>(
                     method, *m_NumberNeighbours, m_ComputeFeatureInfluence,
                     totalNumberRows, partitionNumberRows, numberColumns)
               : maths::COutliers::estimateMemoryUsedByCompute<POINT>(
                     m_ComputeFeatureInfluence, totalNumberRows,
                     partitionNumberRows, numberColumns);
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
