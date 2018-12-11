/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameOutliersRunner.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <maths/CLocalOutlierFactors.h>

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
const char* NUMBER_NEIGHBOURS{"number_neighbours"};
const char* METHOD{"method"};
const char* LOF{"lof"};
const char* LDOF{"ldof"};
const char* DISTANCE_KTH_NN{"distance_kth_nn"};
const char* DISTANCE_KNN{"distance_knn"};
const char* VALID_MEMBER_NAMES[]{NUMBER_NEIGHBOURS, METHOD};

// Output
const std::string OUTLIER_SCORE{"outlier_score"};

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

    auto registerFailure = [this, &params](const char* name) {
        LOG_ERROR(<< "Bad input: '" << toString(params[name]) << "' for '" << name << "'");
        this->setToBad();
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
            m_Method.reset(static_cast<std::size_t>(maths::CLocalOutlierFactors::E_Lof));
        } else if (std::strcmp(LDOF, params[METHOD].GetString()) == 0) {
            m_Method.reset(static_cast<std::size_t>(maths::CLocalOutlierFactors::E_Ldof));
        } else if (std::strcmp(DISTANCE_KTH_NN, params[METHOD].GetString()) == 0) {
            m_Method.reset(static_cast<std::size_t>(maths::CLocalOutlierFactors::E_DistancekNN));
        } else if (std::strcmp(DISTANCE_KNN, params[METHOD].GetString()) == 0) {
            m_Method.reset(static_cast<std::size_t>(maths::CLocalOutlierFactors::E_TotalDistancekNN));
        } else {
            registerFailure(METHOD);
        }
    }

    // Check for any unrecognised fields; these might be typos.
    for (auto i = params.MemberBegin(); i != params.MemberEnd(); ++i) {
        if (isValidMember(*i) == false) {
            LOG_ERROR(<< "Bad input: unexpected member '" << i->name.GetString() << "'")
            this->setToBad();
        }
    }
}

CDataFrameOutliersRunner::CDataFrameOutliersRunner(const CDataFrameAnalysisSpecification& spec)
    : CDataFrameAnalysisRunner{spec}, m_NumberPartitions{1 /*FIXME*/} {
}

std::size_t CDataFrameOutliersRunner::numberOfPartitions() const {
    return m_NumberPartitions;
}

std::size_t CDataFrameOutliersRunner::numberExtraColumns() const {
    // Column for outlier score + explaining features TBD.
    return 1;
}

void CDataFrameOutliersRunner::writeOneRow(TRowRef row,
                                           core::CRapidJsonConcurrentLineWriter& outputWriter) const {
    std::size_t lastColumn{row.numberColumns() - 1};
    outputWriter.StartObject();
    outputWriter.Key(OUTLIER_SCORE);
    outputWriter.Double(row[lastColumn]);
    outputWriter.EndObject();
}

void CDataFrameOutliersRunner::runImpl(core::CDataFrame& frame) {
    maths::computeOutliers(this->spec().numberThreads(), frame);
}

const char* CDataFrameOutliersRunnerFactory::name() const {
    return "outliers";
}

CDataFrameOutliersRunnerFactory::TRunnerUPtr
CDataFrameOutliersRunnerFactory::make(const CDataFrameAnalysisSpecification& spec) const {
    return boost::make_unique<CDataFrameOutliersRunner>(spec);
}

CDataFrameOutliersRunnerFactory::TRunnerUPtr
CDataFrameOutliersRunnerFactory::make(const CDataFrameAnalysisSpecification& spec,
                                      const rapidjson::Value& params) const {
    return boost::make_unique<CDataFrameOutliersRunner>(spec, params);
}
}
}
