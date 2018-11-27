/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisSpecification.h>

#include <core/CLogger.h>
#include <core/Concurrency.h>

#include <api/CDataFrameOutliersRunner.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <boost/make_unique.hpp>

#include <cstring>
#include <iterator>
#include <thread>

namespace ml {
namespace api {
namespace {
// TODO These should be consistent with Java naming where relevant.
const char* ROWS{"rows"};
const char* COLS{"cols"};
const char* MEMORY_LIMIT{"memory_limit"};
const char* THREADS{"threads"};
const char* ANALYSIS{"analysis"};
const char* NAME{"name"};
const char* PARAMETERS{"parameters"};

const char* VALID_MEMBER_NAMES[]{ROWS, COLS, MEMORY_LIMIT, THREADS, ANALYSIS};

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

CDataFrameAnalysisSpecification::CDataFrameAnalysisSpecification(TRunnerFactoryUPtrVec runnerFactories,
                                                                 const std::string& jsonSpecification)
    : m_RunnerFactories{std::move(runnerFactories)} {
    rapidjson::Document document;
    if (document.Parse(jsonSpecification.c_str()) == false) {
        LOG_ERROR(<< "Failed to parse: '" << jsonSpecification << "'");
        m_Bad = true;
    } else {
        auto isPositiveInteger = [](const rapidjson::Value& value) {
            return value.IsUint() && value.GetUint() > 0;
        };
        auto registerFailure = [this, &document](const char* name) {
            if (document.HasMember(name)) {
                LOG_ERROR(<< "Internal error: bad value for '" << name
                          << "': " << toString(document[name]));
            } else {
                LOG_ERROR(<< "Internal error: missing '" << name << "'");
            }
            m_Bad = true;
        };

        if (document.HasMember(ROWS) && isPositiveInteger(document[ROWS])) {
            m_Rows = document[ROWS].GetUint();
        } else {
            registerFailure(ROWS);
        }
        if (document.HasMember(COLS) && isPositiveInteger(document[COLS])) {
            m_Cols = document[COLS].GetUint();
        } else {
            registerFailure(COLS);
        }
        if (document.HasMember(MEMORY_LIMIT) && isPositiveInteger(document[MEMORY_LIMIT])) {
            m_MemoryLimit = document[MEMORY_LIMIT].GetUint();
        } else {
            registerFailure(MEMORY_LIMIT);
        }
        if (document.HasMember(THREADS) && isPositiveInteger(document[THREADS])) {
            m_Threads = document[THREADS].GetUint();
        } else {
            registerFailure(THREADS);
        }

        if (document.HasMember(ANALYSIS) && document[ANALYSIS].IsObject()) {
            const auto& analysis = document[ANALYSIS];
            if (analysis.HasMember(NAME) && analysis[NAME].IsString()) {
                this->initializeRunner(analysis[NAME].GetString(), analysis);
            } else {
                registerFailure(NAME);
            }
        } else {
            registerFailure(ANALYSIS);
        }

        // Check for any unrecognised fields; these might be typos.
        for (auto i = document.MemberBegin(); i != document.MemberEnd(); ++i) {
            if (isValidMember(*i) == false) {
                LOG_ERROR(<< "Bad input: unexpected member '" << i->name.GetString() << "'")
                m_Bad = true;
            }
        }
    }

    if (this->bad() == false && m_Threads > 1) {
        core::startDefaultAsyncExecutor(m_Threads - 1);
    }
}

CDataFrameAnalysisSpecification::~CDataFrameAnalysisSpecification() {
    if (m_Threads > 1) {
        core::stopDefaultAsyncExecutor();
    }
}

bool CDataFrameAnalysisSpecification::bad() const {
    return m_Bad || m_Runner->bad();
}

std::size_t CDataFrameAnalysisSpecification::rows() const {
    return m_Rows;
}

std::size_t CDataFrameAnalysisSpecification::cols() const {
    return m_Cols;
}

std::size_t CDataFrameAnalysisSpecification::memoryLimit() const {
    return m_MemoryLimit;
}

std::size_t CDataFrameAnalysisSpecification::threads() const {
    return m_Threads;
}

CDataFrameAnalysisRunner* CDataFrameAnalysisSpecification::run(core::CDataFrame& frame) const {
    if (m_Runner != nullptr) {
        m_Runner->run(frame);
        return m_Runner.get();
    }
    return nullptr;
}

void CDataFrameAnalysisSpecification::initializeRunner(const char* name,
                                                       const rapidjson::Value& analysis) {
    // We pass of the interpretation of the parameters object to the appropriate
    // analysis runner.
    for (const auto& factory : m_RunnerFactories) {
        if (std::strcmp(factory->name(), name) == 0) {
            m_Runner = analysis.HasMember(PARAMETERS)
                           ? factory->make(*this, analysis[PARAMETERS])
                           : factory->make((*this));
            return;
        }
    }

    LOG_ERROR(<< "Internal error: unexpected value for 'name': '" << name << "'");
    m_Bad = true;
}
}
}
