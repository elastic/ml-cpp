/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisSpecification.h>

#include <core/CDataFrame.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CRapidJsonLineWriter.h>

#include <api/CDataFrameOutliersRunner.h>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>

#include <boost/filesystem/path.hpp>
#include <boost/make_unique.hpp>

#include <cstring>
#include <iterator>
#include <thread>

namespace ml {
namespace api {
namespace {
using TRunnerFactoryUPtrVec = ml::api::CDataFrameAnalysisSpecification::TRunnerFactoryUPtrVec;

TRunnerFactoryUPtrVec analysisFactories() {
    TRunnerFactoryUPtrVec factories;
    factories.push_back(boost::make_unique<ml::api::CDataFrameOutliersRunnerFactory>());
    // Add new analysis types here.
    return factories;
}

// These must be consistent with Java names.
const char* ROWS{"rows"};
const char* COLS{"cols"};
const char* MEMORY_LIMIT{"memory_limit"};
const char* THREADS{"threads"};
const char* TEMPORARY_DIRECTORY{"temp_dir"};
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
    std::ostringstream valueAsString;
    rapidjson::OStreamWrapper shim{valueAsString};
    ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper> writer{shim};
    writer.write(value);
    return valueAsString.str();
}
}

CDataFrameAnalysisSpecification::CDataFrameAnalysisSpecification(const std::string& jsonSpecification)
    : CDataFrameAnalysisSpecification{analysisFactories(), jsonSpecification} {
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
            m_NumberRows = document[ROWS].GetUint();
        } else {
            registerFailure(ROWS);
        }
        if (document.HasMember(COLS) && isPositiveInteger(document[COLS])) {
            m_NumberColumns = document[COLS].GetUint();
        } else {
            registerFailure(COLS);
        }
        if (document.HasMember(MEMORY_LIMIT) && isPositiveInteger(document[MEMORY_LIMIT])) {
            m_MemoryLimit = document[MEMORY_LIMIT].GetUint();
        } else {
            registerFailure(MEMORY_LIMIT);
        }
        if (document.HasMember(THREADS) && isPositiveInteger(document[THREADS])) {
            m_NumberThreads = document[THREADS].GetUint();
        } else {
            registerFailure(THREADS);
        }
        // TODO Remove hack when being passed.
        if (false) {
            if (document.HasMember(TEMPORARY_DIRECTORY) &&
                document[TEMPORARY_DIRECTORY].IsString() &&
                boost::filesystem::portable_name(document[TEMPORARY_DIRECTORY].GetString())) {
                m_TemporaryDirectory = document[TEMPORARY_DIRECTORY].GetString();
            } else {
                registerFailure(TEMPORARY_DIRECTORY);
            }
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
}

bool CDataFrameAnalysisSpecification::bad() const {
    return m_Bad || m_Runner->bad();
}

std::size_t CDataFrameAnalysisSpecification::numberRows() const {
    return m_NumberRows;
}

std::size_t CDataFrameAnalysisSpecification::numberColumns() const {
    return m_NumberColumns;
}

std::size_t CDataFrameAnalysisSpecification::numberExtraColumns() const {
    return m_Runner != nullptr ? m_Runner->numberExtraColumns() : 0;
}

std::size_t CDataFrameAnalysisSpecification::memoryLimit() const {
    return m_MemoryLimit;
}

std::size_t CDataFrameAnalysisSpecification::numberThreads() const {
    return m_NumberThreads;
}

CDataFrameAnalysisSpecification::TDataFrameUPtr
CDataFrameAnalysisSpecification::makeDataFrame() const {
    if (m_Bad) {
        return {};
    }

    // TODO Remove hack when passing directory in config.
    if (m_Runner->storeDataFrameInMainMemory()) {
        return {};
    }

    TDataFrameUPtr result{m_Runner->storeDataFrameInMainMemory()
                              ? core::makeMainStorageDataFrame(m_NumberColumns)
                              : core::makeDiskStorageDataFrame(
                                    m_TemporaryDirectory, m_NumberColumns, m_NumberRows)};
    result->reserve(m_NumberThreads, m_NumberColumns + this->numberExtraColumns());

    return result;
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
