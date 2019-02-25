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

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameOutliersRunner.h>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>

#include <boost/filesystem.hpp>

#include <cstring>
#include <iterator>
#include <memory>
#include <thread>

namespace ml {
namespace api {
namespace {
using TRunnerFactoryUPtrVec = ml::api::CDataFrameAnalysisSpecification::TRunnerFactoryUPtrVec;

TRunnerFactoryUPtrVec analysisFactories() {
    TRunnerFactoryUPtrVec factories;
    factories.push_back(std::make_unique<ml::api::CDataFrameOutliersRunnerFactory>());
    // Add new analysis types here.
    return factories;
}

// These must be consistent with Java names.
const char* const ROWS{"rows"};
const char* const COLS{"cols"};
const char* const MEMORY_LIMIT{"memory_limit"};
const char* const THREADS{"threads"};
const char* const TEMPORARY_DIRECTORY{"temp_dir"};
const char* const ANALYSIS{"analysis"};
const char* const NAME{"name"};
const char* const PARAMETERS{"parameters"};

const CDataFrameAnalysisConfigReader CONFIG_READER{[] {
    CDataFrameAnalysisConfigReader theReader;
    theReader.addParameter(ROWS, CDataFrameAnalysisConfigReader::E_RequiredParameter);
    theReader.addParameter(COLS, CDataFrameAnalysisConfigReader::E_RequiredParameter);
    theReader.addParameter(MEMORY_LIMIT, CDataFrameAnalysisConfigReader::E_RequiredParameter);
    theReader.addParameter(THREADS, CDataFrameAnalysisConfigReader::E_RequiredParameter);
    // TODO required
    theReader.addParameter(TEMPORARY_DIRECTORY,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(ANALYSIS, CDataFrameAnalysisConfigReader::E_RequiredParameter);
    return theReader;
}()};

const CDataFrameAnalysisConfigReader ANALYSIS_READER{[] {
    CDataFrameAnalysisConfigReader theReader;
    theReader.addParameter(NAME, CDataFrameAnalysisConfigReader::E_RequiredParameter);
    theReader.addParameter(PARAMETERS, CDataFrameAnalysisConfigReader::E_OptionalParameter);
    return theReader;
}()};
}

CDataFrameAnalysisSpecification::CDataFrameAnalysisSpecification(const std::string& jsonSpecification)
    : CDataFrameAnalysisSpecification{analysisFactories(), jsonSpecification} {
}

CDataFrameAnalysisSpecification::CDataFrameAnalysisSpecification(TRunnerFactoryUPtrVec runnerFactories,
                                                                 const std::string& jsonSpecification)
    : m_RunnerFactories{std::move(runnerFactories)} {

    rapidjson::Document specification;
    if (specification.Parse(jsonSpecification.c_str()) == false) {
        HANDLE_FATAL(<< "Input error: failed to parse analysis specification '"
                     << jsonSpecification << "'. Please report this problem.");
    } else {

        auto parameters = CONFIG_READER.read(specification);

        for (auto name : {ROWS, COLS, MEMORY_LIMIT, THREADS}) {
            if (parameters[name].as<std::size_t>() == 0) {
                HANDLE_FATAL(<< "Input error: '" << name << "' must be non-zero");
            }
        }
        m_NumberRows = parameters[ROWS].as<std::size_t>();
        m_NumberColumns = parameters[COLS].as<std::size_t>();
        m_MemoryLimit = parameters[MEMORY_LIMIT].as<std::size_t>();
        m_NumberThreads = parameters[THREADS].as<std::size_t>();
        m_TemporaryDirectory = parameters[TEMPORARY_DIRECTORY].fallback(boost::filesystem::current_path().string());

        auto jsonAnalysis = parameters[ANALYSIS].jsonObject();
        if (jsonAnalysis != nullptr) {
            this->initializeRunner(*jsonAnalysis);
        }
    }
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

CDataFrameAnalysisSpecification::TDataFrameUPtrTemporaryDirectoryPtrPr
CDataFrameAnalysisSpecification::makeDataFrame() {
    if (m_Runner == nullptr) {
        return {};
    }

    auto result = m_Runner->storeDataFrameInMainMemory()
                      ? core::makeMainStorageDataFrame(m_NumberColumns)
                      : core::makeDiskStorageDataFrame(m_TemporaryDirectory,
                                                       m_NumberColumns, m_NumberRows);
    result.first->reserve(m_NumberThreads, m_NumberColumns + this->numberExtraColumns());

    return result;
}

CDataFrameAnalysisRunner* CDataFrameAnalysisSpecification::run(core::CDataFrame& frame) const {
    if (m_Runner != nullptr) {
        m_Runner->run(frame);
        return m_Runner.get();
    }
    return nullptr;
}

void CDataFrameAnalysisSpecification::initializeRunner(const rapidjson::Value& jsonAnalysis) {
    // We pass of the interpretation of the parameters object to the appropriate
    // analysis runner.

    auto analysis = ANALYSIS_READER.read(jsonAnalysis);

    std::string name{analysis[NAME].as<std::string>()};

    for (const auto& factory : m_RunnerFactories) {
        if (name == factory->name()) {
            auto parameters = analysis[PARAMETERS].jsonObject();
            m_Runner = parameters != nullptr
                           ? factory->make(*this, *parameters)
                           : factory->make(*this);
            return;
        }
    }

    HANDLE_FATAL(<< "Input error: unexpected analysis name '" << name
                 << "'. Please report this problem.");
}
}
}
