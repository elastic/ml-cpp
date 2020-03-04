/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisSpecification.h>

#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameOutliersRunner.h>
#include <api/CDataFrameTrainBoostedTreeClassifierRunner.h>
#include <api/CDataFrameTrainBoostedTreeRegressionRunner.h>
#include <api/CMemoryUsageEstimationResultJsonWriter.h>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>

#include <boost/filesystem.hpp>

#include <cstring>
#include <iterator>
#include <memory>

namespace ml {
namespace api {

// These must be consistent with Java names.
const std::string CDataFrameAnalysisSpecification::JOB_ID{"job_id"};
const std::string CDataFrameAnalysisSpecification::ROWS{"rows"};
const std::string CDataFrameAnalysisSpecification::COLS{"cols"};
const std::string CDataFrameAnalysisSpecification::MEMORY_LIMIT{"memory_limit"};
const std::string CDataFrameAnalysisSpecification::THREADS{"threads"};
const std::string CDataFrameAnalysisSpecification::TEMPORARY_DIRECTORY{"temp_dir"};
const std::string CDataFrameAnalysisSpecification::RESULTS_FIELD{"results_field"};
const std::string CDataFrameAnalysisSpecification::MISSING_FIELD_VALUE{"missing_field_value"};
const std::string CDataFrameAnalysisSpecification::CATEGORICAL_FIELD_NAMES{"categorical_fields"};
const std::string CDataFrameAnalysisSpecification::DISK_USAGE_ALLOWED{"disk_usage_allowed"};
const std::string CDataFrameAnalysisSpecification::ANALYSIS{"analysis"};
const std::string CDataFrameAnalysisSpecification::NAME{"name"};
const std::string CDataFrameAnalysisSpecification::PARAMETERS{"parameters"};

namespace {
using TBoolVec = std::vector<bool>;
using TRunnerFactoryUPtrVec = CDataFrameAnalysisSpecification::TRunnerFactoryUPtrVec;

TRunnerFactoryUPtrVec analysisFactories() {
    TRunnerFactoryUPtrVec factories;
    factories.push_back(std::make_unique<CDataFrameOutliersRunnerFactory>());
    factories.push_back(
        std::make_unique<CDataFrameTrainBoostedTreeRegressionRunnerFactory>());
    factories.push_back(
        std::make_unique<CDataFrameTrainBoostedTreeClassifierRunnerFactory>());
    // Add new analysis types here.
    return factories;
}

const std::string DEFAULT_RESULT_FIELD("ml");
const bool DEFAULT_DISK_USAGE_ALLOWED(false);

const CDataFrameAnalysisConfigReader CONFIG_READER{[] {
    CDataFrameAnalysisConfigReader theReader;
    theReader.addParameter(CDataFrameAnalysisSpecification::JOB_ID,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::ROWS,
                           CDataFrameAnalysisConfigReader::E_RequiredParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::COLS,
                           CDataFrameAnalysisConfigReader::E_RequiredParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::MEMORY_LIMIT,
                           CDataFrameAnalysisConfigReader::E_RequiredParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::THREADS,
                           CDataFrameAnalysisConfigReader::E_RequiredParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::TEMPORARY_DIRECTORY,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::RESULTS_FIELD,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::MISSING_FIELD_VALUE,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::CATEGORICAL_FIELD_NAMES,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::DISK_USAGE_ALLOWED,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::ANALYSIS,
                           CDataFrameAnalysisConfigReader::E_RequiredParameter);
    return theReader;
}()};

const CDataFrameAnalysisConfigReader ANALYSIS_READER{[] {
    CDataFrameAnalysisConfigReader theReader;
    theReader.addParameter(CDataFrameAnalysisSpecification::NAME,
                           CDataFrameAnalysisConfigReader::E_RequiredParameter);
    theReader.addParameter(CDataFrameAnalysisSpecification::PARAMETERS,
                           CDataFrameAnalysisConfigReader::E_OptionalParameter);
    return theReader;
}()};
}

CDataFrameAnalysisSpecification::CDataFrameAnalysisSpecification(
    const std::string& jsonSpecification,
    TPersisterSupplier persisterSupplier,
    TRestoreSearcherSupplier restoreSearcherSupplier)
    : CDataFrameAnalysisSpecification{analysisFactories(), jsonSpecification,
                                      std::move(persisterSupplier),
                                      std::move(restoreSearcherSupplier)} {
}

CDataFrameAnalysisSpecification::CDataFrameAnalysisSpecification(
    TRunnerFactoryUPtrVec runnerFactories,
    const std::string& jsonSpecification,
    TPersisterSupplier persisterSupplier,
    TRestoreSearcherSupplier restoreSearcherSupplier)
    : m_RunnerFactories{std::move(runnerFactories)}, m_PersisterSupplier{std::move(persisterSupplier)},
      m_RestoreSearcherSupplier{std::move(restoreSearcherSupplier)} {

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
        m_TemporaryDirectory = parameters[TEMPORARY_DIRECTORY].fallback(std::string{});
        m_JobId = parameters[JOB_ID].fallback(std::string{});
        m_ResultsField = parameters[RESULTS_FIELD].fallback(DEFAULT_RESULT_FIELD);
        m_MissingFieldValue = parameters[MISSING_FIELD_VALUE].fallback(
            core::CDataFrame::DEFAULT_MISSING_STRING);
        m_CategoricalFieldNames = parameters[CATEGORICAL_FIELD_NAMES].fallback(TStrVec{});
        m_DiskUsageAllowed = parameters[DISK_USAGE_ALLOWED].fallback(DEFAULT_DISK_USAGE_ALLOWED);

        double missing;
        if (m_MissingFieldValue != core::CDataFrame::DEFAULT_MISSING_STRING &&
            core::CStringUtils::stringToTypeSilent(m_MissingFieldValue, missing)) {
            HANDLE_FATAL(<< "Input error: you can't use a number (" << missing
                         << ") to denote a missing field value.")
        }
        if (m_DiskUsageAllowed && m_TemporaryDirectory.empty()) {
            HANDLE_FATAL(<< "Input error: temporary directory path should be explicitly set if disk"
                            " usage is allowed! Please report this problem.")
        }

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

const std::string& CDataFrameAnalysisSpecification::resultsField() const {
    return m_ResultsField;
}

const CDataFrameAnalysisSpecification::TStrVec&
CDataFrameAnalysisSpecification::categoricalFieldNames() const {
    return m_CategoricalFieldNames;
}

bool CDataFrameAnalysisSpecification::diskUsageAllowed() const {
    return m_DiskUsageAllowed;
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
    result.first->missingString(m_MissingFieldValue);
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

void CDataFrameAnalysisSpecification::estimateMemoryUsage(CMemoryUsageEstimationResultJsonWriter& writer) const {
    if (m_Runner == nullptr) {
        HANDLE_FATAL(<< "Internal error: no runner available so can't estimate memory."
                     << " Please report this problem.");
        return;
    }
    m_Runner->estimateMemoryUsage(writer);
}

TBoolVec CDataFrameAnalysisSpecification::columnsForWhichEmptyIsMissing(const TStrVec& fieldNames) const {
    if (m_Runner == nullptr) {
        HANDLE_FATAL(<< "Internal error: no runner available. Please report this problem.");
        return TBoolVec(fieldNames.size(), false);
    }
    return m_Runner->columnsForWhichEmptyIsMissing(fieldNames);
}

void CDataFrameAnalysisSpecification::initializeRunner(const rapidjson::Value& jsonAnalysis) {
    // We pass of the interpretation of the parameters object to the appropriate
    // analysis runner.

    auto analysis = ANALYSIS_READER.read(jsonAnalysis);

    m_AnalysisName = analysis[NAME].as<std::string>();
    LOG_TRACE(<< "Parsed analysis with name '" << m_AnalysisName << "'");

    for (const auto& factory : m_RunnerFactories) {
        if (m_AnalysisName == factory->name()) {
            auto parameters = analysis[PARAMETERS].jsonObject();
            m_Runner = parameters != nullptr ? factory->make(*this, *parameters)
                                             : factory->make(*this);
            return;
        }
    }

    HANDLE_FATAL(<< "Input error: unexpected analysis name '" << m_AnalysisName
                 << "'. Please report this problem.");
}

CDataFrameAnalysisSpecification::TDataAdderUPtr
CDataFrameAnalysisSpecification::persister() const {
    return m_PersisterSupplier();
}

CDataFrameAnalysisSpecification::TDataSearcherUPtr
CDataFrameAnalysisSpecification::restoreSearcher() const {
    return m_RestoreSearcherSupplier();
}

CDataFrameAnalysisSpecification::TDataAdderUPtr
CDataFrameAnalysisSpecification::noopPersisterSupplier() {
    return nullptr;
}

CDataFrameAnalysisSpecification::TDataSearcherUPtr
CDataFrameAnalysisSpecification::noopRestoreSearcherSupplier() {
    return nullptr;
}

const std::string& CDataFrameAnalysisSpecification::jobId() const {
    return m_JobId;
}

const std::string& CDataFrameAnalysisSpecification::analysisName() const {
    return m_AnalysisName;
}

CDataFrameAnalysisRunner* CDataFrameAnalysisSpecification::runner() {
    return m_Runner.get();
}
}
}
