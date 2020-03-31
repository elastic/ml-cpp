/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
//! \brief
//! Applies a range of ML analyses on a data frame.
//!
//! DESCRIPTION:\n
//! Expects to be streamed CSV or length encoded data on STDIN or a named pipe,
//! and sends its JSON results to STDOUT or another named pipe.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Standalone program.
//!
#include <core/CDataFrameRowSlice.h>
#include <core/CDataSearcher.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CNonInstantiatable.h>
#include <core/CProcessPriority.h>
#include <core/CProgramCounters.h>
#include <core/Concurrency.h>

#include <ver/CBuildInfo.h>

#include <api/CCsvInputParser.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalyzer.h>
#include <api/CIoManager.h>
#include <api/CLengthEncodedInputParser.h>
#include <api/CMemoryUsageEstimationResultJsonWriter.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include "CCmdLineParser.h"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
std::pair<std::string, bool> readFileToString(const std::string& fileName) {
    std::ifstream fileStream{fileName};
    if (fileStream.is_open() == false) {
        LOG_FATAL(<< "Environment error: failed to open file '" << fileName << "'.");
        return {std::string{}, false};
    }
    return {std::string{std::istreambuf_iterator<char>{fileStream},
                        std::istreambuf_iterator<char>{}},
            true};
}

class CCleanUpOnExit : private ml::core::CNonInstantiatable {
public:
    using TTemporaryDirectoryPtr = std::shared_ptr<ml::core::CTemporaryDirectory>;

public:
    static void add(TTemporaryDirectoryPtr directory) {
        m_DataFrameDirectory = directory;
    }

    static void run() {
        if (m_DataFrameDirectory != nullptr) {
            m_DataFrameDirectory->removeAll();
        }
    }

private:
    static TTemporaryDirectoryPtr m_DataFrameDirectory;
};
CCleanUpOnExit::TTemporaryDirectoryPtr CCleanUpOnExit::m_DataFrameDirectory{};
}

int main(int argc, char** argv) {
    // Register the set of counters in which this program is interested
    const ml::counter_t::TCounterTypeSet counters{
        ml::counter_t::E_DFOEstimatedPeakMemoryUsage,
        ml::counter_t::E_DFOPeakMemoryUsage,
        ml::counter_t::E_DFOTimeToCreateEnsemble,
        ml::counter_t::E_DFOTimeToComputeScores,
        ml::counter_t::E_DFONumberPartitions,
        ml::counter_t::E_DFTPMEstimatedPeakMemoryUsage,
        ml::counter_t::E_DFTPMPeakMemoryUsage,
        ml::counter_t::E_DFTPMTimeToTrain,
        ml::counter_t::E_DFTPMTrainedForestNumberTrees};
    ml::core::CProgramCounters::registerProgramCounterTypes(counters);

    // Read command line options
    std::string configFile;
    bool memoryUsageEstimationOnly(false);
    std::string logProperties;
    std::string logPipe;
    bool lengthEncodedInput(false);
    std::string inputFileName;
    bool isInputFileNamedPipe(false);
    std::string outputFileName;
    bool isOutputFileNamedPipe(false);
    std::string restoreFileName;
    bool isRestoreFileNamedPipe(false);
    std::string persistFileName;
    bool isPersistFileNamedPipe(false);
    if (ml::data_frame_analyzer::CCmdLineParser::parse(
            argc, argv, configFile, memoryUsageEstimationOnly, logProperties, logPipe,
            lengthEncodedInput, inputFileName, isInputFileNamedPipe, outputFileName,
            isOutputFileNamedPipe, restoreFileName, isRestoreFileNamedPipe,
            persistFileName, isPersistFileNamedPipe) == false) {
        return EXIT_FAILURE;
    }

    // The static members of CCleanUpOnExit will all be fully initialised before
    // this call (before the beginning main). Therefore, they will be destructed
    // after CCleanUpOnExit::run is run and it is safe to act on them there.
    std::atexit(CCleanUpOnExit::run);

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    ml::api::CIoManager ioMgr(inputFileName, isInputFileNamedPipe, outputFileName,
                              isOutputFileNamedPipe, restoreFileName, isRestoreFileNamedPipe,
                              persistFileName, isPersistFileNamedPipe);

    if (ml::core::CLogger::instance().reconfigure(logPipe, logProperties) == false) {
        LOG_FATAL(<< "Could not reconfigure logging");
        return EXIT_FAILURE;
    }

    // Log the program version immediately after reconfiguring the logger.  This
    // must be done from the program, and NOT a shared library, as each program
    // statically links its own version library.
    LOG_DEBUG(<< ml::ver::CBuildInfo::fullInfo());

    ml::core::CProcessPriority::reduceMemoryPriority();

    if (ioMgr.initIo() == false) {
        LOG_FATAL(<< "Failed to initialise IO");
        return EXIT_FAILURE;
    }

    // Reduce CPU priority after connecting named pipes so the JVM gets more
    // time when CPU is constrained
    ml::core::CProcessPriority::reduceCpuPriority();

    using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;

    std::string analysisSpecificationJson;
    bool couldReadConfigFile;
    std::tie(analysisSpecificationJson, couldReadConfigFile) = readFileToString(configFile);
    if (couldReadConfigFile == false) {
        LOG_FATAL(<< "Failed to read config file '" << configFile << "'");
        return EXIT_FAILURE;
    }

    auto resultsStreamSupplier = [&ioMgr]() {
        return std::make_unique<ml::core::CJsonOutputStreamWrapper>(ioMgr.outputStream());
    };

    using TDataAdderUPtr = std::unique_ptr<ml::core::CDataAdder>;
    auto persisterSupplier = [&ioMgr]() -> TDataAdderUPtr {
        if (ioMgr.persistStream() != nullptr) {

            return std::make_unique<ml::api::CSingleStreamDataAdder>(ioMgr.persistStream());
        }
        return nullptr;
    };

    using TDataSearcherUPtr = std::unique_ptr<ml::core::CDataSearcher>;
    auto restoreSearcherSupplier = [isRestoreFileNamedPipe, &ioMgr]() -> TDataSearcherUPtr {
        if (ioMgr.restoreStream()) {
            // Check whether state is restored from a file, if so we assume that this is a debugging case
            // and therefore does not originate from the ML Java code.
            if (isRestoreFileNamedPipe == false) {
                // apply a filter to overcome differences in the way persistence vs. restore works
                auto strm = std::make_shared<boost::iostreams::filtering_istream>();
                strm->push(ml::api::CStateRestoreStreamFilter());
                strm->push(*ioMgr.restoreStream());
                return std::make_unique<ml::api::CSingleStreamSearcher>(strm);
            }
            return std::make_unique<ml::api::CSingleStreamSearcher>(ioMgr.restoreStream());
        }
        return nullptr;
    };

    auto analysisSpecification = std::make_unique<ml::api::CDataFrameAnalysisSpecification>(
        analysisSpecificationJson, std::move(persisterSupplier),
        std::move(restoreSearcherSupplier));

    if (memoryUsageEstimationOnly) {
        auto outStream = resultsStreamSupplier();
        ml::api::CMemoryUsageEstimationResultJsonWriter writer(*outStream);
        analysisSpecification->estimateMemoryUsage(writer);
        return EXIT_SUCCESS;
    }

    if (analysisSpecification->numberThreads() > 1) {
        ml::core::startDefaultAsyncExecutor(analysisSpecification->numberThreads());
    }

    ml::api::CDataFrameAnalyzer dataFrameAnalyzer{
        std::move(analysisSpecification), std::move(resultsStreamSupplier)};

    CCleanUpOnExit::add(dataFrameAnalyzer.dataFrameDirectory());

    auto inputParser{[lengthEncodedInput, &ioMgr]() -> TInputParserUPtr {
        if (lengthEncodedInput) {
            return std::make_unique<ml::api::CLengthEncodedInputParser>(ioMgr.inputStream());
        }
        return std::make_unique<ml::api::CCsvInputParser>(ioMgr.inputStream());
    }()};
    if (inputParser->readStreamIntoVecs(
            [&dataFrameAnalyzer](const auto& fieldNames, const auto& fieldValues) {
                return dataFrameAnalyzer.handleRecord(fieldNames, fieldValues);
            }) == false) {
        LOG_FATAL(<< "Failed to handle input to be analyzed");
        return EXIT_FAILURE;
    }

    if (dataFrameAnalyzer.usingControlMessages() == false) {
        // To make running from the command line easy, we'll run the analysis
        // after closing the input pipe if control messages are not in use.
        dataFrameAnalyzer.receivedAllRows();
        dataFrameAnalyzer.run();
    }

    // Print out the runtime counters generated during this execution context.
    LOG_INFO(<< ml::core::CProgramCounters::instance());

    // This message makes it easier to spot process crashes in a log file - if
    // this isn't present in the log for a given PID and there's no other log
    // message indicating early exit then the process has probably core dumped.
    LOG_DEBUG(<< "Ml data frame analyzer exiting");

    return EXIT_SUCCESS;
}
