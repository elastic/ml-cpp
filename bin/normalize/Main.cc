/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
//! \brief
//! Normalise anomaly scores and/or probabilties in results
//!
//! DESCRIPTION:\n
//! Expects to be streamed CSV data on STDIN,
//! and sends its JSON results to STDOUT.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Standalone program.
//!
#include <core/CBlockingCallCancellingTimer.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include <model/CAnomalyDetectorModelConfig.h>

#include <api/CAnomalyJobConfig.h>
#include <api/CCsvInputParser.h>
#include <api/CCsvOutputWriter.h>
#include <api/CIoManager.h>
#include <api/CLengthEncodedInputParser.h>
#include <api/CNdJsonOutputWriter.h>
#include <api/CResultNormalizer.h>

#include <seccomp/CSystemCallFilter.h>

#include "CCmdLineParser.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    // Read command line options
    std::string modelConfigFile;
    std::string logProperties;
    std::string logPipe;
    ml::core_t::TTime bucketSpan{0};
    bool lengthEncodedInput{false};
    ml::core_t::TTime namedPipeConnectTimeout{
        ml::core::CBlockingCallCancellingTimer::DEFAULT_TIMEOUT_SECONDS};
    std::string inputFileName;
    bool isInputFileNamedPipe{false};
    std::string outputFileName;
    bool isOutputFileNamedPipe{false};
    std::string quantilesStateFile;
    bool deleteStateFiles{false};
    bool writeCsv{false};
    bool validElasticLicenseKeyConfirmed{false};
    if (ml::normalize::CCmdLineParser::parse(
            argc, argv, modelConfigFile, logProperties, logPipe, bucketSpan,
            lengthEncodedInput, namedPipeConnectTimeout, inputFileName,
            isInputFileNamedPipe, outputFileName, isOutputFileNamedPipe, quantilesStateFile,
            deleteStateFiles, writeCsv, validElasticLicenseKeyConfirmed) == false) {
        return EXIT_FAILURE;
    }

    ml::core::CBlockingCallCancellingTimer cancellerThread{
        ml::core::CThread::currentThreadId(), std::chrono::seconds{namedPipeConnectTimeout}};

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    ml::api::CIoManager ioMgr{cancellerThread, inputFileName, isInputFileNamedPipe,
                              outputFileName, isOutputFileNamedPipe};

    if (cancellerThread.start() == false) {
        // This log message will probably never been seen as it will go to the
        // real stderr of this process rather than the log pipe...
        LOG_FATAL(<< "Could not start blocking call canceller thread");
        return EXIT_FAILURE;
    }
    if (ml::core::CLogger::instance().reconfigure(
            logPipe, logProperties, cancellerThread.hasCancelledBlockingCall()) == false) {
        LOG_FATAL(<< "Could not reconfigure logging");
        cancellerThread.stop();
        return EXIT_FAILURE;
    }
    cancellerThread.stop();

    // Log the program version immediately after reconfiguring the logger.  This
    // must be done from the program, and NOT a shared library, as each program
    // statically links its own version library.
    LOG_DEBUG(<< ml::ver::CBuildInfo::fullInfo());

    if (validElasticLicenseKeyConfirmed == false) {
        LOG_FATAL(<< "Failed to confirm valid license key.");
        return EXIT_FAILURE;
    }

    // Reduce memory priority before installing system call filters.
    ml::core::CProcessPriority::reduceMemoryPriority();

    ml::seccomp::CSystemCallFilter::installSystemCallFilter();

    if (ioMgr.initIo() == false) {
        LOG_FATAL(<< "Failed to initialise IO");
        return EXIT_FAILURE;
    }

    // Reduce CPU priority after connecting named pipes so the JVM gets more
    // time when CPU is constrained.  Named pipe connection is time-sensitive,
    // hence is done before reducing CPU priority.
    ml::core::CProcessPriority::reduceCpuPriority();

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(bucketSpan);
    if (!modelConfigFile.empty() && modelConfig.init(modelConfigFile) == false) {
        LOG_FATAL(<< "ML model config file '" << modelConfigFile << "' could not be loaded");
        return EXIT_FAILURE;
    }

    // There's a choice of input and output formats for the numbers to be normalised
    using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;
    const TInputParserUPtr inputParser{[lengthEncodedInput, &ioMgr]() -> TInputParserUPtr {
        if (lengthEncodedInput) {
            return std::make_unique<ml::api::CLengthEncodedInputParser>(ioMgr.inputStream());
        }
        return std::make_unique<ml::api::CCsvInputParser>(ioMgr.inputStream());
    }()};

    using TSimpleOutputWriterUPtr = std::unique_ptr<ml::api::CSimpleOutputWriter>;
    const TSimpleOutputWriterUPtr outputWriter{[writeCsv, &ioMgr]() -> TSimpleOutputWriterUPtr {
        if (writeCsv) {
            return std::make_unique<ml::api::CCsvOutputWriter>(ioMgr.outputStream());
        }
        return std::make_unique<ml::api::CNdJsonOutputWriter>(
            ml::api::CNdJsonOutputWriter::TStrSet{ml::api::CResultNormalizer::PROBABILITY_NAME,
                                                  ml::api::CResultNormalizer::NORMALIZED_SCORE_NAME},
            ioMgr.outputStream());
    }()};

    // Initialize memory limits with default values.
    // This is fine as the normalizer doesn't use the memory limit.
    ml::model::CLimits limits{false};

    // This object will do the work
    ml::api::CResultNormalizer normalizer{modelConfig, *outputWriter, limits};

    // Restore state
    if (!quantilesStateFile.empty()) {
        if (normalizer.initNormalizer(quantilesStateFile) == false) {
            LOG_FATAL(<< "Failed to initialize normalizer");
            return EXIT_FAILURE;
        }
        if (deleteStateFiles) {
            std::remove(quantilesStateFile.c_str());
        }
    }

    // Now handle the numbers to be normalised from stdin
    if (inputParser->readStreamIntoMaps(
            std::bind(&ml::api::CResultNormalizer::handleRecord, &normalizer,
                      std::placeholders::_1)) == false) {
        LOG_FATAL(<< "Failed to handle input to be normalized");
        return EXIT_FAILURE;
    }

    // This message makes it easier to spot process crashes in a log file - if
    // this isn't present in the log for a given PID and there's no other log
    // message indicating early exit then the process has probably core dumped
    LOG_DEBUG(<< "ML normalizer exiting");

    return EXIT_SUCCESS;
}
