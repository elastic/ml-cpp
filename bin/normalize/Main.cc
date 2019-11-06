/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include <model/CAnomalyDetectorModelConfig.h>

#include <api/CCsvInputParser.h>
#include <api/CCsvOutputWriter.h>
#include <api/CIoManager.h>
#include <api/CLengthEncodedInputParser.h>
#include <api/CNdJsonOutputWriter.h>
#include <api/CResultNormalizer.h>

#include <seccomp/CSystemCallFilter.h>

#include "CCmdLineParser.h"

#include <functional>
#include <memory>
#include <string>

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    // Read command line options
    std::string modelConfigFile;
    std::string logProperties;
    std::string logPipe;
    ml::core_t::TTime bucketSpan(0);
    bool lengthEncodedInput(false);
    std::string inputFileName;
    bool isInputFileNamedPipe(false);
    std::string outputFileName;
    bool isOutputFileNamedPipe(false);
    std::string quantilesStateFile;
    bool deleteStateFiles(false);
    bool writeCsv(false);
    if (ml::normalize::CCmdLineParser::parse(
            argc, argv, modelConfigFile, logProperties, logPipe, bucketSpan, lengthEncodedInput,
            inputFileName, isInputFileNamedPipe, outputFileName, isOutputFileNamedPipe,
            quantilesStateFile, deleteStateFiles, writeCsv) == false) {
        return EXIT_FAILURE;
    }

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    ml::api::CIoManager ioMgr(inputFileName, isInputFileNamedPipe,
                              outputFileName, isOutputFileNamedPipe);

    if (ml::core::CLogger::instance().reconfigure(logPipe, logProperties) == false) {
        LOG_FATAL(<< "Could not reconfigure logging");
        return EXIT_FAILURE;
    }

    // Log the program version immediately after reconfiguring the logger.  This
    // must be done from the program, and NOT a shared library, as each program
    // statically links its own version library.
    LOG_DEBUG(<< ml::ver::CBuildInfo::fullInfo());

    ml::core::CProcessPriority::reducePriority();

    ml::seccomp::CSystemCallFilter::installSystemCallFilter();

    if (ioMgr.initIo() == false) {
        LOG_FATAL(<< "Failed to initialise IO");
        return EXIT_FAILURE;
    }

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(bucketSpan);
    if (!modelConfigFile.empty() && modelConfig.init(modelConfigFile) == false) {
        LOG_FATAL(<< "Ml model config file '" << modelConfigFile << "' could not be loaded");
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

    using TOutputHandlerUPtr = std::unique_ptr<ml::api::COutputHandler>;
    const TOutputHandlerUPtr outputWriter{[writeCsv, &ioMgr]() -> TOutputHandlerUPtr {
        if (writeCsv) {
            return std::make_unique<ml::api::CCsvOutputWriter>(ioMgr.outputStream());
        }
        return std::make_unique<ml::api::CNdJsonOutputWriter>(
            ml::api::CNdJsonOutputWriter::TStrSet{ml::api::CResultNormalizer::PROBABILITY_NAME,
                                                  ml::api::CResultNormalizer::NORMALIZED_SCORE_NAME},
            ioMgr.outputStream());
    }()};

    // This object will do the work
    ml::api::CResultNormalizer normalizer(modelConfig, *outputWriter);

    // Restore state
    if (!quantilesStateFile.empty()) {
        if (normalizer.initNormalizer(quantilesStateFile) == false) {
            LOG_FATAL(<< "Failed to initialize normalizer");
            return EXIT_FAILURE;
        }
        if (deleteStateFiles) {
            ::remove(quantilesStateFile.c_str());
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
    LOG_DEBUG(<< "Ml normalizer exiting");

    return EXIT_SUCCESS;
}
