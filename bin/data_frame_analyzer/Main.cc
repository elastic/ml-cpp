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
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/Concurrency.h>

#include <ver/CBuildInfo.h>

#include <api/CCsvInputParser.h>
#include <api/CDataFrameAnalysisRunner.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalyzer.h>
#include <api/CDataFrameOutliersRunner.h>
#include <api/CIoManager.h>
#include <api/CLengthEncodedInputParser.h>

#include "CCmdLineParser.h"

#include <boost/make_unique.hpp>

#include <fstream>
#include <string>

#include <stdlib.h>

// TODO That might belong in CDataFrameAnalyzer
// TODO Error handling: split out reading file.
ml::api::CDataFrameAnalysisSpecification
makeDataFrameAnalysisSpecification(const std::string& configFile) {
    using TRunnerFactoryUPtrVec = ml::api::CDataFrameAnalysisSpecification::TRunnerFactoryUPtrVec;
    TRunnerFactoryUPtrVec factories;
    factories.push_back(boost::make_unique<ml::api::CDataFrameOutliersRunnerFactory>());

    std::ifstream configFileStream(configFile);
    std::string dataFrameConfig(std::istreambuf_iterator<char>{configFileStream},
                                std::istreambuf_iterator<char>{});
    return ml::api::CDataFrameAnalysisSpecification{std::move(factories), dataFrameConfig};
}

int main(int argc, char** argv) {
    // Read command line options
    std::string configFile;
    std::string logProperties;
    std::string logPipe;
    bool lengthEncodedInput(false);
    std::string inputFileName;
    bool isInputFileNamedPipe(false);
    std::string outputFileName;
    bool isOutputFileNamedPipe(false);
    if (ml::data_frame_analyzer::CCmdLineParser::parse(
            argc, argv, configFile, logProperties, logPipe, lengthEncodedInput, inputFileName,
            isInputFileNamedPipe, outputFileName, isOutputFileNamedPipe) == false) {
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

    if (ioMgr.initIo() == false) {
        LOG_FATAL(<< "Failed to initialise IO");
        return EXIT_FAILURE;
    }

    using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;
    auto inputParser{[lengthEncodedInput, &ioMgr]() -> TInputParserUPtr {
        if (lengthEncodedInput) {
            return boost::make_unique<ml::api::CLengthEncodedInputParser>(ioMgr.inputStream());
        }
        return boost::make_unique<ml::api::CCsvInputParser>(
            ioMgr.inputStream(), ml::api::CCsvInputParser::COMMA);
    }()};

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(ioMgr.outputStream());

    // TODO Actually use the specification to create the data frame and run the analysis
    ml::api::CDataFrameAnalysisSpecification dataFrameAnalysisSpecification{
        makeDataFrameAnalysisSpecification(configFile)};
    if (dataFrameAnalysisSpecification.bad()) {
        LOG_FATAL("Failed to parse analysis specification");
        return EXIT_FAILURE;
    }
    if (dataFrameAnalysisSpecification.threads() > 1) {
        ml::core::startDefaultAsyncExecutor(dataFrameAnalysisSpecification.threads());
    }

    ml::api::CDataFrameAnalyzer dataFrameAnalyzer;

    if (inputParser->readStreamIntoVecs(
            [&dataFrameAnalyzer](const auto& fieldNames, const auto& fieldValues) {
                return dataFrameAnalyzer.handleRecord(fieldNames, fieldValues);
            }) == false) {
        LOG_FATAL(<< "Failed to handle input to be analyzed");
        return EXIT_FAILURE;
    }

    // This message makes it easier to spot process crashes in a log file - if
    // this isn't present in the log for a given PID and there's no other log
    // message indicating early exit then the process has probably core dumped
    LOG_DEBUG(<< "Ml data frame analyzer exiting");

    return EXIT_SUCCESS;
}
