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

#include <ver/CBuildInfo.h>

#include <api/CCsvInputParser.h>
#include <api/CDataFrameAnalyzer.h>
#include <api/CIoManager.h>
#include <api/CLengthEncodedInputParser.h>

#include "CCmdLineParser.h"

#include <boost/make_unique.hpp>

#include <string>

#include <stdlib.h>

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

    // auto inputParser =
    //     boost::make_unique<ml::api::CLengthEncodedInputParser>(ioMgr.inputStream());
    using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;
    auto inputParser{[lengthEncodedInput, &ioMgr]() -> TInputParserUPtr {
        if (lengthEncodedInput) {
            return boost::make_unique<ml::api::CLengthEncodedInputParser>(ioMgr.inputStream());
        }
        return boost::make_unique<ml::api::CCsvInputParser>(
            ioMgr.inputStream(), ml::api::CCsvInputParser::COMMA);
    }()};

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(ioMgr.outputStream());

    ml::api::CDataFrameAnalyzer dataFrameAnalyzer;

    if (inputParser->readStream([&dataFrameAnalyzer](const auto& record) {
            return dataFrameAnalyzer.handleRecord(record);
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
