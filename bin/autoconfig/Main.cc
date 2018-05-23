/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

//! \brief Understand field characteristics and suggest sensible
//! configuration for the analytics.
//!
//! DESCRIPTION:\n
//! Expects to be streamed CSV data on STDIN.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Standalone program.
//!
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include <api/CCmdSkeleton.h>
#include <api/CCsvInputParser.h>
#include <api/CIoManager.h>
#include <api/CLengthEncodedInputParser.h>

#include <config/CAutoconfigurer.h>
#include <config/CAutoconfigurerParams.h>
#include <config/CReportWriter.h>

#include <seccomp/CSystemCallFilter.h>

#include "CCmdLineParser.h"

#include <boost/bind.hpp>

#include <memory>
#include <string>

#include <stdlib.h>

int main(int argc, char** argv) {
    // Read command line options
    std::string logProperties;
    std::string logPipe;
    char delimiter(',');
    bool lengthEncodedInput(false);
    std::string timeField("time");
    std::string timeFormat;
    std::string configFile;
    std::string inputFileName;
    bool isInputFileNamedPipe(false);
    std::string outputFileName;
    bool isOutputFileNamedPipe(false);
    bool verbose(false);
    bool writeDetectorConfigs(false);
    if (ml::autoconfig::CCmdLineParser::parse(
            argc, argv, logProperties, logPipe, delimiter, lengthEncodedInput, timeField,
            timeFormat, configFile, inputFileName, isInputFileNamedPipe, outputFileName,
            isOutputFileNamedPipe, verbose, writeDetectorConfigs) == false) {
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

    using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;
    const TInputParserUPtr inputParser{[lengthEncodedInput, &ioMgr, delimiter]() -> TInputParserUPtr {
        if (lengthEncodedInput) {
            return std::make_unique<ml::api::CLengthEncodedInputParser>(ioMgr.inputStream());
        }
        return std::make_unique<ml::api::CCsvInputParser>(ioMgr.inputStream(), delimiter);
    }()};

    // This manages the full parameterization of the autoconfigurer.
    ml::config::CAutoconfigurerParams params(timeField, timeFormat, verbose, writeDetectorConfigs);
    params.init(configFile);

    // This is responsible for outputting the config.
    ml::config::CReportWriter writer(ioMgr.outputStream());

    // Need a new CAutoconfigurer for doing the actual heavy lifting.
    ml::config::CAutoconfigurer configurer(params, writer);

    // The skeleton avoids the need to duplicate a lot of boilerplate code
    ml::api::CCmdSkeleton skeleton(nullptr, // no restoration at present
                                   nullptr, // no persistence at present
                                   *inputParser, configurer);
    if (skeleton.ioLoop() == false) {
        LOG_FATAL(<< "Ml autoconfig failed");
        return EXIT_FAILURE;
    }

    // This message makes it easier to spot process crashes in a log file - if
    // this isn't present in the log for a given PID and there's no other log
    // message indicating early exit then the process has probably core dumped
    LOG_DEBUG(<< "Ml autoconfig exiting");

    return EXIT_SUCCESS;
}
