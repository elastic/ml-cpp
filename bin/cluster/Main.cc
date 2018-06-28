/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
//! \brief Clusters a collection of numeric vectors.
//!
//! DESCRIPTION:\n
//! Expects to be streamed CSV or length encoded data on STDIN or a named pipe,
//! and sends its JSON results to STDOUT or another named pipe.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Standalone program.
//!
#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>

#include <ver/CBuildInfo.h>

#include <api/CClusterer.h>
#include <api/CCmdSkeleton.h>
#include <api/CCsvInputParser.h>
#include <api/CIoManager.h>
#include <api/CLengthEncodedInputParser.h>
#include <api/CSingleStreamSearcher.h>

#include "CCmdLineParser.h"

#include <boost/scoped_ptr.hpp>

#include <iostream>
#include <string>

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    using TScopedDataSearcherPtr = boost::scoped_ptr<ml::core::CDataSearcher>;
    using TScopedInputParserPtr = boost::scoped_ptr<ml::api::CInputParser>;

    // Read command line options
    std::string clusterField;
    std::string featureField;
    std::string paramsFile;
    std::string logProperties;
    std::string logPipe;
    char delimiter('\t');
    bool lengthEncodedInput(false);
    std::string inputFileName;
    bool isInputFileNamedPipe(false);
    std::string outputFileName;
    bool isOutputFileNamedPipe(false);
    if (ml::cluster::CCmdLineParser::parse(
            argc, argv, clusterField, featureField, paramsFile, logProperties,
            logPipe, delimiter, lengthEncodedInput, inputFileName,
            isInputFileNamedPipe, outputFileName, isOutputFileNamedPipe) == false) {
        return EXIT_FAILURE;
    }

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    ml::api::CIoManager ioMgr(inputFileName, isInputFileNamedPipe,
                              outputFileName, isOutputFileNamedPipe);

    if (ml::core::CLogger::instance().reconfigure(logPipe, logProperties) == false) {
        LOG_FATAL("Could not reconfigure logging");
        return EXIT_FAILURE;
    }

    // Log the program version immediately after reconfiguring the logger.  This
    // must be done from the program, and NOT a shared library, as each program
    // statically links its own version library.
    LOG_DEBUG(ml::ver::CBuildInfo::fullInfo());

    ml::core::CProcessPriority::reducePriority();

    if (ioMgr.initIo() == false) {
        LOG_FATAL("Failed to initialise IO");
        return EXIT_FAILURE;
    }

    if (clusterField.empty()) {
        LOG_FATAL("ML must specify the name of the field to cluster");
        return EXIT_FAILURE;
    }

    if (featureField.empty()) {
        LOG_FATAL("ML must specify the name of the field identifying features");
        return EXIT_FAILURE;
    }

    ml::api::CClusterer::CParams params;
    params.clusterField(clusterField).featureField(featureField);
    if (!paramsFile.empty() && params.init(paramsFile) == false) {
        LOG_FATAL("ML clustering parameters file '" << paramsFile << "' could not be loaded");
        return EXIT_FAILURE;
    }

    TScopedDataSearcherPtr restoreSearcher;
    if (ioMgr.restoreStream() != 0) {
        restoreSearcher.reset(new ml::api::CSingleStreamSearcher(ioMgr.restoreStream()));
    }
    TScopedInputParserPtr inputParser;
    if (lengthEncodedInput) {
        inputParser.reset(new ml::api::CLengthEncodedInputParser(ioMgr.inputStream()));
    } else {
        inputParser.reset(new ml::api::CCsvInputParser(ioMgr.inputStream(), delimiter));
    }
    ml::api::CClustererOutputWriter writer(ioMgr.outputStream());

    // Create the clusterer.
    ml::api::CClusterer clusterer(params, writer);

    // The skeleton avoids the need to duplicate a lot of boilerplate code.
    ml::api::CCmdSkeleton skeleton(restoreSearcher.get(), 0, *inputParser, clusterer);
    bool ioLoopSucceeded(skeleton.ioLoop());

    // Unfortunately, we cannot rely on destruction to finalise the output
    // writer as it must be finalised before the skeleton is destroyed, and C++
    // destruction order means the skeleton will be destroyed before the output
    // writer as it was constructed after it.
    writer.finalise();

    if (!ioLoopSucceeded) {
        LOG_FATAL("ML clustering failed");
        return EXIT_FAILURE;
    }

    // This message makes it easier to spot process crashes in a log file - if
    // this isn't present in the log for a given PID and there's no other log
    // message indicating early exit then the process has probably core dumped
    LOG_DEBUG("ML clustering exiting");

    return EXIT_SUCCESS;
}
