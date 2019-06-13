/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CAnomalyJob.h>
#include <api/CFieldConfig.h>
#include <api/CIoManager.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include <core/CDataSearcher.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CStateDecompressor.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>
#include <core/COsFileFuncs.h>

#include <ver/CBuildInfo.h>

#include "CCmdLineParser.h"

#include <fstream>
#include <ios>
#include <iostream>
#include <memory>

#include <stdlib.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/filesystem.hpp>

// We use short field names to reduce the state size
namespace {


}

using namespace ml;

int main(int argc, char** argv) {

    // Read command line options
    std::string logProperties;
    std::string inputFileName;
    bool isInputFileNamedPipe(false);
    std::string outputFileName;
    bool isOutputFileNamedPipe(false);
    if (model_extractor::CCmdLineParser::parse(
            argc, argv,  logProperties, inputFileName, isInputFileNamedPipe, outputFileName,
            isOutputFileNamedPipe) == false) {
        return EXIT_FAILURE;
    }

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    ml::api::CIoManager ioMgr(inputFileName, isInputFileNamedPipe, outputFileName, isOutputFileNamedPipe);

    const std::string logPipe{};
    if (ml::core::CLogger::instance().reconfigure(logPipe, logProperties) == false) {
        LOG_FATAL(<< "Could not reconfigure logging");
        return EXIT_FAILURE;
    }

    // Log the program version immediately after reconfiguring the logger.  This
    // must be done from the program, and NOT a shared library, as each program
    // statically links its own version library.
    LOG_DEBUG(<< ml::ver::CBuildInfo::fullInfo());

    if (ioMgr.initIo() == false) {
        LOG_FATAL(<< "Failed to initialise IO");
        return EXIT_FAILURE;
    }

    // apply a filter to allow us to restore from an autodetect persistence (debug) dump
    ml::api::CSingleStreamSearcher restoreSearcher([&ioMgr]() {
        auto strm = std::make_shared<boost::iostreams::filtering_istream>();
        strm->push(ml::api::CStateRestoreStreamFilter());
        strm->push(ioMgr.inputStream());
        return strm;
    }());

    // create a job with basic config sufficient enough to restore state.
    static const ml::core_t::TTime BUCKET_SIZE(600);
    static const std::string JOB_ID("job");

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig;

    if (!fieldConfig.initFromFile(ml::core::COsFileFuncs::NULL_FILENAME)) {
        std::cerr << "Failed to initialize field configuration." << std::endl;
        exit(EXIT_FAILURE);
    }
    const int latencyBuckets{0};

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(
            BUCKET_SIZE, ml::model_t::E_None, "", BUCKET_SIZE * latencyBuckets, false);

    // dummy job output stream to satisfy CAnomalyJob ctor requirements
    std::ofstream jobOutputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    if(!jobOutputStrm.is_open()) {
        LOG_ERROR(<< "Failed to open output stream.");
    }
    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(jobOutputStrm);

    ml::api::CAnomalyJob restoredJob(JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream, [](ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport){});

    ml::core_t::TTime completeToTime(0);
    if (!restoredJob.restoreState(restoreSearcher, completeToTime)) {
        LOG_ERROR(<< "Failed to restore state from file \"" << inputFileName << "\"");
        exit(EXIT_FAILURE);
    }
    assert(completeToTime > 0);
    LOG_DEBUG( << "Restore complete to time " << completeToTime << std::endl);

    core::CNamedPipeFactory::TOStreamP persistStrm{&ioMgr.outputStream(), [](std::ostream*){}};
    ml::api::CSingleStreamDataAdder persister(persistStrm);

    // Attempt to persist state in a plain JSON formatted file or stream or stream
    if (restoredJob.persistResidualModelsState(persister) == false) {
        LOG_FATAL(<< "Failed to persist state as JSON");
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}
