/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CBlockingCallCancellingTimer.h>
#include <core/CDataSearcher.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CStateDecompressor.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include <api/CAnomalyJob.h>
#include <api/CAnomalyJobConfig.h>
#include <api/CIoManager.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include "CCmdLineParser.h"

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <fstream>
#include <ios>
#include <iostream>
#include <memory>

#include <stdlib.h>

// We use short field names to reduce the state size

using namespace ml;

int main(int argc, char** argv) {

    // Read command line options
    std::string logProperties;
    ml::core_t::TTime namedPipeConnectTimeout{
        ml::core::CBlockingCallCancellingTimer::DEFAULT_TIMEOUT_SECONDS};
    std::string inputFileName;
    bool isInputFileNamedPipe{false};
    std::string outputFileName;
    bool isOutputFileNamedPipe{false};
    std::string restoreFileName;
    bool isRestoreFileNamedPipe{false};
    std::string persistFileName;
    bool isPersistFileNamedPipe{false};
    std::string outputFormat;
    if (model_extractor::CCmdLineParser::parse(
            argc, argv, logProperties, namedPipeConnectTimeout, inputFileName, isInputFileNamedPipe,
            outputFileName, isOutputFileNamedPipe, restoreFileName, isRestoreFileNamedPipe,
            persistFileName, isPersistFileNamedPipe, outputFormat) == false) {
        return EXIT_FAILURE;
    }

    ml::core::CBlockingCallCancellingTimer cancellerThread{
        ml::core::CThread::currentThreadId(), std::chrono::seconds{namedPipeConnectTimeout}};

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    ml::api::CIoManager ioMgr{
        cancellerThread,        inputFileName,         isInputFileNamedPipe,
        outputFileName,         isOutputFileNamedPipe, restoreFileName,
        isRestoreFileNamedPipe, persistFileName,       isPersistFileNamedPipe};

    if (cancellerThread.start() == false) {
        // This log message will probably never been seen as it will go to the
        // real stderr of this process rather than the log pipe...
        LOG_FATAL(<< "Could not start blocking call canceller thread");
        return EXIT_FAILURE;
    }

    const std::string logPipe;
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
    static const ml::core_t::TTime BUCKET_SIZE{600};
    static const std::string JOB_ID{"job"};

    ml::model::CLimits limits;
    ml::api::CAnomalyJobConfig jobConfig;

    const int latencyBuckets{0};

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(
            BUCKET_SIZE, ml::model_t::E_None, "", BUCKET_SIZE * latencyBuckets, false);

    // dummy job output stream to satisfy CAnomalyJob ctor requirements
    std::ofstream jobOutputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    if (jobOutputStrm.is_open() == false) {
        LOG_ERROR(<< "Failed to open output stream.");
    }
    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(jobOutputStrm);

    ml::api::CAnomalyJob restoredJob{
        JOB_ID,
        limits,
        jobConfig,
        modelConfig,
        wrappedOutputStream,
        [](ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport) {},
        nullptr,
        -1,
        ml::api::CAnomalyJob::DEFAULT_TIME_FIELD_NAME,
        ml::api::CAnomalyJob::EMPTY_STRING,
        0};

    ml::core_t::TTime completeToTime{0};
    ml::core_t::TTime prevCompleteToTime{0};
    while (restoredJob.restoreState(restoreSearcher, completeToTime)) {
        assert(completeToTime > prevCompleteToTime);
        prevCompleteToTime = completeToTime;
        LOG_DEBUG(<< "Restore complete to time " << completeToTime << std::endl);

        core::CNamedPipeFactory::TOStreamP persistStrm{&ioMgr.outputStream(),
                                                       [](std::ostream*) {}};
        ml::api::CSingleStreamDataAdder persister{persistStrm};

        // Attempt to persist state in a plain JSON formatted file or stream
        if (restoredJob.persistModelsState(persister, completeToTime, outputFormat) == false) {
            LOG_FATAL(<< "Failed to persist state as JSON");
            exit(EXIT_FAILURE);
        }
    }

    exit(EXIT_SUCCESS);
}
