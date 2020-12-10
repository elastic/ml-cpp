/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CIoManager.h>

#include <core/CBlockingCallCancellingTimer.h>
#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CProgramCounters.h>
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include <seccomp/CSystemCallFilter.h>


// TODO including torch/all.h causes problems. 
// Which headers are required?
#include <torch/script.h>

#include "CBufferedIStreamAdapter.h"
#include "CCmdLineParser.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>


int main(int argc, char** argv) {
    // Read command line options
    std::string modelId;
    ml::core_t::TTime namedPipeConnectTimeout{
        ml::core::CBlockingCallCancellingTimer::DEFAULT_TIMEOUT_SECONDS};

    if (ml::torch::CCmdLineParser::parse(argc, argv, modelId) == false) {
        return EXIT_FAILURE;
    }

    ml::core::CBlockingCallCancellingTimer cancellerThread{
        ml::core::CThread::currentThreadId(), std::chrono::seconds{namedPipeConnectTimeout}};

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    const std::string EMPTY;
    ml::api::CIoManager ioMgr{cancellerThread, 
        EMPTY, false,
        EMPTY, false, 
        EMPTY, false,  
        EMPTY, false};

    if (cancellerThread.start() == false) {
        // This log message will probably never been seen as it will go to the
        // real stderr of this process rather than the log pipe...
        LOG_FATAL(<< "Could not start blocking call canceller thread");
        return EXIT_FAILURE;
    }
    if (ml::core::CLogger::instance().reconfigure(
            EMPTY, EMPTY, cancellerThread.hasCancelledBlockingCall()) == false) {
        LOG_FATAL(<< "Could not reconfigure logging");
        cancellerThread.stop();
        return EXIT_FAILURE;
    }
    cancellerThread.stop();



    // Reduce memory priority before installing system call filters.
    ml::core::CProcessPriority::reduceMemoryPriority();

    // ml::seccomp::CSystemCallFilter::installSystemCallFilter();


    if (ioMgr.initIo() == false) {
        LOG_FATAL(<< "Failed to initialise IO");        
        return EXIT_FAILURE;
    }


    

    torch::jit::script::Module module;
    try {    
        // ioMgr.inputStream().seekg(0);
        // if (!ioMgr.inputStream().good()) {
            // LOG_INFO(<< "bad stream");
            // return EXIT_FAILURE;
        // }

        auto readAdapter = std::make_unique<ml::torch::CBufferedIStreamAdapter>(1330816933, ioMgr.inputStream());
        LOG_INFO(<< "load");
        module = torch::jit::load(std::move(readAdapter));


        // module = torch::jit::load(ioMgr.inputStream());
        // module = torch::jit::load("/Users/davidkyle/source/ml-search/projects/universal/torchscript/dbmdz-ner/conll03_traced_ner.pt");
        LOG_INFO(<< "model loaded");
    }
    catch (const c10::Error& e) {                        
        LOG_FATAL(<< "Error loading the model: " << e.msg());
        return EXIT_FAILURE;
    }




    // Print out the runtime counters generated during this execution context
    LOG_DEBUG(<< ml::core::CProgramCounters::instance());


    LOG_DEBUG(<< "ML Torch model prototype exiting");

    return EXIT_SUCCESS;
}
