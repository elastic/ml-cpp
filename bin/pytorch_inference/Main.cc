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

#include <core/CBlockingCallCancellingTimer.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CSetEnv.h>
#include <core/CStopWatch.h>
#include <core/CStringUtils.h>
#include <core/Concurrency.h>

#include <seccomp/CSystemCallFilter.h>

#include <ver/CBuildInfo.h>

#include <api/CIoManager.h>

#include "CBufferedIStreamAdapter.h"
#include "CCmdLineParser.h"
#include "CCommandParser.h"
#include "CResultWriter.h"
#include "CThreadSettings.h"

#include <ATen/Parallel.h>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/script.h>

#include <cstdint>
#include <memory>
#include <string>

torch::Tensor infer(torch::jit::script::Module& module_,
                    ml::torch::CCommandParser::SRequest& request) {

    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(1 + request.s_SecondaryArguments.size());

    std::array<std::int64_t, 2> dimensions = {request.s_NumberInferences,
                                              request.s_NumberInputTokens};
    at::IntArrayRef inputSize{dimensions};

    // Sequence tokens.
    inputs.emplace_back(torch::from_blob(static_cast<void*>(request.s_Tokens.data()),
                                         inputSize, at::dtype(torch::kInt64)));
    // Attention mask.
    for (auto& args : request.s_SecondaryArguments) {
        inputs.emplace_back(torch::from_blob(static_cast<void*>(args.data()),
                                             inputSize, at::dtype(torch::kInt64)));
    }

    torch::InferenceMode inferenceModeGuard;
    auto result = module_.forward(inputs);
    if (result.isTuple()) {
        // For transformers the result tensor is the first element in a tuple.
        return result.toTuple()->elements()[0].toTensor();
    }
    return result.toTensor();
}

bool handleRequest(ml::torch::CCommandParser::CRequestCacheInterface& cache,
                   ml::torch::CCommandParser::SRequest request,
                   torch::jit::script::Module& module_,
                   ml::torch::CResultWriter& resultWriter) {

    ml::core::async(ml::core::defaultAsyncExecutor(), [
        &cache, capturedRequest = std::move(request), &module_, &resultWriter
    ]() mutable {
        std::string requestId{capturedRequest.s_RequestId};
        // We time the combination of the cache lookup and (if necessary)
        // the inference.
        ml::core::CStopWatch stopWatch(true);
        cache.lookup(std::move(capturedRequest),
                     [&](auto request_) -> std::string {
                         torch::Tensor results = infer(module_, request_);
                         return resultWriter.createInnerResult(results);
                     },
                     [&](const auto& innerResponseJson_, bool isCacheHit) {
                         resultWriter.wrapAndWriteInnerResponse(innerResponseJson_,
                                                                requestId, isCacheHit,
                                                                stopWatch.stop());
                     });
    });
    return true;
}

void handleControlMessage(const ml::torch::CCommandParser::SControlMessage& controlMessage,
                          ml::torch::CThreadSettings& threadSettings,
                          ml::torch::CCommandParser::CRequestCacheInterface& cache,
                          ml::torch::CResultWriter& resultWriter) {

    switch (controlMessage.s_MessageType) {
    case ml::torch::CCommandParser::E_NumberOfAllocations:
        threadSettings.numAllocations(controlMessage.s_NumAllocations);
        ml::core::defaultAsyncExecutor().numberThreadsInUse(threadSettings.numAllocations());
        LOG_DEBUG(<< "Updated number of allocations to [" << threadSettings.numAllocations()
                  << "] ([" << controlMessage.s_NumAllocations << "] requested)");
        resultWriter.writeThreadSettings(controlMessage.s_RequestId, threadSettings);
        break;
    case ml::torch::CCommandParser::E_ClearCache:
        cache.clear();
        resultWriter.writeSimpleAck(controlMessage.s_RequestId);
        break;
    case ml::torch::CCommandParser::E_Unknown:
        LOG_ERROR(<< "Attempt to handle unknown control message");
        break;
    }
}

int main(int argc, char** argv) {
    // command line options
    std::string modelId;
    std::string inputFileName;
    bool isInputFileNamedPipe{false};
    std::string outputFileName;
    bool isOutputFileNamedPipe{false};
    std::string restoreFileName;
    bool isRestoreFileNamedPipe{false};
    std::string logFileName;
    std::string logProperties;
    ml::core_t::TTime namedPipeConnectTimeout{
        ml::core::CBlockingCallCancellingTimer::DEFAULT_TIMEOUT_SECONDS};
    std::int32_t numThreadsPerAllocation{1};
    std::int32_t numAllocations{1};
    std::size_t cacheMemorylimitBytes{0};
    bool validElasticLicenseKeyConfirmed{false};

    if (ml::torch::CCmdLineParser::parse(
            argc, argv, modelId, namedPipeConnectTimeout, inputFileName, isInputFileNamedPipe,
            outputFileName, isOutputFileNamedPipe, restoreFileName, isRestoreFileNamedPipe,
            logFileName, logProperties, numThreadsPerAllocation, numAllocations,
            cacheMemorylimitBytes, validElasticLicenseKeyConfirmed) == false) {
        return EXIT_FAILURE;
    }

    ml::torch::CThreadSettings threadSettings{
        static_cast<std::int32_t>(std::thread::hardware_concurrency()),
        numThreadsPerAllocation, numAllocations};

    // Setting the number of threads used by libtorch also sets
    // the number of threads used by MKL or OMP libs. However,
    // this doesn't address the Accelerate framework found on Macs.
    // Thus, we set the environment variable that controls threading for that one.
    // It doesn't hurt to set variables that won't have any effect on some platforms.
    ml::core::CSetEnv::setEnv(
        "VECLIB_MAXIMUM_THREADS",
        ml::core::CStringUtils::typeToString(threadSettings.numThreadsPerAllocation())
            .c_str(),
        0);

    ml::core::CBlockingCallCancellingTimer cancellerThread{
        ml::core::CThread::currentThreadId(), std::chrono::seconds{namedPipeConnectTimeout}};

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    const std::string EMPTY;
    ml::api::CIoManager ioMgr{cancellerThread,
                              inputFileName,
                              isInputFileNamedPipe,
                              outputFileName,
                              isOutputFileNamedPipe,
                              restoreFileName,
                              isRestoreFileNamedPipe,
                              EMPTY,
                              false};

    if (cancellerThread.start() == false) {
        // This log message will probably never been seen as it will go to the
        // real stderr of this process rather than the log pipe...
        LOG_FATAL(<< "Could not start blocking call canceller thread");
        return EXIT_FAILURE;
    }

    if (ml::core::CLogger::instance().reconfigure(
            logFileName, logProperties, cancellerThread.hasCancelledBlockingCall()) == false) {
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

    // On Linux we use libgomp (GNU's OMP implementation) for threading and have
    // found that setting this to "threads per allocation" really does allow
    // that number of threads to be used per allocation. On other platforms,
    // particularly macOS where we use the Accelerate framework instead of OMP,
    // it may be that this sets a number of threads to be shared across all
    // allocations rather than per allocation. But macOS is not supported for
    // production, but just as a convenience for developers. So the most
    // important thing is that the threading works as intended on Linux.
    at::set_num_threads(threadSettings.numThreadsPerAllocation());

    // This is not used as we don't call at::launch anywhere.
    // Setting it to 1 to ensure there is no thread pool sitting around.
    at::set_num_interop_threads(1);

    LOG_DEBUG(<< at::get_parallel_info());
    LOG_DEBUG(<< "Number of allocations: " << threadSettings.numAllocations());

    ml::torch::CResultWriter resultWriter{ioMgr.outputStream()};
    resultWriter.writeThreadSettings(ml::torch::CCommandParser::RESERVED_REQUEST_ID,
                                     threadSettings);

    torch::jit::script::Module module_;
    try {
        auto readAdapter = std::make_unique<ml::torch::CBufferedIStreamAdapter>(
            *ioMgr.restoreStream());
        if (readAdapter->init() == false) {
            return EXIT_FAILURE;
        }
        module_ = torch::jit::load(std::move(readAdapter));
        module_.eval();

        LOG_DEBUG(<< "model loaded");
    } catch (const c10::Error& e) {
        LOG_FATAL(<< "Error loading the model: " << e.msg());
        return EXIT_FAILURE;
    }

    ml::torch::CCommandParser commandParser{ioMgr.inputStream(), cacheMemorylimitBytes};

    // Size the threadpool to the number of hardware threads
    // so we can grow and shrink the threadpool dynamically
    ml::core::startDefaultAsyncExecutor();
    // Set the number of threads to use
    ml::core::defaultAsyncExecutor().numberThreadsInUse(threadSettings.numAllocations());

    commandParser.ioLoop(
        [&module_, &resultWriter](ml::torch::CCommandParser::CRequestCacheInterface& cache,
                                  ml::torch::CCommandParser::SRequest request) -> bool {
            return handleRequest(cache, std::move(request), module_, resultWriter);
        },
        [&resultWriter, &threadSettings](
            ml::torch::CCommandParser::CRequestCacheInterface& cache,
            const ml::torch::CCommandParser::SControlMessage& controlMessage) {
            return handleControlMessage(controlMessage, threadSettings, cache, resultWriter);
        },
        [&resultWriter](const std::string& requestId, const std::string& message) {
            resultWriter.writeError(requestId, message);
        });

    // Stopping the executor forces this to block until all work is done
    ml::core::stopDefaultAsyncExecutor();

    LOG_DEBUG(<< "ML PyTorch inference process exiting");

    return EXIT_SUCCESS;
}
