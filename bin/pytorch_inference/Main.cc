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
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CRapidJsonConcurrentLineWriter.h>
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
#include "CThreadSettings.h"

#include <ATen/Parallel.h>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/script.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

namespace {
using TRapidJsonLineWriter = ml::core::CRapidJsonLineWriter<rapidjson::StringBuffer>;

const std::string RESULT{"result"};
const std::string INFERENCE{"inference"};
const std::string ERROR{"error"};
const std::string TIME_MS{"time_ms"};
const std::string CACHE_HIT{"cache_hit"};
const std::string THREAD_SETTINGS{"thread_settings"};
const std::string ACK{"ack"};
const std::string ACKNOWLEDGED{"acknowledged"};
const std::string NUM_ALLOCATIONS{"num_allocations"};
const std::string NUM_THREADS_PER_ALLOCATION{"num_threads_per_allocation"};
}

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

template<typename T>
void writeTensor(const torch::TensorAccessor<T, 1UL>& accessor,
                 TRapidJsonLineWriter& jsonWriter) {
    jsonWriter.StartArray();
    for (int i = 0; i < accessor.size(0); ++i) {
        jsonWriter.Double(static_cast<double>(accessor[i]));
    }
    jsonWriter.EndArray();
}

template<typename T, std::size_t N_DIMS>
void writeTensor(const torch::TensorAccessor<T, N_DIMS>& accessor,
                 TRapidJsonLineWriter& jsonWriter) {
    jsonWriter.StartArray();
    for (int i = 0; i < accessor.size(0); ++i) {
        writeTensor(accessor[i], jsonWriter);
    }
    jsonWriter.EndArray();
}

template<typename T>
void writeInferenceResults(const torch::TensorAccessor<T, 3UL>& accessor,
                           TRapidJsonLineWriter& jsonWriter) {

    jsonWriter.Key(RESULT);
    jsonWriter.StartObject();
    jsonWriter.Key(INFERENCE);
    writeTensor(accessor, jsonWriter);
    jsonWriter.EndObject();
}

template<typename T>
void writeInferenceResults(const torch::TensorAccessor<T, 2UL>& accessor,
                           TRapidJsonLineWriter& jsonWriter) {

    jsonWriter.Key(RESULT);
    jsonWriter.StartObject();
    jsonWriter.Key(INFERENCE);
    // Output must be a 3D array so wrap the 2D result in an outer array.
    jsonWriter.StartArray();
    writeTensor(accessor, jsonWriter);
    jsonWriter.EndArray();
    jsonWriter.EndObject();
}

void writeInnerError(const std::string& message, TRapidJsonLineWriter& jsonWriter) {
    jsonWriter.Key(ERROR);
    jsonWriter.StartObject();
    jsonWriter.Key(ERROR);
    jsonWriter.String(message);
    jsonWriter.EndObject();
}

void writeError(const std::string& requestId,
                const std::string& message,
                TRapidJsonLineWriter& jsonWriter) {
    jsonWriter.StartObject();
    jsonWriter.Key(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    writeInnerError(message, jsonWriter);
    jsonWriter.EndObject();
}

std::string wrapInnerResponse(const std::string& innerResponse,
                              const std::string& requestId,
                              bool isCacheHit,
                              std::uint64_t timeMs) {
    std::ostringstream strm;
    strm << "{\"" << ml::torch::CCommandParser::REQUEST_ID << "\":\"" << requestId
         << "\",\"" << CACHE_HIT << "\":" << std::boolalpha << isCacheHit
         << ",\"" << TIME_MS << "\":" << timeMs << "," << innerResponse << "}";
    return strm.str();
}

void writeThreadSettings(ml::core::CJsonOutputStreamWrapper& wrappedOutputStream,
                         const std::string& requestId,
                         const ml::torch::CThreadSettings& threadSettings) {
    ml::core::CRapidJsonConcurrentLineWriter jsonWriter(wrappedOutputStream);
    jsonWriter.StartObject();
    jsonWriter.Key(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(THREAD_SETTINGS);
    jsonWriter.StartObject();
    jsonWriter.Key(NUM_THREADS_PER_ALLOCATION);
    jsonWriter.Uint(threadSettings.numThreadsPerAllocation());
    jsonWriter.Key(NUM_ALLOCATIONS);
    jsonWriter.Uint(threadSettings.numAllocations());
    jsonWriter.EndObject();
    jsonWriter.EndObject();
}

void writeSimpleAck(ml::core::CJsonOutputStreamWrapper& wrappedOutputStream,
                    const std::string& requestId) {
    ml::core::CRapidJsonConcurrentLineWriter jsonWriter(wrappedOutputStream);
    jsonWriter.StartObject();
    jsonWriter.Key(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(ACK);
    jsonWriter.StartObject();
    jsonWriter.Key(ACKNOWLEDGED);
    jsonWriter.Bool(true);
    jsonWriter.EndObject();
    jsonWriter.EndObject();
}

template<std::size_t N>
void writePrediction(const torch::Tensor& prediction, TRapidJsonLineWriter& jsonWriter) {

    // Creating the accessor will throw if the tensor does not have exactly
    // N dimensions. Do this before writing any output so the error message
    // isn't mingled with a partial result.

    if (prediction.dtype() == torch::kFloat32) {
        auto accessor = prediction.accessor<float, N>();
        writeInferenceResults(accessor, jsonWriter);

    } else if (prediction.dtype() == torch::kFloat64) {
        auto accessor = prediction.accessor<double, N>();
        writeInferenceResults(accessor, jsonWriter);

    } else {
        std::ostringstream ss;
        ss << "Cannot process result tensor of type [" << prediction.dtype() << "]";
        writeInnerError(ss.str(), jsonWriter);
    }
}

void inferAndWriteInnerResult(ml::torch::CCommandParser::SRequest& request,
                              torch::jit::script::Module& module_,
                              TRapidJsonLineWriter& jsonWriter) {
    try {
        torch::Tensor results = infer(module_, request);
        auto sizes = results.sizes();

        switch (sizes.size()) {
        case 3:
            writePrediction<3>(results, jsonWriter);
            break;
        case 2:
            writePrediction<2>(results, jsonWriter);
            break;
        default: {
            std::ostringstream ss;
            ss << "Cannot convert results tensor of size [" << sizes << "]";
            writeInnerError(ss.str(), jsonWriter);
            break;
        }
        }
    } catch (const c10::Error& e) {
        writeInnerError(e.what(), jsonWriter);
    } catch (const std::runtime_error& e) {
        writeInnerError(e.what(), jsonWriter);
    }
    jsonWriter.Flush();
}

bool handleRequest(ml::torch::CCommandParser::CRequestCacheInterface& cache,
                   ml::torch::CCommandParser::SRequest request,
                   torch::jit::script::Module& module_,
                   ml::core::CJsonOutputStreamWrapper& wrappedOutputStream) {

    ml::core::async(ml::core::defaultAsyncExecutor(), [
        &cache, capturedRequest = std::move(request), &module_, &wrappedOutputStream
    ]() mutable {
        std::string requestId{capturedRequest.s_RequestId};
        std::string responseJson;
        // We time the combination of the cache lookup and (if necessary)
        // the inference.
        ml::core::CStopWatch stopWatch(true);
        cache.lookup(std::move(capturedRequest),
                     [&](auto request_) -> std::string {
                         rapidjson::StringBuffer stringBuffer;
                         TRapidJsonLineWriter jsonWriter;
                         jsonWriter.Reset(stringBuffer);
                         inferAndWriteInnerResult(request_, module_, jsonWriter);
                         return stringBuffer.GetString();
                     },
                     [&](const auto& innerResponseJson_, bool isCacheHit) {
                         responseJson = wrapInnerResponse(innerResponseJson_,
                                                          requestId, isCacheHit,
                                                          stopWatch.stop());
                     });
        wrappedOutputStream.writeJson(responseJson);
        wrappedOutputStream.flush();
    });
    return true;
}

void handleControlMessage(const ml::torch::CCommandParser::SControlMessage& controlMessage,
                          ml::torch::CThreadSettings& threadSettings,
                          ml::torch::CCommandParser::CRequestCacheInterface& cache,
                          ml::core::CJsonOutputStreamWrapper& wrappedOutputStream) {

    switch (controlMessage.s_MessageType) {
    case ml::torch::CCommandParser::E_NumberOfAllocations:
        threadSettings.numAllocations(controlMessage.s_NumAllocations);
        ml::core::defaultAsyncExecutor().numberThreadsInUse(threadSettings.numAllocations());
        LOG_DEBUG(<< "Updated number of allocations to [" << threadSettings.numAllocations()
                  << "] ([" << controlMessage.s_NumAllocations << "] requested)");
        writeThreadSettings(wrappedOutputStream, controlMessage.s_RequestId, threadSettings);
        break;
    case ml::torch::CCommandParser::E_ClearCache:
        cache.clear();
        writeSimpleAck(wrappedOutputStream, controlMessage.s_RequestId);
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

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream{ioMgr.outputStream()};

    writeThreadSettings(wrappedOutputStream,
                        ml::torch::CCommandParser::RESERVED_REQUEST_ID, threadSettings);

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
        [&module_, &wrappedOutputStream](
            ml::torch::CCommandParser::CRequestCacheInterface& cache,
            ml::torch::CCommandParser::SRequest request) -> bool {
            return handleRequest(cache, std::move(request), module_, wrappedOutputStream);
        },
        [&wrappedOutputStream, &threadSettings](
            ml::torch::CCommandParser::CRequestCacheInterface& cache,
            const ml::torch::CCommandParser::SControlMessage& controlMessage) {
            return handleControlMessage(controlMessage, threadSettings, cache,
                                        wrappedOutputStream);
        },
        [&wrappedOutputStream](const std::string& requestId, const std::string& message) {
            ml::core::CRapidJsonConcurrentLineWriter errorWriter(wrappedOutputStream);
            writeError(requestId, message, errorWriter);
        });

    // Stopping the executor forces this to block until all work is done
    ml::core::stopDefaultAsyncExecutor();

    LOG_DEBUG(<< "ML PyTorch inference process exiting");

    return EXIT_SUCCESS;
}
