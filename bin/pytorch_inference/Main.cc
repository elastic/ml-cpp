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
#include "SettingsValidator.h"

#include <ATen/Parallel.h>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/script.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

namespace {
const std::string RESULT{"result"};
const std::string INFERENCE{"inference"};
const std::string ERROR{"error"};
const std::string TIME_MS{"time_ms"};
const std::string THREAD_SETTINGS{"thread_settings"};
const std::string INFERENCE_THREADS{"inference_threads"};
const std::string MODEL_THREADS{"model_threads"};
}

torch::Tensor infer(torch::jit::script::Module& module,
                    ml::torch::CCommandParser::SRequest& request) {

    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(1 + request.s_SecondaryArguments.size());

    std::array<std::int64_t, 2> dimensions = {request.s_NumberInferences,
                                              request.s_NumberInputTokens};
    at::IntArrayRef inputSize{dimensions};

    // BERT UInt tokens
    inputs.emplace_back(torch::from_blob(static_cast<void*>(request.s_Tokens.data()),
                                         inputSize, at::dtype(torch::kInt64)));

    for (auto& args : request.s_SecondaryArguments) {
        inputs.emplace_back(torch::from_blob(static_cast<void*>(args.data()),
                                             inputSize, at::dtype(torch::kInt64)));
    }

    torch::InferenceMode inferenceModeGuard;
    auto result = module.forward(inputs);
    if (result.isTuple()) {
        // For BERT models the result tensor is the first element in a tuple
        return result.toTuple()->elements()[0].toTensor();
    }
    return result.toTensor();
}

template<typename T>
void writeTensor(const torch::TensorAccessor<T, 1UL>& accessor,
                 ml::core::CRapidJsonConcurrentLineWriter& jsonWriter) {
    jsonWriter.StartArray();
    for (int i = 0; i < accessor.size(0); ++i) {
        jsonWriter.Double(static_cast<double>(accessor[i]));
    }
    jsonWriter.EndArray();
}

template<typename T, std::size_t N_DIMS>
void writeTensor(const torch::TensorAccessor<T, N_DIMS>& accessor,
                 ml::core::CRapidJsonConcurrentLineWriter& jsonWriter) {
    jsonWriter.StartArray();
    for (int i = 0; i < accessor.size(0); ++i) {
        writeTensor(accessor[i], jsonWriter);
    }
    jsonWriter.EndArray();
}

template<typename T>
void writeInferenceResults(const torch::TensorAccessor<T, 3UL>& accessor,
                           ml::core::CRapidJsonConcurrentLineWriter& jsonWriter) {

    jsonWriter.Key(INFERENCE);
    writeTensor(accessor, jsonWriter);
}

template<typename T>
void writeInferenceResults(const torch::TensorAccessor<T, 2UL>& accessor,
                           ml::core::CRapidJsonConcurrentLineWriter& jsonWriter) {

    jsonWriter.Key(INFERENCE);
    // output must be a 3D array so wrap the 2D result in an outer array
    jsonWriter.StartArray();
    writeTensor(accessor, jsonWriter);
    jsonWriter.EndArray();
}

void writeError(const std::string& requestId,
                const std::string& message,
                ml::core::CRapidJsonConcurrentLineWriter& jsonWriter) {
    jsonWriter.StartObject();
    jsonWriter.Key(RESULT);
    jsonWriter.StartObject();
    jsonWriter.Key(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(ERROR);
    jsonWriter.String(message);
    jsonWriter.EndObject();
    jsonWriter.EndObject();
}

void writeDocumentOpening(const std::string& requestId,
                          std::uint64_t timeMs,
                          ml::core::CRapidJsonConcurrentLineWriter& jsonWriter) {
    jsonWriter.StartObject();
    jsonWriter.Key(RESULT);
    jsonWriter.StartObject();
    jsonWriter.Key(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(TIME_MS);
    jsonWriter.Uint64(timeMs);
}

void writeDocumentClosing(ml::core::CRapidJsonConcurrentLineWriter& jsonWriter) {
    jsonWriter.EndObject();
    jsonWriter.EndObject();
}

void writeThreadSettings(ml::core::CJsonOutputStreamWrapper& wrappedOutputStream,
                         std::int32_t inferenceThreads,
                         std::int32_t modelThreads) {
    ml::core::CRapidJsonConcurrentLineWriter jsonWriter(wrappedOutputStream);
    jsonWriter.StartObject();
    jsonWriter.Key(THREAD_SETTINGS);
    jsonWriter.StartObject();
    jsonWriter.Key(INFERENCE_THREADS);
    jsonWriter.Uint64(inferenceThreads);
    jsonWriter.Key(MODEL_THREADS);
    jsonWriter.Uint64(modelThreads);
    jsonWriter.EndObject();
    jsonWriter.EndObject();
}

template<std::size_t N>
void writePrediction(const torch::Tensor& prediction,
                     const std::string& requestId,
                     std::uint64_t timeMs,
                     ml::core::CRapidJsonConcurrentLineWriter& jsonWriter) {

    // creating the accessor will throw if the tensor does
    // not have exactly N dimensions. Do this before writing
    // any output so the error message isn't mingled with
    // a partial result

    if (prediction.dtype() == torch::kFloat32) {
        auto accessor = prediction.accessor<float, N>();

        writeDocumentOpening(requestId, timeMs, jsonWriter);
        writeInferenceResults(accessor, jsonWriter);
        writeDocumentClosing(jsonWriter);

    } else if (prediction.dtype() == torch::kFloat64) {
        auto accessor = prediction.accessor<double, N>();

        writeDocumentOpening(requestId, timeMs, jsonWriter);
        writeInferenceResults(accessor, jsonWriter);
        writeDocumentClosing(jsonWriter);
    } else {
        std::ostringstream ss;
        ss << "cannot process result tensor of type [" << prediction.dtype() << "]";
        writeError(requestId, ss.str(), jsonWriter);
    }
}

void inferAndWriteResult(ml::torch::CCommandParser::SRequest& request,
                         torch::jit::script::Module& module,
                         ml::core::CRapidJsonConcurrentLineWriter& jsonWriter) {
    try {
        ml::core::CStopWatch stopWatch(true);
        torch::Tensor results = infer(module, request);
        std::uint64_t timeMs = stopWatch.stop();
        auto sizes = results.sizes();

        // The output is always a 3D array, in the case of a 2D result
        // it must be wrapped in an outer array
        if (sizes.size() == 3) {
            writePrediction<3>(results, request.s_RequestId, timeMs, jsonWriter);
        } else if (sizes.size() == 2) {
            writePrediction<2>(results, request.s_RequestId, timeMs, jsonWriter);
        } else {
            std::ostringstream ss;
            ss << "Cannot convert results tensor of size [" << sizes << "]";
            writeError(request.s_RequestId, ss.str(), jsonWriter);
        }
    } catch (const c10::Error& e) {
        writeError(request.s_RequestId, e.what(), jsonWriter);
    } catch (std::runtime_error& e) {
        writeError(request.s_RequestId, e.what(), jsonWriter);
    }
    jsonWriter.Flush();
}

bool handleRequest(const ml::torch::CCommandParser::SRequest& request,
                   torch::jit::script::Module& module,
                   ml::core::CJsonOutputStreamWrapper& wrappedOutputStream) {

    ml::core::async(
        ml::core::defaultAsyncExecutor(),
        [ requestCopy = request, &module, &wrappedOutputStream ]() mutable {
            ml::core::CRapidJsonConcurrentLineWriter jsonWriter(wrappedOutputStream);
            inferAndWriteResult(requestCopy, module, jsonWriter);
        });
    return true;
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
    std::int32_t inferenceThreads{1};
    std::int32_t modelThreads{1};
    bool validElasticLicenseKeyConfirmed{false};

    if (ml::torch::CCmdLineParser::parse(
            argc, argv, modelId, namedPipeConnectTimeout, inputFileName,
            isInputFileNamedPipe, outputFileName, isOutputFileNamedPipe, restoreFileName,
            isRestoreFileNamedPipe, logFileName, logProperties, inferenceThreads,
            modelThreads, validElasticLicenseKeyConfirmed) == false) {
        return EXIT_FAILURE;
    }

    ml::torch::validateThreadingParameters(
        static_cast<int32_t>(std::thread::hardware_concurrency()),
        inferenceThreads, modelThreads);

    // Setting the number of threads used by libtorch also sets
    // the number of threads used by MKL or OMP libs. However,
    // this doesn't address the Accelerated.Framework found on macs.
    // Thus, we set the environment variable that controls threading for that one.
    // It doesn't hurt to set variables that won't have any effect on some platforms.
    ml::core::CSetEnv::setEnv(
        "VECLIB_MAXIMUM_THREADS",
        ml::core::CStringUtils::typeToString(inferenceThreads).c_str(), 0);

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

    at::set_num_threads(inferenceThreads);

    // This is not used as we don't call at::launch anywhere.
    // Setting it to 1 to ensure there is no thread pool sitting around.
    at::set_num_interop_threads(1);

    LOG_DEBUG(<< at::get_parallel_info());
    LOG_DEBUG(<< "Model threads: " << modelThreads);

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream{ioMgr.outputStream()};

    writeThreadSettings(wrappedOutputStream, inferenceThreads, modelThreads);

    torch::jit::script::Module module;
    try {
        auto readAdapter = std::make_unique<ml::torch::CBufferedIStreamAdapter>(
            *ioMgr.restoreStream());
        if (readAdapter->init() == false) {
            return EXIT_FAILURE;
        }
        module = torch::jit::load(std::move(readAdapter));
        module.eval();

        LOG_DEBUG(<< "model loaded");
    } catch (const c10::Error& e) {
        LOG_FATAL(<< "Error loading the model: " << e.msg());
        return EXIT_FAILURE;
    }

    ml::torch::CCommandParser commandParser{ioMgr.inputStream()};

    // Starting the executor with 1 thread will use an extra thread that isn't necessary
    // so we only start it when more than 1 threads are set.
    if (modelThreads > 1) {
        ml::core::startDefaultAsyncExecutor(modelThreads);
    }

    commandParser.ioLoop(
        [&module, &wrappedOutputStream](const ml::torch::CCommandParser::SRequest& request) {
            return handleRequest(request, module, wrappedOutputStream);
        },
        [&wrappedOutputStream](const std::string& requestId, const std::string& message) {
            ml::core::CRapidJsonConcurrentLineWriter errorWriter(wrappedOutputStream);
            writeError(requestId, message, errorWriter);
        });

    // Stopping the executor forces this to block until all work is done
    ml::core::stopDefaultAsyncExecutor();

    LOG_DEBUG(<< "ML Torch model prototype exiting");

    return EXIT_SUCCESS;
}
