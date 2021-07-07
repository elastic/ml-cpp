/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CBlockingCallCancellingTimer.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CRapidJsonLineWriter.h>
#include <core/CStopWatch.h>

#include <seccomp/CSystemCallFilter.h>

#include <torch/csrc/api/include/torch/types.h>
#include <ver/CBuildInfo.h>

#include <api/CIoManager.h>

#include "CBufferedIStreamAdapter.h"
#include "CCmdLineParser.h"
#include "CCommandParser.h"

#include <rapidjson/ostreamwrapper.h>
#include <torch/script.h>

#include <memory>
#include <sstream>
#include <string>

namespace {
const std::string INFERENCE{"inference"};
const std::string ERROR{"error"};
const std::string TIME_MS{"time_ms"};

ml::core::CStopWatch stopWatch;
}

torch::Tensor infer(torch::jit::script::Module& module,
                    ml::torch::CCommandParser::SRequest& request) {

    std::vector<torch::jit::IValue> inputs;

    if (request.hasTokens()) {
        inputs.reserve(1 + request.s_SecondaryArguments.size());

        // BERT UInt tokens
        inputs.emplace_back(
            torch::from_blob(static_cast<void*>(request.s_Tokens.data()),
                             {1, static_cast<std::int64_t>(request.s_Tokens.size())},
                             at::dtype(torch::kInt64)));

        for (auto& args : request.s_SecondaryArguments) {
            inputs.emplace_back(torch::from_blob(
                static_cast<void*>(args.data()),
                {1, static_cast<std::int64_t>(args.size())}, at::dtype(torch::kInt64)));
        }
    } else {
        // floating point inputs
        inputs.emplace_back(
            torch::from_blob(static_cast<void*>(request.s_Inputs.data()),
                             {1, static_cast<std::int64_t>(request.s_Inputs.size())},
                             at::dtype(torch::kFloat64))
                .to(torch::kFloat32));
    }

    torch::NoGradGuard noGrad;
    auto result = module.forward(inputs);
    if (result.isTuple()) {
        // For BERT models the result tensor is the first element in a tuple
        return result.toTuple()->elements()[0].toTensor();
    }
    return result.toTensor();
}

template<typename T>
void writeTensor(const torch::TensorAccessor<T, 1UL>& accessor,
                 ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>& jsonWriter) {
    jsonWriter.StartArray();
    for (int i = 0; i < accessor.size(0); ++i) {
        jsonWriter.Double(static_cast<double>(accessor[i]));
    }
    jsonWriter.EndArray();
}

template<typename T>
void writeTensor(const torch::TensorAccessor<T, 2UL>& accessor,
                 ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>& jsonWriter) {
    for (int i = 0; i < accessor.size(0); ++i) {
        writeTensor(accessor[i], jsonWriter);
    }
}

void writeError(const std::string& requestId,
                const std::string& message,
                ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>& jsonWriter) {
    jsonWriter.StartObject();
    jsonWriter.Key(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(ERROR);
    jsonWriter.String(message);
    jsonWriter.EndObject();
}

void writeDocumentOpening(const std::string& requestId,
                          std::uint64_t timeMs,
                          ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>& jsonWriter) {
    jsonWriter.StartObject();
    jsonWriter.Key(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(TIME_MS);
    jsonWriter.Uint64(timeMs);
    jsonWriter.Key(INFERENCE);
    jsonWriter.StartArray();
}

void writeDocumentClosing(ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>& jsonWriter) {
    jsonWriter.EndArray();
    jsonWriter.EndObject();
}

template<std::size_t N>
void writePrediction(const torch::Tensor& prediction,
                     const std::string& requestId,
                     std::uint64_t timeMs,
                     ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>& jsonWriter) {

    // creating the accessor will throw if the tensor does
    // not have exactly N dimensions. Do this before writing
    // any output so the error message isn't mingled with
    // a partial result

    if (prediction.dtype() == torch::kFloat32) {
        auto accessor = prediction.accessor<float, N>();

        writeDocumentOpening(requestId, timeMs, jsonWriter);
        writeTensor(accessor, jsonWriter);
        writeDocumentClosing(jsonWriter);

    } else if (prediction.dtype() == torch::kFloat64) {
        auto accessor = prediction.accessor<double, N>();

        writeDocumentOpening(requestId, timeMs, jsonWriter);
        writeTensor(accessor, jsonWriter);
        writeDocumentClosing(jsonWriter);
    } else {
        std::ostringstream ss;
        ss << "cannot process result tensor of type [" << prediction.dtype() << "]";
        writeError(requestId, ss.str(), jsonWriter);
    }
}

bool handleRequest(ml::torch::CCommandParser::SRequest& request,
                   torch::jit::script::Module& module,
                   ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>& jsonWriter) {

    try {
        LOG_DEBUG(<< "Inference request with id: " << request.s_RequestId);
        stopWatch.reset(true);
        torch::Tensor results = infer(module, request);
        std::uint64_t timeMs = stopWatch.stop();
        LOG_DEBUG(<< "Got results for request with id: " << request.s_RequestId);
        auto sizes = results.sizes();
        // Some models return a 3D tensor in which case
        // the first dimension must have size == 1
        if (sizes.size() == 3 && sizes[0] == 1) {
            writePrediction<2>(results[0], request.s_RequestId, timeMs, jsonWriter);
        } else if (sizes.size() == 2) {
            writePrediction<2>(results, request.s_RequestId, timeMs, jsonWriter);
        } else if (sizes.size() == 1) {
            writePrediction<1>(results, request.s_RequestId, timeMs, jsonWriter);
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
    bool validElasticLicenseKeyConfirmed{true};

    if (ml::torch::CCmdLineParser::parse(
            argc, argv, modelId, namedPipeConnectTimeout, inputFileName, isInputFileNamedPipe,
            outputFileName, isOutputFileNamedPipe, restoreFileName, isRestoreFileNamedPipe,
            logFileName, logProperties, validElasticLicenseKeyConfirmed) == false) {
        return EXIT_FAILURE;
    }

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

    rapidjson::OStreamWrapper writeStream(ioMgr.outputStream());
    ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper> jsonWriter(writeStream);

    jsonWriter.StartArray();
    commandParser.ioLoop(
        [&module, &jsonWriter](ml::torch::CCommandParser::SRequest& request) {
            return handleRequest(request, module, jsonWriter);
        },
        [&jsonWriter](const std::string& requestId, const std::string& message) {
            writeError(requestId, message, jsonWriter);
        });
    jsonWriter.EndArray();
    LOG_DEBUG(<< "ML Torch model prototype exiting");

    return EXIT_SUCCESS;
}
