/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CBlockingCallCancellingTimer.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CRapidJsonLineWriter.h>

#include <seccomp/CSystemCallFilter.h>

#include <ver/CBuildInfo.h>

#include <api/CIoManager.h>

#include "CBufferedIStreamAdapter.h"
#include "CCmdLineParser.h"
#include "CCommandParser.h"

#include <rapidjson/ostreamwrapper.h>
#include <torch/script.h>

#include <memory>
#include <string>

namespace {
const std::string INFERENCE{"inference"};
const std::string ERROR{"error"};
}

torch::Tensor infer(torch::jit::script::Module& module,
                    ml::torch::CCommandParser::SRequest& request) {

    torch::Tensor tokensTensor =
        torch::from_blob(static_cast<void*>(request.s_Tokens.data()),
                         {1, static_cast<std::int64_t>(request.s_Tokens.size())},
                         at::dtype(torch::kInt64));

    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(1 + request.s_SecondaryArguments.size());
    inputs.push_back(tokensTensor);

    for (auto& args : request.s_SecondaryArguments) {
        inputs.emplace_back(torch::from_blob(
            static_cast<void*>(args.data()),
            {1, static_cast<std::int64_t>(args.size())}, at::dtype(torch::kInt64)));
    }

    torch::NoGradGuard noGrad;
    auto tuple = module.forward(inputs).toTuple();
    return tuple->elements()[0].toTensor();
}

void writePrediction(const torch::Tensor& prediction,
                     const std::string& requestId,
                     std::ostream& outputStream) {

    torch::Tensor view;
    auto sizes = prediction.sizes();
    // Some models return a 3D tensor in which case
    // the first dimension must have size == 1
    if (sizes.size() == 3 && sizes[0] == 1) {
        view = prediction[0];
    } else {
        view = prediction;
    }

    // creating the accessor will throw if view does not
    // have exactly 2 dimensions. Do this before writing
    // any output so the error message isn't mingled with
    // a partial result
    auto accessor = view.accessor<float, 2>();

    rapidjson::OStreamWrapper writeStream(outputStream);
    ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper> jsonWriter(writeStream);
    jsonWriter.StartObject();
    jsonWriter.Key(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(INFERENCE);
    jsonWriter.StartArray();

    for (int i = 0; i < accessor.size(0); ++i) {
        jsonWriter.StartArray();
        for (int j = 0; j < accessor.size(1); ++j) {
            jsonWriter.Double(static_cast<double>(accessor[i][j]));
        }
        jsonWriter.EndArray();
    }

    jsonWriter.EndArray();
    jsonWriter.EndObject();
}

void writeError(const std::string& requestId, const std::string& message, std::ostream& outputStream) {
    rapidjson::OStreamWrapper writeStream(outputStream);
    ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper> jsonWriter(writeStream);
    jsonWriter.StartObject();
    jsonWriter.Key(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(ERROR);
    jsonWriter.String(message);
    jsonWriter.EndObject();
}

bool handleRequest(ml::torch::CCommandParser::SRequest& request,
                   torch::jit::script::Module& module,
                   std::ostream& outputStream) {

    try {
        torch::Tensor results = infer(module, request);
        writePrediction(results, request.s_RequestId, outputStream);
    } catch (std::runtime_error& e) {
        writeError(request.s_RequestId, e.what(), outputStream);
    }

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

    if (ml::torch::CCmdLineParser::parse(
            argc, argv, modelId, namedPipeConnectTimeout, inputFileName,
            isInputFileNamedPipe, outputFileName, isOutputFileNamedPipe, restoreFileName,
            isRestoreFileNamedPipe, logFileName, logProperties) == false) {
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

    commandParser.ioLoop([&module, &ioMgr](ml::torch::CCommandParser::SRequest& request) {
        return handleRequest(request, module, ioMgr.outputStream());
    });

    LOG_DEBUG(<< "ML Torch model prototype exiting");

    return EXIT_SUCCESS;
}
