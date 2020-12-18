/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CIoManager.h>
#include <core/CBlockingCallCancellingTimer.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CRapidJsonLineWriter.h>
#include <core/CoreTypes.h>
#include <seccomp/CSystemCallFilter.h>

#include <rapidjson/ostreamwrapper.h>
#include <torch/script.h>

#include "CBufferedIStreamAdapter.h"
#include "CCmdLineParser.h"

#include <string>


torch::Tensor infer(torch::jit::script::Module& module, std::vector<float>& data) {
    torch::Tensor tokens_tensor = torch::from_blob(data.data(), {1, static_cast<long long>(data.size())}).to(torch::kInt64);    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tokens_tensor);
    inputs.push_back(torch::ones({1, static_cast<std::int64_t>(data.size())}));  // attention mask
    inputs.push_back(torch::zeros({1, static_cast<std::int64_t>(data.size())}).to(torch::kInt64)); // token type ids
    inputs.push_back(torch::arange(static_cast<std::int64_t>(data.size())).to(torch::kInt64)); // position ids

    torch::NoGradGuard no_grad;
    auto tuple = module.forward(inputs).toTuple();  
    auto predictions = tuple->elements()[0].toTensor();

    return torch::argmax(predictions, 2);
}

std::uint32_t readUInt32(std::istream& stream) {
    std::uint32_t netNum{0};
    stream.read(reinterpret_cast<char*>(&netNum), sizeof(std::uint32_t));
    return ntohl(netNum);    
}

std::vector<float> readTokens(std::istream& inputStream) {
    // return a float vector rather than integers because 
    // float is need to create the tensor    
    std::vector<float> tokens;
    std::uint32_t numTokens = readUInt32(inputStream);
    for (uint32_t i=0; i<numTokens; ++i) {
        tokens.push_back(readUInt32(inputStream));
    }

    return tokens;
}

void writePrediction(torch::Tensor& prediction, std::ostream& outputStream) {
    rapidjson::OStreamWrapper writeStream(outputStream);
    ml::core::CRapidJsonLineWriter<rapidjson::OStreamWrapper> jsonWriter(writeStream);
    jsonWriter.StartObject();
    jsonWriter.Key("inference");
    jsonWriter.StartArray();
    auto arr = prediction.accessor<std::int64_t, 2>();
    for(int i = 0; i < arr.size(1); i++) {                
        jsonWriter.Int64(arr[0][i]);
    }
    jsonWriter.EndArray();
    jsonWriter.EndObject();
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
    std::string persistFileName;
    ml::core_t::TTime namedPipeConnectTimeout{
        ml::core::CBlockingCallCancellingTimer::DEFAULT_TIMEOUT_SECONDS};

    if (ml::torch::CCmdLineParser::parse(argc, argv, 
                                        modelId, 
                                        namedPipeConnectTimeout,
                                        inputFileName,
                                        isInputFileNamedPipe,
                                        outputFileName,
                                        isOutputFileNamedPipe,
                                        restoreFileName, 
                                        isRestoreFileNamedPipe) == false) {
        return EXIT_FAILURE;
    }

    ml::core::CBlockingCallCancellingTimer cancellerThread{
        ml::core::CThread::currentThreadId(), std::chrono::seconds{namedPipeConnectTimeout}};

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    const std::string EMPTY;
    ml::api::CIoManager ioMgr{cancellerThread, 
        inputFileName, isInputFileNamedPipe,
        outputFileName, isOutputFileNamedPipe,         
        restoreFileName, isRestoreFileNamedPipe,
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
    ml::seccomp::CSystemCallFilter::installSystemCallFilter();


    if (ioMgr.initIo() == false) {
        LOG_FATAL(<< "Failed to initialise IO");        
        return EXIT_FAILURE;
    }

    
    torch::jit::script::Module module;
    try {    
        auto readAdapter = std::make_unique<ml::torch::CBufferedIStreamAdapter>(ioMgr.restoreStream());            
        module = torch::jit::load(std::move(readAdapter));        
        module.eval();

        LOG_DEBUG(<< "model loaded");
    }
    catch (const c10::Error& e) {                        
        LOG_FATAL(<< "Error loading the model: " << e.msg());
        return EXIT_FAILURE;
    }

    std::vector<float> tokens = readTokens(ioMgr.inputStream());
    torch::Tensor results = infer(module, tokens);
    writePrediction(results, ioMgr.outputStream());


    LOG_DEBUG(<< "ML Torch model prototype exiting");

    return EXIT_SUCCESS;
}
