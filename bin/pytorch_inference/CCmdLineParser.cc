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
#include "CCmdLineParser.h"

#include <ver/CBuildInfo.h>

#include <boost/program_options.hpp>

#include <iostream>

namespace ml {
namespace torch {

const std::string CCmdLineParser::DESCRIPTION = "Usage: pytorch_inference [options]\n"
                                                "Options:";

bool CCmdLineParser::parse(int argc,
                           const char* const* argv,
                           std::string& modelId,
                           core_t::TTime& namedPipeConnectTimeout,
                           std::string& inputFileName,
                           bool& isInputFileNamedPipe,
                           std::string& outputFileName,
                           bool& isOutputFileNamedPipe,
                           std::string& restoreFileName,
                           bool& isRestoreFileNamedPipe,
                           std::string& loggingFileName,
                           std::string& logProperties,
                           std::int32_t& numLibTorchThreads,
                           std::int32_t& numLibTorchInterOpThreads,
                           std::int32_t& numParallelForwardingThreads,
                           bool& validElasticLicenseKeyConfirmed) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        // clang-format off
        desc.add_options()
            ("help", "Display this information and exit")
            ("version", "Display version information and exit")
            ("modelid", boost::program_options::value<std::string>(),
                        "The TorchScript model this process is associated with")
            ("namedPipeConnectTimeout", boost::program_options::value<core_t::TTime>(),
                        "Optional timeout (in seconds) for connecting named pipes on startup - default is 300 seconds")
            ("input", boost::program_options::value<std::string>(),
                        "Optional file to read input from - not present means read from STDIN")
            ("inputIsPipe", "Specified input file is a named pipe")        
            ("output", boost::program_options::value<std::string>(),
                        "Optional file to write output to - not present means write to STDOUT")
            ("outputIsPipe", "Specified output file is a named pipe")
            ("restore", boost::program_options::value<std::string>(),
                        "Named pipe to read model from")    
            ("restoreIsPipe", "Specified restore file is a named pipe")
            ("logPipe", boost::program_options::value<std::string>(),
                        "Named pipe to write log messages to")
            ("logProperties", "Optional logger properties file")
            ("numLibTorchThreads", boost::program_options::value<std::int32_t>(),
                        "Optionaly set number of threads LibTorch can use for inference - not present means use the LibTorch defaults")
            ("numLibTorchInterOpThreads", boost::program_options::value<std::int32_t>(),
                        "Optionaly set number of threads LibTorch can use for inter operation parallelism - not present means use the LibTorch defaults")
            ("numParallelForwardingThreads", boost::program_options::value<std::int32_t>(),
                        "Optionaly set number of threads to parallelize model forwarding - not present means 1")
            ("validElasticLicenseKeyConfirmed", boost::program_options::value<bool>(),
             "Confirmation that a valid Elastic license key is in use.")
            ;
        // clang-format on

        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed =
            boost::program_options::command_line_parser(argc, argv)
                .options(desc)
                .run();
        boost::program_options::store(parsed, vm);

        if (vm.count("help") > 0) {
            std::cerr << desc << std::endl;
            return false;
        }
        if (vm.count("version") > 0) {
            std::cerr << ver::CBuildInfo::fullInfo() << std::endl;
            return false;
        }
        if (vm.count("modelid") > 0) {
            modelId = vm["modelid"].as<std::string>();
        }
        if (vm.count("namedPipeConnectTimeout") > 0) {
            namedPipeConnectTimeout = vm["namedPipeConnectTimeout"].as<core_t::TTime>();
        }
        if (vm.count("input") > 0) {
            inputFileName = vm["input"].as<std::string>();
        }
        if (vm.count("inputIsPipe") > 0) {
            isInputFileNamedPipe = true;
        }
        if (vm.count("output") > 0) {
            outputFileName = vm["output"].as<std::string>();
        }
        if (vm.count("outputIsPipe") > 0) {
            isOutputFileNamedPipe = true;
        }
        if (vm.count("restore") > 0) {
            restoreFileName = vm["restore"].as<std::string>();
        }
        if (vm.count("restoreIsPipe") > 0) {
            isRestoreFileNamedPipe = true;
        }
        if (vm.count("logPipe") > 0) {
            loggingFileName = vm["logPipe"].as<std::string>();
        }
        if (vm.count("logProperties") > 0) {
            logProperties = vm["logProperties"].as<std::string>();
        }
        if (vm.count("numLibTorchThreads") > 0) {
            numLibTorchThreads = vm["numLibTorchThreads"].as<std::int32_t>();
        }
        if (vm.count("numLibTorchInterOpThreads") > 0) {
            numLibTorchInterOpThreads = vm["numLibTorchInterOpThreads"].as<std::int32_t>();
        }
        if (vm.count("numParallelForwardingThreads") > 0) {
            numParallelForwardingThreads =
                vm["numParallelForwardingThreads"].as<std::int32_t>();
        }
        if (vm.count("validElasticLicenseKeyConfirmed") > 0) {
            validElasticLicenseKeyConfirmed =
                vm["validElasticLicenseKeyConfirmed"].as<bool>();
        }
    } catch (std::exception& e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}
}
}
