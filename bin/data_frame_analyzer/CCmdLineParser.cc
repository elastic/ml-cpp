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
namespace data_frame_analyzer {

const std::string CCmdLineParser::DESCRIPTION = "Usage: data_frame_analyzer [options]\n"
                                                "Options:";

bool CCmdLineParser::parse(int argc,
                           const char* const* argv,
                           std::string& configFile,
                           bool& memoryUsageEstimationOnly,
                           std::string& logProperties,
                           std::string& logPipe,
                           bool& lengthEncodedInput,
                           core_t::TTime& namedPipeConnectTimeout,
                           std::string& inputFileName,
                           bool& isInputFileNamedPipe,
                           std::string& outputFileName,
                           bool& isOutputFileNamedPipe,
                           std::string& restoreFileName,
                           bool& isRestoreFileNamedPipe,
                           std::string& persistFileName,
                           bool& isPersistFileNamedPipe,
                           bool& validElasticLicenseKeyConfirmed) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        // clang-format off
        desc.add_options()
            ("help", "Display this information and exit")
            ("version", "Display version information and exit")
            ("config", boost::program_options::value<std::string>(),
                    "The configuration file")
            ("memoryUsageEstimationOnly", "Whether to perform memory usage estimation only")
            ("logProperties", boost::program_options::value<std::string>(),
                    "Optional logger properties file")
            ("logPipe", boost::program_options::value<std::string>(),
                    "Optional log to named pipe")
            ("lengthEncodedInput",
                    "Take input in length encoded binary format - default is CSV")
            ("namedPipeConnectTimeout", boost::program_options::value<core_t::TTime>(),
                    "Optional timeout (in seconds) for connecting named pipes on startup - default is 300 seconds")
            ("input", boost::program_options::value<std::string>(),
                    "Optional file to read input from - not present means read from STDIN")
            ("inputIsPipe", "Specified input file is a named pipe")
            ("output", boost::program_options::value<std::string>(),
                    "Optional file to write output to - not present means write to STDOUT")
            ("outputIsPipe", "Specified output file is a named pipe")
            ("restore", boost::program_options::value<std::string>(),
                    "Optional file to restore state from - not present means no state restoration")
            ("restoreIsPipe", "Specified restore file is a named pipe")
            ("persist", boost::program_options::value<std::string>(),
                    "File to persist state to - not present means no state persistence")
            ("persistIsPipe", "Specified persist file is a named pipe")
            ("validElasticLicenseKeyConfirmed", boost::program_options::value<bool>(),
             "Confirmation that a valid Elastic license key is in use.")
        ;
        // clang-format on

        boost::program_options::variables_map vm;
        boost::program_options::store(
            boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        if (vm.count("help") > 0) {
            std::cerr << desc << std::endl;
            return false;
        }
        if (vm.count("version") > 0) {
            std::cerr << ver::CBuildInfo::fullInfo() << std::endl;
            return false;
        }
        if (vm.count("config") > 0) {
            configFile = vm["config"].as<std::string>();
        }
        if (vm.count("memoryUsageEstimationOnly") > 0) {
            memoryUsageEstimationOnly = true;
        }
        if (vm.count("logProperties") > 0) {
            logProperties = vm["logProperties"].as<std::string>();
        }
        if (vm.count("logPipe") > 0) {
            logPipe = vm["logPipe"].as<std::string>();
        }
        if (vm.count("lengthEncodedInput") > 0) {
            lengthEncodedInput = true;
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
        if (vm.count("persist") > 0) {
            persistFileName = vm["persist"].as<std::string>();
        }
        if (vm.count("persistIsPipe") > 0) {
            isPersistFileNamedPipe = true;
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
