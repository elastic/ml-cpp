/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CCmdLineParser.h"

#include <ver/CBuildInfo.h>

#include <model/CAnomalyDetector.h>
#include <model/CAnomalyScore.h>

#include <boost/program_options.hpp>

#include <iostream>

namespace ml {
namespace autodetect {

const std::string CCmdLineParser::DESCRIPTION = "Usage: autodetect [options]]\n"
                                                "Options:";

bool CCmdLineParser::parse(int argc,
                           const char* const* argv,
                           std::string& configFile,
                           std::string& filtersConfigFile,
                           std::string& eventsConfigFile,
                           std::string& modelConfigFile,
                           std::string& logProperties,
                           std::string& logPipe,
                           char& delimiter,
                           bool& lengthEncodedInput,
                           std::string& timeFormat,
                           std::string& quantilesState,
                           bool& deleteStateFiles,
                           std::size_t& bucketPersistInterval,
                           core_t::TTime& namedPipeConnectTimeout,
                           std::string& inputFileName,
                           bool& isInputFileNamedPipe,
                           std::string& outputFileName,
                           bool& isOutputFileNamedPipe,
                           std::string& restoreFileName,
                           bool& isRestoreFileNamedPipe,
                           std::string& persistFileName,
                           bool& isPersistFileNamedPipe,
                           bool& isPersistInForeground,
                           std::size_t& maxAnomalyRecords,
                           bool& memoryUsage) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        // clang-format off
        desc.add_options()
            ("help", "Display this information and exit")
            ("version", "Display version information and exit")
            ("config", boost::program_options::value<std::string>(),
                    "The job configuration file")
            ("filtersconfig", boost::program_options::value<std::string>(),
                    "The filters configuration file")
            ("eventsconfig", boost::program_options::value<std::string>(),
                    "The scheduled events configuration file")
            ("modelconfig", boost::program_options::value<std::string>(),
                    "Optional model config file")
            ("logProperties", boost::program_options::value<std::string>(),
                    "Optional logger properties file")
            ("logPipe", boost::program_options::value<std::string>(),
                    "Optional log to named pipe")
            ("delimiter", boost::program_options::value<char>(),
                    "Optional delimiter character for delimited data formats - default is '\t' (tab separated)")
            ("lengthEncodedInput",
                    "Take input in length encoded binary format - default is delimited")
            ("timeformat", boost::program_options::value<std::string>(),
                    "Optional format of the date in the time field in strptime code - default is the epoch time in seconds")
            ("quantilesState", boost::program_options::value<std::string>(),
                    "Optional file to quantiles for normalization")
            ("deleteStateFiles",
                    "If the 'quantilesState' option is used and this flag is set then delete the model state files once they have been read")
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
                    "Optional file to persist state to - not present means no state persistence")
            ("persistIsPipe", "Specified persist file is a named pipe")
            ("persistInForeground", "Persistence occurs in the foreground. Defaults to background persistence.")
            ("bucketPersistInterval", boost::program_options::value<std::size_t>(),
                    "Optional number of buckets after which to periodically persist model state.")
            ("maxAnomalyRecords", boost::program_options::value<std::size_t>(),
                    "The maximum number of records to be outputted for each bucket. Defaults to 100, a value 0 removes the limit.")
            ("memoryUsage",
                    "Log the model memory usage at the end of the job")
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
            std::cerr << "Model State Version "
                      << model::CAnomalyDetector::STATE_VERSION << std::endl
                      << "Quantile State Version "
                      << model::CAnomalyScore::CURRENT_FORMAT_VERSION << std::endl
                      << ver::CBuildInfo::fullInfo() << std::endl;
            return false;
        }
        if (vm.count("config") == 0) {
            std::cerr << "Error processing command line: the option '--config' is required but missing";
            return false;
        }

        configFile = vm["config"].as<std::string>();
        if (vm.count("filtersconfig") > 0) {
            filtersConfigFile = vm["filtersconfig"].as<std::string>();
        }
        if (vm.count("eventsconfig") > 0) {
            eventsConfigFile = vm["eventsconfig"].as<std::string>();
        }
        if (vm.count("modelconfig") > 0) {
            modelConfigFile = vm["modelconfig"].as<std::string>();
        }
        if (vm.count("logProperties") > 0) {
            logProperties = vm["logProperties"].as<std::string>();
        }
        if (vm.count("logPipe") > 0) {
            logPipe = vm["logPipe"].as<std::string>();
        }
        if (vm.count("delimiter") > 0) {
            delimiter = vm["delimiter"].as<char>();
        }
        if (vm.count("lengthEncodedInput") > 0) {
            lengthEncodedInput = true;
        }
        if (vm.count("timeformat") > 0) {
            timeFormat = vm["timeformat"].as<std::string>();
        }
        if (vm.count("quantilesState") > 0) {
            quantilesState = vm["quantilesState"].as<std::string>();
        }
        if (vm.count("deleteStateFiles") > 0) {
            deleteStateFiles = true;
        }
        if (vm.count("bucketPersistInterval") > 0) {
            bucketPersistInterval = vm["bucketPersistInterval"].as<std::size_t>();
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
        if (vm.count("persistInForeground") > 0) {
            isPersistInForeground = true;
        }
        if (vm.count("maxAnomalyRecords") > 0) {
            maxAnomalyRecords = vm["maxAnomalyRecords"].as<std::size_t>();
        }
        if (vm.count("memoryUsage") > 0) {
            memoryUsage = true;
        }
    } catch (std::exception& e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}
}
}
