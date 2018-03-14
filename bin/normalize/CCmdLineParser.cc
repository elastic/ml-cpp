/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include "CCmdLineParser.h"

#include <ver/CBuildInfo.h>

#include <boost/program_options.hpp>

#include <iostream>

namespace ml {
namespace normalize {

const std::string CCmdLineParser::DESCRIPTION = "Usage: normalize [options]\n"
                                                "Options:";

bool CCmdLineParser::parse(int argc,
                           const char* const* argv,
                           std::string& modelConfigFile,
                           std::string& logProperties,
                           std::string& logPipe,
                           core_t::TTime& bucketSpan,
                           bool& lengthEncodedInput,
                           std::string& inputFileName,
                           bool& isInputFileNamedPipe,
                           std::string& outputFileName,
                           bool& isOutputFileNamedPipe,
                           std::string& quantilesState,
                           bool& deleteStateFiles,
                           bool& writeCsv,
                           bool& perPartitionNormalization) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        // clang-format off
        desc.add_options()
            ("help", "Display this information and exit")
            ("version", "Display version information and exit")
            ("modelconfig", boost::program_options::value<std::string>(),
                        "Optional model config file")
            ("logProperties", boost::program_options::value<std::string>(),
                        "Optional logger properties file")
            ("logPipe", boost::program_options::value<std::string>(),
                        "Optional log to named pipe")
            ("bucketspan", boost::program_options::value<core_t::TTime>(),
                        "Optional aggregation bucket span (in seconds) - default is 300")
            ("lengthEncodedInput",
                        "Take input in length encoded binary format - default is CSV")
            ("input", boost::program_options::value<std::string>(),
                        "Optional file to read input from - not present means read from STDIN")
            ("inputIsPipe", "Specified input file is a named pipe")
            ("output", boost::program_options::value<std::string>(),
                        "Optional file to write output to - not present means write to STDOUT")
            ("outputIsPipe", "Specified output file is a named pipe")
            ("quantilesState", boost::program_options::value<std::string>(),
                        "Optional file to initialization data for normalization (in JSON)")
            ("deleteStateFiles",
                        "If this flag is set then delete the normalizer state files once they have been read")
            ("writeCsv",
                        "Write the results in CSV format (default is lineified JSON)")
            ("perPartitionNormalization",
                        "Optional flag to enable per partition normalization")
        ;
        // clang-format on

        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc),
                                      vm);
        boost::program_options::notify(vm);

        if (vm.count("help") > 0) {
            std::cerr << desc << std::endl;
            return false;
        }
        if (vm.count("version") > 0) {
            std::cerr << ver::CBuildInfo::fullInfo() << std::endl;
            return false;
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
        if (vm.count("bucketspan") > 0) {
            bucketSpan = vm["bucketspan"].as<core_t::TTime>();
        }
        if (vm.count("lengthEncodedInput") > 0) {
            lengthEncodedInput = true;
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
        if (vm.count("quantilesState") > 0) {
            quantilesState = vm["quantilesState"].as<std::string>();
        }
        if (vm.count("deleteStateFiles") > 0) {
            deleteStateFiles = true;
        }
        if (vm.count("writeCsv") > 0) {
            writeCsv = true;
        }
        if (vm.count("perPartitionNormalization") > 0) {
            perPartitionNormalization = true;
        }
    } catch (std::exception& e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}
}
}
