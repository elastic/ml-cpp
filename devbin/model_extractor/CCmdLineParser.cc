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
namespace model_extractor {

const std::string CCmdLineParser::DESCRIPTION = "Usage: model_extractor [options]\n"
                                                "Options:";

bool CCmdLineParser::parse(int argc,
                           const char* const* argv,
                           std::string& logProperties,
                           std::string& inputFileName,
                           bool& isInputFileNamedPipe,
                           std::string& outputFileName,
                           bool& isOutputFileNamedPipe,
                           std::string& outputFormat) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        // clang-format off
        desc.add_options()
            ("help", "Display this information and exit")
            ("version", "Display version information and exit")
            ("logProperties", boost::program_options::value<std::string>(),
                        "Optional logger properties file")
            ("input", boost::program_options::value<std::string>(),
                        "Optional file to read input from - not present means read from STDIN")
            ("inputIsPipe", "Specified input file is a named pipe")
            ("output", boost::program_options::value<std::string>(),
                        "Optional file to write output to - not present means write to STDOUT")
            ("outputIsPipe", "Specified output file is a named pipe")
            ("outputFormat", boost::program_options::value<std::string>()->default_value("JSON"), "Format of output documents [JSON|XML].")

        ;
        // clang-format on

        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed =
            boost::program_options::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run();
        boost::program_options::store(parsed, vm);

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
        if (vm.count("logProperties") > 0) {
            logProperties = vm["logProperties"].as<std::string>();
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
        if (vm.count("outputFormat") > 0 &&
            (vm["outputFormat"].as<std::string>() == std::string("XML") ||
             vm["outputFormat"].as<std::string>() == std::string("JSON"))) {
            outputFormat = vm["outputFormat"].as<std::string>();
        } else {
            std::cerr << "Unknown output format \""
                      << vm["outputFormat"].as<std::string>()
                      << "\". Must be either JSON or XML." << std::endl;
            return false;
        }

    } catch (std::exception& e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}
}
}
