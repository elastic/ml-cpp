/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CCmdLineParser.h"


#include <boost/program_options.hpp>

#include <iostream>

namespace ml {
namespace torch {

const std::string CCmdLineParser::DESCRIPTION = "Usage: pytorch_inference [options]\n"
                                                "Options:";

bool CCmdLineParser::parse(int argc,
                           const char* const* argv,
                           std::string& modelId) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        // clang-format off
        desc.add_options()
            ("help", "Display this information and exit")
            ("version", "Display version information and exit")
            ("modelid", boost::program_options::value<std::string>(),
                        "The TorchScript model this process is associated with")
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
            std::cerr << "Pytorch prototype " << std::endl;
            return false;
        }
        if (vm.count("modelid") > 0) {
             modelId = vm["modelid"].as<std::string>();
        }

    } catch (std::exception& e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}
}
}
