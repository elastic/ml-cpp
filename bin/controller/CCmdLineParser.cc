/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CCmdLineParser.h"

#include <ver/CBuildInfo.h>

#include <boost/program_options.hpp>

#include <iostream>

namespace ml {
namespace controller {

const std::string CCmdLineParser::DESCRIPTION = "Usage: controller [options]\n"
                                                "Options";

bool CCmdLineParser::parse(int argc, const char* const* argv, std::string& jvmPidStr, std::string& logPipe, std::string& commandPipe) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        // clang-format off
        desc.add_options()
            ("help", "Display this information and exit")
            ("version", "Display version information and exit")
            ("jvmPid", boost::program_options::value<std::string>(),
                        "Process ID of the JVM to communicate with - default is parent process PID")
            ("logPipe", boost::program_options::value<std::string>(),
                        "Named pipe to log to - default is controller_log_<JVM PID>")
            ("commandPipe", boost::program_options::value<std::string>(),
                        "Named pipe to accept commands from - default is controller_command_<JVM PID>")
        ;
        // clang-format on

        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        if (vm.count("help") > 0) {
            std::cerr << desc << std::endl;
            return false;
        }
        if (vm.count("version") > 0) {
            std::cerr << ver::CBuildInfo::fullInfo() << std::endl;
            return false;
        }
        if (vm.count("jvmPid") > 0) {
            jvmPidStr = vm["jvmPid"].as<std::string>();
        }
        if (vm.count("logPipe") > 0) {
            logPipe = vm["logPipe"].as<std::string>();
        }
        if (vm.count("commandPipe") > 0) {
            commandPipe = vm["commandPipe"].as<std::string>();
        }
    } catch (std::exception& e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}
}
}
