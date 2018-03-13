/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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
namespace categorize {

const std::string CCmdLineParser::DESCRIPTION = "Usage: categorize [options]\n"
                                                "Options:";

bool CCmdLineParser::parse(int argc,
                           const char *const *argv,
                           std::string &limitConfigFile,
                           std::string &jobId,
                           std::string &logProperties,
                           std::string &logPipe,
                           char &delimiter,
                           bool &lengthEncodedInput,
                           core_t::TTime &persistInterval,
                           std::string &inputFileName,
                           bool &isInputFileNamedPipe,
                           std::string &outputFileName,
                           bool &isOutputFileNamedPipe,
                           std::string &restoreFileName,
                           bool &isRestoreFileNamedPipe,
                           std::string &persistFileName,
                           bool &isPersistFileNamedPipe,
                           std::string &categorizationFieldName) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        desc.add_options()("help", "Display this information and exit")(
            "version", "Display version information and exit")(
            "limitconfig",
            boost::program_options::value<std::string>(),
            "Optional limit config file")("jobid",
                                          boost::program_options::value<std::string>(),
                                          "ID of the job this process is associated with")(
            "logProperties",
            boost::program_options::value<std::string>(),
            "Optional logger properties file")(
            "logPipe", boost::program_options::value<std::string>(), "Optional log to named pipe")(
            "delimiter",
            boost::program_options::value<char>(),
            "Optional delimiter character for delimited data formats - default is '\t' (tab "
            "separated)")("lengthEncodedInput",
                          "Take input in length encoded binary format - default is delimited")(
            "input",
            boost::program_options::value<std::string>(),
            "Optional file to read input from - not present means read from STDIN")(
            "inputIsPipe", "Specified input file is a named pipe")(
            "output",
            boost::program_options::value<std::string>(),
            "Optional file to write output to - not present means write to STDOUT")(
            "outputIsPipe", "Specified output file is a named pipe")(
            "restore",
            boost::program_options::value<std::string>(),
            "Optional file to restore state from - not present means no state restoration")(
            "restoreIsPipe", "Specified restore file is a named pipe")(
            "persist",
            boost::program_options::value<std::string>(),
            "Optional file to persist state to - not present means no state persistence")(
            "persistIsPipe", "Specified persist file is a named pipe")(
            "persistInterval",
            boost::program_options::value<core_t::TTime>(),
            "Optional interval at which to periodically persist model state - if not specified "
            "then models will only be persisted at program exit")(
            "categorizationfield",
            boost::program_options::value<std::string>(),
            "Field to compute mlcategory from");

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
        if (vm.count("limitconfig") > 0) {
            limitConfigFile = vm["limitconfig"].as<std::string>();
        }
        if (vm.count("jobid") > 0) {
            jobId = vm["jobid"].as<std::string>();
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
        if (vm.count("persistInterval") > 0) {
            persistInterval = vm["persistInterval"].as<core_t::TTime>();
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
        if (vm.count("categorizationfield") > 0) {
            categorizationFieldName = vm["categorizationfield"].as<std::string>();
        }
    } catch (std::exception &e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}
}
}
