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
namespace autoconfig {

const std::string CCmdLineParser::DESCRIPTION =
    "Usage: autoconfig [options]\n"
    "Options";

bool CCmdLineParser::parse(int argc,
                           const char * const *argv,
                           std::string &logProperties,
                           std::string &logPipe,
                           char &delimiter,
                           bool &lengthEncodedInput,
                           std::string &timeField,
                           std::string &timeFormat,
                           std::string &configFile,
                           std::string &inputFileName,
                           bool &isInputFileNamedPipe,
                           std::string &outputFileName,
                           bool &isOutputFileNamedPipe,
                           bool &verbose,
                           bool &writeDetectorConfigs) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        desc.add_options()
            ("help", "Display this information and exit")
            ("version", "Display version information and exit")
            ("logProperties", boost::program_options::value<std::string>(),
            "Optional logger properties file")
            ("logPipe", boost::program_options::value<std::string>(),
            "Optional log to named pipe")
            ("delimiter", boost::program_options::value<char>(),
            "Optional delimiter character for delimited data formats - default is ',' (comma separated)")
            ("lengthEncodedInput",
            "Take input in length encoded binary format - default is delimited")
            ("timefield", boost::program_options::value<std::string>(),
            "Optional name of the field containing the timestamp - default is 'time'")
            ("timeformat", boost::program_options::value<std::string>(),
            "Optional format of the date in the time field in strptime code - default is the epoch time in seconds")
            ("config", boost::program_options::value<std::string>(),
            "Optional configuration file")
            ("input", boost::program_options::value<std::string>(),
            "Optional file to read input from - not present means read from STDIN")
            ("inputIsPipe", "Specified input file is a named pipe")
            ("output", boost::program_options::value<std::string>(),
            "Optional file to write output to - not present means write to STDOUT")
            ("outputIsPipe", "Specified output file is a named pipe")
            ("verbose", "Output information about all detectors including those that have been discarded")
            ("writeDetectorConfigs",
            "Output the detector configurations in JSON format")
        ;

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
        if (vm.count("timefield") > 0) {
            timeField = vm["timefield"].as<std::string>();
        }
        if (vm.count("timeformat") > 0) {
            timeFormat = vm["timeformat"].as<std::string>();
        }
        if (vm.count("config") > 0) {
            configFile = vm["config"].as<std::string>();
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
        if (vm.count("verbose") > 0) {
            verbose = true;
        }
        if (vm.count("writeDetectorConfigs") > 0) {
            writeDetectorConfigs = true;
        }
    } catch (std::exception &e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}

}
}
