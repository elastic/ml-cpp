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
namespace domain_name_entropy {


const std::string CCmdLineParser::DESCRIPTION =
    "Usage: domain_name_entropy [options]\n"
    "Options:";


bool CCmdLineParser::parse(int argc,
                           const char * const *argv,
                           std::string &csvFileName,
                           std::string &domainNameField,
                           std::string &timeField) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        desc.add_options()
        ("help", "Display this information and exit")
        ("version", "Display version information and exit")
        ("csvfilename", boost::program_options::value<std::string>(),
         "Csv file name string")
        ("domainfieldname", boost::program_options::value<std::string>(),
         "Domain field name string")
        ("timefieldname", boost::program_options::value<std::string>(),
         "Time field name string")
        ;

        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed =
            boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
        boost::program_options::store(parsed, vm);

        if (vm.count("help") > 0) {
            std::cerr << desc << std::endl;
            return false;
        }
        if (vm.count("version") > 0) {
            std::cerr << ver::CBuildInfo::fullInfo() << std::endl;
            return false;
        }
        if (vm.count("domainfieldname") > 0) {
            domainNameField = vm["domainfieldname"].as<std::string>();
        } else {
            std::cerr << desc << std::endl;
            return false;
        }
        if (vm.count("timefieldname") > 0) {
            timeField = vm["timefieldname"].as<std::string>();
        } else {
            std::cerr << desc << std::endl;
            return false;
        }
        if (vm.count("csvfilename") > 0) {
            csvFileName = vm["csvfilename"].as<std::string>();
        } else {
            std::cerr << desc << std::endl;
            return false;
        }
    } catch (std::exception &e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}


}
}

