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
#include <core/CResourceLocator.h>

#include <core/COsFileFuncs.h>
#include <core/CProgName.h>

#include <stdlib.h>

namespace {
const char* CPP_SRC_HOME("CPP_SRC_HOME");
}

namespace ml {
namespace core {

std::string CResourceLocator::resourceDir() {
    // Look relative to the program that's running, assuming this directory layout:
    // $ES_HOME/plugin/<plugin name>/resources
    // $ES_HOME/plugin/<plugin name>/platform/<platform name>/bin

    std::string productionDir(CProgName::progDir() + "/../../../resources");

    // If the production directory doesn't exist, return the dev directory if
    // that does, but if neither exist return the production directory so the
    // error message is nicer for the end user.
    COsFileFuncs::TStat buf;
    if (COsFileFuncs::stat(productionDir.c_str(), &buf) != 0) {
        const char* cppSrcHome(::getenv(CPP_SRC_HOME));
        if (cppSrcHome != nullptr) {
            std::string devDir(cppSrcHome);
            devDir += "/lib/core";
            if (COsFileFuncs::stat(devDir.c_str(), &buf) == 0) {
                return devDir;
            }
        }
    }

    return productionDir;
}

std::string CResourceLocator::logDir() {
    // Look relative to the program that's running, assuming this directory layout:
    // $ES_HOME/logs
    // $ES_HOME/plugin/<plugin name>/platform/<platform name>/bin

    std::string productionDir(CProgName::progDir() + "/../../../../../logs");

    COsFileFuncs::TStat buf;
    if (COsFileFuncs::stat(productionDir.c_str(), &buf) != 0) {
        // Assume we're running as a unit test
        return ".";
    }

    return productionDir;
}

std::string CResourceLocator::cppRootDir() {
    const char* cppSrcHome(::getenv(CPP_SRC_HOME));
    if (cppSrcHome == nullptr) {
        // Assume we're in a unittest directory
        return "../../..";
    }
    return cppSrcHome;
}
}
}
