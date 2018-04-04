/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
        if (cppSrcHome != 0) {
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
    if (cppSrcHome == 0) {
        // Assume we're in a unittest directory
        return "../../..";
    }
    return cppSrcHome;
}
}
}
