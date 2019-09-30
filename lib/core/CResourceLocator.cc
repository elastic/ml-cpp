/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CResourceLocator.h>

#include <core/COsFileFuncs.h>
#include <core/CProgName.h>

#include <cstdlib>

namespace {
// Important: all file scope in this file must be simple types that don't
// require construction, as they could be accessed before the static
// constructors are run.
const char* const CPP_SRC_HOME{"CPP_SRC_HOME"};

#ifdef MacOSX
const char* const RESOURCE_RELATIVE_DIR{"../Resources"};
#else
const char* const RESOURCE_RELATIVE_DIR{"../resources"};
#endif
}

namespace ml {
namespace core {

std::string CResourceLocator::resourceDir() {

    // Look relative to the program that's running, assuming that the resource
    // directory is located relative to the directory the current program is in.
    std::string productionDir(CProgName::progDir() + '/' + RESOURCE_RELATIVE_DIR);

    // If the production directory doesn't exist, return the dev directory if
    // that does, but if neither exist return the production directory so the
    // error message is nicer for the end user.
    COsFileFuncs::TStat buf;
    if (COsFileFuncs::stat(productionDir.c_str(), &buf) != 0) {
        const char* cppSrcHome(std::getenv(CPP_SRC_HOME));
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

std::string CResourceLocator::cppRootDir() {
    const char* cppSrcHome(std::getenv(CPP_SRC_HOME));
    if (cppSrcHome == nullptr) {
        // Assume we're in a unittest directory
        return "../../..";
    }
    return cppSrcHome;
}
}
}
