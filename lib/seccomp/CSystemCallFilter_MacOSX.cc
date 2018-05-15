/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "seccomp/CSystemCallFilter.h"

#include <core/CLogger.h>

#include <fstream>
#include <iostream>

#include <sandbox.h>

namespace ml {
namespace seccomp {

// The Sandbox rules deny all actions apart from reading and writing
// to files in directories under /private/tmp.
// (allow file-write*) is required for mkfifo and that permission
// can not be set using the more granular controls.
// OSX links /tmp to /private/tmp but sandbox insists on using /private/tmp
static const std::string SANDBOX_RULES("\
    (version 1) \
    (deny default) \
    (debug all) \
    (allow file-read*) \
    (allow file-read-data) \
    (allow file-write*) \
    (allow file-write-data)");

static const char* PROFILE_NAME = "/private/tmp/ml-autodetect.sb";

CSystemCallFilter::CSystemCallFilter() {

    std::ofstream tempRulesFile;
    tempRulesFile.open(PROFILE_NAME);
    tempRulesFile << SANDBOX_RULES << std::endl;
    tempRulesFile.close();

    char* errorbuf = nullptr;
    if (sandbox_init(PROFILE_NAME, SANDBOX_NAMED, &errorbuf) != 0) {
        std::string msg("Error initializing macOS sandbox");
        if (errorbuf != nullptr) {
            msg += ": ";
            msg += errorbuf;
            sandbox_free_error(errorbuf);
        }
        LOG_ERROR(<< msg);
    } else {
        LOG_INFO(<< "macOS sandbox initialized");
    }
}
}
}
