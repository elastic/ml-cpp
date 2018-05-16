/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "seccomp/CSystemCallFilter.h"

#include <core/CLogger.h>

#include <paths.h>
#include <cstdio>
#include <fstream>
#include <sandbox.h>

namespace ml {
namespace seccomp {

// The Sandbox rules deny all actions apart from creating fifos,
// opening files and reading and writing.
// (allow file-write*) is required for mkfifo and that permission
// can not be set using the more granular controls.
static const std::string SANDBOX_RULES("\
    (version 1) \
    (deny default) \
    (debug all) \
    (allow file-read*) \
    (allow file-read-data) \
    (allow file-write*) \
    (allow file-write-data)");

std::string getTempDir() {
    // In production this needs to match the setting of java.io.tmpdir.  We rely
    // on the JVM that spawns our controller daemon setting TMPDIR in the
    // environment of the spawned process.
    const char* tmpDir(::getenv("TMPDIR"));

    // Make sure path ends with a slash so it's ready to have a file name
    // appended.  (_PATH_VARTMP already has this on all platforms I've seen,
    // but a user-defined $TMPDIR might not.)
    std::string path((tmpDir == nullptr) ? _PATH_VARTMP : tmpDir);
    if (path[path.length() - 1] != '/') {
        path += '/';
    }
    return path;
}

std::string getTempRulesFilename() {
    std::string tempDir = getTempDir();
    // randomise the sandbox rules filename
    char * tempName = ::tempnam(tempDir.c_str(), "ml");
    if (tempName != nullptr) {
        return std::string(tempName) + ".sb";
        ::free(tempName);
    } else {
        return tempDir + "ml.sb";
    }
}

CSystemCallFilter::CSystemCallFilter() {
    std::string profileFilename = getTempRulesFilename();
    std::ofstream tempRulesFile;
    tempRulesFile.open(profileFilename);
    tempRulesFile << SANDBOX_RULES << std::endl;
    tempRulesFile.close();

    char* errorbuf = nullptr;
    if (sandbox_init(profileFilename.c_str(), SANDBOX_NAMED, &errorbuf) != 0) {
        std::string msg("Error initializing macOS sandbox");
        if (errorbuf != nullptr) {
            msg += ": ";
            msg += errorbuf;
            sandbox_free_error(errorbuf);
        }
        LOG_ERROR(<< msg);
    } else {
        LOG_DEBUG(<< "macOS sandbox initialized");
    }
}
}
}
