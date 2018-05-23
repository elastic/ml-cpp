/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "seccomp/CSystemCallFilter.h"

#include <core/CLogger.h>

#include <paths.h>
#include <sandbox.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace ml {
namespace seccomp {

namespace {
// The Sandbox rules deny all actions apart from creating fifos,
// opening files, reading and writing.
// (allow file-write*) is required for mkfifo and that permission
// can not be set using the more granular controls.
const std::string SANDBOX_RULES("\
    (version 1) \
    (deny default) \
    (allow file-read*) \
    (allow file-read-data) \
    (allow file-write*) \
    (allow file-write-data)");

// mkstemps will replace the Xs with random characters
const char* FILE_NAME_TEMPLATE = "ml.XXXXXX.sb";
// The length of the suffix '.sb'
const int FILE_NAME_TEMPLATE_SUFFIX_LEN = 3;

std::string getTempDir() {
    // Prefer to use the temporary directory set by the Elasticsearch JVM
    const char* tmpDir(::getenv("TMPDIR"));

    // If TMPDIR is not set use _PATH_VARTMP
    std::string path((tmpDir == nullptr) ? _PATH_VARTMP : tmpDir);
    // Make sure path ends with a slash so it's ready to have a file name appended
    if (path[path.length() - 1] != '/') {
        path += '/';
    }
    return path;
}

std::string writeTempRulesFile() {
    std::string profileFilename = getTempDir() + FILE_NAME_TEMPLATE;

    // Create and open a temporary file with a random name
    // profileFilename is updated with the new filename.
    int fd = mkstemps(&profileFilename[0], FILE_NAME_TEMPLATE_SUFFIX_LEN);
    if (fd == -1) {
        LOG_ERROR(<< "Opening a temporary file with mkstemps failed: " << std::strerror(errno));
        return std::string();
    }
    write(fd, SANDBOX_RULES.c_str(), SANDBOX_RULES.size());
    close(fd);

    return profileFilename;
}
}

void CSystemCallFilter::installSystemCallFilter() {
    std::string profileFilename = writeTempRulesFile();
    if (profileFilename.empty()) {
        LOG_WARN(<< "Cannot write sandbox rules. macOS sandbox will not be initialized");
        return;
    }

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

    std::remove(profileFilename.c_str());
}
}
}
