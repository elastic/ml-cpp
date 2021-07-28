/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#include <core/CProgName.h>

#include <unistd.h>

// Secret global variable in the Linux glibc...
extern char* __progname;

namespace ml {
namespace core {

std::string CProgName::progName() {
    if (__progname == 0) {
        return std::string();
    }

    return __progname;
}

std::string CProgName::progDir() {
    static const size_t BUFFER_SIZE(2048);
    std::string path(BUFFER_SIZE, '\0');
    ssize_t len(::readlink("/proc/self/exe", &path[0], BUFFER_SIZE));
    if (len == -1) {
        return std::string();
    }
    size_t lastSlash(path.rfind('/', static_cast<size_t>(len)));
    if (lastSlash == std::string::npos) {
        return std::string();
    }
    path.resize(lastSlash);
    return path;
}
}
}
