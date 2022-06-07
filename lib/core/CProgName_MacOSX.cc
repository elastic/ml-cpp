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

#include <boost/filesystem.hpp>

#include <cstdint>

#include <mach-o/dyld.h>
#include <stdlib.h>

namespace ml {
namespace core {

std::string CProgName::progName() {
    const char* progName(::getprogname());
    if (progName == nullptr) {
        return std::string();
    }

    return progName;
}

std::string CProgName::progDir() {
    std::uint32_t bufferSize(2048);
    std::string path(bufferSize, '\0');
    if (_NSGetExecutablePath(&path[0], &bufferSize) != 0) {
        return std::string();
    }
    std::size_t lastSlash(path.rfind('/'));
    if (lastSlash == std::string::npos) {
        return std::string();
    }
    path.resize(lastSlash);

    // On Mac OS X the path returned from _NSGetExecutablePath() is not always
    // canonical, e.g. containing /./
    return boost::filesystem::canonical(boost::filesystem::path(path)).string();
}
}
}
