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
#include <test/CTestTmpDir.h>

#include <core/CLogger.h>
#include <core/CResourceLocator.h>

#include <boost/filesystem.hpp>

namespace ml {
namespace test {

std::string CTestTmpDir::tmpDir() {
    // Try to create a temporary sub-directory under the build directory at the
    // top level of the repo.  This ensures multiple users sharing the same
    // server don't clash.
    boost::filesystem::path tmpPath{core::CResourceLocator::cppRootDir()};
    tmpPath /= "build";
    tmpPath /= "tmp";
    try {
        // Prior existence of the directory is not considered an error by
        // boost::filesystem, and this is what we want
        boost::filesystem::create_directories(tmpPath);
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to create directory " << tmpPath << " - " << e.what());
        return ".";
    }

    return tmpPath.string();
}
}
}
