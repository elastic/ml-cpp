/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CTestTmpDir.h>

#include <core/CLogger.h>
#include <core/CResourceLocator.h>

#include <boost/filesystem.hpp>

namespace ml {
namespace test {

std::string CTestTmpDir::tmpDir() {
    // Try create a temp sub-directory under the build directory at the top
    // level of the repo.  This ensures multiple users sharing the same server
    // don't clash
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
