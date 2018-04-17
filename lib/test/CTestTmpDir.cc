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
#include <test/CTestTmpDir.h>

#include <core/CLogger.h>

#include <boost/filesystem.hpp>

#include <errno.h>
#include <pwd.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

namespace ml {
namespace test {

std::string CTestTmpDir::tmpDir() {
    // Try to create a user-specific sub-directory of the temporary directory so
    // that multiple users sharing the same server don't clash.  However, if
    // this fails for any reason drop back to just raw /tmp.
    struct passwd pwd;
    ::memset(&pwd, 0, sizeof(pwd));
    static const size_t BUFSIZE(16384);
    char buffer[BUFSIZE] = {'\0'};
    struct passwd* result(nullptr);
    ::getpwuid_r(::getuid(), &pwd, buffer, BUFSIZE, &result);
    if (result == nullptr || result->pw_name == nullptr) {
        LOG_ERROR(<< "Could not get current user name: " << ::strerror(errno));
        return "/tmp";
    }

    std::string userSubdir("/tmp/");
    userSubdir += result->pw_name;

    try {
        // Prior existence of the directory is not considered an error by
        // boost::filesystem, and this is what we want
        boost::filesystem::path directoryPath(userSubdir);
        boost::filesystem::create_directories(directoryPath);
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to create directory " << userSubdir << " - " << e.what());
        return "/tmp";
    }

    return userSubdir;
}
}
}
