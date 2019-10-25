/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CStringUtils.h>
#include <core/CUname.h>

#include <boost/test/unit_test.hpp>

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

BOOST_AUTO_TEST_SUITE(CProcessPriorityTest)

namespace {

bool readFromSystemFile(const std::string& fileName, std::string& content) {
    char buffer[16] = {'\0'};

    // Use low level functions to read rather than C++ wrappers, as these are
    // system files.
    int fd = ::open(fileName.c_str(), O_RDONLY);
    if (fd == -1) {
        LOG_INFO(<< "Could not open " << fileName << ": " << ::strerror(errno));
        return false;
    }

    ssize_t bytesRead = ::read(fd, buffer, sizeof(buffer));
    ::close(fd);

    if (bytesRead < 0) {
        LOG_ERROR(<< "Error reading from " << fileName << ": " << ::strerror(errno));
        return false;
    }

    if (bytesRead == 0) {
        LOG_WARN(<< "Read nothing from " << fileName);
        return false;
    }

    content.assign(buffer, 0, static_cast<size_t>(bytesRead));
    ml::core::CStringUtils::trimWhitespace(content);

    return true;
}
}

BOOST_AUTO_TEST_CASE(testReducePriority) {
    ml::core::CProcessPriority::reducePriority();

    bool readFromOneOrOther(false);

    std::string content;
    if (readFromSystemFile("/proc/self/oom_score_adj", content) == true) {
        BOOST_REQUIRE_EQUAL(std::string("667"), content);
        readFromOneOrOther = true;
    }
    if (readFromSystemFile("/proc/self/oom_adj", content) == true) {
        if (readFromOneOrOther) {
            LOG_DEBUG(<< "oom_score_adj 667 corresponds to oom_adj " << content
                      << " on kernel " << ml::core::CUname::release());
            int oomAdj = 0;
            BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(content, oomAdj));
            // For the kernel versions that support both, there's variation in
            // what an oom_score_adj of 667 maps to - the range seems to be 8-11
            BOOST_TEST_REQUIRE(oomAdj >= 8);
            BOOST_TEST_REQUIRE(oomAdj <= 11);
        } else {
            BOOST_REQUIRE_EQUAL(std::string("10"), content);
        }
        readFromOneOrOther = true;
    }

    BOOST_TEST_REQUIRE(readFromOneOrOther);
}

BOOST_AUTO_TEST_SUITE_END()
