/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
#include "CProcessPriorityTest.h"

#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CStringUtils.h>
#include <core/CUname.h>

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

namespace {

bool readFromSystemFile(const std::string& fileName, std::string& content) {
    char buffer[16] = {'\0'};

    // Use low level functions to read rather than C++ wrappers, as these are
    // system files.
    int fd = ::open(fileName.c_str(), O_RDONLY);
    if (fd == -1) {
        LOG_INFO("Could not open " << fileName << ": " << ::strerror(errno));
        return false;
    }

    ssize_t bytesRead = ::read(fd, buffer, sizeof(buffer));
    ::close(fd);

    if (bytesRead < 0) {
        LOG_ERROR("Error reading from " << fileName << ": " << ::strerror(errno));
        return false;
    }

    if (bytesRead == 0) {
        LOG_WARN("Read nothing from " << fileName);
        return false;
    }

    content.assign(buffer, 0, static_cast<size_t>(bytesRead));
    ml::core::CStringUtils::trimWhitespace(content);

    return true;
}
}

CppUnit::Test* CProcessPriorityTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CProcessPriorityTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CProcessPriorityTest>("CProcessPriorityTest::testReducePriority",
                                                      &CProcessPriorityTest::testReducePriority));

    return suiteOfTests;
}

void CProcessPriorityTest::testReducePriority(void) {
    ml::core::CProcessPriority::reducePriority();

    bool readFromOneOrOther(false);

    std::string content;
    if (readFromSystemFile("/proc/self/oom_score_adj", content) == true) {
        CPPUNIT_ASSERT_EQUAL(std::string("667"), content);
        readFromOneOrOther = true;
    }
    if (readFromSystemFile("/proc/self/oom_adj", content) == true) {
        if (readFromOneOrOther) {
            LOG_DEBUG("oom_score_adj 667 corresponds to oom_adj " << content << " on kernel "
                                                                  << ml::core::CUname::release());
            int oomAdj = 0;
            CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(content, oomAdj));
            // For the kernel versions that support both, there's variation in
            // what an oom_score_adj of 667 maps to - the range seems to be 8-11
            CPPUNIT_ASSERT(oomAdj >= 8);
            CPPUNIT_ASSERT(oomAdj <= 11);
        } else {
            CPPUNIT_ASSERT_EQUAL(std::string("10"), content);
        }
        readFromOneOrOther = true;
    }

    CPPUNIT_ASSERT(readFromOneOrOther);
}
