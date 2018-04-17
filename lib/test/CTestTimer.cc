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
#include <test/CTestTimer.h>

#include <core/CLogger.h>

#include <cppunit/Test.h>

namespace ml {
namespace test {

void CTestTimer::startTest(CppUnit::Test* test) {
    m_StopWatch.reset(true);
    if (test != nullptr) {
        std::string testName{"|  " + test->getName() + "  |"};
        LOG_DEBUG(<< "+" << std::string(testName.length() - 2, '-') << "+");
        LOG_DEBUG(<< testName);
        LOG_DEBUG(<< "+" << std::string(testName.length() - 2, '-') << "+");
    }
}

void CTestTimer::endTest(CppUnit::Test* test) {
    if (test == nullptr) {
        LOG_ERROR(<< "Unexpected NULL pointer");
        return;
    }

    uint64_t duration(m_StopWatch.stop());

    const std::string& testName = test->getName();
    m_TestTimes[testName] = duration;

    LOG_INFO(<< "Unit test timing - " << testName << " took " << duration << "ms");
}

uint64_t CTestTimer::timeForTest(const std::string& testName) const {
    TStrUInt64MapCItr iter = m_TestTimes.find(testName);
    if (iter == m_TestTimes.end()) {
        LOG_WARN(<< "No timing for test named " << testName);
        return 0;
    }

    return iter->second;
}

uint64_t CTestTimer::totalTime() const {
    uint64_t result(0);

    for (TStrUInt64MapCItr iter = m_TestTimes.begin(); iter != m_TestTimes.end(); ++iter) {
        result += iter->second;
    }

    return result;
}

uint64_t CTestTimer::averageTime() const {
    if (m_TestTimes.empty()) {
        return 0;
    }

    return this->totalTime() / m_TestTimes.size();
}
}
}
