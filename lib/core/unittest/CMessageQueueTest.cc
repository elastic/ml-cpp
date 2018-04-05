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
#include "CMessageQueueTest.h"

#include <core/CLogger.h>
#include <core/CMessageQueue.h>
#include <core/CSleep.h>

#include <vector>

#include <stdint.h>

CppUnit::Test* CMessageQueueTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CMessageQueueTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMessageQueueTest>("CMessageQueueTest::testSendReceive", &CMessageQueueTest::testSendReceive));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMessageQueueTest>("CMessageQueueTest::testTiming", &CMessageQueueTest::testTiming));

    return suiteOfTests;
}

namespace {
class CReceiver {
public:
    CReceiver(uint32_t sleepTime = 0) : m_SleepTime(sleepTime) {}

    void processMsg(const std::string& str, size_t /* backlog */) {
        m_Strings.push_back(str);
        if ((m_Strings.size() % 1000) == 0) {
            LOG_DEBUG("Received " << m_Strings.size() << " strings");
        }

        // Delay the processing if requested - this enables us to test
        // the timing functionality
        if (m_SleepTime > 0) {
            ml::core::CSleep::sleep(m_SleepTime);
        }
    }

    size_t size() const { return m_Strings.size(); }

private:
    using TStrVec = std::vector<std::string>;

    TStrVec m_Strings;

    uint32_t m_SleepTime;
};
}

void CMessageQueueTest::testSendReceive() {
    CReceiver receiver;

    ml::core::CMessageQueue<std::string, CReceiver> queue(receiver);

    CPPUNIT_ASSERT(queue.start());

    static const size_t TEST_SIZE(10000);

    LOG_DEBUG("Sending " << TEST_SIZE << " strings");

    for (size_t i = 0; i < TEST_SIZE; ++i) {
        queue.dispatchMsg("Test string");
    }

    LOG_DEBUG("Sent all strings");

    queue.stop();

    CPPUNIT_ASSERT_EQUAL(TEST_SIZE, receiver.size());
}

void CMessageQueueTest::testTiming() {
    // Tell the receiver to delay processing by 29ms for each item (otherwise
    // it will be too fast to time on a modern computer).
    CReceiver receiver(29);

    static const size_t NUM_TO_TIME(100);
    ml::core::CMessageQueue<std::string, CReceiver, NUM_TO_TIME> queue(receiver);

    CPPUNIT_ASSERT(queue.start());

    static const size_t TEST_SIZE(100);

    LOG_DEBUG("Sending " << TEST_SIZE << " strings");

    for (size_t i = 0; i < TEST_SIZE; ++i) {
        queue.dispatchMsg("Test string");
    }

    LOG_DEBUG("Sent all strings");

    queue.stop();

    CPPUNIT_ASSERT_EQUAL(TEST_SIZE, receiver.size());

    double avgProcTimeSec(queue.rollingAverageProcessingTime());
    LOG_DEBUG("Average processing time per item for the last " << NUM_TO_TIME << " items was " << avgProcTimeSec << " seconds");

    // The high side tolerance is greater here, because although the sleep will
    // make up the bulk of the processing time, there is some other processing
    // to be done too.  However, there's also a low side tolerance to account
    // for clock inaccuracies, for example the Windows clock only being accurate
    // to the nearest 15.625 milliseconds.
    CPPUNIT_ASSERT(0.029 - 0.032 / double(TEST_SIZE) < avgProcTimeSec);
    // The bound in this next assert was increased from 0.034 to 0.04 based on
    // experience with the OS X Yosemite build VM - TODO: investigate in detail
    CPPUNIT_ASSERT(0.04 > avgProcTimeSec);
}
