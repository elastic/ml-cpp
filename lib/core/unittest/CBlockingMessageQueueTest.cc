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
#include "CBlockingMessageQueueTest.h"

#include <core/CBlockingMessageQueue.h>
#include <core/CLogger.h>

#include <vector>

CppUnit::Test* CBlockingMessageQueueTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CBlockingMessageQueueTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CBlockingMessageQueueTest>(
        "CBlockingMessageQueueTest::testSendReceive", &CBlockingMessageQueueTest::testSendReceive));

    return suiteOfTests;
}

namespace {
class CReceiver {
public:
    void processMsg(const std::string& str, size_t /* backlog */) {
        m_Strings.push_back(str);
        if ((m_Strings.size() % 1000) == 0) {
            LOG_DEBUG(<< "Received " << m_Strings.size() << " strings");
        }
    }

    size_t size() const { return m_Strings.size(); }

private:
    using TStrVec = std::vector<std::string>;

    TStrVec m_Strings;
};
}

void CBlockingMessageQueueTest::testSendReceive() {
    CReceiver receiver;

    static const size_t QUEUE_SIZE(100);

    ml::core::CBlockingMessageQueue<std::string, CReceiver, QUEUE_SIZE> queue(receiver);

    CPPUNIT_ASSERT(queue.start());

    // Note that the number of strings to be sent is much higher than the queue
    // size, so the message dispatch will probably block sometimes
    static const size_t TEST_SIZE(10000);

    LOG_DEBUG(<< "Sending " << TEST_SIZE << " strings");

    for (size_t i = 0; i < TEST_SIZE; ++i) {
        queue.dispatchMsg("Test string");
    }

    LOG_DEBUG(<< "Sent all strings");

    queue.stop();

    CPPUNIT_ASSERT_EQUAL(TEST_SIZE, receiver.size());
}
