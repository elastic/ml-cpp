/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CBlockingMessageQueueTest.h"

#include <core/CBlockingMessageQueue.h>
#include <core/CLogger.h>

#include <vector>

CppUnit::Test* CBlockingMessageQueueTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CBlockingMessageQueueTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CBlockingMessageQueueTest>("CBlockingMessageQueueTest::testSendReceive",
                                                                             &CBlockingMessageQueueTest::testSendReceive));

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
