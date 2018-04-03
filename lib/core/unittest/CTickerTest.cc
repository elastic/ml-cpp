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
#include "CTickerTest.h"

#include <core/CLogger.h>
#include <core/CSleep.h>
#include <core/CTicker.h>


CppUnit::Test *CTickerTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CTickerTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CTickerTest>(
                               "CTickerTest::testTicker",
                               &CTickerTest::testTicker) );

    return suiteOfTests;
}

namespace {
class CReceiver {
    public:
        CReceiver(void)
            : m_Ticks(0)
        {}

        void tick(void) {
            ++m_Ticks;
        }

        size_t ticks(void) const {
            return m_Ticks;
        }

    private:
        size_t m_Ticks;
};
}

void CTickerTest::testTicker(void) {
    CReceiver receiver;

    ml::core::CTicker<CReceiver> ticker(100, receiver);

    LOG_DEBUG("About to start ticker");
    CPPUNIT_ASSERT(ticker.start());

    ml::core::CSleep::sleep(1000);

    // Should receive 9 or 10 ticks
    size_t tickCount(receiver.ticks());
    LOG_DEBUG("Received " << tickCount << " ticks");

    CPPUNIT_ASSERT(tickCount <= 10);

    // NB: This test has been seen to fail on busy machines where the ticker
    // thread is starved of CPU and doesn't get to run as soon as its condition
    // wait expires.  If this test fails when the machine is busy (and on a VM
    // that might mean the hypervisor was doing something rather than the VM
    // itself) then it's probably not too much of a cause for concern.
    CPPUNIT_ASSERT(tickCount >= 9);
}

