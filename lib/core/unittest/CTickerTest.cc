/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CTickerTest.h"

#include <core/CLogger.h>
#include <core/CSleep.h>
#include <core/CTicker.h>


CppUnit::Test *CTickerTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CTickerTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CTickerTest>(
                                   "CTickerTest::testTicker",
                                   &CTickerTest::testTicker) );

    return suiteOfTests;
}

namespace
{
    class CReceiver
    {
        public:
            CReceiver()
                : m_Ticks(0)
            {
            }

            void tick()
            {
                ++m_Ticks;
            }

            size_t ticks() const
            {
                return m_Ticks;
            }

        private:
            size_t m_Ticks;
    };
}

void CTickerTest::testTicker()
{
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

