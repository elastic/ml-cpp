/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CProgramCountersTest_h
#define INCLUDED_CProgramCountersTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <core/CProgramCounters.h>
#include <core/CThread.h>

class CProgramCountersTest : public CppUnit::TestFixture {
public:
    void testCounters();
    void testPersist();
    void testCacheCounters();
    void testUnknownCounter();
    void testMissingCounter();

    static CppUnit::Test* suite();

private:
    static const int TEST_COUNTER{0u};
    class CProgramCountersTestRunner : public ml::core::CThread {
    public:
        CProgramCountersTestRunner() : m_I(0), m_N(0) {}

        void initialise(int i, int n) {
            m_N = n;
            m_I = i;
        }

    private:
        virtual void run() {
            if (m_I < 6) {
                ++ml::core::CProgramCounters::counter(TEST_COUNTER + m_I);
            } else {
                --ml::core::CProgramCounters::counter(TEST_COUNTER + m_I - m_N);
            }
        }

        virtual void shutdown() {}

        int m_I;
        int m_N;
    };
private:
    std::string persist(bool doCachCounters = true);
    void restore(const std::string& staticsXml);
};

#endif // INCLUDED_CProgramCountersTest_h
