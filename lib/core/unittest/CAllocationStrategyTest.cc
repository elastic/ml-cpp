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
#include "CAllocationStrategyTest.h"

#include <core/CAllocationStrategy.h>
#include <core/CLogger.h>

#include <algorithm>
#include <vector>

using namespace ml;

namespace {
double TOLERANCE = 1.0;
// This is 10% plus a small allowance for rounding error
double RATIO = 1.1 + 0.05;
}

template<typename T>
void assertSize(const T& t) {
    std::size_t s = t.size();
    std::size_t c = t.capacity();
    LOG_DEBUG(<< "Size " << s << ", capacity " << c);
    CPPUNIT_ASSERT(double(c) <= std::max(double(s) * RATIO, double(s) + TOLERANCE));
}

void CAllocationStrategyTest::test() {
    using TIntVec = std::vector<int>;

    {
        TIntVec v;
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), v.capacity());

        core::CAllocationStrategy::resize(v, 1);
        assertSize(v);

        core::CAllocationStrategy::resize(v, 5);
        assertSize(v);

        core::CAllocationStrategy::resize(v, 10);
        assertSize(v);

        core::CAllocationStrategy::resize(v, 20);
        assertSize(v);

        core::CAllocationStrategy::resize(v, 22);
        assertSize(v);

        core::CAllocationStrategy::resize(v, 93);
        assertSize(v);

        core::CAllocationStrategy::resize(v, 103);
        assertSize(v);

        core::CAllocationStrategy::resize(v, 128);
        assertSize(v);
    }
    {
        TIntVec v;
        core::CAllocationStrategy::push_back(v, 55);
        assertSize(v);

        for (std::size_t i = 0; i < 10000; i++) {
            core::CAllocationStrategy::push_back(v, int(55 + i));
            assertSize(v);
        }
    }
}

CppUnit::Test* CAllocationStrategyTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CAllocationStrategyTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CAllocationStrategyTest>(
        "CAllocationStrategyTest::test", &CAllocationStrategyTest::test));
    return suiteOfTests;
}
