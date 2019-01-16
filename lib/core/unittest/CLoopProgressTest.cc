/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CLoopProgressTest.h"

#include <core/CLogger.h>
#include <core/CLoopProgress.h>

#include <test/CRandomNumbers.h>

using namespace ml;

using TSizeVec = std::vector<std::size_t>;

void CLoopProgressTest::testShort() {

    double progress{0.0};

    for (std::size_t n = 1; n < 16; ++n) {

        LOG_DEBUG(<< "Testing " << n);

        progress = 0.0;
        core::CLoopProgress loopProgress{n,
                                         [&progress](double p) { progress += p; }};

        for (std::size_t i = 0; i < n; ++i, loopProgress.increment()) {
            // Empty
        }

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, progress, 1e-15);
    }

    LOG_DEBUG(<< "Test with stride > 1");

    for (std::size_t n = 2; n < 16; ++n) {

        LOG_DEBUG(<< "Testing " << n);

        progress = 0.0;
        core::CLoopProgress loopProgress{n,
                                         [&progress](double p) { progress += p; }};

        for (std::size_t i = 0; i < n; i += 2, loopProgress.increment(2)) {
            // Empty
        }
    }
}

void CLoopProgressTest::testRandom() {

    double progress{0.0};
    test::CRandomNumbers rng;

    for (std::size_t t = 0; t < 100; ++t) {

        TSizeVec size;
        rng.generateUniformSamples(16, 10000, 1, size);

        if (t % 10 == 0) {
            LOG_DEBUG(<< "Loop length = " << size[0]);
        }

        progress = 0.0;
        core::CLoopProgress loopProgress{size[0],
                                         [&progress](double p) { progress += p; }};

        for (std::size_t i = 0; i < size[0]; ++i, loopProgress.increment()) {
            // Empty
        }

        CPPUNIT_ASSERT_EQUAL(1.0, progress);
    }

    LOG_DEBUG(<< "Test with stride > 1");

    for (std::size_t t = 0; t < 100; ++t) {

        TSizeVec size;
        rng.generateUniformSamples(30, 100, 1, size);

        if (t % 10 == 0) {
            LOG_DEBUG(<< "Loop length = " << size[0]);
        }

        progress = 0.0;
        core::CLoopProgress loopProgress{size[0],
                                         [&progress](double p) { progress += p; }};

        for (std::size_t i = 0; i < size[0]; i += 20, loopProgress.increment(20)) {
            // Empty
        }

        CPPUNIT_ASSERT_EQUAL(1.0, progress);
    }
}

void CLoopProgressTest::testScaled() {

    double progress{0.0};
    test::CRandomNumbers rng;

    for (std::size_t t = 0; t < 100; ++t) {

        TSizeVec step;
        rng.generateUniformSamples(1, 10, 1, step);
        TSizeVec size;
        rng.generateUniformSamples(16 * step[0], 10000, 1, size);

        if (t % 10 == 0) {
            LOG_DEBUG(<< "Loop length = " << size[0]);
        }

        progress = 0.0;
        core::CLoopProgress loopProgress{size[0],
                                         [&progress](double p) { progress += p; },
                                         1.0 / static_cast<double>(step[0])};

        for (std::size_t i = 0; i < size[0];
             i += step[0], loopProgress.increment(step[0])) {
            // Empty
        }

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0 / static_cast<double>(step[0]), progress, 1e-15);
    }
}

CppUnit::Test* CLoopProgressTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CLoopProgressTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CLoopProgressTest>(
        "CLoopProgressTest::testShort", &CLoopProgressTest::testShort));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLoopProgressTest>(
        "CLoopProgressTest::testRandom", &CLoopProgressTest::testRandom));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLoopProgressTest>(
        "CLoopProgressTest::testScaled", &CLoopProgressTest::testScaled));

    return suiteOfTests;
}
