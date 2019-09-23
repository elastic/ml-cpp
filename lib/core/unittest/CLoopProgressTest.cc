/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CLoopProgressTest.h"

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CLoopProgress.h>

#include <test/CRandomNumbers.h>

#include <functional>
#include <sstream>

using namespace ml;

using TSizeVec = std::vector<std::size_t>;

void CLoopProgressTest::testShort() {

    double progress{0.0};
    auto recordProgress = [&progress](double p) { progress += p; };

    LOG_DEBUG(<< "Test with stride == 1");

    for (std::size_t n = 1; n < 16; ++n) {

        LOG_DEBUG(<< "Testing " << n);

        progress = 0.0;
        core::CLoopProgress loopProgress{n, recordProgress};

        for (std::size_t i = 0; i < n; ++i, loopProgress.increment()) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                static_cast<double>(i) / static_cast<double>(n), progress, 1e-15);
        }

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, progress, 1e-15);
    }

    LOG_DEBUG(<< "Test with stride > 1");

    for (std::size_t n = 2; n < 16; ++n) {

        LOG_DEBUG(<< "Testing " << n);

        progress = 0.0;
        core::CLoopProgress loopProgress{n, recordProgress};

        for (std::size_t i = 0; i < n; i += 2, loopProgress.increment(2)) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                static_cast<double>(i) / static_cast<double>(n), progress, 1e-15);
        }
    }
}

void CLoopProgressTest::testRandom() {

    double progress{0.0};
    auto recordProgress = [&progress](double p) { progress += p; };

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Test with stride == 1");

    for (std::size_t t = 0; t < 100; ++t) {

        TSizeVec size;
        rng.generateUniformSamples(16, 10000, 1, size);

        if (t % 10 == 0) {
            LOG_DEBUG(<< "Loop length = " << size[0]);
        }

        progress = 0.0;
        core::CLoopProgress loopProgress{size[0], recordProgress};

        for (std::size_t i = 0; i < size[0]; ++i, loopProgress.increment()) {
            CPPUNIT_ASSERT_EQUAL(static_cast<double>(32 * i / size[0]) / 32.0, progress);
        }

        CPPUNIT_ASSERT_EQUAL(1.0, progress);
    }

    LOG_DEBUG(<< "Test with stride > 1");

    for (std::size_t t = 0; t < 100; ++t) {

        TSizeVec size;
        rng.generateUniformSamples(33, 100, 1, size);

        if (t % 10 == 0) {
            LOG_DEBUG(<< "Loop length = " << size[0]);
        }

        progress = 0.0;
        core::CLoopProgress loopProgress{size[0], recordProgress};

        for (std::size_t i = 0; i < size[0]; i += 20, loopProgress.increment(20)) {
            CPPUNIT_ASSERT_EQUAL(static_cast<double>(32 * i / size[0]) / 32.0, progress);
        }

        CPPUNIT_ASSERT_EQUAL(1.0, progress);
    }
}

void CLoopProgressTest::testScaled() {

    double progress{0.0};
    auto recordProgress = [&progress](double p) { progress += p; };

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
        core::CLoopProgress loopProgress{size[0], recordProgress,
                                         1.0 / static_cast<double>(step[0])};

        for (std::size_t i = 0; i < size[0];
             i += step[0], loopProgress.increment(step[0])) {
            // We're only interested in checking the progress at the end.
        }

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0 / static_cast<double>(step[0]), progress, 1e-15);
    }
}

void CLoopProgressTest::testSerialization() {

    double progress{0.0};
    auto recordProgress = [&progress](double p) { progress += p; };

    core::CLoopProgress loopProgress{50, recordProgress};
    for (std::size_t i = 0; i < 20; ++i) {
        loopProgress.increment();
    }

    std::stringstream persistStream;
    {
        core::CJsonStatePersistInserter inserter(persistStream);
        loopProgress.acceptPersistInserter(inserter);
    }

    LOG_DEBUG(<< "state = " << persistStream.str());

    core::CJsonStateRestoreTraverser traverser(persistStream);
    core::CLoopProgress restoredLoopProgress;
    restoredLoopProgress.acceptRestoreTraverser(traverser);

    double restoredProgress{0.0};
    auto restoredRecordProgress = [&restoredProgress](double p) {
        restoredProgress += p;
    };
    restoredLoopProgress.attach(restoredRecordProgress);
    restoredLoopProgress.resumeRestored();

    CPPUNIT_ASSERT_EQUAL(loopProgress.checksum(), restoredLoopProgress.checksum());
    for (std::size_t i = 20; i < 50; ++i) {
        loopProgress.increment();
        restoredLoopProgress.increment();
        CPPUNIT_ASSERT_EQUAL(progress, restoredProgress);
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
    suiteOfTests->addTest(new CppUnit::TestCaller<CLoopProgressTest>(
        "CLoopProgressTest::testSerialization", &CLoopProgressTest::testSerialization));

    return suiteOfTests;
}
