/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CLoopProgress.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <functional>
#include <sstream>

BOOST_AUTO_TEST_SUITE(CLoopProgressTest)

using namespace ml;

using TIntVec = std::vector<int>;
using TSizeVec = std::vector<std::size_t>;

BOOST_AUTO_TEST_CASE(testShort) {

    double progress{0.0};
    auto recordProgress = [&progress](double p) { progress += p; };

    LOG_DEBUG(<< "Test with stride == 1");

    for (std::size_t n = 1; n < 16; ++n) {

        LOG_DEBUG(<< "Testing " << n);

        progress = 0.0;
        core::CLoopProgress loopProgress{n, recordProgress};

        for (std::size_t i = 0; i < n; ++i, loopProgress.increment()) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                static_cast<double>(i) / static_cast<double>(n), progress, 1e-15);
        }

        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, progress, 1e-15);
    }

    LOG_DEBUG(<< "Test with stride > 1");

    for (std::size_t n = 2; n < 16; ++n) {

        LOG_DEBUG(<< "Testing " << n);

        progress = 0.0;
        core::CLoopProgress loopProgress{n, recordProgress};

        for (std::size_t i = 0; i < n; i += 2, loopProgress.increment(2)) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                static_cast<double>(i) / static_cast<double>(n), progress, 1e-15);
        }
    }
}

BOOST_AUTO_TEST_CASE(testRandom) {

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
            BOOST_REQUIRE_EQUAL(static_cast<double>(32 * i / size[0]) / 32.0, progress);
        }

        BOOST_REQUIRE_EQUAL(1.0, progress);
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
            BOOST_REQUIRE_EQUAL(static_cast<double>(32 * i / size[0]) / 32.0, progress);
        }

        BOOST_REQUIRE_EQUAL(1.0, progress);
    }
}

BOOST_AUTO_TEST_CASE(testScaled) {

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

        BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0 / static_cast<double>(step[0]), progress, 1e-15);
    }
}

BOOST_AUTO_TEST_CASE(testIncrementRange) {

    for (std::size_t steps : {30, 100}) {
        double progress{0.0};
        auto recordProgress = [&progress](double p) { progress += p; };
        core::CLoopProgress loopProgress{50, recordProgress, 1.0, steps};

        for (std::size_t i = 0; i < 20; ++i) {
            loopProgress.increment();
        }

        loopProgress.incrementRange(-20);
        BOOST_REQUIRE_CLOSE(20.0 / 30.0, progress, 2.0);

        loopProgress.incrementRange(30);
        for (std::size_t i = 0; i < 40; ++i) {
            loopProgress.increment();
            BOOST_REQUIRE_CLOSE(std::max(static_cast<double>(20 + i) / 60.0, 20.0 / 30.0),
                                progress, 4.0);
        }
    }

    for (std::size_t steps : {30, 100}) {
        double progress{0.0};
        auto recordProgress = [&progress](double p) { progress += p; };
        core::CLoopProgress loopProgress{50, recordProgress, 1.0, steps};

        for (std::size_t i = 0; i < 20; ++i) {
            loopProgress.increment();
        }

        loopProgress.incrementRange(30);
        BOOST_REQUIRE_CLOSE(20.0 / 50.0, progress, 2.0);

        loopProgress.incrementRange(-20);
        for (std::size_t i = 0; i < 40; ++i) {
            loopProgress.increment();
            BOOST_REQUIRE_CLOSE(std::max(static_cast<double>(20 + i) / 60.0, 20.0 / 50.0),
                                progress, 4.0);
        }
    }
}

BOOST_AUTO_TEST_CASE(testSerialization) {

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
    restoredLoopProgress.progressCallback(restoredRecordProgress);
    restoredLoopProgress.resumeRestored();

    BOOST_REQUIRE_EQUAL(loopProgress.checksum(), restoredLoopProgress.checksum());
    for (std::size_t i = 20; i < 50; ++i) {
        loopProgress.increment();
        restoredLoopProgress.increment();
        BOOST_REQUIRE_EQUAL(progress, restoredProgress);
    }
}

BOOST_AUTO_TEST_SUITE_END()
