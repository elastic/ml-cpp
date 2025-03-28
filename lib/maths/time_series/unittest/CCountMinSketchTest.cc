/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>

#include <maths/common/CBasicStatistics.h>

#include <maths/time_series/CCountMinSketch.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CCountMinSketchTest)

using namespace ml;

using TDoubleVec = std::vector<double>;

BOOST_AUTO_TEST_CASE(testCounts) {
    using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;

    test::CRandomNumbers rng;

    // Test that we can estimate counts exactly up to "# rows" x "# columns".
    // and that the error subsequently increases approximately linearly with
    // increasing category count.

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "Test Uniform");

    for (std::size_t n = 100; n < 1500; n += 100) {
        LOG_DEBUG(<< "*** number categories = " << n << " ***");

        TDoubleVec counts;
        rng.generateUniformSamples(2.0, 301.0, n, counts);

        maths::time_series::CCountMinSketch sketch(2, 751);

        for (std::size_t i = 0; i < counts.size(); ++i) {
            counts[i] = std::floor(counts[i]);
            sketch.add(static_cast<std::uint32_t>(i), counts[i]);
        }
        LOG_DEBUG(<< "error = " << sketch.oneMinusDeltaError());

        TMeanAccumulator meanError;
        double errorCount = 0.0;
        for (std::size_t i = 0; i < counts.size(); ++i) {
            double count = counts[i];
            double estimated = sketch.count(static_cast<std::uint32_t>(i));
            if (i % 50 == 0) {
                LOG_DEBUG(<< "category = " << i << ", true count = " << count
                          << ", estimated count = " << estimated);
            }

            meanError.add(std::fabs(estimated - count));
            if (count + sketch.oneMinusDeltaError() < estimated) {
                errorCount += 1.0;
            }
        }
        LOG_DEBUG(<< "mean error = " << maths::common::CBasicStatistics::mean(meanError));
        LOG_DEBUG(<< "error count = " << errorCount);
        if (sketch.oneMinusDeltaError() == 0.0) {
            BOOST_REQUIRE_EQUAL(0.0, maths::common::CBasicStatistics::mean(meanError));
        } else {
            //BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanError) <
            //                   0.1 * static_cast<double>(n));
        }
        //BOOST_REQUIRE_EQUAL(0.0, errorCount);
    }

    // Test the case that a small fraction of categories generate a large
    // fraction of the counts.

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "Test Heavy Hitters");
    {
        TDoubleVec heavyHitters;
        rng.generateUniformSamples(10000.0, 11001.0, 20, heavyHitters);

        TDoubleVec counts;
        rng.generateUniformSamples(2.0, 101.0, 1000, counts);

        maths::time_series::CCountMinSketch sketch(2, 751);

        for (std::size_t i = 0; i < heavyHitters.size(); ++i) {
            heavyHitters[i] = std::floor(heavyHitters[i]);
            sketch.add(static_cast<std::uint32_t>(i), heavyHitters[i]);
        }
        for (std::size_t i = 0; i < counts.size(); ++i) {
            counts[i] = std::floor(counts[i]);
            sketch.add(static_cast<std::uint32_t>(i + heavyHitters.size()), counts[i]);
        }
        LOG_DEBUG(<< "error = " << sketch.oneMinusDeltaError());

        TMeanAccumulator meanRelativeError;
        for (std::size_t i = 0; i < heavyHitters.size(); ++i) {
            double count = heavyHitters[i];
            double estimated = sketch.count(static_cast<std::uint32_t>(i));
            LOG_DEBUG(<< "category = " << i << ", true count = " << count
                      << ", estimated count = " << estimated);

            double relativeError = std::fabs(estimated - count) / count;
            BOOST_TEST_REQUIRE(relativeError < 0.01);

            meanRelativeError.add(relativeError);
        }

        LOG_DEBUG(<< "mean relative error "
                  << maths::common::CBasicStatistics::mean(meanRelativeError));
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(meanRelativeError) < 0.005);
    }
}

BOOST_AUTO_TEST_CASE(testSwap) {
    test::CRandomNumbers rng;

    TDoubleVec counts1;
    TDoubleVec counts2;
    TDoubleVec counts3;
    TDoubleVec counts4;
    rng.generateUniformSamples(2.0, 301.0, 500, counts1);
    rng.generateUniformSamples(2.0, 301.0, 500, counts2);
    rng.generateUniformSamples(2.0, 301.0, 1500, counts3);
    rng.generateUniformSamples(2.0, 301.0, 1500, counts4);

    maths::time_series::CCountMinSketch sketch1(3, 500);
    maths::time_series::CCountMinSketch sketch2(2, 750);
    maths::time_series::CCountMinSketch sketch3(3, 300);
    maths::time_series::CCountMinSketch sketch4(2, 400);
    for (std::size_t i = 0; i < counts1.size(); ++i) {
        sketch1.add(static_cast<std::uint32_t>(i), counts1[i]);
    }
    for (std::size_t i = 0; i < counts2.size(); ++i) {
        sketch2.add(static_cast<std::uint32_t>(i), counts2[i]);
    }
    for (std::size_t i = 0; i < counts3.size(); ++i) {
        sketch3.add(static_cast<std::uint32_t>(i), counts3[i]);
    }
    for (std::size_t i = 0; i < counts4.size(); ++i) {
        sketch4.add(static_cast<std::uint32_t>(i), counts4[i]);
    }

    std::uint64_t checksum1 = sketch1.checksum();
    std::uint64_t checksum2 = sketch2.checksum();
    std::uint64_t checksum3 = sketch3.checksum();
    std::uint64_t checksum4 = sketch4.checksum();
    LOG_DEBUG(<< "checksum1 = " << checksum1);
    LOG_DEBUG(<< "checksum2 = " << checksum2);
    LOG_DEBUG(<< "checksum3 = " << checksum3);
    LOG_DEBUG(<< "checksum4 = " << checksum4);

    sketch1.swap(sketch2);
    BOOST_REQUIRE_EQUAL(checksum2, sketch1.checksum());
    BOOST_REQUIRE_EQUAL(checksum1, sketch2.checksum());
    sketch1.swap(sketch2);

    sketch2.swap(sketch3);
    BOOST_REQUIRE_EQUAL(checksum3, sketch2.checksum());
    BOOST_REQUIRE_EQUAL(checksum2, sketch3.checksum());
    sketch2.swap(sketch3);

    sketch1.swap(sketch4);
    BOOST_REQUIRE_EQUAL(checksum1, sketch4.checksum());
    BOOST_REQUIRE_EQUAL(checksum4, sketch1.checksum());
    sketch1.swap(sketch4);

    sketch3.swap(sketch4);
    BOOST_REQUIRE_EQUAL(checksum3, sketch4.checksum());
    BOOST_REQUIRE_EQUAL(checksum4, sketch3.checksum());
    sketch3.swap(sketch4);
}

void testPersistSketch(const maths::time_series::CCountMinSketch& origSketch) {
    std::ostringstream origJson;
    core::CJsonStatePersistInserter::persist(
        origJson, std::bind_front(&maths::time_series::CCountMinSketch::acceptPersistInserter,
                                  &origSketch));
    LOG_DEBUG(<< "original sketch JSON = " << origJson.str());

    // Restore the JSON into a new sketch.
    std::istringstream is{"{\"topLevel\":" + origJson.str() + "}"};
    core::CJsonStateRestoreTraverser traverser(is);
    maths::time_series::CCountMinSketch restoredSketch(traverser);

    LOG_DEBUG(<< "orig checksum = " << origSketch.checksum()
              << ", new checksum = " << restoredSketch.checksum());
    BOOST_REQUIRE_EQUAL(origSketch.checksum(), restoredSketch.checksum());

    std::ostringstream newJson;
    core::CJsonStatePersistInserter::persist(
        newJson, std::bind_front(&maths::time_series::CCountMinSketch::acceptPersistInserter,
                                 &restoredSketch));
    BOOST_REQUIRE_EQUAL(origJson.str(), newJson.str());
}

BOOST_AUTO_TEST_CASE(testPersist) {
    test::CRandomNumbers rng;

    TDoubleVec counts;
    rng.generateUniformSamples(2.0, 301.0, 500, counts);

    maths::time_series::CCountMinSketch origSketch(2, 600);
    for (std::size_t i = 0; i < counts.size(); ++i) {
        origSketch.add(static_cast<std::uint32_t>(i), counts[i]);
    }

    testPersistSketch(origSketch);

    // Sketch.
    TDoubleVec moreCounts;
    rng.generateUniformSamples(2.0, 301.0, 500, moreCounts);
    for (std::size_t i = 0; i < moreCounts.size(); ++i) {
        origSketch.add(static_cast<std::uint32_t>(counts.size() + i), moreCounts[i]);
    }

    testPersistSketch(origSketch);
}

BOOST_AUTO_TEST_SUITE_END()
