/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBjkstUniqueValues.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CBjkstUniqueValuesTest)

using namespace ml;
using namespace maths;
using namespace test;

namespace {

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TUInt32Set = std::set<uint32_t>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

uint8_t trailingZeros(uint32_t x) {
    uint8_t result = 0;
    for (/**/; (x & 0x1) == 0; x >>= 1) {
        ++result;
    }
    return result;
}
}

BOOST_AUTO_TEST_CASE(testTrailingZeros) {
    uint32_t n = 1;
    for (uint8_t i = 0; i < 32; n <<= 1, ++i) {
        BOOST_REQUIRE_EQUAL(i, CBjkstUniqueValues::trailingZeros(n));
    }

    TDoubleVec samples;

    CRandomNumbers rng;
    rng.generateUniformSamples(0.0, std::numeric_limits<uint32_t>::max(), 10000, samples);

    for (std::size_t i = 0; i < samples.size(); ++i) {
        uint32_t sample = static_cast<uint32_t>(samples[i]);
        BOOST_REQUIRE_EQUAL(trailingZeros(sample),
                            CBjkstUniqueValues::trailingZeros(sample));
    }
}

BOOST_AUTO_TEST_CASE(testNumber) {
    // Test the approximation errors.

    const std::size_t numberTests = 1000;

    CRandomNumbers rng;

    double totalError5 = 0.0;
    std::size_t largeError5Count = 0;
    double totalError6 = 0.0;
    std::size_t largeError6Count = 0;

    for (std::size_t i = 0; i < numberTests; ++i) {
        CBjkstUniqueValues approxUniqueValues5(5, 60);
        CBjkstUniqueValues approxUniqueValues6(6, 60);
        TUInt32Set uniqueValues;

        TDoubleVec samples;
        rng.generateUniformSamples(0.0, 20000.0, 500u + i, samples);

        for (std::size_t j = 0; j < 2 * samples.size(); ++j) {
            uint32_t sample = static_cast<uint32_t>(samples[j % samples.size()]);
            approxUniqueValues5.add(sample);
            approxUniqueValues6.add(sample);
            uniqueValues.insert(sample);
        }

        double n = static_cast<double>(uniqueValues.size());

        double e5 = static_cast<double>(approxUniqueValues5.number());
        double error5 = std::fabs(e5 - n) / std::max(e5, n);

        double e6 = static_cast<double>(approxUniqueValues6.number());
        double error6 = std::fabs(e6 - n) / std::max(e6, n);

        if (i % 20 == 0) {
            LOG_DEBUG(<< "error5 = " << error5 << ", error6 = " << error6);
        }
        BOOST_TEST_REQUIRE(error5 < 0.35);
        BOOST_TEST_REQUIRE(error6 < 0.30);

        if (error5 > 0.14) {
            ++largeError5Count;
        }
        totalError5 += error5;

        if (error6 > 0.12) {
            ++largeError6Count;
        }
        totalError6 += error6;
    }

    totalError5 /= static_cast<double>(numberTests);
    totalError6 /= static_cast<double>(numberTests);

    LOG_DEBUG(<< "totalError5 = " << totalError5 << ", largeErrorCount5 = " << largeError5Count);
    LOG_DEBUG(<< "totalError6 = " << totalError6 << ", largeErrorCount6 = " << largeError6Count);

    BOOST_TEST_REQUIRE(totalError5 < 0.07);
    BOOST_TEST_REQUIRE(largeError5Count < 80);

    BOOST_TEST_REQUIRE(totalError6 < 0.06);
    BOOST_TEST_REQUIRE(largeError6Count < 85);
}

BOOST_AUTO_TEST_CASE(testRemove) {
    // Check that our error is controlled if we add and remove
    // categories. Note because compression is an irreversible
    // operation we expect higher relative error if the number
    // of unique values shrinks a lot from its peak.

    test::CRandomNumbers rng;

    TSizeVec categories;
    rng.generateUniformSamples(0, 50000, 1000, categories);

    std::size_t numberTests = 500;
    TSizeVec toRemove;
    rng.generateUniformSamples(100, 500, numberTests, toRemove);

    TMeanAccumulator meanRelativeErrorBeforeRemove;
    TMeanAccumulator meanRelativeErrorAfterRemove;

    for (std::size_t t = 0; t < numberTests; ++t) {
        if (t % 10 == 0) {
            LOG_DEBUG(<< "*** test = " << t + 1 << " ***");
        }
        maths::CBjkstUniqueValues sketch(2, 150);
        TUInt32Set unique;
        for (std::size_t i = 0; i < categories.size(); ++i) {
            uint32_t category = static_cast<uint32_t>(categories[i]);
            sketch.add(category);
            unique.insert(category);
        }
        if (t % 10 == 0) {
            LOG_DEBUG(<< "exact  = " << unique.size());
            LOG_DEBUG(<< "approx = " << sketch.number());
        }
        BOOST_REQUIRE_CLOSE_ABSOLUTE(static_cast<double>(unique.size()),
                                     static_cast<double>(sketch.number()),
                                     0.3 * static_cast<double>(unique.size()));
        meanRelativeErrorBeforeRemove.add(std::fabs(static_cast<double>(unique.size()) -
                                                    static_cast<double>(sketch.number())) /
                                          static_cast<double>(unique.size()));

        rng.random_shuffle(categories.begin(), categories.end());
        for (std::size_t i = 0; i < toRemove[t]; ++i) {
            uint32_t category = static_cast<uint32_t>(categories[i]);
            sketch.remove(category);
            unique.erase(category);
        }
        if (t % 10 == 0) {
            LOG_DEBUG(<< "exact  = " << unique.size());
            LOG_DEBUG(<< "approx = " << sketch.number());
        }
        BOOST_REQUIRE_CLOSE_ABSOLUTE(static_cast<double>(unique.size()),
                                     static_cast<double>(sketch.number()),
                                     0.25 * static_cast<double>(unique.size()));
        meanRelativeErrorAfterRemove.add(std::fabs(static_cast<double>(unique.size()) -
                                                   static_cast<double>(sketch.number())) /
                                         static_cast<double>(unique.size()));
    }

    LOG_DEBUG(<< "meanRelativeErrorBeforeRemove = "
              << maths::CBasicStatistics::mean(meanRelativeErrorBeforeRemove));
    LOG_DEBUG(<< "meanRelativeErrorAfterRemove  = "
              << maths::CBasicStatistics::mean(meanRelativeErrorAfterRemove));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanRelativeErrorBeforeRemove) < 0.05);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanRelativeErrorAfterRemove) <
                       1.3 * maths::CBasicStatistics::mean(meanRelativeErrorBeforeRemove));
}

BOOST_AUTO_TEST_CASE(testSwap) {
    test::CRandomNumbers rng;

    TSizeVec categories1;
    TSizeVec categories2;
    TSizeVec categories3;
    TSizeVec categories4;
    rng.generateUniformSamples(0, 20000, 100, categories1);
    rng.generateUniformSamples(0, 20000, 110, categories2);
    rng.generateUniformSamples(0, 20000, 1000, categories3);
    rng.generateUniformSamples(0, 20000, 1100, categories4);

    maths::CBjkstUniqueValues sketch1(3, 100);
    maths::CBjkstUniqueValues sketch2(2, 110);
    maths::CBjkstUniqueValues sketch3(3, 120);
    maths::CBjkstUniqueValues sketch4(2, 180);
    for (std::size_t i = 0; i < categories1.size(); ++i) {
        sketch1.add(static_cast<uint32_t>(categories1[i]));
    }
    for (std::size_t i = 0; i < categories2.size(); ++i) {
        sketch2.add(static_cast<uint32_t>(categories2[i]));
    }
    for (std::size_t i = 0; i < categories3.size(); ++i) {
        sketch3.add(static_cast<uint32_t>(categories3[i]));
    }
    for (std::size_t i = 0; i < categories4.size(); ++i) {
        sketch4.add(static_cast<uint32_t>(categories4[i]));
    }

    uint64_t checksum1 = sketch1.checksum();
    uint64_t checksum2 = sketch2.checksum();
    uint64_t checksum3 = sketch3.checksum();
    uint64_t checksum4 = sketch4.checksum();
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

BOOST_AUTO_TEST_CASE(testSmall) {
    // Test that there is zero error for small distinct
    // counts. This is managed by switching to use a sketch
    // only when exceeding the memory threshold.

    test::CRandomNumbers rng;

    TSizeVec categories;
    rng.generateUniformSamples(0, 50000, 1000, categories);

    TMeanAccumulator meanRelativeError;

    maths::CBjkstUniqueValues sketch(3, 100);
    TUInt32Set unique;
    for (std::size_t i = 0; i < 100; ++i) {
        uint32_t category = static_cast<uint32_t>(categories[i]);
        sketch.add(category);
        unique.insert(category);
        BOOST_REQUIRE_EQUAL(unique.size(), std::size_t(sketch.number()));
        meanRelativeError.add(0.0);
    }

    LOG_DEBUG(<< "# categories = " << sketch.number());
    for (std::size_t i = 100; i < categories.size(); ++i) {
        uint32_t category = static_cast<uint32_t>(categories[i]);
        sketch.add(category);
        unique.insert(category);
        if (i % 20 == 0) {
            LOG_DEBUG(<< "exact  = " << unique.size());
            LOG_DEBUG(<< "approx = " << sketch.number());
        }
        BOOST_REQUIRE_CLOSE_ABSOLUTE(static_cast<double>(unique.size()),
                                     static_cast<double>(sketch.number()),
                                     0.15 * static_cast<double>(unique.size()));
        meanRelativeError.add(std::fabs(static_cast<double>(unique.size()) -
                                        static_cast<double>(sketch.number())) /
                              static_cast<double>(unique.size()));
    }

    LOG_DEBUG(<< "meanRelativeError = " << maths::CBasicStatistics::mean(meanRelativeError));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanRelativeError) < 0.05);
}

BOOST_AUTO_TEST_CASE(testPersist) {
    test::CRandomNumbers rng;

    TSizeVec categories;
    rng.generateUniformSamples(0, 50000, 1000, categories);

    maths::CBjkstUniqueValues origSketch(2, 100);
    for (std::size_t i = 0; i < 100; ++i) {
        origSketch.add(static_cast<uint32_t>(categories[i]));
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origSketch.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_DEBUG(<< "original sketch XML = " << origXml);

    // Restore the XML into a new sketch.
    {
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::CBjkstUniqueValues restoredSketch(traverser);

        LOG_DEBUG(<< "orig checksum = " << origSketch.checksum()
                  << ", new checksum = " << restoredSketch.checksum());
        BOOST_REQUIRE_EQUAL(origSketch.checksum(), restoredSketch.checksum());

        std::string newXml;
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredSketch.acceptPersistInserter(inserter);
        inserter.toXml(newXml);

        BOOST_REQUIRE_EQUAL(origXml, newXml);
    }

    for (std::size_t i = 100; i < categories.size(); ++i) {
        origSketch.add(static_cast<uint32_t>(categories[i]));
    }

    origXml.clear();
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origSketch.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_DEBUG(<< "original sketch XML = " << origXml);

    // Restore the XML into a new sketch.
    {
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::CBjkstUniqueValues restoredSketch(traverser);

        LOG_DEBUG(<< "orig checksum = " << origSketch.checksum()
                  << ", new checksum = " << restoredSketch.checksum());
        BOOST_REQUIRE_EQUAL(origSketch.checksum(), restoredSketch.checksum());

        std::string newXml;
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredSketch.acceptPersistInserter(inserter);
        inserter.toXml(newXml);

        BOOST_REQUIRE_EQUAL(origXml, newXml);
    }
}

BOOST_AUTO_TEST_SUITE_END()
