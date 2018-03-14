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

#include "CBjkstUniqueValuesTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBjkstUniqueValues.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

using namespace ml;
using namespace maths;
using namespace test;

namespace {

typedef std::vector<double> TDoubleVec;
typedef std::vector<std::size_t> TSizeVec;
typedef std::set<uint32_t> TUInt32Set;
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;

uint8_t trailingZeros(uint32_t x) {
    uint8_t result = 0;
    for (/**/; (x & 0x1) == 0; x >>= 1) {
        ++result;
    }
    return result;
}

}

void CBjkstUniqueValuesTest::testTrailingZeros(void) {
    LOG_DEBUG("+---------------------------------------------+");
    LOG_DEBUG("|  CBjkstUniqueValuesTest::testTrailingZeros  |");
    LOG_DEBUG("+---------------------------------------------+");

    uint32_t n = 1;
    for (uint8_t i = 0; i < 32; n <<= 1, ++i) {
        CPPUNIT_ASSERT_EQUAL(i, CBjkstUniqueValues::trailingZeros(n));
    }

    TDoubleVec samples;

    CRandomNumbers rng;
    rng.generateUniformSamples(0.0,
                               std::numeric_limits<uint32_t>::max(),
                               10000,
                               samples);

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        uint32_t sample = static_cast<uint32_t>(samples[i]);
        CPPUNIT_ASSERT_EQUAL(trailingZeros(sample),
                             CBjkstUniqueValues::trailingZeros(sample));
    }
}

void CBjkstUniqueValuesTest::testNumber(void) {
    LOG_DEBUG("+--------------------------------------+");
    LOG_DEBUG("|  CBjkstUniqueValuesTest::testNumber  |");
    LOG_DEBUG("+--------------------------------------+");

    // Test the approximation errors.

    const std::size_t numberTests = 1000u;

    CRandomNumbers rng;

    double      totalError5 = 0.0;
    std::size_t largeError5Count = 0u;
    double      totalError6 = 0.0;
    std::size_t largeError6Count = 0u;

    for (std::size_t i = 0u; i < numberTests; ++i) {
        CBjkstUniqueValues approxUniqueValues5(5, 60);
        CBjkstUniqueValues approxUniqueValues6(6, 60);
        TUInt32Set         uniqueValues;

        TDoubleVec samples;
        rng.generateUniformSamples(0.0, 20000.0, 500u + i, samples);

        for (std::size_t j = 0u; j < 2 * samples.size(); ++j) {
            uint32_t sample = static_cast<uint32_t>(samples[j % samples.size()]);
            approxUniqueValues5.add(sample);
            approxUniqueValues6.add(sample);
            uniqueValues.insert(sample);
        }

        double n = static_cast<double>(uniqueValues.size());

        double e5 = static_cast<double>(approxUniqueValues5.number());
        double error5 = ::fabs(e5 - n) / std::max(e5, n);

        double e6 = static_cast<double>(approxUniqueValues6.number());
        double error6 = ::fabs(e6 - n) / std::max(e6, n);

        LOG_DEBUG("error5 = " << error5 << ", error6 = " << error6);
        CPPUNIT_ASSERT(error5 < 0.35);
        CPPUNIT_ASSERT(error6 < 0.30);

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

    LOG_DEBUG("totalError5 = " << totalError5
                               << ", largeErrorCount5 = " << largeError5Count);
    LOG_DEBUG("totalError6 = " << totalError6
                               << ", largeErrorCount6 = " << largeError6Count);

    CPPUNIT_ASSERT(totalError5 < 0.07);
    CPPUNIT_ASSERT(largeError5Count < 80);

    CPPUNIT_ASSERT(totalError6 < 0.06);
    CPPUNIT_ASSERT(largeError6Count < 85);
}

void CBjkstUniqueValuesTest::testRemove(void) {
    LOG_DEBUG("+--------------------------------------+");
    LOG_DEBUG("|  CBjkstUniqueValuesTest::testRemove  |");
    LOG_DEBUG("+--------------------------------------+");

    // Check that our error is controlled if we add and remove
    // categories. Note because compression is an irreversible
    // operation we expect higher relative error if the number
    // of unique values shrinks a lot from its peak.

    test::CRandomNumbers rng;

    TSizeVec categories;
    rng.generateUniformSamples(0, 50000, 1000, categories);

    std::size_t numberTests = 500u;
    TSizeVec    toRemove;
    rng.generateUniformSamples(100, 500, numberTests, toRemove);

    TMeanAccumulator meanRelativeErrorBeforeRemove;
    TMeanAccumulator meanRelativeErrorAfterRemove;

    for (std::size_t t = 0u; t < numberTests; ++t) {
        LOG_DEBUG("*** test = " << t+1 << " ***");

        maths::CBjkstUniqueValues sketch(2, 150);
        TUInt32Set                unique;
        for (std::size_t i = 0u; i < categories.size(); ++i) {
            uint32_t category = static_cast<uint32_t>(categories[i]);
            sketch.add(category);
            unique.insert(category);
        }
        LOG_DEBUG("exact  = " << unique.size());
        LOG_DEBUG("approx = " << sketch.number());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(static_cast<double>(unique.size()),
                                     static_cast<double>(sketch.number()),
                                     0.3 * static_cast<double>(unique.size()));
        meanRelativeErrorBeforeRemove.add(::fabs(  static_cast<double>(unique.size())
                                                   - static_cast<double>(sketch.number()))
                                          / static_cast<double>(unique.size()));


        rng.random_shuffle(categories.begin(), categories.end());
        for (std::size_t i = 0u; i < toRemove[t]; ++i) {
            uint32_t category = static_cast<uint32_t>(categories[i]);
            sketch.remove(category);
            unique.erase(category);
        }
        LOG_DEBUG("exact  = " << unique.size());
        LOG_DEBUG("approx = " << sketch.number());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(static_cast<double>(unique.size()),
                                     static_cast<double>(sketch.number()),
                                     0.25 * static_cast<double>(unique.size()));
        meanRelativeErrorAfterRemove.add(::fabs(  static_cast<double>(unique.size())
                                                  - static_cast<double>(sketch.number()))
                                         / static_cast<double>(unique.size()));
    }

    LOG_DEBUG("meanRelativeErrorBeforeRemove = "
              << maths::CBasicStatistics::mean(meanRelativeErrorBeforeRemove));
    LOG_DEBUG("meanRelativeErrorAfterRemove  = "
              << maths::CBasicStatistics::mean(meanRelativeErrorAfterRemove));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanRelativeErrorBeforeRemove) < 0.05);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanRelativeErrorAfterRemove)
                   < 1.3 * maths::CBasicStatistics::mean(meanRelativeErrorBeforeRemove));
}

void CBjkstUniqueValuesTest::testSwap(void) {
    LOG_DEBUG("+------------------------------------+");
    LOG_DEBUG("|  CBjkstUniqueValuesTest::testSwap  |");
    LOG_DEBUG("+------------------------------------+");

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
    for (std::size_t i = 0u; i < categories1.size(); ++i) {
        sketch1.add(static_cast<uint32_t>(categories1[i]));
    }
    for (std::size_t i = 0u; i < categories2.size(); ++i) {
        sketch2.add(static_cast<uint32_t>(categories2[i]));
    }
    for (std::size_t i = 0u; i < categories3.size(); ++i) {
        sketch3.add(static_cast<uint32_t>(categories3[i]));
    }
    for (std::size_t i = 0u; i < categories4.size(); ++i) {
        sketch4.add(static_cast<uint32_t>(categories4[i]));
    }

    uint64_t checksum1 = sketch1.checksum();
    uint64_t checksum2 = sketch2.checksum();
    uint64_t checksum3 = sketch3.checksum();
    uint64_t checksum4 = sketch4.checksum();
    LOG_DEBUG("checksum1 = " << checksum1);
    LOG_DEBUG("checksum2 = " << checksum2);
    LOG_DEBUG("checksum3 = " << checksum3);
    LOG_DEBUG("checksum4 = " << checksum4);

    sketch1.swap(sketch2);
    CPPUNIT_ASSERT_EQUAL(checksum2, sketch1.checksum());
    CPPUNIT_ASSERT_EQUAL(checksum1, sketch2.checksum());
    sketch1.swap(sketch2);

    sketch2.swap(sketch3);
    CPPUNIT_ASSERT_EQUAL(checksum3, sketch2.checksum());
    CPPUNIT_ASSERT_EQUAL(checksum2, sketch3.checksum());
    sketch2.swap(sketch3);

    sketch1.swap(sketch4);
    CPPUNIT_ASSERT_EQUAL(checksum1, sketch4.checksum());
    CPPUNIT_ASSERT_EQUAL(checksum4, sketch1.checksum());
    sketch1.swap(sketch4);

    sketch3.swap(sketch4);
    CPPUNIT_ASSERT_EQUAL(checksum3, sketch4.checksum());
    CPPUNIT_ASSERT_EQUAL(checksum4, sketch3.checksum());
    sketch3.swap(sketch4);
}

void CBjkstUniqueValuesTest::testSmall(void) {
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CBjkstUniqueValuesTest::testSmall  |");
    LOG_DEBUG("+-------------------------------------+");

    // Test that there is zero error for small distinct
    // counts. This is managed by switching to use a sketch
    // only when exceeding the memory threshold.

    test::CRandomNumbers rng;

    TSizeVec categories;
    rng.generateUniformSamples(0, 50000, 1000, categories);

    TMeanAccumulator meanRelativeError;

    maths::CBjkstUniqueValues sketch(3, 100);
    TUInt32Set                unique;
    for (std::size_t i = 0u; i < 100; ++i) {
        uint32_t category = static_cast<uint32_t>(categories[i]);
        sketch.add(category);
        unique.insert(category);
        CPPUNIT_ASSERT_EQUAL(unique.size(), std::size_t(sketch.number()));
        meanRelativeError.add(0.0);
    }

    LOG_DEBUG("# categories = " << sketch.number());
    for (std::size_t i = 100u; i < categories.size(); ++i) {
        uint32_t category = static_cast<uint32_t>(categories[i]);
        sketch.add(category);
        unique.insert(category);
        LOG_DEBUG("exact  = " << unique.size());
        LOG_DEBUG("approx = " << sketch.number());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(static_cast<double>(unique.size()),
                                     static_cast<double>(sketch.number()),
                                     0.15 * static_cast<double>(unique.size()));
        meanRelativeError.add(::fabs(  static_cast<double>(unique.size())
                                       - static_cast<double>(sketch.number()))
                              / static_cast<double>(unique.size()));
    }

    LOG_DEBUG("meanRelativeError = " << maths::CBasicStatistics::mean(meanRelativeError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanRelativeError) < 0.05);
}

void CBjkstUniqueValuesTest::testPersist(void) {
    LOG_DEBUG("+---------------------------------------+");
    LOG_DEBUG("|  CBjkstUniqueValuesTest::testPersist  |");
    LOG_DEBUG("+---------------------------------------+");

    test::CRandomNumbers rng;

    TSizeVec categories;
    rng.generateUniformSamples(0, 50000, 1000, categories);

    maths::CBjkstUniqueValues origSketch(2, 100);
    for (std::size_t i = 0u; i < 100; ++i) {
        origSketch.add(static_cast<uint32_t>(categories[i]));
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origSketch.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_DEBUG("original sketch XML = " << origXml);

    // Restore the XML into a new sketch.
    {
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::CBjkstUniqueValues            restoredSketch(traverser);

        LOG_DEBUG("orig checksum = " << origSketch.checksum()
                                     << ", new checksum = " << restoredSketch.checksum());
        CPPUNIT_ASSERT_EQUAL(origSketch.checksum(),
                             restoredSketch.checksum());

        std::string                         newXml;
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredSketch.acceptPersistInserter(inserter);
        inserter.toXml(newXml);

        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }

    for (std::size_t i = 100u; i < categories.size(); ++i) {
        origSketch.add(static_cast<uint32_t>(categories[i]));
    }

    origXml.clear();
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origSketch.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_DEBUG("original sketch XML = " << origXml);

    // Restore the XML into a new sketch.
    {
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::CBjkstUniqueValues            restoredSketch(traverser);

        LOG_DEBUG("orig checksum = " << origSketch.checksum()
                                     << ", new checksum = " << restoredSketch.checksum());
        CPPUNIT_ASSERT_EQUAL(origSketch.checksum(),
                             restoredSketch.checksum());

        std::string                         newXml;
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredSketch.acceptPersistInserter(inserter);
        inserter.toXml(newXml);

        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
}

CppUnit::Test *CBjkstUniqueValuesTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CBjkstUniqueValuesTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CBjkstUniqueValuesTest>(
                               "CBjkstUniqueValuesTest::testTrailingZeros",
                               &CBjkstUniqueValuesTest::testTrailingZeros) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBjkstUniqueValuesTest>(
                               "CBjkstUniqueValuesTest::testNumber",
                               &CBjkstUniqueValuesTest::testNumber) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBjkstUniqueValuesTest>(
                               "CBjkstUniqueValuesTest::testRemove",
                               &CBjkstUniqueValuesTest::testRemove) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBjkstUniqueValuesTest>(
                               "CBjkstUniqueValuesTest::testSwap",
                               &CBjkstUniqueValuesTest::testSwap) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBjkstUniqueValuesTest>(
                               "CBjkstUniqueValuesTest::testSmall",
                               &CBjkstUniqueValuesTest::testSmall) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBjkstUniqueValuesTest>(
                               "CBjkstUniqueValuesTest::testPersist",
                               &CBjkstUniqueValuesTest::testPersist) );

    return suiteOfTests;
}
