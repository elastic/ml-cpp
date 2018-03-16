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

#include "CCountMinSketchTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCountMinSketch.h>

#include <test/CRandomNumbers.h>

using namespace ml;

typedef std::vector<double> TDoubleVec;

void CCountMinSketchTest::testCounts(void) {
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CCountMinSketchTest::testCounts  |");
    LOG_DEBUG("+-----------------------------------+");

    typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;

    test::CRandomNumbers rng;

    // Test that we can estimate counts exactly up to "# rows" x "# columns".
    // and that the error subsequently increases approximately linearly with
    // increasing category count.

    LOG_DEBUG("")
    LOG_DEBUG("Test Uniform")

    for (std::size_t t = 0u, n = 100u; n < 1500; ++t, n += 100) {
        LOG_DEBUG("*** number categories = " << n << " ***");

        TDoubleVec counts;
        rng.generateUniformSamples(2.0, 301.0, n, counts);

        maths::CCountMinSketch sketch(2, 751);

        for (std::size_t i = 0u; i < counts.size(); ++i) {
            counts[i] = ::floor(counts[i]);
            sketch.add(static_cast<uint32_t>(i), counts[i]);
        }
        LOG_DEBUG("error = " << sketch.oneMinusDeltaError());

        TMeanAccumulator meanError;
        double errorCount = 0.0;
        for (std::size_t i = 0u; i < counts.size(); ++i) {
            double count = counts[i];
            double estimated = sketch.count(static_cast<uint32_t>(i));
            if (i % 50 == 0) {
                LOG_DEBUG("category = " << i << ", true count = " << count << ", estimated count = " << estimated);
            }

            meanError.add(::fabs(estimated - count));
            if (count + sketch.oneMinusDeltaError() < estimated) {
                errorCount += 1.0;
            }
        }
        LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
        LOG_DEBUG("error count = " << errorCount);
        if (sketch.oneMinusDeltaError() == 0.0) {
            CPPUNIT_ASSERT_EQUAL(0.0, maths::CBasicStatistics::mean(meanError));
        } else {
            //CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError)
            //                   < 0.1 * static_cast<double>(n));
        }
        //CPPUNIT_ASSERT_EQUAL(0.0, errorCount);
    }

    // Test the case that a small fraction of categories generate a large
    // fraction of the counts.

    LOG_DEBUG("");
    LOG_DEBUG("Test Heavy Hitters");
    {
        TDoubleVec heavyHitters;
        rng.generateUniformSamples(10000.0, 11001.0, 20, heavyHitters);

        TDoubleVec counts;
        rng.generateUniformSamples(2.0, 101.0, 1000, counts);

        maths::CCountMinSketch sketch(2, 751);

        for (std::size_t i = 0u; i < heavyHitters.size(); ++i) {
            heavyHitters[i] = ::floor(heavyHitters[i]);
            sketch.add(static_cast<uint32_t>(i), heavyHitters[i]);
        }
        for (std::size_t i = 0u; i < counts.size(); ++i) {
            counts[i] = ::floor(counts[i]);
            sketch.add(static_cast<uint32_t>(i + heavyHitters.size()), counts[i]);
        }
        LOG_DEBUG("error = " << sketch.oneMinusDeltaError());

        TMeanAccumulator meanRelativeError;
        for (std::size_t i = 0u; i < heavyHitters.size(); ++i) {
            double count = heavyHitters[i];
            double estimated = sketch.count(static_cast<uint32_t>(i));
            LOG_DEBUG("category = " << i << ", true count = " << count << ", estimated count = " << estimated);

            double relativeError = ::fabs(estimated - count) / count;
            CPPUNIT_ASSERT(relativeError < 0.01);

            meanRelativeError.add(relativeError);
        }

        LOG_DEBUG("mean relative error " << maths::CBasicStatistics::mean(meanRelativeError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanRelativeError) < 0.005);
    }
}

void CCountMinSketchTest::testSwap(void) {
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  CCountMinSketchTest::testSwap  |");
    LOG_DEBUG("+---------------------------------+");

    test::CRandomNumbers rng;

    TDoubleVec counts1;
    TDoubleVec counts2;
    TDoubleVec counts3;
    TDoubleVec counts4;
    rng.generateUniformSamples(2.0, 301.0, 500, counts1);
    rng.generateUniformSamples(2.0, 301.0, 500, counts2);
    rng.generateUniformSamples(2.0, 301.0, 1500, counts3);
    rng.generateUniformSamples(2.0, 301.0, 1500, counts4);

    maths::CCountMinSketch sketch1(3, 500);
    maths::CCountMinSketch sketch2(2, 750);
    maths::CCountMinSketch sketch3(3, 300);
    maths::CCountMinSketch sketch4(2, 400);
    for (std::size_t i = 0u; i < counts1.size(); ++i) {
        sketch1.add(static_cast<uint32_t>(i), counts1[i]);
    }
    for (std::size_t i = 0u; i < counts2.size(); ++i) {
        sketch2.add(static_cast<uint32_t>(i), counts2[i]);
    }
    for (std::size_t i = 0u; i < counts3.size(); ++i) {
        sketch3.add(static_cast<uint32_t>(i), counts3[i]);
    }
    for (std::size_t i = 0u; i < counts4.size(); ++i) {
        sketch4.add(static_cast<uint32_t>(i), counts4[i]);
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

void CCountMinSketchTest::testPersist(void) {
    LOG_DEBUG("+------------------------------------+");
    LOG_DEBUG("|  CCountMinSketchTest::testPersist  |");
    LOG_DEBUG("+------------------------------------+");

    test::CRandomNumbers rng;

    TDoubleVec counts;
    rng.generateUniformSamples(2.0, 301.0, 500, counts);

    maths::CCountMinSketch origSketch(2, 600);
    for (std::size_t i = 0u; i < counts.size(); ++i) {
        origSketch.add(static_cast<uint32_t>(i), counts[i]);
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
        maths::CCountMinSketch restoredSketch(traverser);

        LOG_DEBUG("orig checksum = " << origSketch.checksum() << ", new checksum = " << restoredSketch.checksum());
        CPPUNIT_ASSERT_EQUAL(origSketch.checksum(), restoredSketch.checksum());

        std::string newXml;
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredSketch.acceptPersistInserter(inserter);
        inserter.toXml(newXml);

        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }

    // Sketch.
    TDoubleVec moreCounts;
    rng.generateUniformSamples(2.0, 301.0, 500, moreCounts);
    for (std::size_t i = 0u; i < moreCounts.size(); ++i) {
        origSketch.add(static_cast<uint32_t>(counts.size() + i), moreCounts[i]);
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
        maths::CCountMinSketch restoredSketch(traverser);

        LOG_DEBUG("orig checksum = " << origSketch.checksum() << ", new checksum = " << restoredSketch.checksum());
        CPPUNIT_ASSERT_EQUAL(origSketch.checksum(), restoredSketch.checksum());

        std::string newXml;
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredSketch.acceptPersistInserter(inserter);
        inserter.toXml(newXml);

        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
}

CppUnit::Test* CCountMinSketchTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CCountMinSketchTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CCountMinSketchTest>("CCountMinSketchTest::testCounts",
                                                                       &CCountMinSketchTest::testCounts));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CCountMinSketchTest>("CCountMinSketchTest::testSwap", &CCountMinSketchTest::testSwap));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCountMinSketchTest>("CCountMinSketchTest::testPersist",
                                                                       &CCountMinSketchTest::testPersist));

    return suiteOfTests;
}
