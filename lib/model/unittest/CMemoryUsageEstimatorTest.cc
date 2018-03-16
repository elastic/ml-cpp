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

#include "CMemoryUsageEstimatorTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <model/CMemoryUsageEstimator.h>

#include <boost/bind.hpp>

using namespace ml;
using namespace model;

namespace {

void addValue(CMemoryUsageEstimator& estimator,
              std::size_t memory,
              std::size_t people,
              std::size_t attributes,
              std::size_t correlations = 0) {
    CMemoryUsageEstimator::TSizeArray predictors;
    predictors[CMemoryUsageEstimator::E_People] = people;
    predictors[CMemoryUsageEstimator::E_Attributes] = attributes;
    predictors[CMemoryUsageEstimator::E_Correlations] = correlations;
    estimator.addValue(predictors, memory);
}

CMemoryUsageEstimator::TOptionalSize
estimate(CMemoryUsageEstimator& estimator, std::size_t people, std::size_t attributes, std::size_t correlations = 0) {
    CMemoryUsageEstimator::TSizeArray predictors;
    predictors[CMemoryUsageEstimator::E_People] = people;
    predictors[CMemoryUsageEstimator::E_Attributes] = attributes;
    predictors[CMemoryUsageEstimator::E_Correlations] = correlations;
    return estimator.estimate(predictors);
}
}

void CMemoryUsageEstimatorTest::testEstimateLinear(void) {
    LOG_DEBUG("Running estimator test estimate linear");

    CMemoryUsageEstimator estimator;

    // Pscale = 54
    // Ascale = 556

    // Test that several values have to be added before estimation starts
    CMemoryUsageEstimator::TOptionalSize mem = estimate(estimator, 1, 1);
    CPPUNIT_ASSERT(!mem);

    addValue(estimator, 610, 1, 1);
    mem = estimate(estimator, 2, 1);
    CPPUNIT_ASSERT(!mem);

    addValue(estimator, 664, 2, 1);
    mem = estimate(estimator, 3, 1);
    CPPUNIT_ASSERT(!mem);

    addValue(estimator, 718, 3, 1);
    mem = estimate(estimator, 4, 1);
    CPPUNIT_ASSERT(mem);
    CPPUNIT_ASSERT_EQUAL(std::size_t(772), mem.get());

    addValue(estimator, 826, 5, 1);
    addValue(estimator, 880, 6, 1);
    addValue(estimator, 934, 7, 1);
    addValue(estimator, 988, 8, 1);
    mem = estimate(estimator, 9, 1);
    CPPUNIT_ASSERT(mem);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1042), mem.get());

    // Test that after 10 estimates we need to add some more real values
    for (std::size_t i = 0; i < 10; i++) {
        mem = estimate(estimator, 4, 1);
    }
    CPPUNIT_ASSERT(!mem);

    // Test that adding values for Attributes scales independently of People
    addValue(estimator, 1274, 3, 2);
    addValue(estimator, 2386, 3, 4);
    mem = estimate(estimator, 4, 4);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2440), mem.get());
    mem = estimate(estimator, 5, 4);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2494), mem.get());
    mem = estimate(estimator, 6, 5);
    CPPUNIT_ASSERT_EQUAL(std::size_t(3104), mem.get());

    // This is outside the variance range of the supplied values
    mem = estimate(estimator, 60, 30);
    CPPUNIT_ASSERT(!mem);
}

void CMemoryUsageEstimatorTest::testEstimateNonlinear(void) {
    LOG_DEBUG("Running estimator test estimate non-linear");

    {
        // intercept = 356
        // Pscale = 54
        // Ascale = 200
        // + noise

        CMemoryUsageEstimator estimator;
        addValue(estimator, 602, 1, 1);
        addValue(estimator, 668, 2, 1);
        addValue(estimator, 716, 3, 1);
        addValue(estimator, 830, 5, 1);
        addValue(estimator, 874, 6, 1);
        addValue(estimator, 938, 7, 1);
        addValue(estimator, 1020, 8, 1);

        CMemoryUsageEstimator::TOptionalSize mem = estimate(estimator, 9, 1);
        CPPUNIT_ASSERT_EQUAL(std::size_t(1080), mem.get());

        addValue(estimator, 1188, 8, 2);
        mem = estimate(estimator, 9, 3);
        CPPUNIT_ASSERT_EQUAL(std::size_t(1443), mem.get());
    }

    {
        // quadratic

        int pScale = 100;
        int aScale = 50;
        int cScale = 30;

        CMemoryUsageEstimator estimator;
        addValue(estimator, pScale * 10 * 10 + aScale * 9 * 9 + cScale * 15 * 15, 10, 9, 15);
        addValue(estimator, pScale * 11 * 11 + aScale * 11 * 11 + cScale * 20 * 20, 11, 11, 20);
        addValue(estimator, pScale * 12 * 12 + aScale * 13 * 13 + cScale * 25 * 25, 12, 13, 25);
        addValue(estimator, pScale * 13 * 13 + aScale * 15 * 15 + cScale * 26 * 26, 13, 15, 26);
        addValue(estimator, pScale * 17 * 17 + aScale * 19 * 19 + cScale * 27 * 27, 17, 19, 27);
        addValue(estimator, pScale * 20 * 20 + aScale * 19 * 19 + cScale * 30 * 30, 20, 19, 30);
        addValue(estimator, pScale * 20 * 20 + aScale * 25 * 25 + cScale * 40 * 40, 20, 25, 40);

        CMemoryUsageEstimator::TOptionalSize mem = estimate(estimator, 25, 35, 45);
        std::size_t actual = pScale * 25 * 25 + aScale * 35 * 35 + cScale * 45 * 45;
        LOG_DEBUG("actual = " << actual << ", estimated = " << mem.get());
        CPPUNIT_ASSERT(static_cast<double>(actual - mem.get()) / static_cast<double>(actual) < 0.15);
    }
}

void CMemoryUsageEstimatorTest::testPersist(void) {
    LOG_DEBUG("Running estimator test persist");

    CMemoryUsageEstimator origEstimator;
    {
        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            origEstimator.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
            LOG_DEBUG("origXml = " << origXml);
        }

        // Restore the XML into a new data gatherer
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        CMemoryUsageEstimator restoredEstimator;
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            boost::bind(&CMemoryUsageEstimator::acceptRestoreTraverser, &restoredEstimator, _1)));

        // The XML representation of the new data gatherer should be the same
        // as the original.
        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restoredEstimator.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
    {
        int pScale = 10000;
        int aScale = 5;
        int cScale = 3;
        addValue(origEstimator, pScale * 10 + aScale * 9 + cScale * 15, 10, 9, 15);
        addValue(origEstimator, pScale * 11 + aScale * 11 + cScale * 20, 11, 11, 20);
        addValue(origEstimator, pScale * 12 + aScale * 13 + cScale * 25, 12, 13, 25);
        addValue(origEstimator, pScale * 13 + aScale * 15 + cScale * 26, 13, 15, 26);
        addValue(origEstimator, pScale * 17 + aScale * 19 + cScale * 27, 17, 19, 27);
        addValue(origEstimator, pScale * 20 + aScale * 19 + cScale * 30, 20, 19, 30);

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            origEstimator.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
            LOG_DEBUG("origXml = " << origXml);
        }

        // Restore the XML into a new data gatherer
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        CMemoryUsageEstimator restoredEstimator;
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            boost::bind(&CMemoryUsageEstimator::acceptRestoreTraverser, &restoredEstimator, _1)));

        // The XML representation of the new data gatherer should be the same
        // as the original.
        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restoredEstimator.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
}

CppUnit::Test* CMemoryUsageEstimatorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CMemoryUsageEstimatorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CMemoryUsageEstimatorTest>(
        "CMemoryUsageEstimatorTest::testEstimateLinear", &CMemoryUsageEstimatorTest::testEstimateLinear));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMemoryUsageEstimatorTest>(
        "CMemoryUsageEstimatorTest::testEstimateNonlinear", &CMemoryUsageEstimatorTest::testEstimateNonlinear));
    suiteOfTests->addTest(new CppUnit::TestCaller<CMemoryUsageEstimatorTest>("CMemoryUsageEstimatorTest::testPersist",
                                                                             &CMemoryUsageEstimatorTest::testPersist));

    return suiteOfTests;
}
