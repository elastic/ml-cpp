/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CInterimBucketCorrectorTest.h"

#include <test/CRandomNumbers.h>

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CSmallVector.h>

#include <model/CInterimBucketCorrector.h>

#include <boost/bind.hpp>

using namespace ml;
using namespace model;

namespace {
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble10Vec = core::CSmallVector<double, 10>;
const double EPSILON = 1e-10;
}

CppUnit::Test* CInterimBucketCorrectorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CInterimBucketCorrectorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CInterimBucketCorrectorTest>(
        "CInterimBucketCorrectorTest::testCorrectionsGivenSingleValue",
        &CInterimBucketCorrectorTest::testCorrectionsGivenSingleValue));
    suiteOfTests->addTest(new CppUnit::TestCaller<CInterimBucketCorrectorTest>(
        "CInterimBucketCorrectorTest::testCorrectionsGivenSingleValueAndNoBaseline",
        &CInterimBucketCorrectorTest::testCorrectionsGivenSingleValueAndNoBaseline));
    suiteOfTests->addTest(new CppUnit::TestCaller<CInterimBucketCorrectorTest>(
        "CInterimBucketCorrectorTest::testCorrectionsGivenMultiValueAndMultiMode",
        &CInterimBucketCorrectorTest::testCorrectionsGivenMultiValueAndMultiMode));
    suiteOfTests->addTest(new CppUnit::TestCaller<CInterimBucketCorrectorTest>(
        "CInterimBucketCorrectorTest::testPersist", &CInterimBucketCorrectorTest::testPersist));

    return suiteOfTests;
}

void CInterimBucketCorrectorTest::testCorrectionsGivenSingleValue() {
    core_t::TTime bucketLength(3600);
    CInterimBucketCorrector corrector(bucketLength);

    core_t::TTime now = 3600;
    core_t::TTime end = now + 24 * bucketLength;
    while (now < end) {
        corrector.finalBucketCount(now, 100);
        now += bucketLength;
    }

    {
        // Value = 1100, Mode = 1000, Completeness = 50%, Expected another 500.
        // Value > mode, correction should be 0.
        double value = 1100.0;
        corrector.currentBucketCount(now, 50);
        double correction = corrector.corrections(1000, value);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, correction, EPSILON);
    }
    {
        // Value = 100, Completeness = 50%, Expected another 500.
        // Correction should be 500.
        double value = 100.0;
        corrector.currentBucketCount(now, 50);
        double correction = corrector.corrections(1000, value);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(500.0, correction, EPSILON);
    }
    {
        // Value = 200, Completeness = 10%, Expected another 4500.
        // Correction should be 4500.
        double value = 200.0;
        corrector.currentBucketCount(now, 10);
        double correction = corrector.corrections(5000, value);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(4500.0, correction, EPSILON);
    }
    {
        // Value = 0, Completeness = 10%, Expected another 900.
        // Correction should be 900.
        double value = 0.0;
        corrector.currentBucketCount(now, 10);
        double correction = corrector.corrections(1000, value);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(900.0, correction, EPSILON);
    }
    {
        // Value = 800, Completeness = 50%, Expected another 500.
        // Correction should be 200.
        double value = 800.0;
        corrector.currentBucketCount(now, 50);
        double correction = corrector.corrections(1000, value);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(200.0, correction, EPSILON);
    }
    {
        // Value = 0, Completeness = 0%, Expected another 1000.
        // Correction should be 200.
        double value = 0.0;
        corrector.currentBucketCount(now, 0);
        double correction = corrector.corrections(1000, value);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1000.0, correction, EPSILON);
    }
    {
        // Value = -800, Completeness = 40%, Expected another -400.
        // Correction should be -200.
        double value = -800.0;
        corrector.currentBucketCount(now, 40);
        double correction = corrector.corrections(-1000, value);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-200.0, correction, EPSILON);
    }
    {
        // Value = -100, Completeness = 40%, Expected another -600.
        // Correction should be -400.
        double value = -100.0;
        corrector.currentBucketCount(now, 40);
        double correction = corrector.corrections(-1000, value);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-600.0, correction, EPSILON);
    }
}

void CInterimBucketCorrectorTest::testCorrectionsGivenSingleValueAndNoBaseline() {
    core_t::TTime bucketLength(3600);
    CInterimBucketCorrector corrector(bucketLength);

    double value = 100.0;
    corrector.currentBucketCount(3600, 10);
    double correction = corrector.corrections(1000, value);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, correction, EPSILON);
}

void CInterimBucketCorrectorTest::testCorrectionsGivenMultiValueAndMultiMode() {
    core_t::TTime bucketLength(3600);
    CInterimBucketCorrector corrector(bucketLength);

    core_t::TTime now = 3600;
    core_t::TTime end = now + 24 * bucketLength;
    while (now < end) {
        corrector.finalBucketCount(now, 100);
        now += bucketLength;
    }

    TDouble10Vec value(10, 0.0);
    value[0] = 0.0;
    value[1] = 500.0;
    value[2] = 500.0;
    value[3] = 500.0;
    value[4] = 500.0;
    value[5] = 500.0;
    value[6] = 500.0;
    value[7] = 500.0;
    value[8] = 500.0;
    value[9] = 901.0;

    TDouble10Vec mode(10, 0.0);
    mode[0] = 1000.0;
    mode[1] = 100.0;
    mode[2] = 200.0;
    mode[3] = 300.0;
    mode[4] = 400.0;
    mode[5] = 500.0;
    mode[6] = 600.0;
    mode[7] = 700.0;
    mode[8] = 800.0;
    mode[9] = 900.0;

    corrector.currentBucketCount(now, 50);
    TDouble10Vec correction = corrector.corrections(mode, value);
    CPPUNIT_ASSERT_EQUAL(std::size_t(10), correction.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(500.0, correction[0], EPSILON);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, correction[1], EPSILON);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, correction[2], EPSILON);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, correction[3], EPSILON);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, correction[4], EPSILON);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, correction[5], EPSILON);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(100.0, correction[6], EPSILON);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(200.0, correction[7], EPSILON);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(300.0, correction[8], EPSILON);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, correction[9], EPSILON);
}

void CInterimBucketCorrectorTest::testPersist() {
    core_t::TTime bucketLength(300);
    CInterimBucketCorrector corrector(bucketLength);

    core_t::TTime now = 300;
    core_t::TTime end = now + 24 * bucketLength;
    while (now < end) {
        corrector.finalBucketCount(now, 100);
        now += bucketLength;
    }

    double value = 100.0;
    corrector.currentBucketCount(now, 50);
    double correction = corrector.corrections(1000, value);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(500.0, correction, EPSILON);

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        corrector.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_TRACE(<< "XML:\n" << origXml);

    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);
    CInterimBucketCorrector restoredCorrector(bucketLength);
    traverser.traverseSubLevel(boost::bind(
        &CInterimBucketCorrector::acceptRestoreTraverser, &restoredCorrector, _1));

    corrector.currentBucketCount(now, 50);
    correction = restoredCorrector.corrections(1000, value);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(500.0, correction, EPSILON);
}
