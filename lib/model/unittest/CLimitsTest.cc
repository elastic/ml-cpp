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
#include "CLimitsTest.h"

#include <model/CLimits.h>


CppUnit::Test *CLimitsTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CLimitsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CLimitsTest>(
                               "CLimitsTest::testTrivial",
                               &CLimitsTest::testTrivial) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLimitsTest>(
                               "CLimitsTest::testValid",
                               &CLimitsTest::testValid) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLimitsTest>(
                               "CLimitsTest::testInvalid",
                               &CLimitsTest::testInvalid) );

    return suiteOfTests;
}

void CLimitsTest::testTrivial(void) {
    ml::model::CLimits config;

    CPPUNIT_ASSERT_EQUAL(ml::model::CLimits::DEFAULT_AUTOCONFIG_EVENTS, config.autoConfigEvents());
    CPPUNIT_ASSERT_EQUAL(ml::model::CLimits::DEFAULT_ANOMALY_MAX_TIME_BUCKETS, config.anomalyMaxTimeBuckets());
    CPPUNIT_ASSERT_EQUAL(ml::model::CLimits::DEFAULT_RESULTS_MAX_EXAMPLES, config.maxExamples());
    CPPUNIT_ASSERT_EQUAL(ml::model::CLimits::DEFAULT_RESULTS_UNUSUAL_PROBABILITY_THRESHOLD / 100.0,
                         config.unusualProbabilityThreshold());
    CPPUNIT_ASSERT_EQUAL(ml::model::CResourceMonitor::DEFAULT_MEMORY_LIMIT_MB, config.memoryLimitMB());
}

void CLimitsTest::testValid(void) {
    ml::model::CLimits config;
    CPPUNIT_ASSERT(config.init("testfiles/mllimits.conf"));

    // This one isn't present in the config file so should be defaulted
    CPPUNIT_ASSERT_EQUAL(ml::model::CLimits::DEFAULT_ANOMALY_MAX_TIME_BUCKETS,
                         config.anomalyMaxTimeBuckets());

    CPPUNIT_ASSERT_EQUAL(size_t(8), config.maxExamples());

    CPPUNIT_ASSERT_EQUAL(0.005, config.unusualProbabilityThreshold());

    CPPUNIT_ASSERT_EQUAL(size_t(4567), config.memoryLimitMB());
}

void CLimitsTest::testInvalid(void) {
    ml::model::CLimits config;
    CPPUNIT_ASSERT(!config.init("testfiles/invalidmllimits.conf"));
}

