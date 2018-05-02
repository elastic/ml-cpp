/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CTimeSeriesDecompositionTest_h
#define INCLUDED_CTimeSeriesDecompositionTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTimeSeriesDecompositionTest : public CppUnit::TestFixture {
public:
    void testSuperpositionOfSines();
    void testDistortedPeriodic();
    void testMinimizeLongComponents();
    void testWeekend();
    void testSinglePeriodicity();
    void testSeasonalOnset();
    void testVarianceScale();
    void testSpikeyDataProblemCase();
    void testVeryLargeValuesProblemCase();
    void testMixedSmoothAndSpikeyDataProblemCase();
    void testDiurnalPeriodicityWithMissingValues();
    void testLongTermTrend();
    void testLongTermTrendAndPeriodicity();
    void testNonDiurnal();
    void testYearly();
    void testWithOutliers();
    void testCalendar();
    void testConditionOfTrend();
    void testSwap();
    void testPersist();
    void testUpgrade();

    static CppUnit::Test* suite();
    virtual void setUp();
    virtual void tearDown();
};

#endif // INCLUDED_CTimeSeriesDecompositionTest_h
