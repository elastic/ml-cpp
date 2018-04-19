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
    void testCalendar();
    void testConditionOfTrend();
    void testSwap();
    void testPersist();
    void testUpgrade();

    static CppUnit::Test* suite();
    virtual void setUp();
    virtual void tearDown();

private:
    std::string m_TimeZone;
};

#endif // INCLUDED_CTimeSeriesDecompositionTest_h
