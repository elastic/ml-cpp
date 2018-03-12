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
        void testSuperpositionOfSines(void);
        void testDistortedPeriodic(void);
        void testMinimizeLongComponents(void);
        void testWeekend(void);
        void testSinglePeriodicity(void);
        void testSeasonalOnset(void);
        void testVarianceScale(void);
        void testSpikeyDataProblemCase(void);
        void testDiurnalProblemCase(void);
        void testComplexDiurnalProblemCase(void);
        void testDiurnalPeriodicityWithMissingValues(void);
        void testLongTermTrend(void);
        void testLongTermTrendAndPeriodicity(void);
        void testNonDiurnal(void);
        void testYearly(void);
        void testCalendar(void);
        void testConditionOfTrend(void);
        void testSwap(void);
        void testPersist(void);
        void testUpgrade(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CTimeSeriesDecompositionTest_h
