/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CTimeSeriesDecompositionTest_h
#define INCLUDED_CTimeSeriesDecompositionTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTimeSeriesDecompositionTest : public CppUnit::TestFixture
{
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
