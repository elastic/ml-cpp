/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMetricModelTest_h
#define INCLUDED_CMetricModelTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CMetricModelTest : public CppUnit::TestFixture
{
    public:
        void testSample(void);
        void testMultivariateSample(void);
        void testProbabilityCalculationForMetric(void);
        void testProbabilityCalculationForMedian(void);
        void testProbabilityCalculationForLowMedian(void);
        void testProbabilityCalculationForHighMedian(void);
        void testProbabilityCalculationForLowMean(void);
        void testProbabilityCalculationForHighMean(void);
        void testProbabilityCalculationForLowSum(void);
        void testProbabilityCalculationForHighSum(void);
        void testProbabilityCalculationForLatLong(void);
        void testInfluence(void);
        void testLatLongInfluence(void);
        void testPrune(void);
        void testSkipSampling(void);
        void testExplicitNulls(void);
        void testKey(void);
        void testVarp(void);
        void testInterimCorrections(void);
        void testInterimCorrectionsWithCorrelations(void);
        void testCorrelatePersist(void);
        void testSummaryCountZeroRecordsAreIgnored(void);
        void testDecayRateControl(void);
        void testIgnoreSamplingGivenDetectionRules(void);

        static CppUnit::Test *suite(void);

    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricModelTest_h
