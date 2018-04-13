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
        void testSample();
        void testMultivariateSample();
        void testProbabilityCalculationForMetric();
        void testProbabilityCalculationForMedian();
        void testProbabilityCalculationForLowMedian();
        void testProbabilityCalculationForHighMedian();
        void testProbabilityCalculationForLowMean();
        void testProbabilityCalculationForHighMean();
        void testProbabilityCalculationForLowSum();
        void testProbabilityCalculationForHighSum();
        void testProbabilityCalculationForLatLong();
        void testInfluence();
        void testLatLongInfluence();
        void testPrune();
        void testSkipSampling();
        void testExplicitNulls();
        void testKey();
        void testVarp();
        void testInterimCorrections();
        void testInterimCorrectionsWithCorrelations();
        void testCorrelatePersist();
        void testSummaryCountZeroRecordsAreIgnored();
        void testDecayRateControl();
        void testIgnoreSamplingGivenDetectionRules();

        static CppUnit::Test *suite();

    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricModelTest_h
