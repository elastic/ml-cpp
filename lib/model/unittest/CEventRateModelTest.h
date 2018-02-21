/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CEventRateModelTest_h
#define INCLUDED_CEventRateModelTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CEventRateModelTest : public CppUnit::TestFixture
{
    public:
        void testOnlineCountSample(void);
        void testOnlineNonZeroCountSample(void);
        void testOnlineRare(void);
        void testOnlineProbabilityCalculation(void);
        void testOnlineProbabilityCalculationForLowNonZeroCount(void);
        void testOnlineProbabilityCalculationForHighNonZeroCount(void);
        void testOnlineCorrelatedNoTrend(void);
        void testOnlineCorrelatedTrend(void);
        void testPrune(void);
        void testKey(void);
        void testModelsWithValueFields(void);
        void testCountProbabilityCalculationWithInfluence(void);
        void testDistinctCountProbabilityCalculationWithInfluence(void);
        void testOnlineRareWithInfluence(void);
        void testSkipSampling(void);
        void testExplicitNulls(void);
        void testInterimCorrections(void);
        void testInterimCorrectionsWithCorrelations(void);
        void testSummaryCountZeroRecordsAreIgnored(void);
        void testComputeProbabilityGivenDetectionRule(void);
        void testDecayRateControl(void);
        void testIgnoreSamplingGivenDetectionRules(void);
        static CppUnit::Test *suite(void);

    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CEventRateModelTest_h

