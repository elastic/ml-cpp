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

#ifndef INCLUDED_CEventRateModelTest_h
#define INCLUDED_CEventRateModelTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CEventRateModelTest : public CppUnit::TestFixture {
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

