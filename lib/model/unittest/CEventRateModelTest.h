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

class CEventRateModelTest : public CppUnit::TestFixture
{
    public:
        void testOnlineCountSample();
        void testOnlineNonZeroCountSample();
        void testOnlineRare();
        void testOnlineProbabilityCalculation();
        void testOnlineProbabilityCalculationForLowNonZeroCount();
        void testOnlineProbabilityCalculationForHighNonZeroCount();
        void testOnlineCorrelatedNoTrend();
        void testOnlineCorrelatedTrend();
        void testPrune();
        void testKey();
        void testModelsWithValueFields();
        void testCountProbabilityCalculationWithInfluence();
        void testDistinctCountProbabilityCalculationWithInfluence();
        void testOnlineRareWithInfluence();
        void testSkipSampling();
        void testExplicitNulls();
        void testInterimCorrections();
        void testInterimCorrectionsWithCorrelations();
        void testSummaryCountZeroRecordsAreIgnored();
        void testComputeProbabilityGivenDetectionRule();
        void testDecayRateControl();
        void testIgnoreSamplingGivenDetectionRules();
        static CppUnit::Test *suite();

    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CEventRateModelTest_h

