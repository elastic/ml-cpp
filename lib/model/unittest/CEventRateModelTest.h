/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CEventRateModelTest_h
#define INCLUDED_CEventRateModelTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CEventRateModelTest : public CppUnit::TestFixture {
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
    static CppUnit::Test* suite();

private:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CEventRateModelTest_h
