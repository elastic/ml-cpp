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

#ifndef INCLUDED_CMetricModelTest_h
#define INCLUDED_CMetricModelTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CMetricModelTest : public CppUnit::TestFixture {
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

    static CppUnit::Test* suite();

private:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricModelTest_h
