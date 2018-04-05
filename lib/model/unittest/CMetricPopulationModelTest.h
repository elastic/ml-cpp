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

#ifndef INCLUDED_CMetricPopulationModelTest_h
#define INCLUDED_CMetricPopulationModelTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CMetricPopulationModelTest : public CppUnit::TestFixture {
public:
    void testBasicAccessors();
    void testMinMaxAndMean();
    void testVarp();
    void testComputeProbability();
    void testPrune();
    void testKey();
    void testFrequency();
    void testSampleRateWeight();
    void testPeriodicity();
    void testPersistence();
    void testIgnoreSamplingGivenDetectionRules();

    static CppUnit::Test* suite();

private:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricPopulationModelTest_h
