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

#ifndef INCLUDED_CMetricPopulationDataGathererTest_h
#define INCLUDED_CMetricPopulationDataGathererTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CMetricPopulationDataGathererTest : public CppUnit::TestFixture {
public:
    void testMean(void);
    void testMin(void);
    void testMax(void);
    void testSum(void);
    void testSampleCount(void);
    void testFeatureData(void);
    void testRemovePeople(void);
    void testRemoveAttributes(void);
    void testInfluenceStatistics(void);
    void testPersistence(void);
    void testReleaseMemory(void);

    static CppUnit::Test* suite(void);

private:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricPopulationDataGathererTest_h
