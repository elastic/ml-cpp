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

#ifndef INCLUDED_CMetricDataGathererTest_h
#define INCLUDED_CMetricDataGathererTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CMetricDataGathererTest : public CppUnit::TestFixture {
public:
    void singleSeriesTests(void);
    void multipleSeriesTests(void);
    void testSampleCount(void);
    void testRemovePeople(void);
    void testSum(void);
    void singleSeriesOutOfOrderTests(void);
    void testResetBucketGivenSingleSeries(void);
    void testResetBucketGivenMultipleSeries(void);
    void testInfluenceStatistics(void);
    void testMultivariate(void);
    void testStatisticsPersist(void);
    void testVarp(void);

    static CppUnit::Test* suite(void);

private:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricDataGathererTest_h
