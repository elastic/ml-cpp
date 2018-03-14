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

#ifndef INCLUDED_CEventRateDataGathererTest_h
#define INCLUDED_CEventRateDataGathererTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CEventRateDataGathererTest : public CppUnit::TestFixture {
public:
    void singleSeriesTests(void);
    void multipleSeriesTests(void);
    void testRemovePeople(void);
    void singleSeriesOutOfOrderFinalResultTests(void);
    void singleSeriesOutOfOrderInterimResultTests(void);
    void multipleSeriesOutOfOrderFinalResultTests(void);
    void testArrivalBeforeLatencyWindowIsIgnored(void);
    void testResetBucketGivenSingleSeries(void);
    void testResetBucketGivenMultipleSeries(void);
    void testResetBucketGivenBucketNotAvailable(void);
    void testInfluencerBucketStatistics(void);
    void testDistinctStrings(void);
    void testLatencyPersist(void);
    void testDiurnalFeatures(void);

    static CppUnit::Test* suite(void);

private:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CEventRateDataGathererTest_h
