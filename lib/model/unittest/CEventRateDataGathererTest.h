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
    void singleSeriesTests();
    void multipleSeriesTests();
    void testRemovePeople();
    void singleSeriesOutOfOrderFinalResultTests();
    void singleSeriesOutOfOrderInterimResultTests();
    void multipleSeriesOutOfOrderFinalResultTests();
    void testArrivalBeforeLatencyWindowIsIgnored();
    void testResetBucketGivenSingleSeries();
    void testResetBucketGivenMultipleSeries();
    void testResetBucketGivenBucketNotAvailable();
    void testInfluencerBucketStatistics();
    void testDistinctStrings();
    void testLatencyPersist();
    void testDiurnalFeatures();

    static CppUnit::Test* suite();

private:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CEventRateDataGathererTest_h
