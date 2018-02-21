/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CEventRateDataGathererTest_h
#define INCLUDED_CEventRateDataGathererTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CEventRateDataGathererTest : public CppUnit::TestFixture
{
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

        static CppUnit::Test *suite(void);

    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CEventRateDataGathererTest_h
