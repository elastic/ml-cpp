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

        static CppUnit::Test *suite();

    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CEventRateDataGathererTest_h
