/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMetricDataGathererTest_h
#define INCLUDED_CMetricDataGathererTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CMetricDataGathererTest : public CppUnit::TestFixture
{
    public:
        void singleSeriesTests();
        void multipleSeriesTests();
        void testSampleCount();
        void testRemovePeople();
        void testSum();
        void singleSeriesOutOfOrderTests();
        void testResetBucketGivenSingleSeries();
        void testResetBucketGivenMultipleSeries();
        void testInfluenceStatistics();
        void testMultivariate();
        void testStatisticsPersist();
        void testVarp();

        static CppUnit::Test *suite();
    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricDataGathererTest_h
