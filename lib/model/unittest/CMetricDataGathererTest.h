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

        static CppUnit::Test *suite(void);
    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricDataGathererTest_h
