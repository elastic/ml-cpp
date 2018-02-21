/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMetricPopulationDataGathererTest_h
#define INCLUDED_CMetricPopulationDataGathererTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>


class CMetricPopulationDataGathererTest : public CppUnit::TestFixture
{
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

        static CppUnit::Test *suite(void);

    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricPopulationDataGathererTest_h
