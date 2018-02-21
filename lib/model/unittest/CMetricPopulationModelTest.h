/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMetricPopulationModelTest_h
#define INCLUDED_CMetricPopulationModelTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CMetricPopulationModelTest : public CppUnit::TestFixture
{
    public:
        void testBasicAccessors(void);
        void testMinMaxAndMean(void);
        void testVarp(void);
        void testComputeProbability(void);
        void testPrune(void);
        void testKey(void);
        void testFrequency(void);
        void testSampleRateWeight(void);
        void testPeriodicity(void);
        void testPersistence(void);
        void testIgnoreSamplingGivenDetectionRules(void);

        static CppUnit::Test *suite(void);
    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricPopulationModelTest_h
