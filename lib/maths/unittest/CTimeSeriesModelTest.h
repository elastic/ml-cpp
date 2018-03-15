/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_CTimeSeriesModelTest_h
#define INCLUDED_CTimeSeriesModelTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTimeSeriesModelTest : public CppUnit::TestFixture
{
    public:
        void testClone(void);
        void testMode(void);
        void testAddBucketValue(void);
        void testAddSamples(void);
        void testPredict(void);
        void testProbability(void);
        void testWeights(void);
        void testMemoryUsage(void);
        void testPersist(void);
        void testUpgrade(void);
        void testAddSamplesWithCorrelations(void);
        void testProbabilityWithCorrelations(void);
        void testAnomalyModel(void);
        void testStepChangeDiscontinuities(void);
        void daylightSaving(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CTimeSeriesModelTest_h
