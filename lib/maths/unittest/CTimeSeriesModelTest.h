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

class CTimeSeriesModelTest : public CppUnit::TestFixture {
public:
    void testClone();
    void testMode();
    void testAddBucketValue();
    void testAddSamples();
    void testPredict();
    void testProbability();
    void testWeights();
    void testMemoryUsage();
    void testPersist();
    void testUpgrade();
    void testAddSamplesWithCorrelations();
    void testProbabilityWithCorrelations();
    void testAnomalyModel();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CTimeSeriesModelTest_h
