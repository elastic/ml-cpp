/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
    void testStepChangeDiscontinuities();
    void testLinearScaling();
    void testDaylightSaving();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CTimeSeriesModelTest_h
