/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CBoostedTreeTest_h
#define INCLUDED_CBoostedTreeTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBoostedTreeTest : public CppUnit::TestFixture {
public:
    void testPiecewiseConstant();
    void testLinear();
    void testNonLinear();
    void testThreading();
    void testConstantFeatures();
    void testConstantTarget();
    void testCategoricalRegressors();
    void testEstimateMemoryUsedByTrain();
    void testProgressMonitoring();
    void testMissingData();
    // TODO void testFeatureWeights();
    // TODO void testNuisanceFeatures();
    void testPersistRestore();
    void testRestoreErrorHandling();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CBoostedTreeTest_h
