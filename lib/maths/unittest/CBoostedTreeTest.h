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
    void testConstantFeatures();
    void testConstantObjective();
    void testMissingData();
    // TODO void testCategoricalRegressors();
    // TODO void testFeatureWeights();
    // TODO void testNuisanceFeatures();
    // TODO void testModelReflection();
    void testErrors();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CBoostedTreeTest_h
