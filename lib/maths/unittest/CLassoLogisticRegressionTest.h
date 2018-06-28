/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CLassoLogisticRegressionTest_h
#define INCLUDED_CLassoLogisticRegressionTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLassoLogisticRegressionTest : public CppUnit::TestFixture {
public:
    void testCyclicCoordinateDescent();
    void testCyclicCoordinateDescentLargeSparse();
    void testCyclicCoordinateDescentIncremental();
    void testNormBasedLambda();
    void testCrossValidatedLambda();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CLassoLogisticRegressionTest_h
