/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CLeastSquaresOnlineRegressionTest_h
#define INCLUDED_CLeastSquaresOnlineRegressionTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLeastSquaresOnlineRegressionTest : public CppUnit::TestFixture {
public:
    void testInvariants();
    void testFit();
    void testShiftAbscissa();
    void testShiftOrdinate();
    void testShiftGradient();
    void testLinearScale();
    void testAge();
    void testPrediction();
    void testCombination();
    void testSingular();
    void testScale();
    void testMean();
    void testCovariances();
    void testParameters();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CLeastSquaresOnlineRegressionTest_h
