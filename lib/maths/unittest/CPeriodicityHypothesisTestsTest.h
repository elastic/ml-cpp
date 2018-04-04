/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CPeriodicityHypothesisTestsTest_h
#define INCLUDED_CPeriodicityHypothesisTestsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPeriodicityHypothesisTestsTest : public CppUnit::TestFixture {
public:
    void testNonPeriodic();
    void testDiurnal();
    void testNonDiurnal();
    void testWithSparseData();
    void testTestForPeriods();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CPeriodicityHypothesisTestsTest_h
