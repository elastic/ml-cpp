/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_COutliersTest_h
#define INCLUDED_COutliersTest_h

#include <cppunit/extensions/HelperMacros.h>

class COutliersTest : public CppUnit::TestFixture {
public:
    void testLof();
    void testDlof();
    void testDistancekNN();
    void testTotalDistancekNN();
    void testEnsemble();
    void testFeatureInfluences();
    void testEstimateMemoryUsedByCompute();
    void testProgressMonitoring();
    void testMostlyDuplicate();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_COutliersTest_h
