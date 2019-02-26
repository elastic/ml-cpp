/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CKMeansTest_h
#define INCLUDED_CKMeansTest_h

#include <cppunit/extensions/HelperMacros.h>

class CKMeansTest : public CppUnit::TestFixture {
public:
    void testDataPropagation();
    void testFilter();
    void testCentroids();
    void testClosestPoints();
    void testRun();
    void testRunWithSphericalClusters();
    void testPlusPlus();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CKMeansTest_h
