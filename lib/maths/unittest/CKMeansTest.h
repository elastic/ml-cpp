/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CKMeansTest_h
#define INCLUDED_CKMeansTest_h

#include <cppunit/extensions/HelperMacros.h>

class CKMeansTest : public CppUnit::TestFixture
{
    public:
        void testDataPropagation(void);
        void testFilter(void);
        void testCentroids(void);
        void testClosestPoints(void);
        void testRun(void);
        void testRunWithSphericalClusters(void);
        void testPlusPlus(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CKMeansTest_h
