/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CXMeansTest_h
#define INCLUDED_CXMeansTest_h

#include <cppunit/extensions/HelperMacros.h>

class CXMeansTest : public CppUnit::TestFixture
{
    public:
        void testCluster(void);
        void testImproveParams(void);
        void testImproveStructure(void);
        void testOneCluster(void);
        void testFiveClusters(void);
        void testTwentyClusters(void);
        void testPoorlyConditioned(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CXMeansTest_h
