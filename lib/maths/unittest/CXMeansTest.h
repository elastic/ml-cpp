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
        void testCluster();
        void testImproveParams();
        void testImproveStructure();
        void testOneCluster();
        void testFiveClusters();
        void testTwentyClusters();
        void testPoorlyConditioned();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CXMeansTest_h
