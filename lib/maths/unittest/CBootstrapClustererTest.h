/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CBootstrapClustererTest_h
#define INCLUDED_CBootstrapClustererTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBootstrapClustererTest : public CppUnit::TestFixture
{
    public:
        void testFacade(void);
        void testBuildClusterGraph(void);
        void testCutSearch(void);
        void testSeparate(void);
        void testThickets(void);
        void testNonConvexClustering(void);
        void testClusteringStability(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CBootstrapClustererTest_h
