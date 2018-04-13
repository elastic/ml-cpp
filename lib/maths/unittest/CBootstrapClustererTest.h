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
        void testFacade();
        void testBuildClusterGraph();
        void testCutSearch();
        void testSeparate();
        void testThickets();
        void testNonConvexClustering();
        void testClusteringStability();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CBootstrapClustererTest_h
