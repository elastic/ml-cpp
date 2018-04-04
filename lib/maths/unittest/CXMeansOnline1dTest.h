/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CXMeansOnline1dTest_h
#define INCLUDED_CXMeansOnline1dTest_h

#include <cppunit/extensions/HelperMacros.h>

class CXMeansOnline1dTest : public CppUnit::TestFixture
{
    public:
        void testCluster();
        void testMixtureOfGaussians();
        void testMixtureOfUniforms();
        void testMixtureOfLogNormals();
        void testOutliers();
        void testManyClusters();
        void testLowVariation();
        void testAdaption();
        void testLargeHistory();
        void testPersist();
        void testPruneEmptyCluster();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CXMeansOnline1dTest_h
