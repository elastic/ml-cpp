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
        void testCluster(void);
        void testMixtureOfGaussians(void);
        void testMixtureOfUniforms(void);
        void testMixtureOfLogNormals(void);
        void testOutliers(void);
        void testManyClusters(void);
        void testLowVariation(void);
        void testAdaption(void);
        void testLargeHistory(void);
        void testPersist(void);
        void testPruneEmptyCluster(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CXMeansOnline1dTest_h
