/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CXMeansOnlineTest_h
#define INCLUDED_CXMeansOnlineTest_h

#include <cppunit/extensions/HelperMacros.h>

class CXMeansOnlineTest : public CppUnit::TestFixture
{
    public:
        void testCluster(void);
        void testClusteringVanilla(void);
        void testClusteringWithOutliers(void);
        void testManyClusters(void);
        void testAdaption(void);
        void testLargeHistory(void);
        void testLatLongData(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CXMeansOnlineTest_h
