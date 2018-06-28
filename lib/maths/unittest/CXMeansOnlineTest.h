/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CXMeansOnlineTest_h
#define INCLUDED_CXMeansOnlineTest_h

#include <cppunit/extensions/HelperMacros.h>

class CXMeansOnlineTest : public CppUnit::TestFixture {
public:
    void testCluster();
    void testClusteringVanilla();
    void testClusteringWithOutliers();
    void testManyClusters();
    void testAdaption();
    void testLargeHistory();
    void testLatLongData();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CXMeansOnlineTest_h
