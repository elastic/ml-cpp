/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CRandomProjectionClustererTest_h
#define INCLUDED_CRandomProjectionClustererTest_h

#include <cppunit/extensions/HelperMacros.h>

class CRandomProjectionClustererTest : public CppUnit::TestFixture {
public:
    void testGenerateProjections();
    void testClusterProjections();
    void testNeighbourhoods();
    void testSimilarities();
    void testClusterNeighbourhoods();
    void testAccuracy();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CRandomProjectionClustererTest_h
