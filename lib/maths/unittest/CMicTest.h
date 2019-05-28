/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMicTest_h
#define INCLUDED_CMicTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMicTest : public CppUnit::TestFixture {
public:
    void testOptimizeXAxis();
    void testInvariants();
    void testIndependent();
    void testOneToOne();
    void testCorrelated();
    void testVsMutualInformation();
    void testEdgeCases();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CMicTest_h
