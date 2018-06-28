/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CKMostCorrelatedTest_h
#define INCLUDED_CKMostCorrelatedTest_h

#include <cppunit/extensions/HelperMacros.h>

class CKMostCorrelatedTest : public CppUnit::TestFixture {
public:
    void testCorrelation();
    void testNextProjection();
    void testMostCorrelated();
    void testRemoveVariables();
    void testAccuracy();
    void testStability();
    void testChangingCorrelation();
    void testMissingData();
    void testPersistence();
    void testScale();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CKMostCorrelatedTest_h
