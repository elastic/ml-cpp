/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_COrderingsTest_h
#define INCLUDED_COrderingsTest_h

#include <cppunit/extensions/HelperMacros.h>

class COrderingsTest : public CppUnit::TestFixture {
public:
    void testOptionalOrdering();
    void testPtrOrdering();
    void testLess();
    void testFirstLess();
    void testFirstGreater();
    void testSecondLess();
    void testSecondGreater();
    void testDereference();
    void testLexicographicalCompare();
    void testSimultaneousSort();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_COrderingsTest_h
