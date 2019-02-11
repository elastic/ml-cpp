/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CLoopProgressTest_h
#define INCLUDED_CLoopProgressTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLoopProgressTest : public CppUnit::TestFixture {
public:
    void testShort();
    void testRandom();
    void testScaled();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CLoopProgressTest_h
