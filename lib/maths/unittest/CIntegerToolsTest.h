/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CIntegerToolsTest_h
#define INCLUDED_CIntegerToolsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CIntegerToolsTest : public CppUnit::TestFixture {
public:
    void testNextPow2();
    void testReverseBits();
    void testGcd();
    void testBinomial();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CIntegerToolsTest_h
