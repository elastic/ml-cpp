/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CByteSwapperTest_h
#define INCLUDED_CByteSwapperTest_h

#include <cppunit/extensions/HelperMacros.h>

class CByteSwapperTest : public CppUnit::TestFixture {
public:
    void testByteSwaps();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CByteSwapperTest_h
