/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMathsMemoryTest_h
#define INCLUDED_CMathsMemoryTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMathsMemoryTest : public CppUnit::TestFixture {
public:
    void testPriors();
    void testBjkstVec();
    void testTimeSeriesDecompositions();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CMathsMemoryTest_h
