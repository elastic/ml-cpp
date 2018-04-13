/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CSolversTest_h
#define INCLUDED_CSolversTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSolversTest : public CppUnit::TestFixture {
public:
    void testBracket();
    void testBisection();
    void testBrent();
    void testSublevelSet();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CSolversTest_h
