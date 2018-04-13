/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMathsFuncsTest_h
#define INCLUDED_CMathsFuncsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMathsFuncsTest : public CppUnit::TestFixture {
public:
    void testIsNan();
    void testIsInf();
    void testIsFinite();
    void testFpStatus();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CMathsFuncsTest_h
