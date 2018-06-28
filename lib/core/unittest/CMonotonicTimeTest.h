/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CMonotonicTimeTest_h
#define INCLUDED_CMonotonicTimeTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMonotonicTimeTest : public CppUnit::TestFixture {
public:
    void testMilliseconds();
    void testNanoseconds();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CMonotonicTimeTest_h
