/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CLimitsTest_h
#define INCLUDED_CLimitsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLimitsTest : public CppUnit::TestFixture {
public:
    void testTrivial();
    void testValid();
    void testInvalid();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CLimitsTest_h
