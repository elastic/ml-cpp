/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_COutputChainerTest_h
#define INCLUDED_COutputChainerTest_h

#include <cppunit/extensions/HelperMacros.h>

class COutputChainerTest : public CppUnit::TestFixture {
public:
    void testChaining();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_COutputChainerTest_h
