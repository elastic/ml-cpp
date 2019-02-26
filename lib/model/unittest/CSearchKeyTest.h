/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CSearchKeyTest_h
#define INCLUDED_CSearchKeyTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSearchKeyTest : public CppUnit::TestFixture {
public:
    void testSimpleCountComesFirst();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CSearchKeyTest_h
