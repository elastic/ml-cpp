/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CTickerTest_h
#define INCLUDED_CTickerTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTickerTest : public CppUnit::TestFixture {
public:
    void testTicker();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CTickerTest_h
