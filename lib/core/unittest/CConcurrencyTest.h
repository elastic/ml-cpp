/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CConcurrencyTest_h
#define INCLUDED_CConcurrencyTest_h

#include <cppunit/extensions/HelperMacros.h>

class CConcurrencyTest : public CppUnit::TestFixture {
public:
    void testAsyncWithExecutors();
    void testParallelForEach();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataFrameTest_h
