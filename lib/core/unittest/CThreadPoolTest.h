/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CThreadPoolTest_h
#define INCLUDED_CThreadPoolTest_h

#include <cppunit/extensions/HelperMacros.h>

class CThreadPoolTest : public CppUnit::TestFixture {
public:
    void testPool();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CThreadPoolTest_h
