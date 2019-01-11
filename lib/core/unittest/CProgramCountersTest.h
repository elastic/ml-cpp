/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CProgramCountersTest_h
#define INCLUDED_CProgramCountersTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <core/CProgramCounters.h>

class CProgramCountersTest : public CppUnit::TestFixture {
public:
    void testCounters();
    void testPersist();
    void testCacheCounters();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CProgramCountersTest_h
