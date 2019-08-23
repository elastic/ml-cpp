/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CPersistenceTagTest_h
#define INCLUDED_CPersistenceTagTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPersistenceTagTest : public CppUnit::TestFixture {
public:
    void testName();
    void testComparisons();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CPersistenceTagTest_h
