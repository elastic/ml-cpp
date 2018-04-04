/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CPersistUtilsTest_h
#define INCLUDED_CPersistUtilsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPersistUtilsTest : public CppUnit::TestFixture {
public:
    void testPersistContainers();
    void testPersistIterators();
    void testAppend();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CPersistUtilsTest_h
