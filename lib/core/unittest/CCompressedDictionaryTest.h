/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CCompressedDictionaryTest_h
#define INCLUDED_CCompressedDictionaryTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCompressedDictionaryTest : public CppUnit::TestFixture {
public:
    void testAll();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CCompressedDictionaryTest_h
