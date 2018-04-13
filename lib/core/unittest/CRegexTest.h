/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CRegexTest_h
#define INCLUDED_CRegexTest_h

#include <cppunit/extensions/HelperMacros.h>

class CRegexTest : public CppUnit::TestFixture {
public:
    void testInit();
    void testSearch();
    void testSplit();
    void testTokenise1();
    void testTokenise2();
    void testEscape();
    void testLiteralCount();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CRegexTest_h
