/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CLineifiedJsonInputParserTest_h
#define INCLUDED_CLineifiedJsonInputParserTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLineifiedJsonInputParserTest : public CppUnit::TestFixture {
public:
    void testThroughputArbitrary();
    void testThroughputCommon();

    static CppUnit::Test* suite();

private:
    void runTest(bool allDocsSameStructure);
};

#endif // INCLUDED_CLineifiedJsonInputParserTest_h
