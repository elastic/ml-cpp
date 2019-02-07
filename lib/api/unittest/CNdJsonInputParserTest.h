/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CNdJsonInputParserTest_h
#define INCLUDED_CNdJsonInputParserTest_h

#include <cppunit/extensions/HelperMacros.h>

class CNdJsonInputParserTest : public CppUnit::TestFixture {
public:
    void testThroughputArbitraryMapHandler();
    void testThroughputCommonMapHandler();
    void testThroughputArbitraryVecHandler();
    void testThroughputCommonVecHandler();

    static CppUnit::Test* suite();

private:
    void runTest(bool allDocsSameStructure, bool parseAsVecs);
};

#endif // INCLUDED_CNdJsonInputParserTest_h
