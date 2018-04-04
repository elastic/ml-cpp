/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CStringUtilsTest_h
#define INCLUDED_CStringUtilsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CStringUtilsTest : public CppUnit::TestFixture {
public:
    void testNumMatches();
    void testReplace();
    void testReplaceFirst();
    void testTypeToString();
    void testTypeToStringPrecise();
    void testTypeToStringPretty();
    void testStringToType();
    void testTokeniser();
    void testTrim();
    void testJoin();
    void testLower();
    void testUpper();
    void testNarrowWiden();
    void testEscape();
    void testUnEscape();
    void testLongestSubstr();
    void testLongestSubseq();
    void testNormaliseWhitespace();
    void testPerformance();
    void testUtf8ByteType();
    void testRoundtripMaxDouble();

    static CppUnit::Test* suite();

private:
    void testTokeniser(const std::string& delim, const std::string& str);
};

#endif // INCLUDED_CStringUtilsTest_h
