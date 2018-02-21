/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CStringUtilsTest_h
#define INCLUDED_CStringUtilsTest_h

#include <cppunit/extensions/HelperMacros.h>


class CStringUtilsTest : public CppUnit::TestFixture
{
    public:
        void testNumMatches(void);
        void testReplace(void);
        void testReplaceFirst(void);
        void testTypeToString(void);
        void testTypeToStringPrecise(void);
        void testTypeToStringPretty(void);
        void testStringToType(void);
        void testTokeniser(void);
        void testTrim(void);
        void testJoin(void);
        void testLower(void);
        void testUpper(void);
        void testNarrowWiden(void);
        void testEscape(void);
        void testUnEscape(void);
        void testLongestSubstr(void);
        void testLongestSubseq(void);
        void testNormaliseWhitespace(void);
        void testPerformance(void);
        void testUtf8ByteType(void);
        void testRoundtripMaxDouble(void);

        static CppUnit::Test *suite();

    private:
        void testTokeniser(const std::string &delim,
                           const std::string &str);
};

#endif // INCLUDED_CStringUtilsTest_h

