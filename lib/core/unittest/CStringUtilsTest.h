/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#ifndef INCLUDED_CStringUtilsTest_h
#define INCLUDED_CStringUtilsTest_h

#include <cppunit/extensions/HelperMacros.h>


class CStringUtilsTest : public CppUnit::TestFixture
{
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

        static CppUnit::Test *suite();

    private:
        void testTokeniser(const std::string &delim,
                           const std::string &str);
};

#endif // INCLUDED_CStringUtilsTest_h

