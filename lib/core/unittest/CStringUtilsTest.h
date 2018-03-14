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

class CStringUtilsTest : public CppUnit::TestFixture {
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

    static CppUnit::Test* suite();

private:
    void testTokeniser(const std::string& delim, const std::string& str);
};

#endif // INCLUDED_CStringUtilsTest_h
