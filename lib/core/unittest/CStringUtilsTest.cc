/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CStringUtilsTest.h"

#include <core/CLogger.h>
#include <core/CStopWatch.h>
#include <core/CStrTokR.h>
#include <core/CStringUtils.h>

#include <boost/lexical_cast.hpp>

#include <set>
#include <vector>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

CppUnit::Test* CStringUtilsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CStringUtilsTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testNumMatches", &CStringUtilsTest::testNumMatches));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testReplace", &CStringUtilsTest::testReplace));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testReplaceFirst", &CStringUtilsTest::testReplaceFirst));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testTypeToString", &CStringUtilsTest::testTypeToString));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testTypeToStringPrecise", &CStringUtilsTest::testTypeToStringPrecise));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testTypeToStringPretty", &CStringUtilsTest::testTypeToStringPretty));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testStringToType", &CStringUtilsTest::testStringToType));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testTokeniser", &CStringUtilsTest::testTokeniser));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testTrim", &CStringUtilsTest::testTrim));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testJoin", &CStringUtilsTest::testJoin));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testLower", &CStringUtilsTest::testLower));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testUpper", &CStringUtilsTest::testUpper));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testNarrowWiden", &CStringUtilsTest::testNarrowWiden));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testEscape", &CStringUtilsTest::testEscape));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testUnEscape", &CStringUtilsTest::testUnEscape));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testLongestSubstr", &CStringUtilsTest::testLongestSubstr));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testLongestSubseq", &CStringUtilsTest::testLongestSubseq));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testNormaliseWhitespace", &CStringUtilsTest::testNormaliseWhitespace));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testPerformance", &CStringUtilsTest::testPerformance));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testUtf8ByteType", &CStringUtilsTest::testUtf8ByteType));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStringUtilsTest>(
        "CStringUtilsTest::testRoundtripMaxDouble", &CStringUtilsTest::testRoundtripMaxDouble));

    return suiteOfTests;
}

void CStringUtilsTest::testNumMatches() {
    {
        std::string str("%d %M %Y %f %D  %t");

        CPPUNIT_ASSERT_EQUAL(size_t(6), ml::core::CStringUtils::numMatches(str, "%"));
        CPPUNIT_ASSERT_EQUAL(size_t(0), ml::core::CStringUtils::numMatches(str, "q"));
    }
}

void CStringUtilsTest::testReplace() {
    {
        std::string in("%d%M%Y%f%D%t");
        const std::string out(" %d %M %Y %f %D %t");

        CPPUNIT_ASSERT_EQUAL(size_t(6), ml::core::CStringUtils::replace("%", " %", in));

        CPPUNIT_ASSERT_EQUAL(out, in);
    }
    {
        std::string in("%d%M%Y%f%D%t");
        const std::string out("%d%M%Y%f%D%t");

        CPPUNIT_ASSERT_EQUAL(size_t(0), ml::core::CStringUtils::replace("X", "Y", in));

        CPPUNIT_ASSERT_EQUAL(out, in);
    }
}

void CStringUtilsTest::testReplaceFirst() {
    {
        std::string in("%d%M%Y%f%D%t");
        const std::string out(" %d%M%Y%f%D%t");

        CPPUNIT_ASSERT_EQUAL(size_t(1),
                             ml::core::CStringUtils::replaceFirst("%", " %", in));

        CPPUNIT_ASSERT_EQUAL(out, in);
    }
    {
        std::string in("%d%M%Y%f%D%t");
        const std::string out("%d%M%Y%f%D%t");

        CPPUNIT_ASSERT_EQUAL(size_t(0), ml::core::CStringUtils::replaceFirst("X", "Y", in));

        CPPUNIT_ASSERT_EQUAL(out, in);
    }
}

void CStringUtilsTest::testTypeToString() {
    {
        uint64_t i(18446744073709551615ULL);
        std::string expected("18446744073709551615");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        CPPUNIT_ASSERT_EQUAL(expected, actual);

        uint64_t j(0);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(actual, j));
        CPPUNIT_ASSERT_EQUAL(i, j);
    }
    {
        uint32_t i(123456U);
        std::string expected("123456");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        CPPUNIT_ASSERT_EQUAL(expected, actual);

        uint32_t j(0);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(actual, j));
        CPPUNIT_ASSERT_EQUAL(i, j);
    }
    {
        uint16_t i(12345U);
        std::string expected("12345");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        CPPUNIT_ASSERT_EQUAL(expected, actual);

        uint16_t j(0);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(actual, j));
        CPPUNIT_ASSERT_EQUAL(i, j);
    }
    {
        int32_t i(123456);
        std::string expected("123456");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        CPPUNIT_ASSERT_EQUAL(expected, actual);

        int32_t j(0);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(actual, j));
        CPPUNIT_ASSERT_EQUAL(i, j);
    }
    {
        double i(0.123456);
        std::string expected("0.123456");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.123456e10);
        std::string expected("1234560000.000000");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
}

void CStringUtilsTest::testTypeToStringPrecise() {
    {
        double i(1.0);
        std::string expected("1");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(1.0);
        std::string expected("1");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.123456);
        std::string expected("1.23456e-1");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.123456);
        std::string expected("1.23456e-1");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.123456e10);
        std::string expected("1.23456e9");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.123456e10);
        std::string expected("1234560000");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.123456e-10);
        std::string expected("1.23456e-11");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.123456e-10);
        std::string expected("1.23456e-11");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.123456787654321e-10);
        std::string expected("1.234568e-11");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.123456787654321e-10);
        std::string expected("1.23456787654321e-11");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.00000000012345678765432123456);
        std::string expected("1.234568e-10");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(0.00000000012345678765432123456);
        std::string expected("1.23456787654321e-10");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(123456787654321.23456);
        std::string expected("1.234568e14");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    {
        double i(123456787654321.23456);
        std::string expected("123456787654321");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
}

void CStringUtilsTest::testTypeToStringPretty() {
    // This doesn't assert because the format differs between operating systems
    LOG_DEBUG(<< "1.0 -> " << ml::core::CStringUtils::typeToStringPretty(1.0));
    LOG_DEBUG(<< "0.123456 -> " << ml::core::CStringUtils::typeToStringPretty(0.123456));
    LOG_DEBUG(<< "0.123456e10 -> " << ml::core::CStringUtils::typeToStringPretty(0.123456e10));
    LOG_DEBUG(<< "0.123456e-10 -> "
              << ml::core::CStringUtils::typeToStringPretty(0.123456e-10));
    LOG_DEBUG(<< "0.123456787654321e-10 -> "
              << ml::core::CStringUtils::typeToStringPretty(0.123456787654321e-10));
    LOG_DEBUG(<< "0.00000000012345678765432123456 -> "
              << ml::core::CStringUtils::typeToStringPretty(0.00000000012345678765432123456));
    LOG_DEBUG(<< "123456787654321.23456 -> "
              << ml::core::CStringUtils::typeToStringPretty(123456787654321.23456));
}

void CStringUtilsTest::testStringToType() {
    {
        // All good conversions
        bool ret;
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("yes", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("no", ret));
        CPPUNIT_ASSERT(!ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("yES", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("NO", ret));
        CPPUNIT_ASSERT(!ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("true", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("false", ret));
        CPPUNIT_ASSERT(!ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("TRUE", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("False", ret));
        CPPUNIT_ASSERT(!ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("on", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("off", ret));
        CPPUNIT_ASSERT(!ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("On", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("OFF", ret));
        CPPUNIT_ASSERT(!ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("y", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("n", ret));
        CPPUNIT_ASSERT(!ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("Y", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("N", ret));
        CPPUNIT_ASSERT(!ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("t", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("f", ret));
        CPPUNIT_ASSERT(!ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("T", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("F", ret));
        CPPUNIT_ASSERT(!ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("1", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("10", ret));
        CPPUNIT_ASSERT(ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("0", ret));
        CPPUNIT_ASSERT(!ret);
    }
    {
        // All good conversions
        int32_t ret;
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("1000", ret));
        CPPUNIT_ASSERT_EQUAL(int32_t(1000), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("-1000", ret));
        CPPUNIT_ASSERT_EQUAL(int32_t(-1000), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("0", ret));
        CPPUNIT_ASSERT_EQUAL(int32_t(0), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("0x1000", ret));
        CPPUNIT_ASSERT_EQUAL(int32_t(0x1000), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("2147483647", ret));
        CPPUNIT_ASSERT_EQUAL(int32_t(2147483647), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("-2147483647", ret));
        CPPUNIT_ASSERT_EQUAL(int32_t(-2147483647), ret);
    }
    {
        // All good conversions
        uint64_t ret;
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("1000", ret));
        CPPUNIT_ASSERT_EQUAL(uint64_t(1000), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("0", ret));
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("0x1000", ret));
        CPPUNIT_ASSERT_EQUAL(uint64_t(0x1000), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("18446744073709551615", ret));
        CPPUNIT_ASSERT_EQUAL(uint64_t(18446744073709551615ULL), ret);
    }
    {
        // All good conversions
        uint32_t ret;
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("1000", ret));
        CPPUNIT_ASSERT_EQUAL(uint32_t(1000), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("0", ret));
        CPPUNIT_ASSERT_EQUAL(uint32_t(0), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("0x1000", ret));
        CPPUNIT_ASSERT_EQUAL(uint32_t(0x1000), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("2147483650", ret));
        CPPUNIT_ASSERT_EQUAL(uint32_t(2147483650UL), ret);
    }
    {
        // All good conversions
        uint16_t ret;
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("1000", ret));
        CPPUNIT_ASSERT_EQUAL(uint16_t(1000), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("0", ret));
        CPPUNIT_ASSERT_EQUAL(uint16_t(0), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("0x1000", ret));
        CPPUNIT_ASSERT_EQUAL(uint16_t(0x1000), ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("65535", ret));
        CPPUNIT_ASSERT_EQUAL(uint16_t(65535), ret);
    }
    {
        // All good conversions
        double ret;
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("50.256", ret));
        CPPUNIT_ASSERT_EQUAL(50.256, ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("-50.256", ret));
        CPPUNIT_ASSERT_EQUAL(-50.256, ret);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType("0", ret));
        CPPUNIT_ASSERT_EQUAL(0.0, ret);
    }
    {
        // All bad conversions
        bool ret;
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("tr", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("fa", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("s1235sd", ret));
    }
    {
        // All bad conversions
        int64_t ret;
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("abc", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("9223372036854775808", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("-9223372036854775809", ret));
    }
    {
        // All bad conversions
        int32_t ret;
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("abc", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("2147483648", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("-2147483649", ret));
    }
    {
        // All bad conversions
        int16_t ret;
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("abc", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("32768", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("-32769", ret));
    }
    {
        // All bad conversions
        uint64_t ret;
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("abc", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("18446744073709551616", ret));
    }
    {
        // All bad conversions
        uint32_t ret;
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("abc", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("4294967296", ret));
    }
    {
        // All bad conversions
        uint16_t ret;
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("abc", ret));
        CPPUNIT_ASSERT(!ml::core::CStringUtils::stringToType("65536", ret));
    }
}

void CStringUtilsTest::testTokeniser() {
    std::string str = "sadcasd csac asdcasdc asdc asdc sadc sadc asd csdc ewwef f sdf sd f sdf  sdfsadfasdf\n"
                      "adscasdcadsc\n"
                      "asdfcasdcadsds<ENDsa wefasdsadc<END>asdcsadcadsc\n"
                      "asdcasdcsdcasdc\n"
                      "sdcsdacsdac\n"
                      "sdcasdacs<END>";

    // Note: the test is done with strtok, which uses ANY ONE character in the
    // delimiter string to split on, so the delimiters for this test have to be
    // one character
    this->testTokeniser(">", str);
    this->testTokeniser("\n", str);
    this->testTokeniser("f", str);
}

void CStringUtilsTest::testTokeniser(const std::string& delim, const std::string& str) {
    // Tokenise using ml
    ml::core::CStringUtils::TStrVec tokens;
    std::string remainder;

    ml::core::CStringUtils::tokenise(delim, str, tokens, remainder);

    LOG_DEBUG(<< str << " DELIM = '" << delim << "' REMAINDER = '" << remainder << "'");

    for (ml::core::CStringUtils::TStrVecItr itr = tokens.begin();
         itr != tokens.end(); ++itr) {
        LOG_DEBUG(<< "'" << *itr << "'");
    }

    // Tokenise using strtok
    char* test = ::strdup(str.c_str());
    CPPUNIT_ASSERT(test);

    ml::core::CStringUtils::TStrVec strtokVec;

    // Note: strtok, uses ANY ONE character in the delimiter string to split on,
    // so the delimiters for this test have to be one character
    char* brk = nullptr;
    for (char* line = ml::core::CStrTokR::strTokR(test, delim.c_str(), &brk); line != nullptr;
         line = ml::core::CStrTokR::strTokR(nullptr, delim.c_str(), &brk)) {
        strtokVec.push_back(line);
        LOG_DEBUG(<< "'" << line << "'");
    }

    free(test);
    test = nullptr;

    if (remainder.empty() == false) {
        tokens.push_back(remainder);
    }

    std::string::size_type pos = str.rfind(delim);
    if (pos != std::string::npos) {
        std::string remainderExpected = str.substr(pos + delim.size());

        CPPUNIT_ASSERT_EQUAL(remainderExpected, remainder);
    }

    // Compare ml to strtok
    CPPUNIT_ASSERT_EQUAL(strtokVec.size(), tokens.size());
    CPPUNIT_ASSERT(strtokVec == tokens);
}

void CStringUtilsTest::testTrim() {
    std::string testStr;

    testStr = "  hello\r\n";
    ml::core::CStringUtils::trimWhitespace(testStr);
    CPPUNIT_ASSERT_EQUAL(std::string("hello"), testStr);

    testStr = "  hello world ";
    ml::core::CStringUtils::trimWhitespace(testStr);
    CPPUNIT_ASSERT_EQUAL(std::string("hello world"), testStr);

    testStr = "\t  hello \t world \t\n";
    ml::core::CStringUtils::trimWhitespace(testStr);
    CPPUNIT_ASSERT_EQUAL(std::string("hello \t world"), testStr);

    testStr = " ";
    ml::core::CStringUtils::trimWhitespace(testStr);
    CPPUNIT_ASSERT_EQUAL(std::string(""), testStr);

    testStr = "\t ";
    ml::core::CStringUtils::trimWhitespace(testStr);
    CPPUNIT_ASSERT_EQUAL(std::string(""), testStr);

    testStr = "\t  hello \t world \t\n";
    ml::core::CStringUtils::trim(" \th", testStr);
    CPPUNIT_ASSERT_EQUAL(std::string("ello \t world \t\n"), testStr);

    testStr = "\t h h \t  \thhh";
    ml::core::CStringUtils::trim(" \th", testStr);
    CPPUNIT_ASSERT_EQUAL(std::string(""), testStr);
}

void CStringUtilsTest::testJoin() {
    LOG_DEBUG(<< "*** testJoin ***")
    using namespace ml;
    using namespace core;
    using TStrVec = std::vector<std::string>;
    using TStrSet = std::set<std::string>;

    TStrVec strVec;

    LOG_DEBUG(<< "Test empty container")
    CPPUNIT_ASSERT_EQUAL(std::string(""), CStringUtils::join(strVec, std::string(",")));

    LOG_DEBUG(<< "Test container has empty strings")
    strVec.push_back(std::string());
    strVec.push_back(std::string());
    CPPUNIT_ASSERT_EQUAL(std::string(","), CStringUtils::join(strVec, std::string(",")));

    LOG_DEBUG(<< "Test container has empty strings and delimiter is also empty")
    CPPUNIT_ASSERT_EQUAL(std::string(""), CStringUtils::join(strVec, std::string("")));

    strVec.clear();

    LOG_DEBUG(<< "Test only one item")
    strVec.push_back(std::string("aaa"));
    CPPUNIT_ASSERT_EQUAL(std::string("aaa"), CStringUtils::join(strVec, std::string(",")));

    LOG_DEBUG(<< "Test three items")
    strVec.push_back(std::string("bbb"));
    strVec.push_back(std::string("ccc"));

    CPPUNIT_ASSERT_EQUAL(std::string("aaa,bbb,ccc"),
                         CStringUtils::join(strVec, std::string(",")));

    LOG_DEBUG(<< "Test delimiter has more than one characters")
    CPPUNIT_ASSERT_EQUAL(std::string("aaa::bbb::ccc"),
                         CStringUtils::join(strVec, std::string("::")));

    LOG_DEBUG(<< "Test set instead of vector")
    TStrSet strSet;
    strSet.insert(std::string("aaa"));
    strSet.insert(std::string("bbb"));
    strSet.insert(std::string("ccc"));
    CPPUNIT_ASSERT_EQUAL(std::string("aaa,bbb,ccc"),
                         CStringUtils::join(strSet, std::string(",")));
}

void CStringUtilsTest::testLower() {
    CPPUNIT_ASSERT_EQUAL(std::string("hello"), ml::core::CStringUtils::toLower("hello"));
    CPPUNIT_ASSERT_EQUAL(std::string("hello"), ml::core::CStringUtils::toLower("Hello"));
    CPPUNIT_ASSERT_EQUAL(std::string("hello"), ml::core::CStringUtils::toLower("HELLO"));

    CPPUNIT_ASSERT_EQUAL(std::string("123hello"),
                         ml::core::CStringUtils::toLower("123hello"));
    CPPUNIT_ASSERT_EQUAL(std::string("hello  "), ml::core::CStringUtils::toLower("Hello  "));
    CPPUNIT_ASSERT_EQUAL(std::string("_-+hello"),
                         ml::core::CStringUtils::toLower("_-+HELLO"));
}

void CStringUtilsTest::testUpper() {
    CPPUNIT_ASSERT_EQUAL(std::string("HELLO"), ml::core::CStringUtils::toUpper("hello"));
    CPPUNIT_ASSERT_EQUAL(std::string("HELLO"), ml::core::CStringUtils::toUpper("Hello"));
    CPPUNIT_ASSERT_EQUAL(std::string("HELLO"), ml::core::CStringUtils::toUpper("HELLO"));

    CPPUNIT_ASSERT_EQUAL(std::string("123HELLO"),
                         ml::core::CStringUtils::toUpper("123hello"));
    CPPUNIT_ASSERT_EQUAL(std::string("HELLO  "), ml::core::CStringUtils::toUpper("Hello  "));
    CPPUNIT_ASSERT_EQUAL(std::string("_-+HELLO"),
                         ml::core::CStringUtils::toUpper("_-+HELLO"));
}

void CStringUtilsTest::testNarrowWiden() {
    std::string hello1("Hello");
    std::wstring hello2(L"Hello");

    CPPUNIT_ASSERT_EQUAL(hello1.length(),
                         ml::core::CStringUtils::narrowToWide(hello1).length());
    CPPUNIT_ASSERT_EQUAL(hello2.length(),
                         ml::core::CStringUtils::wideToNarrow(hello2).length());

    CPPUNIT_ASSERT(ml::core::CStringUtils::narrowToWide(hello1) == hello2);
    CPPUNIT_ASSERT(ml::core::CStringUtils::wideToNarrow(hello2) == hello1);
}

void CStringUtilsTest::testEscape() {
    const std::string toEscape("\"'\\");

    const std::string escaped1("\\\"quoted\\\"");
    std::string unEscaped1("\"quoted\"");

    ml::core::CStringUtils::escape('\\', toEscape, unEscaped1);
    CPPUNIT_ASSERT_EQUAL(escaped1, unEscaped1);

    const std::string escaped2("\\\\\\\"with escaped quotes\\\\\\\"");
    std::string unEscaped2("\\\"with escaped quotes\\\"");

    ml::core::CStringUtils::escape('\\', toEscape, unEscaped2);
    CPPUNIT_ASSERT_EQUAL(escaped2, unEscaped2);
}

void CStringUtilsTest::testUnEscape() {
    std::string escaped1("\\\"quoted\\\"");
    const std::string unEscaped1("\"quoted\"");

    ml::core::CStringUtils::unEscape('\\', escaped1);
    CPPUNIT_ASSERT_EQUAL(unEscaped1, escaped1);

    std::string escaped2("\\\\\\\"with escaped quotes\\\\\\\"");
    const std::string unEscaped2("\\\"with escaped quotes\\\"");

    ml::core::CStringUtils::unEscape('\\', escaped2);
    CPPUNIT_ASSERT_EQUAL(unEscaped2, escaped2);

    // This should print a warning about the last character being an escape
    std::string dodgy("\\\"dodgy\\");
    ml::core::CStringUtils::unEscape('\\', dodgy);
}

void CStringUtilsTest::testLongestSubstr() {
    {
        std::string str1;
        std::string str2;

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2;

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2("Hello mum");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string("Hello "), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2("Say hello");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string("ello"), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("abc");
        std::string str2("def");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("abc xyz defgv hij");
        std::string str2("abc w defgtu hij");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string(" defg"), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Source LOTS on 13080:742 has shut down.");
        std::string str2("Source INTERN_IPT on 13080:2260 has shut down.");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string(" has shut down."), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("No filter list defined for .");
        std::string str2("No filter list defined for cube_int.");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string("No filter list defined for "), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
}

void CStringUtilsTest::testLongestSubseq() {
    {
        std::string str1;
        std::string str2;

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2;

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2("Hello mum");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string("Hello "), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2("Say hello");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string("ello"), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("abc");
        std::string str2("def");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("abc xyz defgv hij");
        std::string str2("abc w defgtu hij");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string("abc  defg hij"), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Source LOTS on 13080:742 has shut down.");
        std::string str2("Source INTERN_IPT on 13080:2260 has shut down.");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string("Source T on 13080:2 has shut down."), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("No filter list defined for .");
        std::string str2("No filter list defined for cube_int.");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        CPPUNIT_ASSERT_EQUAL(std::string("No filter list defined for ."), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
}

void CStringUtilsTest::testNormaliseWhitespace() {
    std::string spacey(" what\ta   lot \tof\n"
                       "spaces");
    std::string normalised(" what a lot of spaces");

    CPPUNIT_ASSERT_EQUAL(normalised, ml::core::CStringUtils::normaliseWhitespace(spacey));
}

void CStringUtilsTest::testPerformance() {
    static const size_t TEST_SIZE(1000000);
    static const double TEST_SIZE_D(static_cast<double>(TEST_SIZE));

    ml::core::CStopWatch stopWatch;

    {
        LOG_DEBUG(<< "Before CStringUtils::typeToString integer test");

        stopWatch.start();
        for (size_t count = 0; count < TEST_SIZE; ++count) {
            std::string result(ml::core::CStringUtils::typeToString(count));
            ml::core::CStringUtils::stringToType(result, count);
        }
        uint64_t timeMs(stopWatch.stop());
        LOG_DEBUG(<< "After CStringUtils::typeToString integer test");
        LOG_DEBUG(<< "CStringUtils::typeToString integer test took " << timeMs << "ms");
    }

    stopWatch.reset();

    {
        LOG_DEBUG(<< "Before boost::lexical_cast integer test");
        stopWatch.start();
        for (size_t count = 0; count < TEST_SIZE; ++count) {
            std::string result(boost::lexical_cast<std::string>(count));
            count = boost::lexical_cast<size_t>(result);
        }
        uint64_t timeMs(stopWatch.stop());
        LOG_DEBUG(<< "After boost::lexical_cast integer test");
        LOG_DEBUG(<< "boost::lexical_cast integer test took " << timeMs << "ms");
    }

    stopWatch.reset();

    {
        LOG_DEBUG(<< "Before CStringUtils::typeToString floating point test");

        stopWatch.start();
        for (double count = 0.0; count < TEST_SIZE_D; count += 1.41) {
            std::string result(ml::core::CStringUtils::typeToString(count));
            ml::core::CStringUtils::stringToType(result, count);
        }
        uint64_t timeMs(stopWatch.stop());
        LOG_DEBUG(<< "After CStringUtils::typeToString floating point test");
        LOG_DEBUG(<< "CStringUtils::typeToString floating point test took " << timeMs << "ms");
    }

    stopWatch.reset();

    {
        LOG_DEBUG(<< "Before boost::lexical_cast floating point test");
        stopWatch.start();
        for (double count = 0.0; count < TEST_SIZE_D; count += 1.41) {
            std::string result(boost::lexical_cast<std::string>(count));
            count = boost::lexical_cast<double>(result);
        }
        uint64_t timeMs(stopWatch.stop());
        LOG_DEBUG(<< "After boost::lexical_cast floating point test");
        LOG_DEBUG(<< "boost::lexical_cast floating point test took " << timeMs << "ms");
    }
}

void CStringUtilsTest::testUtf8ByteType() {
    std::string testStr;
    // single byte UTF-8 character
    testStr += "a";
    // two byte UTF-8 character
    testStr += "é";
    // three byte UTF-8 character
    testStr += "中";
    // four byte UTF-8 character
    testStr += "𩸽";
    CPPUNIT_ASSERT_EQUAL(size_t(10), testStr.length());
    CPPUNIT_ASSERT_EQUAL(1, ml::core::CStringUtils::utf8ByteType(testStr[0]));
    CPPUNIT_ASSERT_EQUAL(2, ml::core::CStringUtils::utf8ByteType(testStr[1]));
    CPPUNIT_ASSERT_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[2]));
    CPPUNIT_ASSERT_EQUAL(3, ml::core::CStringUtils::utf8ByteType(testStr[3]));
    CPPUNIT_ASSERT_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[4]));
    CPPUNIT_ASSERT_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[5]));
    CPPUNIT_ASSERT_EQUAL(4, ml::core::CStringUtils::utf8ByteType(testStr[6]));
    CPPUNIT_ASSERT_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[7]));
    CPPUNIT_ASSERT_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[8]));
    CPPUNIT_ASSERT_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[9]));
}

void CStringUtilsTest::testRoundtripMaxDouble() {
    ml::core::CIEEE754::EPrecision precisions[] = {
        ml::core::CIEEE754::E_SinglePrecision, ml::core::CIEEE754::E_DoublePrecision};
    double tolerances[] = {5e-7, 5e-15};
    for (std::size_t i = 0u; i < boost::size(precisions); ++i) {
        double max = std::numeric_limits<double>::max();
        std::string str = ml::core::CStringUtils::typeToStringPrecise(max, precisions[i]);
        double d = 0.0;
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(str, d));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(max, d, tolerances[i] * max);
    }
    for (std::size_t i = 0u; i < boost::size(precisions); ++i) {
        double min = -std::numeric_limits<double>::max();
        std::string str = ml::core::CStringUtils::typeToStringPrecise(min, precisions[i]);
        double d = 0.0;
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(str, d));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(min, d, -tolerances[i] * min);
    }
}
