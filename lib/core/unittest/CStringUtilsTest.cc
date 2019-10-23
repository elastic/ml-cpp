/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CStopWatch.h>
#include <core/CStrTokR.h>
#include <core/CStringUtils.h>

#include <test/BoostTestCloseAbsolute.h>

#include <boost/lexical_cast.hpp>
#include <boost/test/unit_test.hpp>

#include <set>
#include <vector>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::wstring)

BOOST_AUTO_TEST_SUITE(CStringUtilsTest)

void testTokeniserHelper(const std::string& delim, const std::string& str) {
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
    BOOST_TEST_REQUIRE(test);

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

        BOOST_REQUIRE_EQUAL(remainderExpected, remainder);
    }

    // Compare ml to strtok
    BOOST_REQUIRE_EQUAL(strtokVec.size(), tokens.size());
    BOOST_TEST_REQUIRE(strtokVec == tokens);
}

BOOST_AUTO_TEST_CASE(testNumMatches) {
    {
        std::string str("%d %M %Y %f %D  %t");

        BOOST_REQUIRE_EQUAL(size_t(6), ml::core::CStringUtils::numMatches(str, "%"));
        BOOST_REQUIRE_EQUAL(size_t(0), ml::core::CStringUtils::numMatches(str, "q"));
    }
}

BOOST_AUTO_TEST_CASE(testReplace) {
    {
        std::string in("%d%M%Y%f%D%t");
        const std::string out(" %d %M %Y %f %D %t");

        BOOST_REQUIRE_EQUAL(size_t(6), ml::core::CStringUtils::replace("%", " %", in));

        BOOST_REQUIRE_EQUAL(out, in);
    }
    {
        std::string in("%d%M%Y%f%D%t");
        const std::string out("%d%M%Y%f%D%t");

        BOOST_REQUIRE_EQUAL(size_t(0), ml::core::CStringUtils::replace("X", "Y", in));

        BOOST_REQUIRE_EQUAL(out, in);
    }
}

BOOST_AUTO_TEST_CASE(testReplaceFirst) {
    {
        std::string in("%d%M%Y%f%D%t");
        const std::string out(" %d%M%Y%f%D%t");

        BOOST_REQUIRE_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst("%", " %", in));

        BOOST_REQUIRE_EQUAL(out, in);
    }
    {
        std::string in("%d%M%Y%f%D%t");
        const std::string out("%d%M%Y%f%D%t");

        BOOST_REQUIRE_EQUAL(size_t(0), ml::core::CStringUtils::replaceFirst("X", "Y", in));

        BOOST_REQUIRE_EQUAL(out, in);
    }
}

BOOST_AUTO_TEST_CASE(testTypeToString) {
    {
        uint64_t i(18446744073709551615ULL);
        std::string expected("18446744073709551615");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        BOOST_REQUIRE_EQUAL(expected, actual);

        uint64_t j(0);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(actual, j));
        BOOST_REQUIRE_EQUAL(i, j);
    }
    {
        uint32_t i(123456U);
        std::string expected("123456");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        BOOST_REQUIRE_EQUAL(expected, actual);

        uint32_t j(0);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(actual, j));
        BOOST_REQUIRE_EQUAL(i, j);
    }
    {
        uint16_t i(12345U);
        std::string expected("12345");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        BOOST_REQUIRE_EQUAL(expected, actual);

        uint16_t j(0);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(actual, j));
        BOOST_REQUIRE_EQUAL(i, j);
    }
    {
        int32_t i(123456);
        std::string expected("123456");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        BOOST_REQUIRE_EQUAL(expected, actual);

        int32_t j(0);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(actual, j));
        BOOST_REQUIRE_EQUAL(i, j);
    }
    {
        double i(0.123456);
        std::string expected("0.123456");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.123456e10);
        std::string expected("1234560000.000000");

        std::string actual = ml::core::CStringUtils::typeToString(i);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
}

BOOST_AUTO_TEST_CASE(testTypeToStringPrecise) {
    {
        double i(1.0);
        std::string expected("1");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(1.0);
        std::string expected("1");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.123456);
        std::string expected("1.23456e-1");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.123456);
        std::string expected("1.23456e-1");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.123456e10);
        std::string expected("1.23456e9");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.123456e10);
        std::string expected("1234560000");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.123456e-10);
        std::string expected("1.23456e-11");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.123456e-10);
        std::string expected("1.23456e-11");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.123456787654321e-10);
        std::string expected("1.234568e-11");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.123456787654321e-10);
        std::string expected("1.23456787654321e-11");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.00000000012345678765432123456);
        std::string expected("1.234568e-10");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(0.00000000012345678765432123456);
        std::string expected("1.23456787654321e-10");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(123456787654321.23456);
        std::string expected("1.234568e14");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_SinglePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
    {
        double i(123456787654321.23456);
        std::string expected("123456787654321");

        std::string actual = ml::core::CStringUtils::typeToStringPrecise(
            i, ml::core::CIEEE754::E_DoublePrecision);
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
}

BOOST_AUTO_TEST_CASE(testTypeToStringPretty) {
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

BOOST_AUTO_TEST_CASE(testStringToType) {
    {
        // All good conversions
        bool ret;
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("yes", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("no", ret));
        BOOST_TEST_REQUIRE(!ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("yES", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("NO", ret));
        BOOST_TEST_REQUIRE(!ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("true", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("false", ret));
        BOOST_TEST_REQUIRE(!ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("TRUE", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("False", ret));
        BOOST_TEST_REQUIRE(!ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("on", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("off", ret));
        BOOST_TEST_REQUIRE(!ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("On", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("OFF", ret));
        BOOST_TEST_REQUIRE(!ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("y", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("n", ret));
        BOOST_TEST_REQUIRE(!ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("Y", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("N", ret));
        BOOST_TEST_REQUIRE(!ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("t", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("f", ret));
        BOOST_TEST_REQUIRE(!ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("T", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("F", ret));
        BOOST_TEST_REQUIRE(!ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("1", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("10", ret));
        BOOST_TEST_REQUIRE(ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("0", ret));
        BOOST_TEST_REQUIRE(!ret);
    }
    {
        // All good conversions
        int32_t ret;
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("1000", ret));
        BOOST_REQUIRE_EQUAL(int32_t(1000), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("-1000", ret));
        BOOST_REQUIRE_EQUAL(int32_t(-1000), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("0", ret));
        BOOST_REQUIRE_EQUAL(int32_t(0), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("0x1000", ret));
        BOOST_REQUIRE_EQUAL(int32_t(0x1000), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("2147483647", ret));
        BOOST_REQUIRE_EQUAL(int32_t(2147483647), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("-2147483647", ret));
        BOOST_REQUIRE_EQUAL(int32_t(-2147483647), ret);
    }
    {
        // All good conversions
        uint64_t ret;
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("1000", ret));
        BOOST_REQUIRE_EQUAL(uint64_t(1000), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("0", ret));
        BOOST_REQUIRE_EQUAL(uint64_t(0), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("0x1000", ret));
        BOOST_REQUIRE_EQUAL(uint64_t(0x1000), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("18446744073709551615", ret));
        BOOST_REQUIRE_EQUAL(uint64_t(18446744073709551615ULL), ret);
    }
    {
        // All good conversions
        uint32_t ret;
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("1000", ret));
        BOOST_REQUIRE_EQUAL(uint32_t(1000), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("0", ret));
        BOOST_REQUIRE_EQUAL(uint32_t(0), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("0x1000", ret));
        BOOST_REQUIRE_EQUAL(uint32_t(0x1000), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("2147483650", ret));
        BOOST_REQUIRE_EQUAL(uint32_t(2147483650UL), ret);
    }
    {
        // All good conversions
        uint16_t ret;
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("1000", ret));
        BOOST_REQUIRE_EQUAL(uint16_t(1000), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("0", ret));
        BOOST_REQUIRE_EQUAL(uint16_t(0), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("0x1000", ret));
        BOOST_REQUIRE_EQUAL(uint16_t(0x1000), ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("65535", ret));
        BOOST_REQUIRE_EQUAL(uint16_t(65535), ret);
    }
    {
        // All good conversions
        double ret;
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("50.256", ret));
        BOOST_REQUIRE_EQUAL(50.256, ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("-50.256", ret));
        BOOST_REQUIRE_EQUAL(-50.256, ret);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType("0", ret));
        BOOST_REQUIRE_EQUAL(0.0, ret);
    }
    {
        // All bad conversions
        bool ret;
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("tr", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("fa", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("s1235sd", ret));
    }
    {
        // All bad conversions
        int64_t ret;
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("abc", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("9223372036854775808", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("-9223372036854775809", ret));
    }
    {
        // All bad conversions
        int32_t ret;
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("abc", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("2147483648", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("-2147483649", ret));
    }
    {
        // All bad conversions
        int16_t ret;
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("abc", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("32768", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("-32769", ret));
    }
    {
        // All bad conversions
        uint64_t ret;
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("abc", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("18446744073709551616", ret));
    }
    {
        // All bad conversions
        uint32_t ret;
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("abc", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("4294967296", ret));
    }
    {
        // All bad conversions
        uint16_t ret;
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("abc", ret));
        BOOST_TEST_REQUIRE(!ml::core::CStringUtils::stringToType("65536", ret));
    }
}

BOOST_AUTO_TEST_CASE(testTokeniser) {
    std::string str = "sadcasd csac asdcasdc asdc asdc sadc sadc asd csdc ewwef f sdf sd f sdf  sdfsadfasdf\n"
                      "adscasdcadsc\n"
                      "asdfcasdcadsds<ENDsa wefasdsadc<END>asdcsadcadsc\n"
                      "asdcasdcsdcasdc\n"
                      "sdcsdacsdac\n"
                      "sdcasdacs<END>";

    // Note: the test is done with strtok, which uses ANY ONE character in the
    // delimiter string to split on, so the delimiters for this test have to be
    // one character
    testTokeniserHelper(">", str);
    testTokeniserHelper("\n", str);
    testTokeniserHelper("f", str);
}

BOOST_AUTO_TEST_CASE(testTrim) {
    std::string testStr;

    testStr = "  hello\r\n";
    ml::core::CStringUtils::trimWhitespace(testStr);
    BOOST_REQUIRE_EQUAL(std::string("hello"), testStr);

    testStr = "  hello world ";
    ml::core::CStringUtils::trimWhitespace(testStr);
    BOOST_REQUIRE_EQUAL(std::string("hello world"), testStr);

    testStr = "\t  hello \t world \t\n";
    ml::core::CStringUtils::trimWhitespace(testStr);
    BOOST_REQUIRE_EQUAL(std::string("hello \t world"), testStr);

    testStr = " ";
    ml::core::CStringUtils::trimWhitespace(testStr);
    BOOST_REQUIRE_EQUAL(std::string(""), testStr);

    testStr = "\t ";
    ml::core::CStringUtils::trimWhitespace(testStr);
    BOOST_REQUIRE_EQUAL(std::string(""), testStr);

    testStr = "\t  hello \t world \t\n";
    ml::core::CStringUtils::trim(" \th", testStr);
    BOOST_REQUIRE_EQUAL(std::string("ello \t world \t\n"), testStr);

    testStr = "\t h h \t  \thhh";
    ml::core::CStringUtils::trim(" \th", testStr);
    BOOST_REQUIRE_EQUAL(std::string(""), testStr);
}

BOOST_AUTO_TEST_CASE(testJoin) {
    using namespace ml;
    using namespace core;
    using TStrVec = std::vector<std::string>;
    using TStrSet = std::set<std::string>;

    TStrVec strVec;

    LOG_DEBUG(<< "Test empty container");
    BOOST_REQUIRE_EQUAL(std::string(""), CStringUtils::join(strVec, std::string(",")));

    LOG_DEBUG(<< "Test container has empty strings");
    strVec.push_back(std::string());
    strVec.push_back(std::string());
    BOOST_REQUIRE_EQUAL(std::string(","), CStringUtils::join(strVec, std::string(",")));

    LOG_DEBUG(<< "Test container has empty strings and delimiter is also empty");
    BOOST_REQUIRE_EQUAL(std::string(""), CStringUtils::join(strVec, std::string("")));

    strVec.clear();

    LOG_DEBUG(<< "Test only one item");
    strVec.push_back(std::string("aaa"));
    BOOST_REQUIRE_EQUAL(std::string("aaa"), CStringUtils::join(strVec, std::string(",")));

    LOG_DEBUG(<< "Test three items");
    strVec.push_back(std::string("bbb"));
    strVec.push_back(std::string("ccc"));

    BOOST_REQUIRE_EQUAL(std::string("aaa,bbb,ccc"),
                      CStringUtils::join(strVec, std::string(",")));

    LOG_DEBUG(<< "Test delimiter has more than one characters");
    BOOST_REQUIRE_EQUAL(std::string("aaa::bbb::ccc"),
                      CStringUtils::join(strVec, std::string("::")));

    LOG_DEBUG(<< "Test set instead of vector");
    TStrSet strSet;
    strSet.insert(std::string("aaa"));
    strSet.insert(std::string("bbb"));
    strSet.insert(std::string("ccc"));
    BOOST_REQUIRE_EQUAL(std::string("aaa,bbb,ccc"),
                      CStringUtils::join(strSet, std::string(",")));
}

BOOST_AUTO_TEST_CASE(testLower) {
    BOOST_REQUIRE_EQUAL(std::string("hello"), ml::core::CStringUtils::toLower("hello"));
    BOOST_REQUIRE_EQUAL(std::string("hello"), ml::core::CStringUtils::toLower("Hello"));
    BOOST_REQUIRE_EQUAL(std::string("hello"), ml::core::CStringUtils::toLower("HELLO"));

    BOOST_REQUIRE_EQUAL(std::string("123hello"), ml::core::CStringUtils::toLower("123hello"));
    BOOST_REQUIRE_EQUAL(std::string("hello  "), ml::core::CStringUtils::toLower("Hello  "));
    BOOST_REQUIRE_EQUAL(std::string("_-+hello"), ml::core::CStringUtils::toLower("_-+HELLO"));
}

BOOST_AUTO_TEST_CASE(testUpper) {
    BOOST_REQUIRE_EQUAL(std::string("HELLO"), ml::core::CStringUtils::toUpper("hello"));
    BOOST_REQUIRE_EQUAL(std::string("HELLO"), ml::core::CStringUtils::toUpper("Hello"));
    BOOST_REQUIRE_EQUAL(std::string("HELLO"), ml::core::CStringUtils::toUpper("HELLO"));

    BOOST_REQUIRE_EQUAL(std::string("123HELLO"), ml::core::CStringUtils::toUpper("123hello"));
    BOOST_REQUIRE_EQUAL(std::string("HELLO  "), ml::core::CStringUtils::toUpper("Hello  "));
    BOOST_REQUIRE_EQUAL(std::string("_-+HELLO"), ml::core::CStringUtils::toUpper("_-+HELLO"));
}

BOOST_AUTO_TEST_CASE(testNarrowWiden) {
    std::string hello1("Hello");
    std::wstring hello2(L"Hello");

    BOOST_REQUIRE_EQUAL(hello1.length(),
                      ml::core::CStringUtils::narrowToWide(hello1).length());
    BOOST_REQUIRE_EQUAL(hello2.length(),
                      ml::core::CStringUtils::wideToNarrow(hello2).length());

    BOOST_TEST_REQUIRE(ml::core::CStringUtils::narrowToWide(hello1) == hello2);
    BOOST_TEST_REQUIRE(ml::core::CStringUtils::wideToNarrow(hello2) == hello1);
}

BOOST_AUTO_TEST_CASE(testEscape) {
    const std::string toEscape("\"'\\");

    const std::string escaped1("\\\"quoted\\\"");
    std::string unEscaped1("\"quoted\"");

    ml::core::CStringUtils::escape('\\', toEscape, unEscaped1);
    BOOST_REQUIRE_EQUAL(escaped1, unEscaped1);

    const std::string escaped2("\\\\\\\"with escaped quotes\\\\\\\"");
    std::string unEscaped2("\\\"with escaped quotes\\\"");

    ml::core::CStringUtils::escape('\\', toEscape, unEscaped2);
    BOOST_REQUIRE_EQUAL(escaped2, unEscaped2);
}

BOOST_AUTO_TEST_CASE(testUnEscape) {
    std::string escaped1("\\\"quoted\\\"");
    const std::string unEscaped1("\"quoted\"");

    ml::core::CStringUtils::unEscape('\\', escaped1);
    BOOST_REQUIRE_EQUAL(unEscaped1, escaped1);

    std::string escaped2("\\\\\\\"with escaped quotes\\\\\\\"");
    const std::string unEscaped2("\\\"with escaped quotes\\\"");

    ml::core::CStringUtils::unEscape('\\', escaped2);
    BOOST_REQUIRE_EQUAL(unEscaped2, escaped2);

    // This should print a warning about the last character being an escape
    std::string dodgy("\\\"dodgy\\");
    ml::core::CStringUtils::unEscape('\\', dodgy);
}

BOOST_AUTO_TEST_CASE(testLongestSubstr) {
    {
        std::string str1;
        std::string str2;

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2;

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2("Hello mum");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string("Hello "), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2("Say hello");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string("ello"), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("abc");
        std::string str2("def");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("abc xyz defgv hij");
        std::string str2("abc w defgtu hij");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string(" defg"), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Source LOTS on 13080:742 has shut down.");
        std::string str2("Source INTERN_IPT on 13080:2260 has shut down.");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string(" has shut down."), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("No filter list defined for .");
        std::string str2("No filter list defined for cube_int.");

        std::string common(ml::core::CStringUtils::longestCommonSubstr(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string("No filter list defined for "), common);

        LOG_DEBUG(<< "Longest common substring of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
}

BOOST_AUTO_TEST_CASE(testLongestSubseq) {
    {
        std::string str1;
        std::string str2;

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2;

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2("Hello mum");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string("Hello "), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Hello world");
        std::string str2("Say hello");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string("ello"), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("abc");
        std::string str2("def");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string(""), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("abc xyz defgv hij");
        std::string str2("abc w defgtu hij");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string("abc  defg hij"), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("Source LOTS on 13080:742 has shut down.");
        std::string str2("Source INTERN_IPT on 13080:2260 has shut down.");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string("Source T on 13080:2 has shut down."), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
    {
        std::string str1("No filter list defined for .");
        std::string str2("No filter list defined for cube_int.");

        std::string common(ml::core::CStringUtils::longestCommonSubsequence(str1, str2));

        BOOST_REQUIRE_EQUAL(std::string("No filter list defined for ."), common);

        LOG_DEBUG(<< "Longest common subsequence of '" << str1 << "' and '"
                  << str2 << "' is '" << common << "'");
    }
}

BOOST_AUTO_TEST_CASE(testNormaliseWhitespace) {
    std::string spacey(" what\ta   lot \tof\n"
                       "spaces");
    std::string normalised(" what a lot of spaces");

    BOOST_REQUIRE_EQUAL(normalised, ml::core::CStringUtils::normaliseWhitespace(spacey));
}

BOOST_AUTO_TEST_CASE(testPerformance) {
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

BOOST_AUTO_TEST_CASE(testUtf8ByteType) {
    std::string testStr;
    // single byte UTF-8 character
    testStr += "a";
    // two byte UTF-8 character
    testStr += "é";
    // three byte UTF-8 character
    testStr += "中";
    // four byte UTF-8 character
    testStr += "𩸽";
    BOOST_REQUIRE_EQUAL(size_t(10), testStr.length());
    BOOST_REQUIRE_EQUAL(1, ml::core::CStringUtils::utf8ByteType(testStr[0]));
    BOOST_REQUIRE_EQUAL(2, ml::core::CStringUtils::utf8ByteType(testStr[1]));
    BOOST_REQUIRE_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[2]));
    BOOST_REQUIRE_EQUAL(3, ml::core::CStringUtils::utf8ByteType(testStr[3]));
    BOOST_REQUIRE_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[4]));
    BOOST_REQUIRE_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[5]));
    BOOST_REQUIRE_EQUAL(4, ml::core::CStringUtils::utf8ByteType(testStr[6]));
    BOOST_REQUIRE_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[7]));
    BOOST_REQUIRE_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[8]));
    BOOST_REQUIRE_EQUAL(-1, ml::core::CStringUtils::utf8ByteType(testStr[9]));
}

BOOST_AUTO_TEST_CASE(testRoundtripMaxDouble) {
    ml::core::CIEEE754::EPrecision precisions[] = {
        ml::core::CIEEE754::E_SinglePrecision, ml::core::CIEEE754::E_DoublePrecision};
    double tolerances[] = {5e-7, 5e-15};
    for (std::size_t i = 0u; i < boost::size(precisions); ++i) {
        double max = std::numeric_limits<double>::max();
        std::string str = ml::core::CStringUtils::typeToStringPrecise(max, precisions[i]);
        double d = 0.0;
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(str, d));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(max, d, tolerances[i] * max);
    }
    for (std::size_t i = 0u; i < boost::size(precisions); ++i) {
        double min = -std::numeric_limits<double>::max();
        std::string str = ml::core::CStringUtils::typeToStringPrecise(min, precisions[i]);
        double d = 0.0;
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(str, d));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(min, d, -tolerances[i] * min);
    }
}

BOOST_AUTO_TEST_SUITE_END()
