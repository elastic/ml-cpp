/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CCsvLineParser.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CCsvLineParserTest)

BOOST_AUTO_TEST_CASE(testLineParser) {
    ml::core::CCsvLineParser lineParser;
    std::string token;

    {
        std::string simple{"a,b,c"};
        lineParser.reset(simple);

        BOOST_TEST_REQUIRE(!lineParser.atEnd());
        BOOST_TEST_REQUIRE(lineParser.parseNext(token));
        BOOST_REQUIRE_EQUAL(std::string("a"), token);

        BOOST_TEST_REQUIRE(!lineParser.atEnd());
        BOOST_TEST_REQUIRE(lineParser.parseNext(token));
        BOOST_REQUIRE_EQUAL(std::string("b"), token);

        BOOST_TEST_REQUIRE(!lineParser.atEnd());
        BOOST_TEST_REQUIRE(lineParser.parseNext(token));
        BOOST_REQUIRE_EQUAL(std::string("c"), token);

        BOOST_TEST_REQUIRE(lineParser.atEnd());
        BOOST_TEST_REQUIRE(!lineParser.parseNext(token));
    }
    {
        std::string quoted{"\"a,b,c\",b and some spaces,\"c quoted unecessarily\",\"d with a literal \"\"\""};
        lineParser.reset(quoted);

        BOOST_TEST_REQUIRE(!lineParser.atEnd());
        BOOST_TEST_REQUIRE(lineParser.parseNext(token));
        BOOST_REQUIRE_EQUAL(std::string("a,b,c"), token);

        BOOST_TEST_REQUIRE(!lineParser.atEnd());
        BOOST_TEST_REQUIRE(lineParser.parseNext(token));
        BOOST_REQUIRE_EQUAL(std::string("b and some spaces"), token);

        BOOST_TEST_REQUIRE(!lineParser.atEnd());
        BOOST_TEST_REQUIRE(lineParser.parseNext(token));
        BOOST_REQUIRE_EQUAL(std::string("c quoted unecessarily"), token);

        BOOST_TEST_REQUIRE(!lineParser.atEnd());
        BOOST_TEST_REQUIRE(lineParser.parseNext(token));
        BOOST_REQUIRE_EQUAL(std::string("d with a literal \""), token);

        BOOST_TEST_REQUIRE(lineParser.atEnd());
        BOOST_TEST_REQUIRE(!lineParser.parseNext(token));
    }
    {
        std::string cjk{"编码,コーディング,코딩"};
        lineParser.reset(cjk);

        BOOST_TEST_REQUIRE(!lineParser.atEnd());
        BOOST_TEST_REQUIRE(lineParser.parseNext(token));
        BOOST_REQUIRE_EQUAL(std::string("编码"), token);

        BOOST_TEST_REQUIRE(!lineParser.atEnd());
        BOOST_TEST_REQUIRE(lineParser.parseNext(token));
        BOOST_REQUIRE_EQUAL(std::string("コーディング"), token);

        BOOST_TEST_REQUIRE(!lineParser.atEnd());
        BOOST_TEST_REQUIRE(lineParser.parseNext(token));
        BOOST_REQUIRE_EQUAL(std::string("코딩"), token);

        BOOST_TEST_REQUIRE(lineParser.atEnd());
        BOOST_TEST_REQUIRE(!lineParser.parseNext(token));
    }
}

BOOST_AUTO_TEST_SUITE_END()
