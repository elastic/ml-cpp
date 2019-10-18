/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonStateRestoreTraverser.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CJsonStateRestoreTraverserTest)

namespace {

bool traverse2ndLevel(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_CHECK_EQUAL(std::string("level2A"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("3.14"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level2B"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("z"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(!traverser.next());

    return true;
}

bool traverse1stLevel1(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_CHECK_EQUAL(std::string("level1A"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("a"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1B"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("25"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1C"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse2ndLevel));
    BOOST_TEST(!traverser.next());

    return true;
}

bool traverse1stLevel2(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_CHECK_EQUAL(std::string("level1A"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("a"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1B"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("25"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1C"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse2ndLevel));
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1D"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("afterAscending"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(!traverser.next());

    return true;
}

bool traverse2ndLevelEmpty(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_TEST(traverser.name().empty());
    BOOST_TEST(traverser.value().empty());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(!traverser.next());

    return true;
}

bool traverse1stLevel3(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_CHECK_EQUAL(std::string("level1A"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("a"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1B"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("25"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1C"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse2ndLevelEmpty));
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1D"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("afterAscending"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(!traverser.next());

    return true;
}

bool traverse1stLevel4(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_CHECK_EQUAL(std::string("level1A"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("a"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1B"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("25"), traverser.value());
    BOOST_TEST(!traverser.hasSubLevel());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("level1C"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    // For this test we ignore the contents of the sub-level
    BOOST_TEST(!traverser.next());

    return true;
}
}

BOOST_AUTO_TEST_CASE(testRestore1) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_CHECK_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse1stLevel1));
    BOOST_TEST(!traverser.next());
}

BOOST_AUTO_TEST_CASE(testRestore2) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"},\"level1D\":"
                     "\"afterAscending\"}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_CHECK_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse1stLevel2));
    BOOST_TEST(!traverser.next());
}

BOOST_AUTO_TEST_CASE(testRestore3) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{},\"level1D\":\"afterAscending\"}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_CHECK_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse1stLevel3));
    BOOST_TEST(!traverser.next());
}

BOOST_AUTO_TEST_CASE(testRestore4) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_CHECK_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse1stLevel4));
    BOOST_TEST(!traverser.next());
}

BOOST_AUTO_TEST_CASE(testParsingBooleanFields) {
    // Even though the parser doesn't handle boolean fields it should not hiccup over them
    std::string json =
        std::string("{\"_index\" : \"categorization-test\", \"_type\" : \"categorizerState\",") +
        std::string("\"_id\" : \"1\",  \"_version\" : 2, \"found\" : true, ") +
        std::string("\"_source\":{\"a\" :\"1\"}");

    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_CHECK_EQUAL(std::string("_index"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("categorization-test"), traverser.value());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("_type"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("categorizerState"), traverser.value());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("_id"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("1"), traverser.value());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("_version"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("2"), traverser.value());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("found"), traverser.name());
    BOOST_CHECK_EQUAL(std::string("true"), traverser.value());
    BOOST_TEST(traverser.next());
    BOOST_CHECK_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
}

BOOST_AUTO_TEST_CASE(testRestore1IgnoreArrays) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"someArray\":[42],\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_CHECK_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse1stLevel1));
    BOOST_TEST(!traverser.next());
}

BOOST_AUTO_TEST_CASE(testRestore1IgnoreArraysNested) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"someArray\":[{\"nestedArray\":[42]}],\"level1B\":\"25\",\"level1C\":{\"level2A\":"
                     "\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_CHECK_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST(traverser.hasSubLevel());
    BOOST_TEST(traverser.traverseSubLevel(&traverse1stLevel1));
    BOOST_TEST(!traverser.next());
}

BOOST_AUTO_TEST_SUITE_END()
