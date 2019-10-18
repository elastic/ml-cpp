/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CTokenListReverseSearchCreator.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CTokenListReverseSearchCreatorTest)

using namespace ml;
using namespace api;


BOOST_AUTO_TEST_CASE(testCostOfToken) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");
    BOOST_CHECK_EQUAL(std::size_t(110),
                         reverseSearchCreator.costOfToken("someToken", 5));
}

BOOST_AUTO_TEST_CASE(testCreateNullSearch) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    BOOST_TEST(reverseSearchCreator.createNullSearch(reverseSearchPart1, reverseSearchPart2));

    BOOST_CHECK_EQUAL(std::string(""), reverseSearchPart1);
    BOOST_CHECK_EQUAL(std::string(""), reverseSearchPart2);
}

BOOST_AUTO_TEST_CASE(testCreateNoUniqueTokenSearch) {
    CTokenListReverseSearchCreator reverseSearchCreator("status");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    BOOST_TEST(reverseSearchCreator.createNoUniqueTokenSearch(
        1, "404", 4, reverseSearchPart1, reverseSearchPart2));

    BOOST_CHECK_EQUAL(std::string(""), reverseSearchPart1);
    BOOST_CHECK_EQUAL(std::string(""), reverseSearchPart2);
}

BOOST_AUTO_TEST_CASE(testInitStandardSearch) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    reverseSearchCreator.initStandardSearch(1, "User 'foo' logged in host '0.0.0.0'",
                                            1, reverseSearchPart1, reverseSearchPart2);

    BOOST_CHECK_EQUAL(std::string(""), reverseSearchPart1);
    BOOST_CHECK_EQUAL(std::string(""), reverseSearchPart2);
}

BOOST_AUTO_TEST_CASE(testAddCommonUniqueToken) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    reverseSearchCreator.addCommonUniqueToken("user", reverseSearchPart1, reverseSearchPart2);
    reverseSearchCreator.addCommonUniqueToken("logged", reverseSearchPart1, reverseSearchPart2);

    BOOST_CHECK_EQUAL(std::string(""), reverseSearchPart1);
    BOOST_CHECK_EQUAL(std::string(""), reverseSearchPart2);
}

BOOST_AUTO_TEST_CASE(testAddInOrderCommonToken) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    reverseSearchCreator.addInOrderCommonToken("user", true, reverseSearchPart1,
                                               reverseSearchPart2);
    reverseSearchCreator.addInOrderCommonToken("logged", false, reverseSearchPart1,
                                               reverseSearchPart2);
    reverseSearchCreator.addInOrderCommonToken("b=0.15+a", false, reverseSearchPart1,
                                               reverseSearchPart2);
    reverseSearchCreator.addInOrderCommonToken("logged", false, reverseSearchPart1,
                                               reverseSearchPart2);

    BOOST_CHECK_EQUAL(std::string("user logged b=0.15+a logged"), reverseSearchPart1);
    BOOST_CHECK_EQUAL(std::string(".*?user.+?logged.+?b=0\\.15\\+a.+?logged"),
                         reverseSearchPart2);
}

BOOST_AUTO_TEST_CASE(testCloseStandardSearch) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    reverseSearchCreator.closeStandardSearch(reverseSearchPart1, reverseSearchPart2);

    BOOST_CHECK_EQUAL(std::string(""), reverseSearchPart1);
    BOOST_CHECK_EQUAL(std::string(".*"), reverseSearchPart2);
}

BOOST_AUTO_TEST_SUITE_END()
