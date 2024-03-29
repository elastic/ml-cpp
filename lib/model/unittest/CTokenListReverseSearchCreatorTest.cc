/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <model/CTokenListReverseSearchCreator.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CTokenListReverseSearchCreatorTest)

using namespace ml;
using namespace model;

BOOST_AUTO_TEST_CASE(testCostOfToken) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");
    BOOST_REQUIRE_EQUAL(110, reverseSearchCreator.costOfToken("someToken", 5));
}

BOOST_AUTO_TEST_CASE(testCreateNoUniqueTokenSearch) {
    CTokenListReverseSearchCreator reverseSearchCreator("status");

    std::string terms;
    std::string regex;

    BOOST_TEST_REQUIRE(reverseSearchCreator.createNoUniqueTokenSearch(
        CLocalCategoryId{1}, "404", 4, terms, regex));

    BOOST_REQUIRE_EQUAL(std::string(), terms);
    BOOST_REQUIRE_EQUAL(std::string(".*"), regex);
}

BOOST_AUTO_TEST_CASE(testInitStandardSearch) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string terms;
    std::string regex;

    reverseSearchCreator.initStandardSearch(
        CLocalCategoryId{1}, "User 'foo' logged in host '0.0.0.0'", 1, terms, regex);

    BOOST_REQUIRE_EQUAL(std::string(), terms);
    BOOST_REQUIRE_EQUAL(std::string(), regex);
}

BOOST_AUTO_TEST_CASE(testAddInOrderCommonToken) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string terms;
    std::string regex;

    reverseSearchCreator.addInOrderCommonToken("user", terms, regex);
    reverseSearchCreator.addInOrderCommonToken("logged", terms, regex);
    reverseSearchCreator.addInOrderCommonToken("b=0.15+a", terms, regex);
    reverseSearchCreator.addInOrderCommonToken("logged", terms, regex);

    BOOST_REQUIRE_EQUAL(std::string("user logged b=0.15+a logged"), terms);
    BOOST_REQUIRE_EQUAL(std::string(".*?user.+?logged.+?b=0\\.15\\+a.+?logged"), regex);
}

BOOST_AUTO_TEST_CASE(testAddOutOfOrderCommonToken) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string terms;
    std::string regex;

    reverseSearchCreator.addOutOfOrderCommonToken("user", terms, regex);
    reverseSearchCreator.addOutOfOrderCommonToken("logged", terms, regex);

    BOOST_REQUIRE_EQUAL(std::string("user logged"), terms);
    BOOST_REQUIRE_EQUAL(std::string(), regex);
}

BOOST_AUTO_TEST_CASE(testCloseStandardSearch) {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string terms;
    std::string regex;

    reverseSearchCreator.closeStandardSearch(terms, regex);

    BOOST_REQUIRE_EQUAL(std::string(), terms);
    BOOST_REQUIRE_EQUAL(std::string(".*"), regex);
}

BOOST_AUTO_TEST_SUITE_END()
