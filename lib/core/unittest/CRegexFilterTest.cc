/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRegexFilter.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CRegexFilterTest)


BOOST_AUTO_TEST_CASE(testConfigure_GivenInvalidRegex) {
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string(".*"));
    regexVector.push_back(std::string("("));

    ml::core::CRegexFilter filter;
    BOOST_TEST(filter.configure(regexVector) == false);
    BOOST_TEST(filter.empty());
}

BOOST_AUTO_TEST_CASE(testApply_GivenEmptyFilter) {
    ml::core::CRegexFilter filter;
    BOOST_TEST(filter.empty());

    BOOST_CHECK_EQUAL(std::string("foo"), filter.apply(std::string("foo")));
}

BOOST_AUTO_TEST_CASE(testApply_GivenSingleMatchAllRegex) {
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string(".*"));

    ml::core::CRegexFilter filter;
    BOOST_TEST(filter.configure(regexVector));

    BOOST_CHECK_EQUAL(std::string(), filter.apply(std::string("foo")));
}

BOOST_AUTO_TEST_CASE(testApply_GivenSingleRegex) {
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string("f"));

    ml::core::CRegexFilter filter;
    BOOST_TEST(filter.configure(regexVector));

    BOOST_CHECK_EQUAL(std::string("a"), filter.apply(std::string("fffa")));
}

BOOST_AUTO_TEST_CASE(testApply_GivenMultipleRegex) {
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string("f[o]+"));
    regexVector.push_back(std::string("bar"));
    regexVector.push_back(std::string(" "));

    ml::core::CRegexFilter filter;
    BOOST_TEST(filter.configure(regexVector));

    BOOST_CHECK_EQUAL(std::string("a"), filter.apply(std::string("foo bar fooooobar a")));
}

BOOST_AUTO_TEST_SUITE_END()
