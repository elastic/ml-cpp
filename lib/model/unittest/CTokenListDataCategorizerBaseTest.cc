/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CTokenListDataCategorizerBase.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CTokenListDataCategorizerBaseTest)

BOOST_AUTO_TEST_CASE(testMinMatchingWeights) {
    BOOST_REQUIRE_EQUAL(
        size_t(0), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(0, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(1), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(1, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(2), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(2, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(3), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(3, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(3), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(4, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(4), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(5, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(5), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(6, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(5), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(7, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(6), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(8, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(7), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(9, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(8), ml::model::CTokenListDataCategorizerBase::minMatchingWeight(10, 0.7));
}

BOOST_AUTO_TEST_CASE(testMaxMatchingWeights) {
    BOOST_REQUIRE_EQUAL(
        size_t(0), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(0, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(1), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(1, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(2), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(2, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(4), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(3, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(5), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(4, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(7), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(5, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(8), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(6, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(9), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(7, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(11), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(8, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(12), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(9, 0.7));
    BOOST_REQUIRE_EQUAL(
        size_t(14), ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(10, 0.7));
}

BOOST_AUTO_TEST_SUITE_END()
