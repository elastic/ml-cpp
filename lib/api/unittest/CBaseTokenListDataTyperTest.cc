/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CBaseTokenListDataTyper.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CBaseTokenListDataTyperTest)

BOOST_AUTO_TEST_CASE(testMinMatchingWeights) {
    BOOST_CHECK_EQUAL(size_t(0),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(0, 0.7));
    BOOST_CHECK_EQUAL(size_t(1),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(1, 0.7));
    BOOST_CHECK_EQUAL(size_t(2),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(2, 0.7));
    BOOST_CHECK_EQUAL(size_t(3),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(3, 0.7));
    BOOST_CHECK_EQUAL(size_t(3),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(4, 0.7));
    BOOST_CHECK_EQUAL(size_t(4),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(5, 0.7));
    BOOST_CHECK_EQUAL(size_t(5),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(6, 0.7));
    BOOST_CHECK_EQUAL(size_t(5),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(7, 0.7));
    BOOST_CHECK_EQUAL(size_t(6),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(8, 0.7));
    BOOST_CHECK_EQUAL(size_t(7),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(9, 0.7));
    BOOST_CHECK_EQUAL(size_t(8),
                      ml::api::CBaseTokenListDataTyper::minMatchingWeight(10, 0.7));
}

BOOST_AUTO_TEST_CASE(testMaxMatchingWeights) {
    BOOST_CHECK_EQUAL(size_t(0),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(0, 0.7));
    BOOST_CHECK_EQUAL(size_t(1),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(1, 0.7));
    BOOST_CHECK_EQUAL(size_t(2),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(2, 0.7));
    BOOST_CHECK_EQUAL(size_t(4),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(3, 0.7));
    BOOST_CHECK_EQUAL(size_t(5),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(4, 0.7));
    BOOST_CHECK_EQUAL(size_t(7),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(5, 0.7));
    BOOST_CHECK_EQUAL(size_t(8),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(6, 0.7));
    BOOST_CHECK_EQUAL(size_t(9),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(7, 0.7));
    BOOST_CHECK_EQUAL(size_t(11),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(8, 0.7));
    BOOST_CHECK_EQUAL(size_t(12),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(9, 0.7));
    BOOST_CHECK_EQUAL(size_t(14),
                      ml::api::CBaseTokenListDataTyper::maxMatchingWeight(10, 0.7));
}

BOOST_AUTO_TEST_SUITE_END()
