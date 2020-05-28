/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CNoopCategoryIdMapper.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CNoopCategoryIdMapperTest)

BOOST_AUTO_TEST_CASE(testLocalToGlobal) {
    ml::api::CNoopCategoryIdMapper categoryIdMapper;

    BOOST_REQUIRE_EQUAL(-2, categoryIdMapper.globalCategoryIdForLocalCategoryId("", -2));
    BOOST_REQUIRE_EQUAL("-2", categoryIdMapper.printMapping("", -2));
    BOOST_REQUIRE_EQUAL(-1, categoryIdMapper.globalCategoryIdForLocalCategoryId("", -1));
    BOOST_REQUIRE_EQUAL("-1", categoryIdMapper.printMapping("", -1));
    BOOST_REQUIRE_EQUAL(1, categoryIdMapper.globalCategoryIdForLocalCategoryId("", 1));
    BOOST_REQUIRE_EQUAL("1", categoryIdMapper.printMapping("", 1));
    BOOST_REQUIRE_EQUAL(2, categoryIdMapper.globalCategoryIdForLocalCategoryId("", 2));
    BOOST_REQUIRE_EQUAL("2", categoryIdMapper.printMapping("", 2));
}

BOOST_AUTO_TEST_CASE(testGlobalToLocal) {
    ml::api::CNoopCategoryIdMapper categoryIdMapper;

    BOOST_REQUIRE_EQUAL("", categoryIdMapper.categorizerKeyForGlobalCategoryId(-2));
    BOOST_REQUIRE_EQUAL(-2, categoryIdMapper.localCategoryIdForGlobalCategoryId(-2));
    BOOST_REQUIRE_EQUAL("-2", categoryIdMapper.printMapping(-2));
    BOOST_REQUIRE_EQUAL("", categoryIdMapper.categorizerKeyForGlobalCategoryId(-1));
    BOOST_REQUIRE_EQUAL(-1, categoryIdMapper.localCategoryIdForGlobalCategoryId(-1));
    BOOST_REQUIRE_EQUAL("-1", categoryIdMapper.printMapping(-1));
    BOOST_REQUIRE_EQUAL("", categoryIdMapper.categorizerKeyForGlobalCategoryId(1));
    BOOST_REQUIRE_EQUAL(1, categoryIdMapper.localCategoryIdForGlobalCategoryId(1));
    BOOST_REQUIRE_EQUAL("1", categoryIdMapper.printMapping(1));
    BOOST_REQUIRE_EQUAL("", categoryIdMapper.categorizerKeyForGlobalCategoryId(2));
    BOOST_REQUIRE_EQUAL(2, categoryIdMapper.localCategoryIdForGlobalCategoryId(2));
    BOOST_REQUIRE_EQUAL("2", categoryIdMapper.printMapping(2));
}

BOOST_AUTO_TEST_SUITE_END()
