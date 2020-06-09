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

    BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId{-2},
                        categoryIdMapper.map(ml::model::CLocalCategoryId{-2}));
    BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId{-1},
                        categoryIdMapper.map(ml::model::CLocalCategoryId{-1}));
    BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId{1},
                        categoryIdMapper.map(ml::model::CLocalCategoryId{1}));
    BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId{2},
                        categoryIdMapper.map(ml::model::CLocalCategoryId{2}));
}

BOOST_AUTO_TEST_SUITE_END()
