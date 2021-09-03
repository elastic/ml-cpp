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
