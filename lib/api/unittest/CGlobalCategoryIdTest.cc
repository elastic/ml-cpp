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

#include <model/CLocalCategoryId.h>

#include <api/CGlobalCategoryId.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CGlobalCategoryIdTest)

BOOST_AUTO_TEST_CASE(testDefaultConstructor) {
    ml::api::CGlobalCategoryId globalCategoryId;
    BOOST_TEST_REQUIRE(globalCategoryId.isValid() == false);
    BOOST_TEST_REQUIRE(globalCategoryId.isSoftFailure());
    BOOST_TEST_REQUIRE(globalCategoryId.isHardFailure() == false);
    BOOST_REQUIRE_EQUAL(ml::model::CLocalCategoryId::SOFT_CATEGORIZATION_FAILURE_ERROR,
                        globalCategoryId.globalId());
    BOOST_REQUIRE_EQUAL("-1", globalCategoryId.print());
    BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId::softFailure(), globalCategoryId);
}

BOOST_AUTO_TEST_CASE(testIdConstructor) {
    ml::api::CGlobalCategoryId globalCategoryId{7};
    BOOST_TEST_REQUIRE(globalCategoryId.isValid());
    BOOST_TEST_REQUIRE(globalCategoryId.isSoftFailure() == false);
    BOOST_TEST_REQUIRE(globalCategoryId.isHardFailure() == false);
    BOOST_REQUIRE_EQUAL(7, globalCategoryId.globalId());
    BOOST_REQUIRE_EQUAL("", globalCategoryId.categorizerKey());
    BOOST_REQUIRE_EQUAL(ml::model::CLocalCategoryId{7}, globalCategoryId.localId());
    BOOST_REQUIRE_EQUAL("7", globalCategoryId.print());
}

BOOST_AUTO_TEST_CASE(testIdKeyIdConstructor) {
    std::string categorizerKey{"foo"};
    ml::api::CGlobalCategoryId globalCategoryId{5, categorizerKey,
                                                ml::model::CLocalCategoryId{2}};
    BOOST_TEST_REQUIRE(globalCategoryId.isValid());
    BOOST_TEST_REQUIRE(globalCategoryId.isSoftFailure() == false);
    BOOST_TEST_REQUIRE(globalCategoryId.isHardFailure() == false);
    BOOST_REQUIRE_EQUAL(5, globalCategoryId.globalId());
    BOOST_REQUIRE_EQUAL("foo", globalCategoryId.categorizerKey());
    BOOST_REQUIRE_EQUAL(ml::model::CLocalCategoryId{2}, globalCategoryId.localId());
    BOOST_REQUIRE_EQUAL("foo/2;5", globalCategoryId.print());
}

BOOST_AUTO_TEST_CASE(testFailureHelpers) {
    ml::api::CGlobalCategoryId softFailure{ml::api::CGlobalCategoryId::softFailure()};
    BOOST_TEST_REQUIRE(softFailure.isValid() == false);
    BOOST_TEST_REQUIRE(softFailure.isSoftFailure());
    BOOST_TEST_REQUIRE(softFailure.isHardFailure() == false);

    ml::api::CGlobalCategoryId hardFailure{ml::api::CGlobalCategoryId::hardFailure()};
    BOOST_TEST_REQUIRE(hardFailure.isValid() == false);
    BOOST_TEST_REQUIRE(hardFailure.isSoftFailure() == false);
    BOOST_TEST_REQUIRE(hardFailure.isHardFailure());

    BOOST_TEST_REQUIRE(softFailure != hardFailure);
    BOOST_TEST_REQUIRE((softFailure == hardFailure) == false);
    BOOST_TEST_REQUIRE(softFailure.print() != hardFailure.print());
}

BOOST_AUTO_TEST_SUITE_END()
