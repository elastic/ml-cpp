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

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CLocalCategoryIdTest)

BOOST_AUTO_TEST_CASE(testDefaultConstructor) {
    ml::model::CLocalCategoryId localCategoryId;
    BOOST_TEST_REQUIRE(localCategoryId.isValid() == false);
    BOOST_TEST_REQUIRE(localCategoryId.isSoftFailure());
    BOOST_TEST_REQUIRE(localCategoryId.isHardFailure() == false);
    BOOST_REQUIRE_EQUAL(ml::model::CLocalCategoryId::SOFT_CATEGORIZATION_FAILURE_ERROR,
                        localCategoryId.id());
    BOOST_REQUIRE_EQUAL("-1", localCategoryId.toString());
    BOOST_REQUIRE_EQUAL(ml::model::CLocalCategoryId::softFailure(), localCategoryId);
}

BOOST_AUTO_TEST_CASE(testIdConstructor) {
    ml::model::CLocalCategoryId localCategoryId{7};
    BOOST_TEST_REQUIRE(localCategoryId.isValid());
    BOOST_TEST_REQUIRE(localCategoryId.isSoftFailure() == false);
    BOOST_TEST_REQUIRE(localCategoryId.isHardFailure() == false);
    BOOST_REQUIRE_EQUAL(7, localCategoryId.id());
    BOOST_REQUIRE_EQUAL(6, localCategoryId.index());
    BOOST_REQUIRE_EQUAL("7", localCategoryId.toString());
    ml::model::CLocalCategoryId otherLocalCategoryId;
    BOOST_TEST_REQUIRE(otherLocalCategoryId.fromString("7"));
    BOOST_REQUIRE_EQUAL(localCategoryId, otherLocalCategoryId);
}

BOOST_AUTO_TEST_CASE(testIndexConstructor) {
    ml::model::CLocalCategoryId localCategoryId{std::size_t(3)};
    BOOST_TEST_REQUIRE(localCategoryId.isValid());
    BOOST_TEST_REQUIRE(localCategoryId.isSoftFailure() == false);
    BOOST_TEST_REQUIRE(localCategoryId.isHardFailure() == false);
    BOOST_REQUIRE_EQUAL(4, localCategoryId.id());
    BOOST_REQUIRE_EQUAL(3, localCategoryId.index());
    BOOST_REQUIRE_EQUAL("4", localCategoryId.toString());
}

BOOST_AUTO_TEST_CASE(testFailureHelpers) {
    ml::model::CLocalCategoryId softFailure{ml::model::CLocalCategoryId::softFailure()};
    BOOST_TEST_REQUIRE(softFailure.isValid() == false);
    BOOST_TEST_REQUIRE(softFailure.isSoftFailure());
    BOOST_TEST_REQUIRE(softFailure.isHardFailure() == false);

    ml::model::CLocalCategoryId hardFailure{ml::model::CLocalCategoryId::hardFailure()};
    BOOST_TEST_REQUIRE(hardFailure.isValid() == false);
    BOOST_TEST_REQUIRE(hardFailure.isSoftFailure() == false);
    BOOST_TEST_REQUIRE(hardFailure.isHardFailure());

    BOOST_TEST_REQUIRE(softFailure != hardFailure);
    BOOST_TEST_REQUIRE((softFailure == hardFailure) == false);
    BOOST_TEST_REQUIRE(softFailure.toString() != hardFailure.toString());
}

BOOST_AUTO_TEST_SUITE_END()
