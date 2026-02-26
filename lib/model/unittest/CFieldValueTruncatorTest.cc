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

#include <model/CFieldValueTruncator.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CFieldValueTruncatorTest)

using namespace ml;
using namespace model;

BOOST_AUTO_TEST_CASE(testShortValueUnchanged) {
    std::string value("short");
    BOOST_REQUIRE_EQUAL(false, CFieldValueTruncator::truncate(value));
    BOOST_REQUIRE_EQUAL("short", value);
}

BOOST_AUTO_TEST_CASE(testExactLimitUnchanged) {
    std::string value(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, 'x');
    BOOST_REQUIRE_EQUAL(false, CFieldValueTruncator::truncate(value));
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, value.size());
}

BOOST_AUTO_TEST_CASE(testOversizedValueTruncated) {
    std::string value(1000, 'x');
    BOOST_REQUIRE_EQUAL(true, CFieldValueTruncator::truncate(value));
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, value.size());
}

BOOST_AUTO_TEST_CASE(testEmptyValueUnchanged) {
    std::string value;
    BOOST_REQUIRE_EQUAL(false, CFieldValueTruncator::truncate(value));
    BOOST_REQUIRE_EQUAL(0, value.size());
}

BOOST_AUTO_TEST_CASE(testConstOverloadReturnsNewString) {
    const std::string longValue(1000, 'x');
    std::string result = CFieldValueTruncator::truncated(longValue);
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, result.size());
    BOOST_REQUIRE_EQUAL(1000, longValue.size());
}

BOOST_AUTO_TEST_CASE(testConstOverloadShortValueReturnsSame) {
    const std::string shortValue("short");
    std::string result = CFieldValueTruncator::truncated(shortValue);
    BOOST_REQUIRE_EQUAL("short", result);
}

BOOST_AUTO_TEST_CASE(testVeryLargeValueTruncated) {
    std::string value(77000, 'y');
    BOOST_REQUIRE_EQUAL(true, CFieldValueTruncator::truncate(value));
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, value.size());
}

BOOST_AUTO_TEST_SUITE_END()
