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

// ============================================================================
// Constraint Enforcement Behavior
// ============================================================================

BOOST_AUTO_TEST_CASE(testShortValueUnchanged) {
    std::string value("short");
    BOOST_REQUIRE_EQUAL(false, CFieldValueTruncator::truncate(value));
    BOOST_REQUIRE_EQUAL("short", value);
}

BOOST_AUTO_TEST_CASE(testValueAtExactLimitUnchanged) {
    std::string value(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, 'x');
    BOOST_REQUIRE_EQUAL(false, CFieldValueTruncator::truncate(value));
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, value.size());
}

BOOST_AUTO_TEST_CASE(testOversizedValueEnforcedTo256Chars) {
    std::string value(1000, 'x');
    BOOST_REQUIRE_EQUAL(true, CFieldValueTruncator::truncate(value));
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, value.size());
}

BOOST_AUTO_TEST_CASE(testEmptyValueUnchanged) {
    std::string value;
    BOOST_REQUIRE_EQUAL(false, CFieldValueTruncator::truncate(value));
    BOOST_REQUIRE_EQUAL(0, value.size());
}

BOOST_AUTO_TEST_CASE(testConstOverloadPreservesOriginal) {
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

BOOST_AUTO_TEST_CASE(testVeryLargeValueFromIssue2796) {
    std::string value(77000, 'y');
    BOOST_REQUIRE_EQUAL(true, CFieldValueTruncator::truncate(value));
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, value.size());
}

BOOST_AUTO_TEST_CASE(testNeedsTruncation) {
    BOOST_REQUIRE_EQUAL(false, CFieldValueTruncator::needsTruncation("short"));
    BOOST_REQUIRE_EQUAL(false, CFieldValueTruncator::needsTruncation(""));
    BOOST_REQUIRE_EQUAL(false, CFieldValueTruncator::needsTruncation(std::string(
                                   CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, 'x')));
    BOOST_REQUIRE_EQUAL(true, CFieldValueTruncator::needsTruncation(std::string(
                                  CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH + 1, 'x')));
    BOOST_REQUIRE_EQUAL(
        true, CFieldValueTruncator::needsTruncation(std::string(77000, 'x')));
}

// ============================================================================
// Hash Suffix Format Validation
// ============================================================================

BOOST_AUTO_TEST_CASE(testTruncatedValueHasCorrectFormat) {
    std::string value(1000, 'x');
    std::string result = CFieldValueTruncator::truncated(value);

    // Format: 240 prefix + '$' + 15 hex chars = 256 total
    BOOST_REQUIRE_EQUAL(256, result.size());
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::HASH_SEPARATOR, result[240]);

    // Prefix should match original
    BOOST_REQUIRE_EQUAL(0, result.compare(0, 240, value, 0, 240));

    // Hash portion should be lowercase hex digits
    for (std::size_t i = 241; i < 256; ++i) {
        BOOST_REQUIRE(std::isxdigit(result[i]));
        BOOST_REQUIRE((result[i] >= '0' && result[i] <= '9') ||
                     (result[i] >= 'a' && result[i] <= 'f'));
    }
}

BOOST_AUTO_TEST_CASE(testInPlaceTruncationPreservesFormat) {
    std::string value(1000, 'z');
    bool wasTruncated = CFieldValueTruncator::truncate(value);

    BOOST_REQUIRE_EQUAL(true, wasTruncated);
    BOOST_REQUIRE_EQUAL(256, value.size());
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::HASH_SEPARATOR, value[240]);

    // Verify hash portion is valid hex
    for (std::size_t i = 241; i < 256; ++i) {
        BOOST_REQUIRE(std::isxdigit(value[i]));
    }
}

// ============================================================================
// Collision Prevention (Data Integrity)
// ============================================================================

BOOST_AUTO_TEST_CASE(testDistinctValuesProduceDistinctResults) {
    std::string prefix(240, 'x');
    std::string value1 = prefix + std::string(1000, 'A');
    std::string value2 = prefix + std::string(1000, 'B');

    std::string truncated1 = CFieldValueTruncator::truncated(value1);
    std::string truncated2 = CFieldValueTruncator::truncated(value2);

    // Same prefix
    BOOST_REQUIRE_EQUAL(truncated1.substr(0, 241), truncated2.substr(0, 241));

    // But different hash suffixes prevent collision
    BOOST_REQUIRE_NE(truncated1.substr(241), truncated2.substr(241));
    BOOST_REQUIRE_NE(truncated1, truncated2);
}

BOOST_AUTO_TEST_CASE(testCollisionsPreventedByHashSuffix) {
    // Two values differing only after position 256 (original collision case)
    std::string value1(300, 'x');
    value1.replace(280, 20, "AAAAAAAAAAAAAAAAAAAA");

    std::string value2(300, 'x');
    value2.replace(280, 20, "BBBBBBBBBBBBBBBBBBBB");

    std::string truncated1 = CFieldValueTruncator::truncated(value1);
    std::string truncated2 = CFieldValueTruncator::truncated(value2);

    // Must be distinct despite identical first 240 chars
    BOOST_REQUIRE_NE(truncated1, truncated2);
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, truncated1.size());
    BOOST_REQUIRE_EQUAL(CFieldValueTruncator::MAX_FIELD_VALUE_LENGTH, truncated2.size());
}

BOOST_AUTO_TEST_CASE(testDeterministicHashing) {
    std::string value(1000, 'y');
    std::string result1 = CFieldValueTruncator::truncated(value);
    std::string result2 = CFieldValueTruncator::truncated(value);

    BOOST_REQUIRE_EQUAL(result1, result2);
}

BOOST_AUTO_TEST_CASE(testVeryLongValueWithDistinctEnding) {
    // Simulate the 77K influencer case from issue #2796
    std::string value1(77000, 'x');
    value1.replace(76990, 10, "VARIANT_A");

    std::string value2(77000, 'x');
    value2.replace(76990, 10, "VARIANT_B");

    std::string truncated1 = CFieldValueTruncator::truncated(value1);
    std::string truncated2 = CFieldValueTruncator::truncated(value2);

    // Must be distinct despite identical first 240 chars
    BOOST_REQUIRE_NE(truncated1, truncated2);
}

BOOST_AUTO_TEST_SUITE_END()
