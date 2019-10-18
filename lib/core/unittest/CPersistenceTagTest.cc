/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStringUtils.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CPersistenceTagTest)


namespace {
constexpr bool useShortTag{false};
constexpr bool useLongTag{true};

const std::string EMPTY_TAG;
const std::string SHORT_TAG1{"a"};
const std::string LONG_TAG1{"attribute1_name"};
const ml::core::TPersistenceTag TAG1{SHORT_TAG1, LONG_TAG1};

const std::string SHORT_TAG2{"b"};
const std::string LONG_TAG2{"attribute2_name"};
const ml::core::TPersistenceTag TAG2{SHORT_TAG2, LONG_TAG2};
}

BOOST_AUTO_TEST_CASE(testName) {
    ml::core::TPersistenceTag emptyTag;
    BOOST_CHECK_EQUAL(EMPTY_TAG, emptyTag.name(useShortTag));
    BOOST_CHECK_EQUAL(EMPTY_TAG, emptyTag.name(useLongTag));

    ml::core::TPersistenceTag fromShortTag(SHORT_TAG1);
    BOOST_CHECK_EQUAL(SHORT_TAG1, fromShortTag.name(useShortTag));
    BOOST_CHECK_EQUAL(SHORT_TAG1, fromShortTag.name(useLongTag));

    BOOST_CHECK_EQUAL(SHORT_TAG1, TAG1.name(useShortTag));
    BOOST_CHECK_EQUAL(LONG_TAG1, TAG1.name(useLongTag));

    BOOST_CHECK_EQUAL(SHORT_TAG2, TAG2.name(useShortTag));
    BOOST_CHECK_EQUAL(LONG_TAG2, TAG2.name(useLongTag));
}

BOOST_AUTO_TEST_CASE(testComparisons) {
    ml::core::TPersistenceTag tag2Copy = TAG2;
    BOOST_CHECK_EQUAL(tag2Copy, TAG2);

    BOOST_TEST(TAG1 != TAG2);
    BOOST_TEST(SHORT_TAG2 != TAG1);

    const std::string shortTag1 = TAG1;
    BOOST_CHECK_EQUAL(shortTag1, SHORT_TAG1);

    const std::string shortTag2{TAG2};
    BOOST_CHECK_EQUAL(shortTag2, SHORT_TAG2);

    BOOST_TEST(SHORT_TAG1 == TAG1);
    BOOST_TEST(LONG_TAG1 == TAG1);
    BOOST_TEST(TAG1 == SHORT_TAG1);
    BOOST_TEST(TAG1 == LONG_TAG1);

    BOOST_TEST(SHORT_TAG2 == TAG2);
    BOOST_TEST(LONG_TAG2 == TAG2);
    BOOST_TEST(TAG2 == SHORT_TAG2);
    BOOST_TEST(TAG2 == LONG_TAG2);
}

BOOST_AUTO_TEST_SUITE_END()
