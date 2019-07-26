/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CPersistenceTagTest.h"

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStringUtils.h>

#include <sstream>

CppUnit::Test* CPersistenceTagTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CPersistenceTagTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CPersistenceTagTest>(
        "CPersistenceTagTest::testName", &CPersistenceTagTest::testName));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPersistenceTagTest>(
        "CPersistenceTagTest::testComparisons", &CPersistenceTagTest::testComparisons));

    return suiteOfTests;
}

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

void CPersistenceTagTest::testName() {
    ml::core::TPersistenceTag emptyTag;
    CPPUNIT_ASSERT_EQUAL(EMPTY_TAG, emptyTag.name(useShortTag));
    CPPUNIT_ASSERT_EQUAL(EMPTY_TAG, emptyTag.name(useLongTag));

    ml::core::TPersistenceTag fromShortTag(SHORT_TAG1);
    CPPUNIT_ASSERT_EQUAL(SHORT_TAG1, fromShortTag.name(useShortTag));
    CPPUNIT_ASSERT_EQUAL(SHORT_TAG1, fromShortTag.name(useLongTag));

    CPPUNIT_ASSERT_EQUAL(SHORT_TAG1, TAG1.name(useShortTag));
    CPPUNIT_ASSERT_EQUAL(LONG_TAG1, TAG1.name(useLongTag));

    CPPUNIT_ASSERT_EQUAL(SHORT_TAG2, TAG2.name(useShortTag));
    CPPUNIT_ASSERT_EQUAL(LONG_TAG2, TAG2.name(useLongTag));
}

void CPersistenceTagTest::testComparisons() {
    ml::core::TPersistenceTag tag2Copy = TAG2;
    CPPUNIT_ASSERT_EQUAL(tag2Copy, TAG2);

    CPPUNIT_ASSERT(TAG1 != TAG2);
    CPPUNIT_ASSERT(SHORT_TAG2 != TAG1);

    const std::string shortTag1 = TAG1;
    CPPUNIT_ASSERT_EQUAL(shortTag1, SHORT_TAG1);

    const std::string shortTag2{TAG2};
    CPPUNIT_ASSERT_EQUAL(shortTag2, SHORT_TAG2);

    CPPUNIT_ASSERT(SHORT_TAG1 == TAG1);
    CPPUNIT_ASSERT(LONG_TAG1 == TAG1);
    CPPUNIT_ASSERT(TAG1 == SHORT_TAG1);
    CPPUNIT_ASSERT(TAG1 == LONG_TAG1);

    CPPUNIT_ASSERT(SHORT_TAG2 == TAG2);
    CPPUNIT_ASSERT(LONG_TAG2 == TAG2);
    CPPUNIT_ASSERT(TAG2 == SHORT_TAG2);
    CPPUNIT_ASSERT(TAG2 == LONG_TAG2);
}
