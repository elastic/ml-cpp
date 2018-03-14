/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include "CBaseTokenListDataTyperTest.h"

#include <api/CBaseTokenListDataTyper.h>


CppUnit::Test *CBaseTokenListDataTyperTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CBaseTokenListDataTyperTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CBaseTokenListDataTyperTest>(
                               "CBaseTokenListDataTyperTest::testMinMatchingWeights",
                               &CBaseTokenListDataTyperTest::testMinMatchingWeights) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBaseTokenListDataTyperTest>(
                               "CBaseTokenListDataTyperTest::testMaxMatchingWeights",
                               &CBaseTokenListDataTyperTest::testMaxMatchingWeights) );

    return suiteOfTests;
}

void CBaseTokenListDataTyperTest::testMinMatchingWeights(void) {
    CPPUNIT_ASSERT_EQUAL(size_t(0), ml::api::CBaseTokenListDataTyper::minMatchingWeight(0, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::api::CBaseTokenListDataTyper::minMatchingWeight(1, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(2), ml::api::CBaseTokenListDataTyper::minMatchingWeight(2, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(3), ml::api::CBaseTokenListDataTyper::minMatchingWeight(3, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(3), ml::api::CBaseTokenListDataTyper::minMatchingWeight(4, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(4), ml::api::CBaseTokenListDataTyper::minMatchingWeight(5, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(5), ml::api::CBaseTokenListDataTyper::minMatchingWeight(6, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(5), ml::api::CBaseTokenListDataTyper::minMatchingWeight(7, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(6), ml::api::CBaseTokenListDataTyper::minMatchingWeight(8, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(7), ml::api::CBaseTokenListDataTyper::minMatchingWeight(9, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(8), ml::api::CBaseTokenListDataTyper::minMatchingWeight(10, 0.7));
}

void CBaseTokenListDataTyperTest::testMaxMatchingWeights(void) {
    CPPUNIT_ASSERT_EQUAL(size_t(0), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(0, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(1, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(2), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(2, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(4), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(3, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(5), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(4, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(7), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(5, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(8), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(6, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(9), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(7, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(11), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(8, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(12), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(9, 0.7));
    CPPUNIT_ASSERT_EQUAL(size_t(14), ml::api::CBaseTokenListDataTyper::maxMatchingWeight(10, 0.7));
}

