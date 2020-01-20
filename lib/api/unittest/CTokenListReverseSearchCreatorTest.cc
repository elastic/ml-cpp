/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CTokenListReverseSearchCreatorTest.h"

#include <api/CTokenListReverseSearchCreator.h>

using namespace ml;
using namespace api;

CppUnit::Test* CTokenListReverseSearchCreatorTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CTokenListReverseSearchCreatorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CTokenListReverseSearchCreatorTest>(
        "CTokenListReverseSearchCreatorTest::testCostOfToken",
        &CTokenListReverseSearchCreatorTest::testCostOfToken));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTokenListReverseSearchCreatorTest>(
        "CTokenListReverseSearchCreatorTest::testCreateNullSearch",
        &CTokenListReverseSearchCreatorTest::testCreateNullSearch));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTokenListReverseSearchCreatorTest>(
        "CTokenListReverseSearchCreatorTest::testCreateNoUniqueTokenSearch",
        &CTokenListReverseSearchCreatorTest::testCreateNoUniqueTokenSearch));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTokenListReverseSearchCreatorTest>(
        "CTokenListReverseSearchCreatorTest::testInitStandardSearch",
        &CTokenListReverseSearchCreatorTest::testInitStandardSearch));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTokenListReverseSearchCreatorTest>(
        "CTokenListReverseSearchCreatorTest::testAddInOrderCommonToken",
        &CTokenListReverseSearchCreatorTest::testAddInOrderCommonToken));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTokenListReverseSearchCreatorTest>(
        "CTokenListReverseSearchCreatorTest::testAddOutOfOrderCommonToken",
        &CTokenListReverseSearchCreatorTest::testAddOutOfOrderCommonToken));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTokenListReverseSearchCreatorTest>(
        "CTokenListReverseSearchCreatorTest::testCloseStandardSearch",
        &CTokenListReverseSearchCreatorTest::testCloseStandardSearch));

    return suiteOfTests;
}

void CTokenListReverseSearchCreatorTest::testCostOfToken() {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");
    CPPUNIT_ASSERT_EQUAL(std::size_t(110),
                         reverseSearchCreator.costOfToken("someToken", 5));
}

void CTokenListReverseSearchCreatorTest::testCreateNullSearch() {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    CPPUNIT_ASSERT(reverseSearchCreator.createNullSearch(reverseSearchPart1, reverseSearchPart2));

    CPPUNIT_ASSERT_EQUAL(std::string(""), reverseSearchPart1);
    CPPUNIT_ASSERT_EQUAL(std::string(""), reverseSearchPart2);
}

void CTokenListReverseSearchCreatorTest::testCreateNoUniqueTokenSearch() {
    CTokenListReverseSearchCreator reverseSearchCreator("status");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    CPPUNIT_ASSERT(reverseSearchCreator.createNoUniqueTokenSearch(
        1, "404", 4, reverseSearchPart1, reverseSearchPart2));

    CPPUNIT_ASSERT_EQUAL(std::string(""), reverseSearchPart1);
    CPPUNIT_ASSERT_EQUAL(std::string(""), reverseSearchPart2);
}

void CTokenListReverseSearchCreatorTest::testInitStandardSearch() {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    reverseSearchCreator.initStandardSearch(1, "User 'foo' logged in host '0.0.0.0'",
                                            1, reverseSearchPart1, reverseSearchPart2);

    CPPUNIT_ASSERT_EQUAL(std::string(""), reverseSearchPart1);
    CPPUNIT_ASSERT_EQUAL(std::string(""), reverseSearchPart2);
}

void CTokenListReverseSearchCreatorTest::testAddInOrderCommonToken() {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    reverseSearchCreator.addInOrderCommonToken("user", true, reverseSearchPart1,
                                               reverseSearchPart2);
    reverseSearchCreator.addInOrderCommonToken("logged", false, reverseSearchPart1,
                                               reverseSearchPart2);
    reverseSearchCreator.addInOrderCommonToken("b=0.15+a", false, reverseSearchPart1,
                                               reverseSearchPart2);
    reverseSearchCreator.addInOrderCommonToken("logged", false, reverseSearchPart1,
                                               reverseSearchPart2);

    CPPUNIT_ASSERT_EQUAL(std::string("user logged b=0.15+a logged"), reverseSearchPart1);
    CPPUNIT_ASSERT_EQUAL(std::string(".*?user.+?logged.+?b=0\\.15\\+a.+?logged"),
                         reverseSearchPart2);
}

void CTokenListReverseSearchCreatorTest::testAddOutOfOrderCommonToken() {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    reverseSearchCreator.addOutOfOrderCommonToken("user", reverseSearchPart1, reverseSearchPart2);
    reverseSearchCreator.addOutOfOrderCommonToken("logged", reverseSearchPart1,
                                                  reverseSearchPart2);

    CPPUNIT_ASSERT_EQUAL(std::string("user logged"), reverseSearchPart1);
    CPPUNIT_ASSERT_EQUAL(std::string(""), reverseSearchPart2);
}

void CTokenListReverseSearchCreatorTest::testCloseStandardSearch() {
    CTokenListReverseSearchCreator reverseSearchCreator("foo");

    std::string reverseSearchPart1;
    std::string reverseSearchPart2;

    reverseSearchCreator.closeStandardSearch(reverseSearchPart1, reverseSearchPart2);

    CPPUNIT_ASSERT_EQUAL(std::string(""), reverseSearchPart1);
    CPPUNIT_ASSERT_EQUAL(std::string(".*"), reverseSearchPart2);
}
