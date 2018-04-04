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
#include "CRegexFilterTest.h"

#include <core/CLogger.h>
#include <core/CRegexFilter.h>

CppUnit::Test* CRegexFilterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CRegexFilterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexFilterTest>("CRegexFilterTest::testConfigure_GivenInvalidRegex",
                                                                    &CRegexFilterTest::testConfigure_GivenInvalidRegex));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexFilterTest>("CRegexFilterTest::testApply_GivenEmptyFilter",
                                                                    &CRegexFilterTest::testApply_GivenEmptyFilter));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexFilterTest>("CRegexFilterTest::testApply_GivenSingleMatchAllRegex",
                                                                    &CRegexFilterTest::testApply_GivenSingleMatchAllRegex));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexFilterTest>("CRegexFilterTest::testApply_GivenSingleRegex",
                                                                    &CRegexFilterTest::testApply_GivenSingleRegex));
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexFilterTest>("CRegexFilterTest::testApply_GivenMultipleRegex",
                                                                    &CRegexFilterTest::testApply_GivenMultipleRegex));

    return suiteOfTests;
}

void CRegexFilterTest::testConfigure_GivenInvalidRegex() {
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string(".*"));
    regexVector.push_back(std::string("("));

    ml::core::CRegexFilter filter;
    CPPUNIT_ASSERT(filter.configure(regexVector) == false);
    CPPUNIT_ASSERT(filter.empty());
}

void CRegexFilterTest::testApply_GivenEmptyFilter() {
    ml::core::CRegexFilter filter;
    CPPUNIT_ASSERT(filter.empty());

    CPPUNIT_ASSERT_EQUAL(std::string("foo"), filter.apply(std::string("foo")));
}

void CRegexFilterTest::testApply_GivenSingleMatchAllRegex() {
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string(".*"));

    ml::core::CRegexFilter filter;
    CPPUNIT_ASSERT(filter.configure(regexVector));

    CPPUNIT_ASSERT_EQUAL(std::string(), filter.apply(std::string("foo")));
}

void CRegexFilterTest::testApply_GivenSingleRegex() {
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string("f"));

    ml::core::CRegexFilter filter;
    CPPUNIT_ASSERT(filter.configure(regexVector));

    CPPUNIT_ASSERT_EQUAL(std::string("a"), filter.apply(std::string("fffa")));
}

void CRegexFilterTest::testApply_GivenMultipleRegex() {
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string("f[o]+"));
    regexVector.push_back(std::string("bar"));
    regexVector.push_back(std::string(" "));

    ml::core::CRegexFilter filter;
    CPPUNIT_ASSERT(filter.configure(regexVector));

    CPPUNIT_ASSERT_EQUAL(std::string("a"), filter.apply(std::string("foo bar fooooobar a")));
}
