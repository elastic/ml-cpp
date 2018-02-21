/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CRegexFilterTest.h"

#include <core/CLogger.h>
#include <core/CRegexFilter.h>


CppUnit::Test *CRegexFilterTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CRegexFilterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexFilterTest>(
                              "CRegexFilterTest::testConfigure_GivenInvalidRegex",
                              &CRegexFilterTest::testConfigure_GivenInvalidRegex) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexFilterTest>(
                              "CRegexFilterTest::testApply_GivenEmptyFilter",
                              &CRegexFilterTest::testApply_GivenEmptyFilter) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexFilterTest>(
                              "CRegexFilterTest::testApply_GivenSingleMatchAllRegex",
                              &CRegexFilterTest::testApply_GivenSingleMatchAllRegex) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexFilterTest>(
                              "CRegexFilterTest::testApply_GivenSingleRegex",
                              &CRegexFilterTest::testApply_GivenSingleRegex) );
    suiteOfTests->addTest(new CppUnit::TestCaller<CRegexFilterTest>(
                              "CRegexFilterTest::testApply_GivenMultipleRegex",
                              &CRegexFilterTest::testApply_GivenMultipleRegex) );

    return suiteOfTests;
}

void CRegexFilterTest::testConfigure_GivenInvalidRegex(void)
{
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string(".*"));
    regexVector.push_back(std::string("("));

    ml::core::CRegexFilter filter;
    CPPUNIT_ASSERT(filter.configure(regexVector) == false);
    CPPUNIT_ASSERT(filter.empty());
}

void CRegexFilterTest::testApply_GivenEmptyFilter(void)
{
    ml::core::CRegexFilter filter;
    CPPUNIT_ASSERT(filter.empty());

    CPPUNIT_ASSERT_EQUAL(std::string("foo"), filter.apply(std::string("foo")));
}

void CRegexFilterTest::testApply_GivenSingleMatchAllRegex(void)
{
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string(".*"));

    ml::core::CRegexFilter filter;
    CPPUNIT_ASSERT(filter.configure(regexVector));

    CPPUNIT_ASSERT_EQUAL(std::string(), filter.apply(std::string("foo")));
}

void CRegexFilterTest::testApply_GivenSingleRegex(void)
{
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string("f"));

    ml::core::CRegexFilter filter;
    CPPUNIT_ASSERT(filter.configure(regexVector));

    CPPUNIT_ASSERT_EQUAL(std::string("a"), filter.apply(std::string("fffa")));
}

void CRegexFilterTest::testApply_GivenMultipleRegex(void)
{
    std::vector<std::string> regexVector;
    regexVector.push_back(std::string("f[o]+"));
    regexVector.push_back(std::string("bar"));
    regexVector.push_back(std::string(" "));

    ml::core::CRegexFilter filter;
    CPPUNIT_ASSERT(filter.configure(regexVector));

    CPPUNIT_ASSERT_EQUAL(std::string("a"), filter.apply(std::string("foo bar fooooobar a")));
}
