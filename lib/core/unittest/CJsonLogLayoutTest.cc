/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CJsonLogLayoutTest.h"

#include <core/CJsonLogLayout.h>
#include <core/CLogger.h>

CppUnit::Test* CJsonLogLayoutTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CJsonLogLayoutTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonLogLayoutTest>(
        "CJsonLogLayoutTest::testPathCropping", &CJsonLogLayoutTest::testPathCropping));

    return suiteOfTests;
}

void CJsonLogLayoutTest::testPathCropping() {
#ifdef Windows
    CPPUNIT_ASSERT_EQUAL(std::string("source.h"),
                         log4cxx::helpers::CJsonLogLayout::cropPath(
                             "c:\\\\home\\hendrik\\src\\include/source.h"));
    CPPUNIT_ASSERT_EQUAL(std::string("source.h"),
                         log4cxx::helpers::CJsonLogLayout::cropPath(
                             "c:\\\\home\\hendrik\\src\\include\\source.h"));
#else
    CPPUNIT_ASSERT_EQUAL(std::string("source.h"),
                         log4cxx::helpers::CJsonLogLayout::cropPath(
                             "/home/hendrik/src/include/source.h"));
    CPPUNIT_ASSERT_EQUAL(std::string("source.h"),
                         log4cxx::helpers::CJsonLogLayout::cropPath(
                             "/home/hendrik/work/../src/include/source.h"));
#endif
}
