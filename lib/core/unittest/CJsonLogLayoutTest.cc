/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CJsonLogLayoutTest.h"

#include <core/CJsonLogLayout.h>
#include <core/CLogger.h>

#include <boost/current_function.hpp>

CppUnit::Test* CJsonLogLayoutTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CJsonLogLayoutTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonLogLayoutTest>(
        "CJsonLogLayoutTest::testPathCropping", &CJsonLogLayoutTest::testPathCropping));
    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonLogLayoutTest>(
        "CJsonLogLayoutTest::testExtractClassAndMethod",
        &CJsonLogLayoutTest::testExtractClassAndMethod));

    return suiteOfTests;
}

void CJsonLogLayoutTest::testPathCropping() {
#ifdef Windows
    CPPUNIT_ASSERT_EQUAL(std::string{"source.h"},
                         ml::core::CJsonLogLayout::cropPath(
                             "c:\\\\home\\hendrik\\src\\include/source.h"));
    CPPUNIT_ASSERT_EQUAL(std::string{"source.h"},
                         ml::core::CJsonLogLayout::cropPath(
                             "c:\\\\home\\hendrik\\src\\include\\source.h"));
#else
    CPPUNIT_ASSERT_EQUAL(std::string{"source.h"},
                         ml::core::CJsonLogLayout::cropPath("/home/hendrik/src/include/source.h"));
    CPPUNIT_ASSERT_EQUAL(std::string{"source.h"},
                         ml::core::CJsonLogLayout::cropPath(
                             "/home/hendrik/work/../src/include/source.h"));
#endif
}

void CJsonLogLayoutTest::testExtractClassAndMethod() {

    std::string className;
    std::string methodName;

    std::tie(className, methodName) = ml::core::CJsonLogLayout::extractClassAndMethod(
        "std::string ns1::ns2::clazz::someMethod(int arg1, char arg2)");
    CPPUNIT_ASSERT_EQUAL(std::string{"ns1::ns2::clazz"}, className);
    CPPUNIT_ASSERT_EQUAL(std::string{"someMethod"}, methodName);

    std::tie(className, methodName) = ml::core::CJsonLogLayout::extractClassAndMethod(
        "static CJsonLogLayout::TStrStrPr ml::core::CJsonLogLayout::extractClassAndMethod(std::string)");
    CPPUNIT_ASSERT_EQUAL(std::string{"ml::core::CJsonLogLayout"}, className);
    CPPUNIT_ASSERT_EQUAL(std::string{"extractClassAndMethod"}, methodName);

    std::tie(className, methodName) =
        ml::core::CJsonLogLayout::extractClassAndMethod(BOOST_CURRENT_FUNCTION);
    CPPUNIT_ASSERT_EQUAL(std::string{"CJsonLogLayoutTest"}, className);
    CPPUNIT_ASSERT_EQUAL(std::string{"testExtractClassAndMethod"}, methodName);
}
