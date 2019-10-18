/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonLogLayout.h>
#include <core/CLogger.h>

#include <boost/current_function.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CJsonLogLayoutTest)

BOOST_AUTO_TEST_CASE(testPathCropping) {
#ifdef Windows
    BOOST_CHECK_EQUAL(std::string{"source.h"},
                      ml::core::CJsonLogLayout::cropPath("c:\\\\home\\hendrik\\src\\include/source.h"));
    BOOST_CHECK_EQUAL(std::string{"source.h"},
                      ml::core::CJsonLogLayout::cropPath("c:\\\\home\\hendrik\\src\\include\\source.h"));
#else
    BOOST_CHECK_EQUAL(std::string{"source.h"},
                      ml::core::CJsonLogLayout::cropPath("/home/hendrik/src/include/source.h"));
    BOOST_CHECK_EQUAL(std::string{"source.h"},
                      ml::core::CJsonLogLayout::cropPath("/home/hendrik/work/../src/include/source.h"));
#endif
}

BOOST_AUTO_TEST_CASE(testExtractClassAndMethod) {

    std::string className;
    std::string methodName;

    std::tie(className, methodName) = ml::core::CJsonLogLayout::extractClassAndMethod(
        "std::string ns1::ns2::clazz::someMethod(int arg1, char arg2)");
    BOOST_CHECK_EQUAL(std::string{"ns1::ns2::clazz"}, className);
    BOOST_CHECK_EQUAL(std::string{"someMethod"}, methodName);

    std::tie(className, methodName) = ml::core::CJsonLogLayout::extractClassAndMethod(
        "static CJsonLogLayout::TStrStrPr ml::core::CJsonLogLayout::extractClassAndMethod(std::string)");
    BOOST_CHECK_EQUAL(std::string{"ml::core::CJsonLogLayout"}, className);
    BOOST_CHECK_EQUAL(std::string{"extractClassAndMethod"}, methodName);

    std::tie(className, methodName) =
        ml::core::CJsonLogLayout::extractClassAndMethod(BOOST_CURRENT_FUNCTION);
    BOOST_CHECK_EQUAL(std::string{"CJsonLogLayoutTest::testExtractClassAndMethod"}, className);
    BOOST_CHECK_EQUAL(std::string{"test_method"}, methodName);
}

BOOST_AUTO_TEST_SUITE_END()
