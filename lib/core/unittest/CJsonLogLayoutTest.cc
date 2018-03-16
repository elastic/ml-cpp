/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
#include "CJsonLogLayoutTest.h"

#include <core/CJsonLogLayout.h>
#include <core/CLogger.h>

CppUnit::Test* CJsonLogLayoutTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CJsonLogLayoutTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CJsonLogLayoutTest>("CJsonLogLayoutTest::testPathCropping",
                                                                      &CJsonLogLayoutTest::testPathCropping));

    return suiteOfTests;
}

void CJsonLogLayoutTest::testPathCropping(void) {
    LOG_DEBUG("CJsonLogLayoutTest::testPathCropping");
#ifdef Windows
    CPPUNIT_ASSERT_EQUAL(std::string("source.h"),
                         log4cxx::helpers::CJsonLogLayout::cropPath("c:\\\\home\\hendrik\\src\\include/source.h"));
    CPPUNIT_ASSERT_EQUAL(std::string("source.h"),
                         log4cxx::helpers::CJsonLogLayout::cropPath("c:\\\\home\\hendrik\\src\\include\\source.h"));
#else
    CPPUNIT_ASSERT_EQUAL(std::string("source.h"),
                         log4cxx::helpers::CJsonLogLayout::cropPath("/home/hendrik/src/include/source.h"));
    CPPUNIT_ASSERT_EQUAL(std::string("source.h"),
                         log4cxx::helpers::CJsonLogLayout::cropPath("/home/hendrik/work/../src/include/source.h"));
#endif
}
