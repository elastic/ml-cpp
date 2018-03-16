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
#include "CResourceLocatorTest.h"

#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CResourceLocator.h>

CppUnit::Test* CResourceLocatorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CResourceLocatorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CResourceLocatorTest>("CResourceLocatorTest::testResourceDir",
                                                                        &CResourceLocatorTest::testResourceDir));
    suiteOfTests->addTest(new CppUnit::TestCaller<CResourceLocatorTest>("CResourceLocatorTest::testLogDir",
                                                                        &CResourceLocatorTest::testLogDir));
    suiteOfTests->addTest(new CppUnit::TestCaller<CResourceLocatorTest>("CResourceLocatorTest::testSrcRootDir",
                                                                        &CResourceLocatorTest::testSrcRootDir));

    return suiteOfTests;
}

void CResourceLocatorTest::testResourceDir(void) {
    std::string resourceDir(ml::core::CResourceLocator::resourceDir());
    LOG_DEBUG("Resource directory is " << resourceDir);

    // It should contain the file ml-en.dict
    ml::core::COsFileFuncs::TStat buf;
    CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::stat((resourceDir + "/ml-en.dict").c_str(), &buf));
}

void CResourceLocatorTest::testLogDir(void) {
    std::string logDir(ml::core::CResourceLocator::logDir());
    LOG_DEBUG("Log directory is " << logDir);

    // Don't assert on this as it will be non-essential once
    // we're an Elasticsearch plugin
}

void CResourceLocatorTest::testSrcRootDir(void) {
    std::string cppRootDir(ml::core::CResourceLocator::cppRootDir());
    LOG_DEBUG("C++ root directory is " << cppRootDir);

    // It should contain the file set_env.sh
    ml::core::COsFileFuncs::TStat buf;
    CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::stat((cppRootDir + "/set_env.sh").c_str(), &buf));
}
