/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CProgNameTest.h"

#include <core/CLogger.h>
#include <core/CProgName.h>
#include <core/CRegex.h>


CppUnit::Test *CProgNameTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CProgNameTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CProgNameTest>(
                                   "CProgNameTest::testProgName",
                                   &CProgNameTest::testProgName) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CProgNameTest>(
                                   "CProgNameTest::testProgDir",
                                   &CProgNameTest::testProgDir) );

    return suiteOfTests;
}

void CProgNameTest::testProgName()
{
    std::string progName(ml::core::CProgName::progName());

    LOG_DEBUG("Current program name is " << progName);

    CPPUNIT_ASSERT_EQUAL(std::string("ml_test"), progName);
}

void CProgNameTest::testProgDir()
{
    std::string progDir(ml::core::CProgName::progDir());

    LOG_DEBUG("Current program directory is " << progDir);

    ml::core::CRegex expectedPathRegex;
    CPPUNIT_ASSERT(expectedPathRegex.init(".+[\\\\/]lib[\\\\/]core[\\\\/]unittest$"));
    CPPUNIT_ASSERT(expectedPathRegex.matches(progDir));

    // Confirm we've stripped any extended length indicator on Windows
    CPPUNIT_ASSERT(progDir.compare(0, 4, "\\\\?\\") != 0);
}

