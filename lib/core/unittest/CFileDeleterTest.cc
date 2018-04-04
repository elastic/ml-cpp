/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CFileDeleterTest.h"

#include <core/CFileDeleter.h>
#include <core/COsFileFuncs.h>

#include <fstream>
#include <string>

#include <errno.h>


CppUnit::Test *CFileDeleterTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CFileDeleterTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CFileDeleterTest>(
                                   "CFileDeleterTest::testDelete",
                                   &CFileDeleterTest::testDelete) );

    return suiteOfTests;
}

void CFileDeleterTest::testDelete()
{
    std::string fileName("CFileDeleterTest.txt");

    {
        ml::core::CFileDeleter deleter(fileName);

        {
            std::ofstream testFile(fileName.c_str());
            testFile << "to be deleted" << std::endl;
        } // The file should exist by the time the stream is closed here

        CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::access(fileName.c_str(),
                                                                    ml::core::COsFileFuncs::EXISTS));
    } // The file should be deleted here

    CPPUNIT_ASSERT_EQUAL(-1, ml::core::COsFileFuncs::access(fileName.c_str(),
                                                                 ml::core::COsFileFuncs::EXISTS));

    CPPUNIT_ASSERT_EQUAL(ENOENT, errno);
}

