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
#include "CFileDeleterTest.h"

#include <core/CFileDeleter.h>
#include <core/COsFileFuncs.h>

#include <fstream>
#include <string>

#include <errno.h>

CppUnit::Test* CFileDeleterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CFileDeleterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CFileDeleterTest>("CFileDeleterTest::testDelete", &CFileDeleterTest::testDelete));

    return suiteOfTests;
}

void CFileDeleterTest::testDelete() {
    std::string fileName("CFileDeleterTest.txt");

    {
        ml::core::CFileDeleter deleter(fileName);

        {
            std::ofstream testFile(fileName.c_str());
            testFile << "to be deleted" << std::endl;
        } // The file should exist by the time the stream is closed here

        CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::access(fileName.c_str(), ml::core::COsFileFuncs::EXISTS));
    } // The file should be deleted here

    CPPUNIT_ASSERT_EQUAL(-1, ml::core::COsFileFuncs::access(fileName.c_str(), ml::core::COsFileFuncs::EXISTS));

    CPPUNIT_ASSERT_EQUAL(ENOENT, errno);
}
