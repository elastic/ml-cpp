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
#include "CCompressUtilsTest.h"

#include <core/CLogger.h>

#include "../CCompressUtils.h"

using namespace ml;
using namespace domain_name_entropy;

CppUnit::Test* CCompressUtilsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CCompressUtilsTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CCompressUtilsTest>(
        "CCompressUtilsTest::testCompressString1", &CCompressUtilsTest::testCompressString1));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCompressUtilsTest>(
        "CCompressUtilsTest::testCompressString2", &CCompressUtilsTest::testCompressString2));
    return suiteOfTests;
}

void CCompressUtilsTest::testCompressString1(void) {
    CCompressUtils tester3;

    std::string str1("1234567890");
    std::string str2("qwertyuiopa1234sdfghjklzxcvbnm");
    std::string str3("!@£$%^&*()_+");

    CPPUNIT_ASSERT(tester3.compressString(false, str1));
    CPPUNIT_ASSERT(tester3.compressString(false, str2));
    CPPUNIT_ASSERT(tester3.compressString(true, str3));

    std::string compressed3;
    size_t length3;

    CPPUNIT_ASSERT(tester3.compressedString(true, compressed3));
    CPPUNIT_ASSERT(tester3.compressedStringLength(true, length3));

    CPPUNIT_ASSERT(compressed3.size() == length3);

    CCompressUtils tester1;

    std::string str = str1 + str2 + str3;

    // NOTE: the string gets bigger!
    //std::cout << str << " " << str.size() << " " << length3 << std::endl;

    CPPUNIT_ASSERT(tester1.compressString(true, str));

    std::string compressed1;
    size_t length1;

    CPPUNIT_ASSERT(tester1.compressedString(true, compressed1));
    CPPUNIT_ASSERT(tester1.compressedStringLength(true, length1));

    CPPUNIT_ASSERT_EQUAL(length1, length3);
    CPPUNIT_ASSERT_EQUAL(compressed1, compressed3);
}

void CCompressUtilsTest::testCompressString2(void) {
    CCompressUtils mTester;

    std::string str1("qwertyuiopa1234sdfghjklzxcvbnm");

    std::string str;

    for (int i = 0; i < 1000; ++i) {
        str += str1;
        CPPUNIT_ASSERT(mTester.compressString(false, str1));
    }

    std::string mCompressed;
    size_t mLength;

    CPPUNIT_ASSERT(mTester.compressedString(true, mCompressed));
    CPPUNIT_ASSERT(mTester.compressedStringLength(true, mLength));

    CPPUNIT_ASSERT(mCompressed.size() == mLength);

    CCompressUtils tester;

    std::string compressed;
    size_t length;

    CPPUNIT_ASSERT(tester.compressString(false, str));

    CPPUNIT_ASSERT(tester.compressedStringLength(true, length));
    CPPUNIT_ASSERT(tester.compressedString(true, compressed));

    CPPUNIT_ASSERT(compressed.size() == length);

    // The string must get smaller here
    CPPUNIT_ASSERT(length < str.size());

    // cout'ing this here will write a null string char to the end of the buffer
    //std::cout << str << std::endl;
    //std::cout << compressed << std::endl;
    //std::cout << length << std::endl;
}
