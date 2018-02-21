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

#include <core/CCompressUtils.h>
#include <core/CLogger.h>

#include <string>


CppUnit::Test *CCompressUtilsTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CCompressUtilsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CCompressUtilsTest>(
                                   "CCompressUtilsTest::testEmptyAdd",
                                   &CCompressUtilsTest::testEmptyAdd) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCompressUtilsTest>(
                                   "CCompressUtilsTest::testOneAdd",
                                   &CCompressUtilsTest::testOneAdd) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCompressUtilsTest>(
                                   "CCompressUtilsTest::testManyAdds",
                                   &CCompressUtilsTest::testManyAdds) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCompressUtilsTest>(
                                   "CCompressUtilsTest::testLengthOnly",
                                   &CCompressUtilsTest::testLengthOnly) );
    return suiteOfTests;
}

void CCompressUtilsTest::testEmptyAdd(void)
{
    ml::core::CCompressUtils compressor(false);

    std::string str;

    CPPUNIT_ASSERT(compressor.addString(str));

    ml::core::CCompressUtils::TByteVec output;
    size_t length(0);

    CPPUNIT_ASSERT(compressor.compressedData(true, output));
    CPPUNIT_ASSERT(compressor.compressedLength(true, length));

    LOG_INFO("Length of nothing compressed is " << length);

    CPPUNIT_ASSERT(length > 0);
    CPPUNIT_ASSERT_EQUAL(length, output.size());
}

void CCompressUtilsTest::testOneAdd(void)
{
    ml::core::CCompressUtils compressor(false);

    std::string str("1234567890");

    CPPUNIT_ASSERT(compressor.addString(str));

    ml::core::CCompressUtils::TByteVec output;
    size_t length(0);

    CPPUNIT_ASSERT(compressor.compressedData(true, output));
    CPPUNIT_ASSERT(compressor.compressedLength(true, length));

    LOG_INFO("Length of " << str << " compressed is " << length);

    CPPUNIT_ASSERT(length > 0);
    CPPUNIT_ASSERT_EQUAL(length, output.size());
}

void CCompressUtilsTest::testManyAdds(void)
{
    ml::core::CCompressUtils compressorMulti(false);

    std::string str1("1234567890");
    std::string str2("qwertyuiopa1234sdfghjklzxcvbnm");
    std::string str3("!@£$%^&*()_+");

    CPPUNIT_ASSERT(compressorMulti.addString(str1));
    CPPUNIT_ASSERT(compressorMulti.addString(str2));
    CPPUNIT_ASSERT(compressorMulti.addString(str3));

    ml::core::CCompressUtils::TByteVec outputMulti;
    size_t lengthMulti(0);

    CPPUNIT_ASSERT(compressorMulti.compressedData(true, outputMulti));
    CPPUNIT_ASSERT(compressorMulti.compressedLength(true, lengthMulti));

    LOG_INFO("Length of " << str1 << str2 << str3 <<
             " compressed is " << lengthMulti);

    CPPUNIT_ASSERT(lengthMulti > 0);
    CPPUNIT_ASSERT_EQUAL(lengthMulti, outputMulti.size());

    ml::core::CCompressUtils compressorSingle(false);

    CPPUNIT_ASSERT(compressorSingle.addString(str1 + str2 + str3));

    ml::core::CCompressUtils::TByteVec outputSingle;
    size_t lengthSingle(0);

    CPPUNIT_ASSERT(compressorSingle.compressedData(true, outputSingle));
    CPPUNIT_ASSERT(compressorSingle.compressedLength(true, lengthSingle));

    CPPUNIT_ASSERT_EQUAL(lengthMulti, lengthSingle);
    CPPUNIT_ASSERT_EQUAL(lengthSingle, outputSingle.size());
    CPPUNIT_ASSERT(outputMulti == outputSingle);
}

void CCompressUtilsTest::testLengthOnly(void)
{
    ml::core::CCompressUtils compressorFull(false);

    std::string str("qwertyuiopa1234sdfghjklzxcvbnm");

    CPPUNIT_ASSERT(compressorFull.addString(str));
    CPPUNIT_ASSERT(compressorFull.addString(str));
    CPPUNIT_ASSERT(compressorFull.addString(str));

    ml::core::CCompressUtils::TByteVec outputFull;
    size_t lengthFull(0);

    CPPUNIT_ASSERT(compressorFull.compressedData(true, outputFull));
    CPPUNIT_ASSERT(compressorFull.compressedLength(true, lengthFull));

    LOG_INFO("Length of " << str << str << str <<
             " compressed is " << lengthFull);

    CPPUNIT_ASSERT(lengthFull > 0);
    CPPUNIT_ASSERT_EQUAL(lengthFull, outputFull.size());

    ml::core::CCompressUtils compressorLengthOnly(true);

    CPPUNIT_ASSERT(compressorLengthOnly.addString(str));
    CPPUNIT_ASSERT(compressorLengthOnly.addString(str));
    CPPUNIT_ASSERT(compressorLengthOnly.addString(str));

    ml::core::CCompressUtils::TByteVec outputLengthOnly;
    size_t lengthLengthOnly(0);

    // Should NOT be possible to get the full compressed data in this case
    CPPUNIT_ASSERT(!compressorLengthOnly.compressedData(true, outputLengthOnly));
    CPPUNIT_ASSERT(compressorLengthOnly.compressedLength(true, lengthLengthOnly));

    CPPUNIT_ASSERT_EQUAL(lengthFull, lengthLengthOnly);
    CPPUNIT_ASSERT_EQUAL(size_t(0), outputLengthOnly.size());
}

