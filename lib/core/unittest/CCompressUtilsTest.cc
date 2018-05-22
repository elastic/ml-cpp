/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CCompressUtilsTest.h"

#include <core/CCompressUtils.h>
#include <core/CLogger.h>

#include <string>

CppUnit::Test* CCompressUtilsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CCompressUtilsTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CCompressUtilsTest>(
        "CCompressUtilsTest::testEmptyAdd", &CCompressUtilsTest::testEmptyAdd));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCompressUtilsTest>(
        "CCompressUtilsTest::testOneAdd", &CCompressUtilsTest::testOneAdd));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCompressUtilsTest>(
        "CCompressUtilsTest::testManyAdds", &CCompressUtilsTest::testManyAdds));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCompressUtilsTest>(
        "CCompressUtilsTest::testLengthOnly", &CCompressUtilsTest::testLengthOnly));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCompressUtilsTest>(
        "CCompressUtilsTest::testInflate", &CCompressUtilsTest::testInflate));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCompressUtilsTest>(
        "CCompressUtilsTest::testTriviallyCopyableTypeVector",
        &CCompressUtilsTest::testTriviallyCopyableTypeVector));
    return suiteOfTests;
}

void CCompressUtilsTest::testEmptyAdd() {
    ml::core::CCompressUtils compressor(ml::core::CCompressUtils::E_Deflate, false);

    std::string str;

    CPPUNIT_ASSERT(compressor.addString(str));

    ml::core::CCompressUtils::TByteVec output;
    size_t length(0);

    CPPUNIT_ASSERT(compressor.data(true, output));
    CPPUNIT_ASSERT(compressor.length(true, length));

    LOG_INFO(<< "Length of nothing compressed is " << length);

    CPPUNIT_ASSERT(length > 0);
    CPPUNIT_ASSERT_EQUAL(length, output.size());
}

void CCompressUtilsTest::testOneAdd() {
    ml::core::CCompressUtils compressor(ml::core::CCompressUtils::E_Deflate, false);

    std::string str("1234567890");

    CPPUNIT_ASSERT(compressor.addString(str));

    ml::core::CCompressUtils::TByteVec output;
    size_t length(0);

    CPPUNIT_ASSERT(compressor.data(true, output));
    CPPUNIT_ASSERT(compressor.length(true, length));

    LOG_INFO(<< "Length of " << str << " compressed is " << length);

    CPPUNIT_ASSERT(length > 0);
    CPPUNIT_ASSERT_EQUAL(length, output.size());
}

void CCompressUtilsTest::testManyAdds() {
    ml::core::CCompressUtils compressorMulti(ml::core::CCompressUtils::E_Deflate, false);

    std::string str1("1234567890");
    std::string str2("qwertyuiopa1234sdfghjklzxcvbnm");
    std::string str3("!@£$%^&*()_+");

    CPPUNIT_ASSERT(compressorMulti.addString(str1));
    CPPUNIT_ASSERT(compressorMulti.addString(str2));
    CPPUNIT_ASSERT(compressorMulti.addString(str3));

    ml::core::CCompressUtils::TByteVec outputMulti;
    size_t lengthMulti(0);

    CPPUNIT_ASSERT(compressorMulti.data(true, outputMulti));
    CPPUNIT_ASSERT(compressorMulti.length(true, lengthMulti));

    LOG_INFO(<< "Length of " << str1 << str2 << str3 << " compressed is " << lengthMulti);

    CPPUNIT_ASSERT(lengthMulti > 0);
    CPPUNIT_ASSERT_EQUAL(lengthMulti, outputMulti.size());

    ml::core::CCompressUtils compressorSingle(ml::core::CCompressUtils::E_Deflate, false);

    CPPUNIT_ASSERT(compressorSingle.addString(str1 + str2 + str3));

    ml::core::CCompressUtils::TByteVec outputSingle;
    size_t lengthSingle(0);

    CPPUNIT_ASSERT(compressorSingle.data(true, outputSingle));
    CPPUNIT_ASSERT(compressorSingle.length(true, lengthSingle));

    CPPUNIT_ASSERT_EQUAL(lengthMulti, lengthSingle);
    CPPUNIT_ASSERT_EQUAL(lengthSingle, outputSingle.size());
    CPPUNIT_ASSERT(outputMulti == outputSingle);
}

void CCompressUtilsTest::testLengthOnly() {
    ml::core::CCompressUtils compressorFull(ml::core::CCompressUtils::E_Deflate, false);

    std::string str("qwertyuiopa1234sdfghjklzxcvbnm");

    CPPUNIT_ASSERT(compressorFull.addString(str));
    CPPUNIT_ASSERT(compressorFull.addString(str));
    CPPUNIT_ASSERT(compressorFull.addString(str));

    ml::core::CCompressUtils::TByteVec outputFull;
    size_t lengthFull(0);

    CPPUNIT_ASSERT(compressorFull.data(true, outputFull));
    CPPUNIT_ASSERT(compressorFull.length(true, lengthFull));

    LOG_INFO(<< "Length of " << str << str << str << " compressed is " << lengthFull);

    CPPUNIT_ASSERT(lengthFull > 0);
    CPPUNIT_ASSERT_EQUAL(lengthFull, outputFull.size());

    ml::core::CCompressUtils compressorLengthOnly(ml::core::CCompressUtils::E_Deflate, true);

    CPPUNIT_ASSERT(compressorLengthOnly.addString(str));
    CPPUNIT_ASSERT(compressorLengthOnly.addString(str));
    CPPUNIT_ASSERT(compressorLengthOnly.addString(str));

    ml::core::CCompressUtils::TByteVec outputLengthOnly;
    size_t lengthLengthOnly(0);

    // Should NOT be possible to get the full compressed data in this case
    CPPUNIT_ASSERT(!compressorLengthOnly.data(true, outputLengthOnly));
    CPPUNIT_ASSERT(compressorLengthOnly.length(true, lengthLengthOnly));

    CPPUNIT_ASSERT_EQUAL(lengthFull, lengthLengthOnly);
    CPPUNIT_ASSERT_EQUAL(size_t(0), outputLengthOnly.size());
}

void CCompressUtilsTest::testInflate() {
    ml::core::CCompressUtils compressor(ml::core::CCompressUtils::E_Deflate, false);

    std::string repeat("qwertyuiopa1234sdfghjklzxcvbnm");
    std::string input(repeat + repeat + repeat);
    LOG_DEBUG(<< "Input size = " << input.size());

    CPPUNIT_ASSERT(compressor.addString(input));

    ml::core::CCompressUtils::TByteVec compressed;
    CPPUNIT_ASSERT(compressor.data(true, compressed));
    LOG_DEBUG(<< "Compressed size = " << compressed.size());

    ml::core::CCompressUtils decompressor(ml::core::CCompressUtils::E_Inflate, false);

    CPPUNIT_ASSERT(decompressor.addVector(compressed));

    ml::core::CCompressUtils::TByteVec decompressed;
    CPPUNIT_ASSERT(decompressor.data(true, decompressed));
    LOG_DEBUG(<< "Decompressed size = " << decompressed.size());

    std::string output{decompressed.begin(), decompressed.end()};

    CPPUNIT_ASSERT_EQUAL(input, output);
}

void CCompressUtilsTest::testTriviallyCopyableTypeVector() {
    std::vector<double> empty;

    ml::core::CCompressUtils compressor(ml::core::CCompressUtils::E_Deflate, false);

    CPPUNIT_ASSERT(compressor.addVector(empty));

    ml::core::CCompressUtils::TByteVec compressed;
    CPPUNIT_ASSERT(!compressor.data(false, compressed));
    CPPUNIT_ASSERT(compressed.empty());

    std::vector<double> input{1.3, 272083891.1, 1.3902e-26, 1.3, 0.0};
    LOG_DEBUG("Input size = " << input.size() * sizeof(double));

    CPPUNIT_ASSERT(compressor.addVector(input));

    compressor.data(true, compressed);
    LOG_DEBUG("Compressed size = " << compressed.size());
    CPPUNIT_ASSERT(compressed.size() < input.size() * sizeof(double));

    ml::core::CCompressUtils decompressor(ml::core::CCompressUtils::E_Inflate, false);
    CPPUNIT_ASSERT(decompressor.addVector(compressed));

    ml::core::CCompressUtils::TByteVec uncompressed;
    CPPUNIT_ASSERT(decompressor.data(true, uncompressed));

    CPPUNIT_ASSERT(std::equal(uncompressed.begin(), uncompressed.end(),
                              reinterpret_cast<Bytef*>(input.data())));
}
