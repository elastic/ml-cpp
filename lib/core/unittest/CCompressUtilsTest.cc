/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CompressUtils.h>

#include <boost/test/unit_test.hpp>

#include <string>

BOOST_AUTO_TEST_SUITE(CCompressUtilsTest)

BOOST_AUTO_TEST_CASE(testEmptyAdd) {
    ml::core::CDeflator compressor(false);

    std::string str;

    BOOST_TEST(compressor.addString(str));

    ml::core::CCompressUtil::TByteVec output;
    size_t length(0);

    BOOST_TEST(compressor.data(true, output));
    BOOST_TEST(compressor.length(true, length));

    LOG_INFO(<< "Length of nothing compressed is " << length);

    BOOST_TEST(length > 0);
    BOOST_CHECK_EQUAL(length, output.size());
}

BOOST_AUTO_TEST_CASE(testOneAdd) {
    ml::core::CDeflator compressor(false);

    std::string str("1234567890");

    BOOST_TEST(compressor.addString(str));

    ml::core::CCompressUtil::TByteVec output;
    size_t length(0);

    BOOST_TEST(compressor.data(true, output));
    BOOST_TEST(compressor.length(true, length));

    LOG_INFO(<< "Length of " << str << " compressed is " << length);

    BOOST_TEST(length > 0);
    BOOST_CHECK_EQUAL(length, output.size());
}

BOOST_AUTO_TEST_CASE(testManyAdds) {
    ml::core::CDeflator compressorMulti(false);

    std::string str1("1234567890");
    std::string str2("qwertyuiopa1234sdfghjklzxcvbnm");
    std::string str3("!@£$%^&*()_+");

    BOOST_TEST(compressorMulti.addString(str1));
    BOOST_TEST(compressorMulti.addString(str2));
    BOOST_TEST(compressorMulti.addString(str3));

    ml::core::CCompressUtil::TByteVec outputMulti;
    size_t lengthMulti(0);

    BOOST_TEST(compressorMulti.data(true, outputMulti));
    BOOST_TEST(compressorMulti.length(true, lengthMulti));

    LOG_INFO(<< "Length of " << str1 << str2 << str3 << " compressed is " << lengthMulti);

    BOOST_TEST(lengthMulti > 0);
    BOOST_CHECK_EQUAL(lengthMulti, outputMulti.size());

    ml::core::CDeflator compressorSingle(false);

    BOOST_TEST(compressorSingle.addString(str1 + str2 + str3));

    ml::core::CCompressUtil::TByteVec outputSingle;
    size_t lengthSingle(0);

    BOOST_TEST(compressorSingle.data(true, outputSingle));
    BOOST_TEST(compressorSingle.length(true, lengthSingle));

    BOOST_CHECK_EQUAL(lengthMulti, lengthSingle);
    BOOST_CHECK_EQUAL(lengthSingle, outputSingle.size());
    BOOST_TEST(outputMulti == outputSingle);
}

BOOST_AUTO_TEST_CASE(testLengthOnly) {
    ml::core::CDeflator compressorFull(false);

    std::string str("qwertyuiopa1234sdfghjklzxcvbnm");

    BOOST_TEST(compressorFull.addString(str));
    BOOST_TEST(compressorFull.addString(str));
    BOOST_TEST(compressorFull.addString(str));

    ml::core::CCompressUtil::TByteVec outputFull;
    size_t lengthFull(0);

    BOOST_TEST(compressorFull.data(true, outputFull));
    BOOST_TEST(compressorFull.length(true, lengthFull));

    LOG_INFO(<< "Length of " << str << str << str << " compressed is " << lengthFull);

    BOOST_TEST(lengthFull > 0);
    BOOST_CHECK_EQUAL(lengthFull, outputFull.size());

    ml::core::CDeflator compressorLengthOnly(true);

    BOOST_TEST(compressorLengthOnly.addString(str));
    BOOST_TEST(compressorLengthOnly.addString(str));
    BOOST_TEST(compressorLengthOnly.addString(str));

    ml::core::CCompressUtil::TByteVec outputLengthOnly;
    size_t lengthLengthOnly(0);

    // Should NOT be possible to get the full compressed data in this case
    BOOST_TEST(!compressorLengthOnly.data(true, outputLengthOnly));
    BOOST_TEST(compressorLengthOnly.length(true, lengthLengthOnly));

    BOOST_CHECK_EQUAL(lengthFull, lengthLengthOnly);
    BOOST_CHECK_EQUAL(size_t(0), outputLengthOnly.size());
}

BOOST_AUTO_TEST_CASE(testInflate) {
    ml::core::CDeflator compressor(false);

    std::string repeat("qwertyuiopa1234sdfghjklzxcvbnm");
    std::string input(repeat + repeat + repeat);
    LOG_DEBUG(<< "Input size = " << input.size());

    BOOST_TEST(compressor.addString(input));

    ml::core::CCompressUtil::TByteVec compressed;
    BOOST_TEST(compressor.data(true, compressed));
    LOG_DEBUG(<< "Compressed size = " << compressed.size());

    ml::core::CInflator decompressor(false);

    BOOST_TEST(decompressor.addVector(compressed));

    ml::core::CCompressUtil::TByteVec decompressed;
    BOOST_TEST(decompressor.data(true, decompressed));
    LOG_DEBUG(<< "Decompressed size = " << decompressed.size());

    std::string output{decompressed.begin(), decompressed.end()};

    BOOST_CHECK_EQUAL(input, output);
}

BOOST_AUTO_TEST_CASE(testTriviallyCopyableTypeVector) {
    std::vector<double> empty;

    ml::core::CDeflator compressor(false);

    BOOST_TEST(compressor.addVector(empty));

    ml::core::CCompressUtil::TByteVec compressed;
    BOOST_TEST(!compressor.data(false, compressed));
    BOOST_TEST(compressed.empty());

    std::vector<double> input{1.3, 272083891.1, 1.3902e-26, 1.3, 0.0};
    LOG_DEBUG(<< "Input size = " << input.size() * sizeof(double));

    BOOST_TEST(compressor.addVector(input));

    compressor.data(true, compressed);
    LOG_DEBUG(<< "Compressed size = " << compressed.size());
    BOOST_TEST(compressed.size() < input.size() * sizeof(double));

    ml::core::CInflator decompressor(false);
    BOOST_TEST(decompressor.addVector(compressed));

    ml::core::CCompressUtil::TByteVec uncompressed;
    BOOST_TEST(decompressor.data(true, uncompressed));

    BOOST_TEST(std::equal(uncompressed.begin(), uncompressed.end(),
                          reinterpret_cast<Bytef*>(input.data())));
}

BOOST_AUTO_TEST_SUITE_END()
