/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPackedBitVector.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

BOOST_AUTO_TEST_SUITE(CPackedBitVectorTest)

using namespace ml;

using TBoolVec = std::vector<bool>;
using TBoolVecVec = std::vector<TBoolVec>;
using TSizeVec = std::vector<std::size_t>;
using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;

namespace {

std::string print(const core::CPackedBitVector& v) {
    std::ostringstream result;
    result << v;
    return result.str();
}

template<typename T>
std::string toBitString(const T& v) {
    std::ostringstream result;
    for (std::size_t i = 0; i < v.size(); ++i) {
        result << static_cast<int>(v[i]);
    }
    return result.str();
}

class CPackedBitVectorInternals : public core::CPackedBitVector {
public:
    static std::size_t maximumOneByteRunLength() {
        return MAXIMUM_ONE_BYTE_RUN_LENGTH;
    }
    static std::size_t maximumTwoByteRunLength() {
        return MAXIMUM_TWO_BYTE_RUN_LENGTH;
    }
    static std::size_t maximumThreeByteRunLength() {
        return MAXIMUM_THREE_BYTE_RUN_LENGTH;
    }
    static std::size_t maximumFourByteRunLength() {
        return MAXIMUM_FOUR_BYTE_RUN_LENGTH;
    }
    static void appendRun(std::size_t runLength, std::uint8_t& lastRunBytes, TUInt8Vec& runLengthBytes) {
        core::CPackedBitVector::appendRun(runLength, lastRunBytes, runLengthBytes);
    }
    static void extendLastRun(std::size_t runLength, std::uint8_t& runBytes, TUInt8Vec& runLengthBytes) {
        core::CPackedBitVector::extendLastRun(runLength, runBytes, runLengthBytes);
    }
    static std::uint8_t bytes(std::size_t runLength) {
        return core::CPackedBitVector::bytes(runLength);
    }
    static std::size_t readLastRunLength(std::uint8_t lastRunBytes,
                                         const TUInt8Vec& runLengthBytes) {
        return core::CPackedBitVector::readLastRunLength(lastRunBytes, runLengthBytes);
    }
    static std::size_t readRunLength(TUInt8VecCItr runLengthBytes) {
        return core::CPackedBitVector::readRunLength(runLengthBytes);
    }
    static std::size_t popRunLength(TUInt8VecCItr& runLengthBytes) {
        return core::CPackedBitVector::popRunLength(runLengthBytes);
    }
    static void writeRunLength(std::size_t runLength, TUInt8VecItr runLengthBytes) {
        core::CPackedBitVector::writeRunLength(runLength, runLengthBytes);
    }
};
}

BOOST_AUTO_TEST_CASE(testInternals) {

    using TUInt8Vec = std::vector<std::uint8_t>;

    LOG_DEBUG(<< "bytes");

    BOOST_TEST_REQUIRE(1, CPackedBitVectorInternals::bytes(1));
    BOOST_TEST_REQUIRE(1, CPackedBitVectorInternals::bytes(
                              CPackedBitVectorInternals::maximumOneByteRunLength()));
    BOOST_TEST_REQUIRE(2, CPackedBitVectorInternals::bytes(
                              CPackedBitVectorInternals::maximumOneByteRunLength() + 1));
    BOOST_TEST_REQUIRE(2, CPackedBitVectorInternals::bytes(
                              CPackedBitVectorInternals::maximumTwoByteRunLength()));
    BOOST_TEST_REQUIRE(3, CPackedBitVectorInternals::bytes(
                              CPackedBitVectorInternals::maximumTwoByteRunLength() + 1));
    BOOST_TEST_REQUIRE(2, CPackedBitVectorInternals::bytes(
                              CPackedBitVectorInternals::maximumThreeByteRunLength()));
    BOOST_TEST_REQUIRE(3, CPackedBitVectorInternals::bytes(
                              CPackedBitVectorInternals::maximumThreeByteRunLength() + 1));
    BOOST_TEST_REQUIRE(4, CPackedBitVectorInternals::bytes(
                              CPackedBitVectorInternals::maximumFourByteRunLength()));

    test::CRandomNumbers rng;
    TSizeVec ranges{1, 1 << 8, 1 << 16, 1 << 24, 1 << 30};
    TUInt8Vec runLengthBytes;
    TSizeVec runLengths;

    LOG_DEBUG(<< "read/writeRunLength");

    // We test some edge cases and then do a randomized test that write followed
    // by read is idempotent.

    runLengthBytes.resize(4);
    for (auto expectedRunLength :
         {std::size_t{1}, CPackedBitVectorInternals::maximumOneByteRunLength(),
          CPackedBitVectorInternals::maximumOneByteRunLength() + 1,
          CPackedBitVectorInternals::maximumTwoByteRunLength(),
          CPackedBitVectorInternals::maximumOneByteRunLength() + 1,
          CPackedBitVectorInternals::maximumThreeByteRunLength(),
          CPackedBitVectorInternals::maximumThreeByteRunLength() + 1,
          CPackedBitVectorInternals::maximumFourByteRunLength()}) {
        CPackedBitVectorInternals::writeRunLength(expectedRunLength,
                                                  runLengthBytes.begin());
        BOOST_REQUIRE_EQUAL(expectedRunLength, CPackedBitVectorInternals::readRunLength(
                                                   runLengthBytes.begin()));
    }

    for (std::size_t i = 0; i < 1000; ++i) {
        for (std::size_t j = 1; j < ranges.size(); ++j) {
            rng.generateUniformSamples(ranges[j - 1], ranges[j], 20, runLengths);
            for (auto expectedRunLength : runLengths) {
                CPackedBitVectorInternals::writeRunLength(expectedRunLength,
                                                          runLengthBytes.begin());
                BOOST_REQUIRE_EQUAL(expectedRunLength, CPackedBitVectorInternals::readRunLength(
                                                           runLengthBytes.begin()));
            }
        }
    }

    LOG_DEBUG(<< "readLastRunLength");

    // Randomized test that write followed by read is idempotent.

    for (std::size_t i = 0; i < 1000; ++i) {
        for (std::size_t j = 1; j < ranges.size(); ++j) {
            rng.generateUniformSamples(ranges[j - 1], ranges[j], 20, runLengths);
            for (auto expectedRunLength : runLengths) {
                std::uint8_t bytes{CPackedBitVectorInternals::bytes(expectedRunLength)};
                runLengthBytes.resize(bytes);
                CPackedBitVectorInternals::writeRunLength(expectedRunLength,
                                                          runLengthBytes.begin());
                BOOST_REQUIRE_EQUAL(expectedRunLength, CPackedBitVectorInternals::readLastRunLength(
                                                           bytes, runLengthBytes));
            }
        }
    }

    LOG_DEBUG(<< "appendRun");

    // Test we append the run length we actually added.

    for (std::size_t i = 0; i < 1000; ++i) {
        runLengthBytes.clear();
        for (std::size_t j = 1; j < ranges.size(); ++j) {
            rng.generateUniformSamples(ranges[j - 1], ranges[j], 1, runLengths);
            std::uint8_t lastRunBytes;
            CPackedBitVectorInternals::appendRun(runLengths[0], lastRunBytes, runLengthBytes);
            BOOST_REQUIRE_EQUAL(CPackedBitVectorInternals::bytes(runLengths[0]), lastRunBytes);
            BOOST_REQUIRE_EQUAL(runLengths[0], CPackedBitVectorInternals::readLastRunLength(
                                                   lastRunBytes, runLengthBytes));
        }
    }

    LOG_DEBUG(<< "extendRun");

    // Test when we extend a range we get the new range we expect.

    runLengthBytes.resize(4);

    for (std::size_t i = 0; i < 1000; ++i) {
        for (auto runLength :
             {CPackedBitVectorInternals::maximumOneByteRunLength(),
              CPackedBitVectorInternals::maximumTwoByteRunLength(),
              CPackedBitVectorInternals::maximumThreeByteRunLength(),
              CPackedBitVectorInternals::maximumFourByteRunLength() - 15}) {
            runLengthBytes.clear();
            for (int j = -10; j < 10; ++j) {
                for (int k = 1; k < 5; ++k) {
                    std::uint8_t lastRunBytes;
                    CPackedBitVectorInternals::appendRun(runLength + j, lastRunBytes,
                                                         runLengthBytes);
                    CPackedBitVectorInternals::extendLastRun(k, lastRunBytes, runLengthBytes);
                    BOOST_REQUIRE_EQUAL(runLength + j + k,
                                        CPackedBitVectorInternals::readLastRunLength(
                                            lastRunBytes, runLengthBytes));
                }
            }
        }
        for (std::size_t j = 1; j < ranges.size(); ++j) {
            rng.generateUniformSamples(ranges[j - 1], ranges[j], 1, runLengths);
            std::uint8_t lastRunBytes;
            CPackedBitVectorInternals::appendRun(runLengths[0], lastRunBytes, runLengthBytes);
            BOOST_REQUIRE_EQUAL(CPackedBitVectorInternals::bytes(runLengths[0]), lastRunBytes);
            BOOST_REQUIRE_EQUAL(runLengths[0], CPackedBitVectorInternals::readLastRunLength(
                                                   lastRunBytes, runLengthBytes));
        }
    }

    LOG_DEBUG(<< "popRunLength");

    // Test reading a sequence with pop run length.

    TSizeVec allRunLengths;
    for (std::size_t i = 0; i < 1000; ++i) {
        allRunLengths.clear();
        for (std::size_t j = 1; j < ranges.size(); ++j) {
            rng.generateUniformSamples(ranges[j - 1], ranges[j], 10, runLengths);
            allRunLengths.insert(allRunLengths.end(), runLengths.begin(),
                                 runLengths.end());
        }
        rng.random_shuffle(allRunLengths.begin(), allRunLengths.end());

        runLengthBytes.clear();
        for (auto runLength : allRunLengths) {
            std::uint8_t dummy;
            CPackedBitVectorInternals::appendRun(runLength, dummy, runLengthBytes);
        }

        auto byteItr = runLengthBytes.cbegin();
        for (std::size_t j = 0; j < allRunLengths.size(); ++j) {
            BOOST_TEST_REQUIRE(allRunLengths[j],
                               CPackedBitVectorInternals::popRunLength(byteItr));
        }
    }
}

BOOST_AUTO_TEST_CASE(testCreation) {
    core::CPackedBitVector test1(3, true);
    LOG_DEBUG(<< "test1 = " << test1);
    BOOST_REQUIRE_EQUAL(std::size_t(3), test1.dimension());
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(TBoolVec(3, true)),
                        core::CContainerPrinter::print(test1.toBitVector()));

    core::CPackedBitVector test2(5, false);
    LOG_DEBUG(<< "test2 = " << test2);
    BOOST_REQUIRE_EQUAL(std::size_t(5), test2.dimension());
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(TBoolVec(5, false)),
                        core::CContainerPrinter::print(test2.toBitVector()));

    core::CPackedBitVector test3(255, true);
    LOG_DEBUG(<< "test3 = " << test3);
    BOOST_REQUIRE_EQUAL(std::size_t(255), test3.dimension());
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(TBoolVec(255, true)),
                        core::CContainerPrinter::print(test3.toBitVector()));

    core::CPackedBitVector test4(279, true);
    LOG_DEBUG(<< "test4 = " << test4);
    BOOST_REQUIRE_EQUAL(std::size_t(279), test4.dimension());
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(TBoolVec(279, true)),
                        core::CContainerPrinter::print(test4.toBitVector()));

    core::CPackedBitVector test5(512, false);
    LOG_DEBUG(<< "test5 = " << test5);
    BOOST_REQUIRE_EQUAL(std::size_t(512), test5.dimension());
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(TBoolVec(512, false)),
                        core::CContainerPrinter::print(test5.toBitVector()));

    core::CPackedBitVector test6((TBoolVec()));
    LOG_DEBUG(<< "test6 = " << test6);
    BOOST_REQUIRE_EQUAL(std::size_t(0), test6.dimension());
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print((TBoolVec())),
                        core::CContainerPrinter::print(test6.toBitVector()));

    TBoolVec bits1{true, true};
    core::CPackedBitVector test7(bits1);
    LOG_DEBUG(<< "test7 = " << test7);
    BOOST_REQUIRE_EQUAL(bits1.size(), test7.dimension());
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(bits1),
                        core::CContainerPrinter::print(test7.toBitVector()));

    TBoolVec bits2{true,  false, false, true, true, false, false,
                   false, false, true,  true, true, true,  false};
    core::CPackedBitVector test8(bits2);
    LOG_DEBUG(<< "test8 = " << test8);
    BOOST_REQUIRE_EQUAL(bits2.size(), test8.dimension());
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(bits2),
                        core::CContainerPrinter::print(test8.toBitVector()));

    test::CRandomNumbers rng;

    TSizeVec components;
    for (std::size_t t = 0; t < 100; ++t) {
        rng.generateUniformSamples(0, 2, 30, components);
        TBoolVec bits3(components.begin(), components.end());
        core::CPackedBitVector test9(bits3);
        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "test9 = " << test9);
        }
        BOOST_REQUIRE_EQUAL(bits3.size(), test9.dimension());
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(bits3),
                            core::CContainerPrinter::print(test9.toBitVector()));
    }
}

BOOST_AUTO_TEST_CASE(testExtend) {
    core::CPackedBitVector test1;
    test1.extend(true);
    BOOST_REQUIRE_EQUAL(std::size_t(1), test1.dimension());
    BOOST_REQUIRE_EQUAL(std::string("[1]"), print(test1));

    test1.extend(true);
    BOOST_REQUIRE_EQUAL(std::size_t(2), test1.dimension());
    BOOST_REQUIRE_EQUAL(std::string("[1 1]"), print(test1));

    test1.extend(false);
    BOOST_REQUIRE_EQUAL(std::size_t(3), test1.dimension());
    BOOST_REQUIRE_EQUAL(std::string("[1 1 0]"), print(test1));

    test1.extend(false);
    BOOST_REQUIRE_EQUAL(std::size_t(4), test1.dimension());
    BOOST_REQUIRE_EQUAL(std::string("[1 1 0 0]"), print(test1));

    test1.extend(true);
    BOOST_REQUIRE_EQUAL(std::size_t(5), test1.dimension());
    BOOST_REQUIRE_EQUAL(std::string("[1 1 0 0 1]"), print(test1));

    for (auto runLength : {CPackedBitVectorInternals::maximumOneByteRunLength(),
                           CPackedBitVectorInternals::maximumTwoByteRunLength()}) {
        LOG_DEBUG(<< "run length = " << runLength);

        core::CPackedBitVector test2(runLength, true);
        test2.extend(true);
        BOOST_REQUIRE_EQUAL(runLength + 1, test2.dimension());
        TBoolVec bits1(runLength + 1, true);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(bits1),
                            core::CContainerPrinter::print(test2.toBitVector()));

        test2.extend(false);
        bits1.push_back(false);
        BOOST_REQUIRE_EQUAL(runLength + 2, test2.dimension());
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(bits1),
                            core::CContainerPrinter::print(test2.toBitVector()));

        core::CPackedBitVector test3(runLength + 1, true);
        test3.extend(false);
        BOOST_REQUIRE_EQUAL(runLength + 2, test3.dimension());
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(bits1),
                            core::CContainerPrinter::print(test3.toBitVector()));
    }

    LOG_DEBUG(<< "randomized");

    test::CRandomNumbers rng;

    TSizeVec components;
    rng.generateUniformSamples(0, 2, 1012, components);

    TBoolVec bits2;
    core::CPackedBitVector test4;

    for (std::size_t i = 0; i < components.size(); ++i) {
        bits2.push_back(components[i] > 0);
        test4.extend(components[i] > 0);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(bits2),
                            core::CContainerPrinter::print(test4.toBitVector()));
    }
}

BOOST_AUTO_TEST_CASE(testContract) {
    core::CPackedBitVector test1;
    test1.extend(true);
    test1.extend(true);
    test1.extend(false);
    test1.extend(true);
    std::string expected[] = {"[1 1 0 1]", "[1 0 1]", "[0 1]", "[1]"};
    for (const std::string* e = expected; test1.dimension() > 0; ++e) {
        BOOST_REQUIRE_EQUAL(*e, print(test1));
        test1.contract();
    }

    for (auto runLength : {CPackedBitVectorInternals::maximumOneByteRunLength() + 2,
                           CPackedBitVectorInternals::maximumTwoByteRunLength() + 2}) {
        LOG_DEBUG(<< "run length = " << runLength);
        TBoolVec bits1(runLength, true);
        bits1.push_back(false);
        bits1.push_back(true);
        bits1.push_back(false);
        core::CPackedBitVector test2(bits1);
        for (std::size_t i = 0; i < 10; ++i) {
            bits1.erase(bits1.begin());
            test2.contract();
            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(bits1),
                                core::CContainerPrinter::print(test2.toBitVector()));
        }
    }
}

BOOST_AUTO_TEST_CASE(testComparisonAndLess) {
    test::CRandomNumbers rng;

    TPackedBitVectorVec test;

    TSizeVec components;
    for (std::size_t t = 0; t < 20; ++t) {
        rng.generateUniformSamples(0, 2, 20, components);
        TBoolVec bits(components.begin(), components.end());
        test.push_back(core::CPackedBitVector(bits));
    }

    for (std::size_t i = 0; i < test.size(); ++i) {
        for (std::size_t j = i; j < test.size(); ++j) {
            BOOST_TEST_REQUIRE(
                (test[i] < test[j] || test[i] > test[j] || test[i] == test[j]));
        }
    }
}

BOOST_AUTO_TEST_CASE(testBitwiseComplement) {
    test::CRandomNumbers rng;

    TSizeVec components;
    for (std::size_t t = 0; t < 200; ++t) {
        rng.generateUniformSamples(0, 2, 350, components);
        TBoolVec bits(components.begin(), components.end());
        auto actualComplement = ~core::CPackedBitVector(bits);
        TBoolVec expectedComplement(bits.size());
        std::transform(bits.begin(), bits.end(), expectedComplement.begin(),
                       std::logical_not<bool>());
        BOOST_REQUIRE_EQUAL(toBitString(expectedComplement), toBitString(actualComplement));
    }
}

BOOST_AUTO_TEST_CASE(testBitwise) {
    std::size_t run{CPackedBitVectorInternals::maximumOneByteRunLength()};

    // Test some run length edge cases.
    {
        core::CPackedBitVector a;
        a.extend(false, run);
        a.extend(true, 2 * run);
        core::CPackedBitVector b;
        b.extend(true, run);
        b.extend(false, 2 * run);

        TBoolVec expectedBitwiseAnd(3 * run, false);
        BOOST_REQUIRE_EQUAL(toBitString(expectedBitwiseAnd), toBitString(a & b));

        TBoolVec expectedBitwiseOr(3 * run, true);
        BOOST_REQUIRE_EQUAL(toBitString(expectedBitwiseOr), toBitString(a | b));

        TBoolVec expectedBitwiseXor(3 * run, true);
        BOOST_REQUIRE_EQUAL(toBitString(expectedBitwiseXor), toBitString(a ^ b));
    }
    {
        core::CPackedBitVector a;
        a.extend(false, run);
        a.extend(true, 2 * run);
        core::CPackedBitVector b;
        b.extend(false, run);
        b.extend(true, 2 * run);

        TBoolVec expectedBitwiseAnd(run, false);
        expectedBitwiseAnd.resize(3 * run, true);
        BOOST_REQUIRE_EQUAL(toBitString(expectedBitwiseAnd), toBitString(a & b));

        TBoolVec expectedBitwiseOr(run, false);
        expectedBitwiseOr.resize(3 * run, true);
        BOOST_REQUIRE_EQUAL(toBitString(expectedBitwiseOr), toBitString(a | b));

        TBoolVec expectedBitwiseXor(3 * run, false);
        BOOST_REQUIRE_EQUAL(toBitString(expectedBitwiseXor), toBitString(a ^ b));
    }

    test::CRandomNumbers rng;

    TPackedBitVectorVec test;
    TBoolVecVec reference;

    TSizeVec components;
    for (std::size_t t = 0; t < 50; ++t) {
        rng.generateUniformSamples(0, 2, 500, components);
        TBoolVec bits(components.begin(), components.end());
        test.push_back(core::CPackedBitVector(bits));
        reference.push_back(std::move(bits));
    }

    TBoolVec expectedBitwiseAnd(500);
    TBoolVec expectedBitwiseOr(500);
    TBoolVec expectedBitwiseXor(500);

    for (std::size_t i = 0; i < test.size(); ++i) {
        for (std::size_t j = 0; j < test.size(); ++j) {
            core::CPackedBitVector actualBitwiseAnd{test[i] & test[j]};
            std::transform(reference[i].begin(), reference[i].end(),
                           reference[j].begin(), expectedBitwiseAnd.begin(),
                           std::logical_and<bool>());
            BOOST_REQUIRE_EQUAL(toBitString(expectedBitwiseAnd),
                                toBitString(actualBitwiseAnd));
            // Also check hidden state...
            BOOST_REQUIRE_EQUAL(core::CPackedBitVector{expectedBitwiseAnd}.checksum(),
                                actualBitwiseAnd.checksum());

            core::CPackedBitVector actualBitwiseOr{test[i] | test[j]};
            std::transform(reference[i].begin(), reference[i].end(),
                           reference[j].begin(), expectedBitwiseOr.begin(),
                           std::logical_or<bool>());
            BOOST_REQUIRE_EQUAL(toBitString(expectedBitwiseOr), toBitString(actualBitwiseOr));
            // Also check hidden state...
            BOOST_REQUIRE_EQUAL(core::CPackedBitVector{expectedBitwiseOr}.checksum(),
                                actualBitwiseOr.checksum());

            core::CPackedBitVector actualBitwiseXor{test[i] ^ test[j]};
            std::transform(reference[i].begin(), reference[i].end(),
                           reference[j].begin(), expectedBitwiseXor.begin(),
                           [](bool lhs, bool rhs) { return lhs ^ rhs; });
            BOOST_REQUIRE_EQUAL(toBitString(expectedBitwiseXor),
                                toBitString(actualBitwiseXor));
            // Also check hidden state...
            BOOST_REQUIRE_EQUAL(core::CPackedBitVector{expectedBitwiseXor}.checksum(),
                                actualBitwiseXor.checksum());
        }
    }
}

BOOST_AUTO_TEST_CASE(testOneBitIterators) {
    {
        // Empty.
        core::CPackedBitVector test;
        TSizeVec indices{test.beginOneBits(), test.endOneBits()};
        BOOST_TEST_REQUIRE(indices.empty());
    }
    {
        // All ones.
        core::CPackedBitVector test{2000, true};
        TSizeVec actualIndices{test.beginOneBits(), test.endOneBits()};
        TSizeVec expectedIndices(2000);
        std::iota(expectedIndices.begin(), expectedIndices.end(), 0);
        BOOST_TEST_REQUIRE(actualIndices == expectedIndices);
    }
    {
        // All zeros.
        core::CPackedBitVector test{1000, false};
        TSizeVec actualIndices{test.beginOneBits(), test.endOneBits()};
        BOOST_TEST_REQUIRE(actualIndices.empty());
    }
    // Test end edge cases.
    for (auto runLength : {CPackedBitVectorInternals::maximumOneByteRunLength(),
                           CPackedBitVectorInternals::maximumTwoByteRunLength()}) {

        core::CPackedBitVector test;
        test.extend(false, runLength);
        test.extend(true, 2 * runLength);

        TSizeVec actualIndices{test.beginOneBits(), test.endOneBits()};
        TSizeVec expectedIndices;
        for (std::size_t i = runLength; i < 3 * runLength; ++i) {
            expectedIndices.push_back(i);
        }
        BOOST_TEST_REQUIRE(actualIndices == expectedIndices);
    }

    test::CRandomNumbers rng;

    TSizeVec components;
    TSizeVec expectedIndices;
    TSizeVec actualIndices;
    for (std::size_t t = 0; t < 1000; ++t) {
        rng.generateUniformSamples(0, 2, 512, components);
        TBoolVec bits(components.begin(), components.end());
        core::CPackedBitVector test{bits};

        actualIndices.assign(test.beginOneBits(), test.endOneBits());

        expectedIndices.clear();
        for (std::size_t i = 0; i < bits.size(); ++i) {
            if (bits[i]) {
                expectedIndices.push_back(i);
            }
        }

        if (actualIndices != expectedIndices) {
            LOG_ERROR(<< "expected = " << core::CContainerPrinter::print(expectedIndices));
            LOG_ERROR(<< "actual   = " << core::CContainerPrinter::print(actualIndices));
        } else if (t % 200 == 0) {
            LOG_DEBUG(<< "indices = " << core::CContainerPrinter::print(expectedIndices));
        }
        BOOST_TEST_REQUIRE(actualIndices == expectedIndices);
    }
}

BOOST_AUTO_TEST_CASE(testInnerProductBitwiseAnd) {

    core::CPackedBitVector test1(10, true);
    core::CPackedBitVector test2(10, false);
    TBoolVec bits1{true,  true,  false, false, true,
                   false, false, false, true,  true};
    core::CPackedBitVector test3(bits1);
    TBoolVec bits2{false, false, true,  false, true,
                   false, false, false, false, false};
    core::CPackedBitVector test4(bits2);

    BOOST_REQUIRE_EQUAL(10.0, test1.inner(test1));
    BOOST_REQUIRE_EQUAL(0.0, test1.inner(test2));
    BOOST_REQUIRE_EQUAL(5.0, test1.inner(test3));
    BOOST_REQUIRE_EQUAL(2.0, test1.inner(test4));
    BOOST_REQUIRE_EQUAL(0.0, test2.inner(test2));
    BOOST_REQUIRE_EQUAL(0.0, test2.inner(test3));
    BOOST_REQUIRE_EQUAL(0.0, test2.inner(test4));
    BOOST_REQUIRE_EQUAL(5.0, test3.inner(test3));
    BOOST_REQUIRE_EQUAL(1.0, test3.inner(test4));
    BOOST_REQUIRE_EQUAL(2.0, test4.inner(test4));

    core::CPackedBitVector test5(570, true);
    test5.extend(false);
    test5.extend(false);
    test5.extend(true);
    core::CPackedBitVector test6(565, true);
    test6.extend(false);
    test6.extend(false);
    test6.extend(false);
    test6.extend(false);
    test6.extend(false);
    test6.extend(false);
    test6.extend(false);
    test6.extend(true);

    BOOST_REQUIRE_EQUAL(566.0, test5.inner(test6));

    test::CRandomNumbers rng;

    TPackedBitVectorVec test7;
    TBoolVecVec reference7;

    TSizeVec components;
    for (std::size_t t = 0; t < 50; ++t) {
        rng.generateUniformSamples(0, 2, 50, components);
        TBoolVec bits3(components.begin(), components.end());
        test7.push_back(core::CPackedBitVector(bits3));
        reference7.push_back(std::move(bits3));
    }

    for (std::size_t i = 0; i < test7.size(); ++i) {
        if (i % 10 == 0) {
            LOG_DEBUG(<< "Testing " << test7[i]);
        }
        for (std::size_t j = 0; j < test7.size(); ++j) {
            double expected{0.0};
            for (std::size_t k = 0; k < 50; ++k) {
                expected += (reference7[i][k] && reference7[j][k] ? 1.0 : 0.0);
            }
            BOOST_REQUIRE_EQUAL(expected, test7[i].inner(test7[j]));
        }
    }
}

BOOST_AUTO_TEST_CASE(testInnerProductBitwiseOr) {

    test::CRandomNumbers rng;

    TPackedBitVectorVec test;
    TBoolVecVec reference;

    TSizeVec components;
    for (std::size_t t = 0; t < 50; ++t) {
        rng.generateUniformSamples(0, 2, 50, components);
        TBoolVec bits(components.begin(), components.end());
        test.push_back(core::CPackedBitVector(bits));
        reference.push_back(std::move(bits));
    }

    for (std::size_t i = 0; i < test.size(); ++i) {
        for (std::size_t j = 0; j < test.size(); ++j) {
            double expected{0.0};
            for (std::size_t k = 0; k < 50; ++k) {
                expected += (reference[i][k] || reference[j][k] ? 1.0 : 0.0);
            }
            BOOST_REQUIRE_EQUAL(
                expected, test[i].inner(test[j], core::CPackedBitVector::E_OR));
            expected = 0.0;
            for (std::size_t k = 0; k < 50; ++k) {
                expected += (reference[i][k] != reference[j][k] ? 1.0 : 0.0);
            }
            BOOST_REQUIRE_EQUAL(
                expected, test[i].inner(test[j], core::CPackedBitVector::E_XOR));
        }
    }
}

BOOST_AUTO_TEST_CASE(testLineScanProblemCase) {

    // Test a problem case with line scan. We were accidentally inverting the bits
    // of the covector in bitwise operations if a run in both vectors stopped at the
    // same position but one of them was encoding a run of length greater than 255.
    // In this test the one bits of rhs are a subset of lhs and so we can assert that
    // the intersection should be equal to rhs and the exclusive or contains the one
    // bits of lhs which aren't the one bits of rhs.

    TBoolVec lhs_{
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
        1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0,
        1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
        1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1,
        0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
        1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0,
        1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0};

    TBoolVec rhs_{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};

    core::CPackedBitVector lhs{lhs_};
    core::CPackedBitVector rhs{rhs_};

    BOOST_REQUIRE_EQUAL(rhs.manhattan(), (lhs & rhs).manhattan());
    BOOST_REQUIRE_EQUAL(lhs.manhattan() - rhs.manhattan(), (lhs ^ rhs).manhattan());
}

BOOST_AUTO_TEST_CASE(testZeroLengthRunProblemCase) {

    test::CRandomNumbers rng;

    auto toBits = [](const TSizeVec& components) {
        TBoolVec result;
        for (auto component : components) {
            std::fill_n(std::back_inserter(result), 255, component);
        }
        return result;
    };

    TSizeVec components;

    for (std::size_t i = 0; i < 100; ++i) {
        TBoolVec bits[3];
        for (std::size_t j = 0; j < 3; ++j) {
            rng.generateUniformSamples(0, 2, 10, components);
            bits[j] = toBits(components);
        }

        core::CPackedBitVector test[3]{core::CPackedBitVector{bits[0]},
                                       core::CPackedBitVector{bits[1]},
                                       core::CPackedBitVector{bits[2]}};

        TBoolVec expected;
        TSizeVec expectedIndices;
        for (std::size_t j = 0; j < bits[0].size(); ++j) {
            expected.push_back((bits[0][j] || bits[1][j]) && bits[2][j]);
            if (expected.back()) {
                expectedIndices.push_back(j);
            }
        }
        core::CPackedBitVector actual{(test[0] | test[1]) & test[2]};
        BOOST_REQUIRE_EQUAL(toBitString(expected), toBitString(actual));

        TSizeVec actualIndices(actual.beginOneBits(), actual.endOneBits());
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedIndices),
                            core::CContainerPrinter::print(actualIndices));
    }
}

BOOST_AUTO_TEST_CASE(testCompression) {

    // Check the number of bytes we use per bit for a range of different one-bit
    // densities.

    using TDoubleVec = std::vector<double>;

    test::CRandomNumbers rng;

    double averageOverhead{0.0};

    for (auto density : {0.5, 0.1, 0.01, 0.001, 0.0002}) {

        std::size_t meanMemoryUsage{0};
        for (std::size_t i = 0; i < 100; ++i) {
            TDoubleVec u01;
            rng.generateUniformSamples(0.0, 1.0, 100000, u01);
            core::CPackedBitVector vector;
            bool parity{true};
            for (auto selector : u01) {
                parity = selector < density ? !parity : parity;
                vector.extend(parity);
            }
            meanMemoryUsage += vector.memoryUsage();
        }
        meanMemoryUsage /= 100;

        // Optimal encoding uses log2(E[run length]) bits per run. This gives
        // expected memory usage of log2(E[run length]) / E[run length] / 8
        // bytes per vector bit.

        double minimumMeanBytesPerBit{density * std::log2(1.0 / density) / 8.0};
        double meanBytesPerBit{static_cast<double>(meanMemoryUsage) / 100000.0};
        LOG_DEBUG(<< "minimum bytes per bit = " << minimumMeanBytesPerBit);
        LOG_DEBUG(<< "bytes per bit         = " << meanBytesPerBit);

        averageOverhead += meanBytesPerBit / minimumMeanBytesPerBit;
        BOOST_TEST_REQUIRE(meanBytesPerBit < 15.0 * minimumMeanBytesPerBit);
    }

    averageOverhead /= 4.0;

    LOG_DEBUG(<< "average overhead = " << averageOverhead);

    BOOST_TEST_REQUIRE(averageOverhead < 6.0);
}

BOOST_AUTO_TEST_CASE(testPersist) {
    TBoolVec bits{true,  true,  false, false, true,
                  false, false, false, true,  true};

    for (std::size_t t = 0; t < bits.size(); ++t) {
        core::CPackedBitVector origVector(bits);

        std::string origXml = origVector.toDelimited();
        LOG_DEBUG(<< "xml = " << origXml);

        core::CPackedBitVector restoredVector;
        restoredVector.fromDelimited(origXml);

        BOOST_REQUIRE_EQUAL(origVector.checksum(), restoredVector.checksum());
    }
}

BOOST_AUTO_TEST_SUITE_END()
