/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CAlignment.h>
#include <core/CLogger.h>

#include <boost/test/unit_test.hpp>

#include <array>
#include <vector>

// TODO: revert this when we upgrade to gcc 9.3
#ifdef __GNUC__
// gcc's alignas is unreliable prior to gcc 9.3, so use the old style attribute
#define ALIGNAS(x) __attribute__((aligned(x)))
#else
#define ALIGNAS(x) alignas(x)
#endif

BOOST_AUTO_TEST_SUITE(CAlignmentTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testMaxAlignment) {

    // Test some known alignments.

    ALIGNAS(32) const char addresses[64]{};
    for (std::size_t i = 0; i < 64; ++i) {
        if (i % 32 == 0) {
            BOOST_TEST_REQUIRE(core::CAlignment::maxAlignment(&addresses[i]) ==
                               core::CAlignment::E_Aligned32);
        } else if (i % 16 == 0) {
            BOOST_TEST_REQUIRE(core::CAlignment::maxAlignment(&addresses[i]) ==
                               core::CAlignment::E_Aligned16);
        } else if (i % 8 == 0) {
            BOOST_TEST_REQUIRE(core::CAlignment::maxAlignment(&addresses[i]) ==
                               core::CAlignment::E_Aligned8);
        } else {
            BOOST_TEST_REQUIRE(core::CAlignment::maxAlignment(&addresses[i]) ==
                               core::CAlignment::E_Unaligned);
        }
    }
}

BOOST_AUTO_TEST_CASE(testIsAligned) {

    // Test some known alignments.

    ALIGNAS(32) const char addresses[64]{};
    for (std::size_t i = 0; i < 64; ++i) {
        if (i % 32 == 0) {
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                &addresses[i], core::CAlignment::E_Aligned32));
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                &addresses[i], core::CAlignment::E_Aligned16));
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                &addresses[i], core::CAlignment::E_Aligned8));
        } else if (i % 16 == 0) {
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                                   &addresses[i], core::CAlignment::E_Aligned32) == false);
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                &addresses[i], core::CAlignment::E_Aligned16));
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                &addresses[i], core::CAlignment::E_Aligned8));
        } else if (i % 8 == 0) {
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                                   &addresses[i], core::CAlignment::E_Aligned32) == false);
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                                   &addresses[i], core::CAlignment::E_Aligned16) == false);
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                &addresses[i], core::CAlignment::E_Aligned8));
        } else {
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                                   &addresses[i], core::CAlignment::E_Aligned32) == false);
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                                   &addresses[i], core::CAlignment::E_Aligned16) == false);
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                                   &addresses[i], core::CAlignment::E_Aligned8) == false);
        }
    }
}

BOOST_AUTO_TEST_CASE(testNextAligned) {

    // Test that next aligned is the first position with the required alignment
    // after the current index.

    ALIGNAS(32) std::array<double, 8> addresses;

    for (std::size_t i = 0; i < 8; ++i) {
        std::size_t i32{core::CAlignment::nextAligned(addresses, i, core::CAlignment::E_Aligned32)};
        BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
            &addresses[i32], core::CAlignment::E_Aligned32));
        for (std::size_t j = i + 1; j < i32; ++j) {
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                                   &addresses[j], core::CAlignment::E_Aligned32) == false);
        }

        std::size_t i16{core::CAlignment::nextAligned(addresses, i, core::CAlignment::E_Aligned16)};
        BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
            &addresses[i16], core::CAlignment::E_Aligned16));
        for (std::size_t j = i + 1; j < i16; ++j) {
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                                   &addresses[j], core::CAlignment::E_Aligned16) == false);
        }

        std::size_t i8{core::CAlignment::nextAligned(addresses, i, core::CAlignment::E_Aligned8)};
        BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
            &addresses[i8], core::CAlignment::E_Aligned8));
        for (std::size_t j = i + 1; j < i8; ++j) {
            BOOST_TEST_REQUIRE(core::CAlignment::isAligned(
                                   &addresses[j], core::CAlignment::E_Aligned8) == false);
        }
    }
}

BOOST_AUTO_TEST_CASE(testRoundup) {

    // Test rounding up the size of a block of char objects generates the expected sizes.

    BOOST_TEST_REQUIRE(
        core::CAlignment::roundup<char>(core::CAlignment::E_Aligned32, 0) == 0);
    BOOST_TEST_REQUIRE(
        core::CAlignment::roundup<char>(core::CAlignment::E_Aligned16, 0) == 0);
    BOOST_TEST_REQUIRE(core::CAlignment::roundup<char>(core::CAlignment::E_Aligned8, 0) == 0);
    BOOST_TEST_REQUIRE(
        core::CAlignment::roundup<char>(core::CAlignment::E_Unaligned, 0) == 0);
    for (std::size_t i = 1; i < 128; ++i) {
        BOOST_TEST_REQUIRE(core::CAlignment::roundup<char>(core::CAlignment::E_Aligned32,
                                                           i) == 32 * ((i + 31) / 32));
        BOOST_TEST_REQUIRE(core::CAlignment::roundup<char>(core::CAlignment::E_Aligned16,
                                                           i) == 16 * ((i + 15) / 16));
        BOOST_TEST_REQUIRE(core::CAlignment::roundup<char>(core::CAlignment::E_Aligned8,
                                                           i) == 8 * ((i + 7) / 8));
        BOOST_TEST_REQUIRE(
            core::CAlignment::roundup<char>(core::CAlignment::E_Unaligned, i) == i);
    }
}

BOOST_AUTO_TEST_CASE(testRoundupSizeof) {

    // Test rounding up the size of a block of float objects generates the expected memory.

    BOOST_TEST_REQUIRE(core::CAlignment::roundupSizeof<float>(
                           core::CAlignment::E_Aligned32, 0) == 0);
    BOOST_TEST_REQUIRE(core::CAlignment::roundupSizeof<float>(
                           core::CAlignment::E_Aligned16, 0) == 0);
    BOOST_TEST_REQUIRE(core::CAlignment::roundupSizeof<float>(
                           core::CAlignment::E_Aligned8, 0) == 0);
    BOOST_TEST_REQUIRE(core::CAlignment::roundupSizeof<float>(
                           core::CAlignment::E_Unaligned, 0) == 0);
    for (std::size_t i = 1; i < 32; ++i) {
        BOOST_TEST_REQUIRE(
            core::CAlignment::roundupSizeof<float>(core::CAlignment::E_Aligned32, i) ==
            32 * ((4 * i + 31) / 32));
        BOOST_TEST_REQUIRE(
            core::CAlignment::roundupSizeof<float>(core::CAlignment::E_Aligned16, i) ==
            16 * ((4 * i + 15) / 16));
        BOOST_TEST_REQUIRE(core::CAlignment::roundupSizeof<float>(
                               core::CAlignment::E_Aligned8, i) == 8 * ((4 * i + 7) / 8));
        BOOST_TEST_REQUIRE(core::CAlignment::roundupSizeof<float>(
                               core::CAlignment::E_Unaligned, i) == 4 * i);
    }
}

BOOST_AUTO_TEST_CASE(testAlignedAllocator) {

    core::CAlignedAllocator<double> allocator;

    std::vector<double*> addresses;

    bool aligned32{true};
    for (std::size_t i = 0; i < 20; ++i) {
        double* address{allocator.allocate(6)};
        addresses.push_back(address);
        aligned32 = aligned32 &&
                    core::CAlignment::isAligned(address, core::CAlignment::E_Aligned32);
    }
    for (auto& address : addresses) {
        allocator.deallocate(address, 6);
    }
    BOOST_TEST_REQUIRE(aligned32);
}

BOOST_AUTO_TEST_SUITE_END()
