/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CIntegerTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <cmath>

BOOST_AUTO_TEST_SUITE(CIntegerToolsTest)

using namespace ml;

namespace {

using TSizeVec = std::vector<std::size_t>;

std::string printBits(std::uint64_t x) {
    std::string result(64, '0');
    for (std::size_t i = 0; i < 64; ++i, x >>= 1) {
        if (x & 0x1) {
            result[63 - i] = '1';
        }
    }
    return result;
}
}

BOOST_AUTO_TEST_CASE(testNextPow2) {
    BOOST_REQUIRE_EQUAL(std::size_t(0), maths::CIntegerTools::nextPow2(0));

    test::CRandomNumbers rng;

    for (std::size_t test = 1u, shift = 1;
         test < (std::numeric_limits<std::size_t>::max() >> 1) + 1; test <<= 1, ++shift) {
        LOG_DEBUG(<< "Testing shift = " << shift);

        // Edge cases.
        BOOST_REQUIRE_EQUAL(shift, maths::CIntegerTools::nextPow2(test));
        BOOST_REQUIRE_EQUAL(shift, maths::CIntegerTools::nextPow2((test << 1) - 1));

        TSizeVec offsets;
        rng.generateUniformSamples(0, test, 100, offsets);
        for (std::size_t i = 0; i < offsets.size(); ++i) {
            BOOST_REQUIRE_EQUAL(shift, maths::CIntegerTools::nextPow2(test + offsets[i]));
        }
    }
}

BOOST_AUTO_TEST_CASE(testReverseBits) {
    test::CRandomNumbers rng;

    TSizeVec values;
    rng.generateUniformSamples(0, boost::numeric::bounds<std::size_t>::highest(),
                               10000, values);

    std::string expected;
    std::string actual;
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint64_t x = static_cast<std::uint64_t>(values[i]);
        expected = printBits(x);
        std::reverse(expected.begin(), expected.end());
        actual = printBits(maths::CIntegerTools::reverseBits(x));
        if (i % 500 == 0) {
            LOG_DEBUG(<< "expected = " << expected);
            LOG_DEBUG(<< "actual   = " << actual);
        }
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
}

BOOST_AUTO_TEST_CASE(testGcd) {
    // Construct a set of integers out of prime factors so we know
    // what the g.c.d. should be.

    std::size_t n1 = 3 * 17 * 29;
    std::size_t n2 = 5 * 17;
    LOG_DEBUG(<< "gcd = " << maths::CIntegerTools::gcd(n2, n1));
    BOOST_REQUIRE_EQUAL(std::size_t(17), maths::CIntegerTools::gcd(n2, n1));

    n1 = 19 * 97;
    n2 = 5 * 7 * 97;
    LOG_DEBUG(<< "gcd = " << maths::CIntegerTools::gcd(n1, n2));
    BOOST_REQUIRE_EQUAL(std::size_t(97), maths::CIntegerTools::gcd(n2, n1));

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "--- gcd(a, b) ---");
    std::size_t primes[] = {2,  3,  5,  7,   11,   13,  17,
                            19, 29, 97, 821, 5851, 7877};
    for (std::size_t i = 0; i < 1000; ++i) {
        TSizeVec indices;
        TSizeVec split;
        rng.generateUniformSamples(0, boost::size(primes), 7, indices);
        rng.generateUniformSamples(2, indices.size() - 2, 1, split);
        std::sort(indices.begin(), indices.begin() + split[0]);
        std::sort(indices.begin() + split[0], indices.end());

        TSizeVec cf;
        std::set_intersection(indices.begin(), indices.begin() + split[0],
                              indices.begin() + split[0], indices.end(),
                              std::back_inserter(cf));

        // Use 64 bit integers here otherwise overflow will occur in 32 bit code
        std::uint64_t bigGcd = 1;
        for (std::size_t j = 0; j < cf.size(); ++j) {
            bigGcd *= primes[cf[j]];
        }

        std::uint64_t big1 = 1;
        for (std::size_t j = 0; j < split[0]; ++j) {
            big1 *= primes[indices[j]];
        }
        std::uint64_t big2 = 1;
        for (std::size_t j = split[0]; j < indices.size(); ++j) {
            big2 *= primes[indices[j]];
        }
        LOG_TRACE(<< "big1 = " << big1 << ", big2 = " << big2 << " - expected gcd = " << bigGcd
                  << ", gcd = " << maths::CIntegerTools::gcd(big1, big2));
        BOOST_REQUIRE_EQUAL(bigGcd, maths::CIntegerTools::gcd(big1, big2));
    }

    LOG_DEBUG(<< "--- gcd(a, b, c) ---");
    for (std::size_t i = 0; i < 1000; ++i) {
        TSizeVec indices;
        rng.generateUniformSamples(0, 10, 9, indices);
        std::sort(indices.begin(), indices.begin() + 3);
        std::sort(indices.begin() + 3, indices.begin() + 6);
        std::sort(indices.begin() + 6, indices.end());

        TSizeVec cf;
        std::set_intersection(indices.begin(), indices.begin() + 3, indices.begin() + 3,
                              indices.begin() + 6, std::back_inserter(cf));
        TSizeVec tmp;
        std::set_intersection(cf.begin(), cf.end(), indices.begin() + 6,
                              indices.end(), std::back_inserter(tmp));
        cf.swap(tmp);
        std::size_t gcd = 1;
        for (std::size_t j = 0; j < cf.size(); ++j) {
            gcd *= primes[cf[j]];
        }

        TSizeVec n(3, 1);
        for (std::size_t j = 0; j < 3; ++j) {
            n[0] *= primes[indices[j]];
        }
        for (std::size_t j = 3; j < 6; ++j) {
            n[1] *= primes[indices[j]];
        }
        for (std::size_t j = 6; j < indices.size(); ++j) {
            n[2] *= primes[indices[j]];
        }
        LOG_TRACE(<< "n = " << core::CContainerPrinter::print(n) << " - expected gcd = "
                  << gcd << ", gcd = " << maths::CIntegerTools::gcd(n));
    }

    LOG_DEBUG(<< "--- gcd(a, b, c, d) ---");
    TSizeVec n(4, 1);
    n[0] = 17 * 19 * 29;
    n[1] = 19 * 97;
    n[2] = 17 * 19 * 83;
    n[3] = 17 * 19 * 29 * 83;
    LOG_DEBUG(<< "n = " << core::CContainerPrinter::print(n) << " - expected gcd = 19"
              << ", gcd = " << maths::CIntegerTools::gcd(n));
    BOOST_REQUIRE_EQUAL(std::size_t(19), maths::CIntegerTools::gcd(n));
}

BOOST_AUTO_TEST_CASE(testLcm) {
    // Check that least common multiple is a multiple of its arguments and that
    // there is no smaller common multiple.

    test::CRandomNumbers rng;

    auto isMultiple = [](std::size_t i, const TSizeVec& integers) {
        for (auto j : integers) {
            if (i % j != 0) {
                return false;
            }
        }
        return true;
    };

    TSizeVec integers;

    BOOST_REQUIRE_EQUAL(0, maths::CIntegerTools::lcm(integers));
    integers.push_back(0);
    BOOST_REQUIRE_EQUAL(0, maths::CIntegerTools::lcm(integers));
    integers.push_back(0);
    BOOST_REQUIRE_EQUAL(0, maths::CIntegerTools::lcm(integers));

    integers.clear();
    integers.push_back(5);
    BOOST_REQUIRE_EQUAL(5, maths::CIntegerTools::lcm(integers));
    integers.push_back(0);
    BOOST_REQUIRE_EQUAL(0, maths::CIntegerTools::lcm(integers));

    for (std::size_t i = 0; i < 100; ++i) {
        rng.generateUniformSamples(1, 200, 2, integers);
        std::size_t lcm{maths::CIntegerTools::lcm(integers[0], integers[1])};
        BOOST_TEST_REQUIRE(isMultiple(lcm, integers));
        for (std::size_t j = std::max(integers[0], integers[1]); j < lcm; ++j) {
            BOOST_TEST_REQUIRE(isMultiple(j, integers) == false);
        }
    }

    for (std::size_t i = 0; i < 100; ++i) {
        rng.generateUniformSamples(1, 20, 5, integers);
        std::size_t lcm{maths::CIntegerTools::lcm(integers)};
        BOOST_TEST_REQUIRE(isMultiple(lcm, integers));
        for (std::size_t j = *std::max_element(integers.begin(), integers.end());
             j < lcm; ++j) {
            BOOST_TEST_REQUIRE(isMultiple(j, integers) == false);
        }
    }
}

BOOST_AUTO_TEST_CASE(testBinomial) {
    unsigned int n[] = {1u, 2u, 5u, 7u, 10u};

    for (std::size_t i = 0; i < boost::size(n); ++i) {
        for (unsigned int j = 0; j <= n[i]; ++j) {
            LOG_DEBUG(<< "j = " << j << ", n = " << n[i]
                      << ", (n j) = " << maths::CIntegerTools::binomial(n[i], j));

            double expected = std::exp(std::lgamma(static_cast<double>(n[i] + 1)) -
                                       std::lgamma(static_cast<double>(n[i] - j + 1)) -
                                       std::lgamma(static_cast<double>(j + 1)));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                expected, maths::CIntegerTools::binomial(n[i], j), 1e-10);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
