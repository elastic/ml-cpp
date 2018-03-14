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

#include "CIntegerToolsTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CIntegerTools.h>

#include <test/CRandomNumbers.h>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>

#include <math.h>

using namespace ml;

namespace {

typedef std::vector<std::size_t> TSizeVec;

std::string printBits(uint64_t x) {
    std::string result(64, '0');
    for (std::size_t i = 0u; i < 64; ++i, x >>= 1) {
        if (x & 0x1) {
            result[63 - i] = '1';
        }
    }
    return result;
}

}

void CIntegerToolsTest::testNextPow2(void) {
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CIntegerToolsTest::testNextPow2  |");
    LOG_DEBUG("+-----------------------------------+");

    CPPUNIT_ASSERT_EQUAL(std::size_t(0), maths::CIntegerTools::nextPow2(0));

    test::CRandomNumbers rng;

    for (std::size_t test = 1u, shift = 1u;
         test < (std::numeric_limits<std::size_t>::max() >> 1) + 1;
         test <<= 1, ++shift) {
        LOG_DEBUG("Testing shift = " << shift);

        // Edge cases.
        CPPUNIT_ASSERT_EQUAL(shift, maths::CIntegerTools::nextPow2(test));
        CPPUNIT_ASSERT_EQUAL(shift, maths::CIntegerTools::nextPow2((test << 1) - 1));

        TSizeVec offsets;
        rng.generateUniformSamples(0, test, 100, offsets);
        for (std::size_t i = 0u; i < offsets.size(); ++i) {
            CPPUNIT_ASSERT_EQUAL(shift, maths::CIntegerTools::nextPow2(test + offsets[i]));
        }
    }
}

void CIntegerToolsTest::testReverseBits(void) {
    LOG_DEBUG("+--------------------------------------+");
    LOG_DEBUG("|  CIntegerToolsTest::testReverseBits  |");
    LOG_DEBUG("+--------------------------------------+");

    test::CRandomNumbers rng;

    TSizeVec values;
    rng.generateUniformSamples(0, boost::numeric::bounds<std::size_t>::highest(), 10000, values);

    std::string expected;
    std::string actual;
    for (std::size_t i = 0u; i < values.size(); ++i) {
        uint64_t x = static_cast<uint64_t>(values[i]);
        expected = printBits(x);
        std::reverse(expected.begin(), expected.end());
        actual = printBits(maths::CIntegerTools::reverseBits(x));
        if (i % 500 == 0) {
            LOG_DEBUG("expected = " << expected);
            LOG_DEBUG("actual   = " << actual);
        }
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
}

void CIntegerToolsTest::testGcd(void) {
    LOG_DEBUG("+------------------------------+");
    LOG_DEBUG("|  CIntegerToolsTest::testGcd  |");
    LOG_DEBUG("+------------------------------+");

    // Construct a set of integers out of prime factors so we know
    // what the g.c.d. should be.

    std::size_t n1 = 3 * 17 * 29;
    std::size_t n2 = 5 * 17;
    LOG_DEBUG("gcd = " << maths::CIntegerTools::gcd(n2, n1));
    CPPUNIT_ASSERT_EQUAL(std::size_t(17), maths::CIntegerTools::gcd(n2, n1));

    n1 = 19 * 97;
    n2 = 5 * 7 * 97;
    LOG_DEBUG("gcd = " << maths::CIntegerTools::gcd(n1, n2));
    CPPUNIT_ASSERT_EQUAL(std::size_t(97), maths::CIntegerTools::gcd(n2, n1));

    test::CRandomNumbers rng;

    LOG_DEBUG("--- gcd(a, b) ---");
    std::size_t primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 29, 97, 821, 5851, 7877 };
    for (std::size_t i = 0u; i < 1000; ++i) {
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
        uint64_t bigGcd = 1;
        for (std::size_t j = 0u; j < cf.size(); ++j) {
            bigGcd *= primes[cf[j]];
        }

        uint64_t big1 = 1;
        for (std::size_t j = 0u; j < split[0]; ++j) {
            big1 *= primes[indices[j]];
        }
        uint64_t big2 = 1;
        for (std::size_t j = split[0]; j < indices.size(); ++j) {
            big2 *= primes[indices[j]];
        }
        LOG_DEBUG("big1 = " << big1
                            << ", big2 = " << big2
                            << " - expected gcd = " << bigGcd
                            << ", gcd = " << maths::CIntegerTools::gcd(big1, big2));
        CPPUNIT_ASSERT_EQUAL(bigGcd, maths::CIntegerTools::gcd(big1, big2));
    }

    LOG_DEBUG("--- gcd(a, b, c) ---");
    for (std::size_t i = 0u; i < 1000; ++i) {
        TSizeVec indices;
        rng.generateUniformSamples(0, 10, 9, indices);
        std::sort(indices.begin(), indices.begin() + 3);
        std::sort(indices.begin() + 3, indices.begin() + 6);
        std::sort(indices.begin() + 6, indices.end());

        TSizeVec cf;
        std::set_intersection(indices.begin(), indices.begin() + 3,
                              indices.begin() + 3, indices.begin() + 6,
                              std::back_inserter(cf));
        TSizeVec tmp;
        std::set_intersection(cf.begin(), cf.end(),
                              indices.begin() + 6, indices.end(),
                              std::back_inserter(tmp));
        cf.swap(tmp);
        std::size_t gcd = 1;
        for (std::size_t j = 0u; j < cf.size(); ++j) {
            gcd *= primes[cf[j]];
        }

        TSizeVec n(3, 1);
        for (std::size_t j = 0u; j < 3; ++j) {
            n[0] *= primes[indices[j]];
        }
        for (std::size_t j = 3; j < 6; ++j) {
            n[1] *= primes[indices[j]];
        }
        for (std::size_t j = 6; j < indices.size(); ++j) {
            n[2] *= primes[indices[j]];
        }
        LOG_DEBUG("n = " << core::CContainerPrinter::print(n)
                         << " - expected gcd = " << gcd
                         << ", gcd = " << maths::CIntegerTools::gcd(n));

    }

    LOG_DEBUG("--- gcd(a, b, c, d) ---");
    TSizeVec n(4, 1);
    n[0] = 17 * 19 * 29;
    n[1] = 19 * 97;
    n[2] = 17 * 19 * 83;
    n[3] = 17 * 19 * 29 * 83;
    LOG_DEBUG("n = " << core::CContainerPrinter::print(n)
                     << " - expected gcd = 19"
                     << ", gcd = " << maths::CIntegerTools::gcd(n));
    CPPUNIT_ASSERT_EQUAL(std::size_t(19), maths::CIntegerTools::gcd(n));
}

void CIntegerToolsTest::testBinomial(void) {
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CIntegerToolsTest::testBinomial  |");
    LOG_DEBUG("+-----------------------------------+");

    unsigned int n[] = { 1u, 2u, 5u, 7u, 10u };

    for (std::size_t i = 0u; i < boost::size(n); ++i) {
        for (unsigned int j = 0u; j <= n[i]; ++j) {
            LOG_DEBUG("j = " << j << ", n = " << n[i]
                             << ", (n j) = " << maths::CIntegerTools::binomial(n[i], j));

            double expected = ::exp(  boost::math::lgamma(static_cast<double>(n[i]+1))
                                      - boost::math::lgamma(static_cast<double>(n[i]-j+1))
                                      - boost::math::lgamma(static_cast<double>(j+1)));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, maths::CIntegerTools::binomial(n[i], j), 1e-10);
        }
    }
}

CppUnit::Test *CIntegerToolsTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CIntegerToolsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CIntegerToolsTest>(
                               "CIntegerToolsTest::testNextPow2",
                               &CIntegerToolsTest::testNextPow2) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CIntegerToolsTest>(
                               "CIntegerToolsTest::testReverseBits",
                               &CIntegerToolsTest::testReverseBits) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CIntegerToolsTest>(
                               "CIntegerToolsTest::testGcd",
                               &CIntegerToolsTest::testGcd) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CIntegerToolsTest>(
                               "CIntegerToolsTest::testBinomial",
                               &CIntegerToolsTest::testBinomial) );

    return suiteOfTests;
}
