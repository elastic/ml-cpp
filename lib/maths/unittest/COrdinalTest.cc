/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "COrdinalTest.h"

#include <core/CLogger.h>

#include <maths/COrdinal.h>

#include <test/CRandomNumbers.h>

#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>
#include <boost/unordered_set.hpp>

#include <iomanip>
#include <sstream>
#include <vector>

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;

template<typename T>
std::string precisePrint(T x) {
    std::ostringstream o;
    o << std::setprecision(18) << x;
    return o.str();
}
}

void COrdinalTest::testEqual() {
    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < 1000; ++i) {
        TDoubleVec sample;
        rng.generateUniformSamples(-10000.0, 10000.0, 1, sample);
        bool equal = maths::COrdinal(static_cast<int64_t>(sample[0])) ==
                     maths::COrdinal(static_cast<int64_t>(sample[0]));
        CPPUNIT_ASSERT_EQUAL(true, equal);
        equal = maths::COrdinal(static_cast<uint64_t>(sample[0])) ==
                maths::COrdinal(static_cast<uint64_t>(sample[0]));
        CPPUNIT_ASSERT_EQUAL(true, equal);
        equal = maths::COrdinal(sample[0]) == maths::COrdinal(sample[0]);
        CPPUNIT_ASSERT_EQUAL(true, equal);
        if (sample[0] >= 0.0) {
            equal = maths::COrdinal(static_cast<int64_t>(sample[0])) ==
                    maths::COrdinal(std::floor(sample[0]));
            CPPUNIT_ASSERT_EQUAL(true, equal);
            equal = maths::COrdinal(std::floor(sample[0])) ==
                    maths::COrdinal(static_cast<int64_t>(sample[0]));
            CPPUNIT_ASSERT_EQUAL(true, equal);
            equal = maths::COrdinal(static_cast<uint64_t>(sample[0])) ==
                    maths::COrdinal(std::floor(sample[0]));
            CPPUNIT_ASSERT_EQUAL(true, equal);
            equal = maths::COrdinal(std::floor(sample[0])) ==
                    maths::COrdinal(static_cast<uint64_t>(sample[0]));
            CPPUNIT_ASSERT_EQUAL(true, equal);
        } else {
            equal = maths::COrdinal(static_cast<int64_t>(sample[0])) ==
                    maths::COrdinal(std::ceil(sample[0]));
            CPPUNIT_ASSERT_EQUAL(true, equal);
            equal = maths::COrdinal(std::ceil(sample[0])) ==
                    maths::COrdinal(static_cast<int64_t>(sample[0]));
            CPPUNIT_ASSERT_EQUAL(true, equal);
        }
    }

    // Test doubles outside the integer range.
    double small = -1e37;
    double large = 1e23;
    CPPUNIT_ASSERT(maths::COrdinal(small) !=
                   maths::COrdinal(boost::numeric::bounds<int64_t>::lowest()));
    CPPUNIT_ASSERT(maths::COrdinal(large) !=
                   maths::COrdinal(boost::numeric::bounds<uint64_t>::highest()));
    CPPUNIT_ASSERT(maths::COrdinal(boost::numeric::bounds<int64_t>::lowest()) !=
                   maths::COrdinal(small));
    CPPUNIT_ASSERT(maths::COrdinal(boost::numeric::bounds<uint64_t>::highest()) !=
                   maths::COrdinal(large));

    // Check some integer values which can't be represented as doubles.
    maths::COrdinal s1[] = {maths::COrdinal(int64_t(-179809067369808278)),
                            maths::COrdinal(int64_t(-179809067369808277)),
                            maths::COrdinal(int64_t(569817345679111267)),
                            maths::COrdinal(int64_t(569817345679111268))};
    maths::COrdinal s2[] = {maths::COrdinal(uint64_t(569817345679111267)),
                            maths::COrdinal(uint64_t(569817345679111268))};
    for (std::size_t i = 0u; i < boost::size(s1); ++i) {
        LOG_DEBUG(<< s1[i] << " (as double " << precisePrint(s1[i].asDouble()) << ")");
        for (std::size_t j = 0u; j < i; ++j) {
            CPPUNIT_ASSERT(s1[i] != s1[j]);
        }
        CPPUNIT_ASSERT(s1[i] == s1[i]);
        CPPUNIT_ASSERT(s1[i] != maths::COrdinal(s1[i].asDouble()));
        for (std::size_t j = i + 1; j < boost::size(s1); ++j) {
            CPPUNIT_ASSERT(s1[i] != s1[j]);
        }
    }
    CPPUNIT_ASSERT(s1[2] != s2[1]);
    CPPUNIT_ASSERT(s1[3] != s2[0]);
    CPPUNIT_ASSERT(s2[0] != s1[3]);
    CPPUNIT_ASSERT(s2[1] != s1[0]);
}

void COrdinalTest::testLess() {
    test::CRandomNumbers rng;

    // Test some random orderings on integer types which don't overflow.

    for (std::size_t i = 0u; i < 1000; ++i) {
        TDoubleVec samples;
        rng.generateUniformSamples(-10000.0, 10000.0, 2, samples);
        bool less = static_cast<int64_t>(samples[0]) < static_cast<int64_t>(samples[1]);
        bool ordinalLess = maths::COrdinal(static_cast<int64_t>(samples[0])) <
                           maths::COrdinal(static_cast<int64_t>(samples[1]));
        CPPUNIT_ASSERT_EQUAL(less, ordinalLess);
        if (samples[0] >= 0.0) {
            less = static_cast<int64_t>(samples[0]) < static_cast<int64_t>(samples[1]);
            ordinalLess = maths::COrdinal(static_cast<uint64_t>(samples[0])) <
                          maths::COrdinal(static_cast<int64_t>(samples[1]));
        }
        if (samples[1] >= 0.0) {
            less = static_cast<int64_t>(samples[0]) < static_cast<int64_t>(samples[1]);
            ordinalLess = maths::COrdinal(static_cast<int64_t>(samples[0])) <
                          maths::COrdinal(static_cast<uint64_t>(samples[1]));
        }
        if (samples[0] >= 0.0 && samples[1] >= 0.0) {
            less = static_cast<uint64_t>(samples[0]) < static_cast<uint64_t>(samples[1]);
            ordinalLess = maths::COrdinal(static_cast<uint64_t>(samples[0])) <
                          maths::COrdinal(static_cast<uint64_t>(samples[1]));
        }
        less = static_cast<double>(static_cast<int64_t>(samples[0])) < samples[1];
        ordinalLess = maths::COrdinal(static_cast<int64_t>(samples[0])) <
                      maths::COrdinal(samples[1]);
        less = samples[0] < static_cast<double>(static_cast<int64_t>(samples[1]));
        ordinalLess = maths::COrdinal(samples[0]) <
                      maths::COrdinal(static_cast<int64_t>(samples[1]));
        less = samples[0] < samples[1];
        if (samples[0] >= 0.0) {
            less = static_cast<double>(static_cast<uint64_t>(samples[0])) < samples[1];
            ordinalLess = maths::COrdinal(static_cast<uint64_t>(samples[0])) <
                          maths::COrdinal(samples[1]);
        }
        if (samples[1] >= 0.0) {
            less = samples[0] < static_cast<double>(static_cast<uint64_t>(samples[1]));
            ordinalLess = maths::COrdinal(samples[0]) <
                          maths::COrdinal(static_cast<uint64_t>(samples[1]));
        }
        ordinalLess = maths::COrdinal(samples[0]) < maths::COrdinal(samples[1]);
        CPPUNIT_ASSERT_EQUAL(less, ordinalLess);
    }

    // Test doubles outside the integer range.
    double small = -1e37;
    double large = 1e23;
    CPPUNIT_ASSERT(maths::COrdinal(small) <
                   maths::COrdinal(boost::numeric::bounds<int64_t>::lowest()));
    CPPUNIT_ASSERT(!(maths::COrdinal(boost::numeric::bounds<int64_t>::lowest()) <
                     maths::COrdinal(small)));
    CPPUNIT_ASSERT(maths::COrdinal(large) >
                   maths::COrdinal(boost::numeric::bounds<int64_t>::highest()));
    CPPUNIT_ASSERT(!(maths::COrdinal(boost::numeric::bounds<int64_t>::highest()) >
                     maths::COrdinal(large)));
    CPPUNIT_ASSERT(maths::COrdinal(large) >
                   maths::COrdinal(boost::numeric::bounds<int64_t>::highest()));
    CPPUNIT_ASSERT(!(maths::COrdinal(boost::numeric::bounds<int64_t>::highest()) >
                     maths::COrdinal(large)));

    // Check some integer values which can't be represented as doubles.
    maths::COrdinal s1[] = {maths::COrdinal(int64_t(-179809067369808278)),
                            maths::COrdinal(int64_t(-179809067369808277)),
                            maths::COrdinal(int64_t(569817345679111267)),
                            maths::COrdinal(int64_t(569817345679111268))};
    maths::COrdinal s2[] = {maths::COrdinal(uint64_t(569817345679111267)),
                            maths::COrdinal(uint64_t(569817345679111268))};
    for (std::size_t i = 0u; i < boost::size(s1); ++i) {
        LOG_DEBUG(<< s1[i] << " (as double " << precisePrint(s1[i].asDouble()) << ")");
        for (std::size_t j = 0u; j < i; ++j) {
            CPPUNIT_ASSERT(!(s1[i] < s1[j]));
        }
        for (std::size_t j = i + 1; j < boost::size(s1); ++j) {
            CPPUNIT_ASSERT(s1[i] < s1[j]);
        }
    }

    CPPUNIT_ASSERT(s1[2] < s2[1]);
    CPPUNIT_ASSERT(!(s2[1] < s1[2]));
    CPPUNIT_ASSERT(s2[0] < s1[3]);
    CPPUNIT_ASSERT(!(s1[3] < s2[0]));
}

void COrdinalTest::testIsNan() {
    maths::COrdinal nan;
    CPPUNIT_ASSERT(nan.isNan());

    CPPUNIT_ASSERT_EQUAL(false, nan < nan);
    CPPUNIT_ASSERT_EQUAL(false, nan > nan);
    CPPUNIT_ASSERT_EQUAL(false, nan == nan);

    {
        maths::COrdinal vsmall(boost::numeric::bounds<int64_t>::lowest());
        maths::COrdinal vlarge(boost::numeric::bounds<int64_t>::highest());
        CPPUNIT_ASSERT_EQUAL(false, nan < vsmall);
        CPPUNIT_ASSERT_EQUAL(false, nan > vsmall);
        CPPUNIT_ASSERT_EQUAL(false, nan == vsmall);
        CPPUNIT_ASSERT_EQUAL(false, nan < vlarge);
        CPPUNIT_ASSERT_EQUAL(false, nan > vlarge);
        CPPUNIT_ASSERT_EQUAL(false, nan == vlarge);
    }
    {
        maths::COrdinal vsmall(boost::numeric::bounds<uint64_t>::lowest());
        maths::COrdinal vlarge(boost::numeric::bounds<uint64_t>::highest());
        CPPUNIT_ASSERT_EQUAL(false, nan < vsmall);
        CPPUNIT_ASSERT_EQUAL(false, nan > vsmall);
        CPPUNIT_ASSERT_EQUAL(false, nan == vsmall);
        CPPUNIT_ASSERT_EQUAL(false, nan < vlarge);
        CPPUNIT_ASSERT_EQUAL(false, nan > vlarge);
        CPPUNIT_ASSERT_EQUAL(false, nan == vlarge);
    }
    {
        maths::COrdinal vsmall(boost::numeric::bounds<double>::lowest());
        maths::COrdinal vlarge(boost::numeric::bounds<double>::highest());
        CPPUNIT_ASSERT_EQUAL(false, nan < vsmall);
        CPPUNIT_ASSERT_EQUAL(false, nan > vsmall);
        CPPUNIT_ASSERT_EQUAL(false, nan == vsmall);
        CPPUNIT_ASSERT_EQUAL(false, nan < vlarge);
        CPPUNIT_ASSERT_EQUAL(false, nan > vlarge);
        CPPUNIT_ASSERT_EQUAL(false, nan == vlarge);
    }
}

void COrdinalTest::testAsDouble() {
    // Check that double conversion is as expected.

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < 100; ++i) {
        TDoubleVec sample;
        rng.generateUniformSamples(-20000.0, 0.0, 1, sample);
        maths::COrdinal signedOrdinal(static_cast<int64_t>(sample[0]));
        CPPUNIT_ASSERT_EQUAL(std::ceil(sample[0]), signedOrdinal.asDouble());

        rng.generateUniformSamples(0.0, 20000.0, 1, sample);
        maths::COrdinal unsignedOrdinal(static_cast<uint64_t>(sample[0]));
        CPPUNIT_ASSERT_EQUAL(std::floor(sample[0]), unsignedOrdinal.asDouble());

        rng.generateUniformSamples(-1.0, 1.0, 1, sample);
        maths::COrdinal doubleOrdinal(sample[0]);
        CPPUNIT_ASSERT_EQUAL(sample[0], doubleOrdinal.asDouble());
    }

    // Check some integer values which can't be represented as doubles.

    int64_t s[] = {-179809067369808278, -179809067369808277, 569817345679111267,
                   569817345679111268};

    for (std::size_t i = 0u; i < boost::size(s); ++i) {
        maths::COrdinal o(s[i]);
        LOG_DEBUG(<< o << " (as double " << precisePrint(o.asDouble()) << ")");
        CPPUNIT_ASSERT_EQUAL(static_cast<double>(s[i]), o.asDouble());
    }
}

void COrdinalTest::testHash() {
    // Test that hashing works over the full range of the distinct types.

    using TSizeUSet = boost::unordered_set<std::size_t>;

    test::CRandomNumbers rng;

    TSizeUSet signedHashes;
    TSizeUSet unsignedHashes;
    TSizeUSet doubleHashes;

    for (std::size_t i = 0u; i < 100; ++i) {
        TDoubleVec sample;
        rng.generateUniformSamples(-20000.0, 0.0, 1, sample);
        maths::COrdinal signedOrdinal(static_cast<int64_t>(sample[0]));
        signedHashes.insert(signedOrdinal.hash());

        rng.generateUniformSamples(0.0, 20000.0, 1, sample);
        maths::COrdinal unsignedOrdinal(static_cast<uint64_t>(sample[0]));
        unsignedHashes.insert(unsignedOrdinal.hash());

        rng.generateUniformSamples(-1.0, 1.0, 1, sample);
        maths::COrdinal doubleOrdinal(sample[0]);
        doubleHashes.insert(doubleOrdinal.hash());
    }

    LOG_DEBUG(<< "# signed hashes   = " << signedHashes.size());
    LOG_DEBUG(<< "# unsigned hashes = " << unsignedHashes.size());
    LOG_DEBUG(<< "# double hashes   = " << doubleHashes.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(100), signedHashes.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(100), unsignedHashes.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(100), doubleHashes.size());
}

CppUnit::Test* COrdinalTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("COrdinalTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<COrdinalTest>(
        "COrdinalTest::testEqual", &COrdinalTest::testEqual));
    suiteOfTests->addTest(new CppUnit::TestCaller<COrdinalTest>(
        "COrdinalTest::testLess", &COrdinalTest::testLess));
    suiteOfTests->addTest(new CppUnit::TestCaller<COrdinalTest>(
        "COrdinalTest::testLess", &COrdinalTest::testLess));
    suiteOfTests->addTest(new CppUnit::TestCaller<COrdinalTest>(
        "COrdinalTest::testIsNan", &COrdinalTest::testIsNan));
    suiteOfTests->addTest(new CppUnit::TestCaller<COrdinalTest>(
        "COrdinalTest::testAsDouble", &COrdinalTest::testAsDouble));
    suiteOfTests->addTest(new CppUnit::TestCaller<COrdinalTest>(
        "COrdinalTest::testHash", &COrdinalTest::testHash));

    return suiteOfTests;
}
