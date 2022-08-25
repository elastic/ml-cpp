/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CLogger.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CChecksum.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <deque>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

BOOST_AUTO_TEST_SUITE(CChecksumTest)

using namespace ml;

namespace {

enum EAnEnum { E_1, E_2, E_3 };

struct SFoo {
    explicit SFoo(std::uint64_t key) : s_Key(key) {}
    std::uint64_t checksum() const { return s_Key; }
    std::uint64_t s_Key;
};

struct SBar {
    explicit SBar(std::uint64_t key) : s_Key(key) {}
    std::uint64_t checksum(std::uint64_t seed) const {
        return core::CHashing::hashCombine(seed, s_Key);
    }
    std::uint64_t s_Key;
};

using TIntVec = std::vector<int>;
using TSizeAnEnumMap = std::map<std::size_t, EAnEnum>;
using TStrSet = std::set<std::string>;
using TStrSetCItr = TStrSet::const_iterator;
using TOptionalDouble = std::optional<double>;
using TOptionalDoubleVec = std::vector<TOptionalDouble>;
using TMeanVarAccumulator =
    maths::common::CBasicStatistics::SSampleMeanVar<maths::common::CFloatStorage>::TAccumulator;
using TMeanVarAccumulatorPtr = std::shared_ptr<TMeanVarAccumulator>;
using TDoubleMeanVarAccumulatorPr = std::pair<double, TMeanVarAccumulator>;
using TDoubleMeanVarAccumulatorPrList = std::list<TDoubleMeanVarAccumulatorPr>;
using TFooDeque = std::deque<SFoo>;
using TBarVec = std::vector<SBar>;

constexpr std::uint64_t seed = 1679023009937ULL;
}

BOOST_AUTO_TEST_CASE(testMemberChecksum) {

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "*** test member functions ***");

    // Test that member functions are invoked.
    SFoo foo(100);
    LOG_DEBUG(<< "checksum foo = " << maths::common::CChecksum::calculate(seed, foo));
    BOOST_REQUIRE_EQUAL(maths::common::CChecksum::calculate(seed, foo),
                        core::CHashing::hashCombine(seed, foo.checksum()));
    SBar bar(200);
    LOG_DEBUG(<< "checksum bar = " << maths::common::CChecksum::calculate(seed, bar));
    BOOST_REQUIRE_EQUAL(maths::common::CChecksum::calculate(seed, bar), bar.checksum(seed));
}

BOOST_AUTO_TEST_CASE(testContainers) {

    test::CRandomNumbers rng;

    // Test that containers are correctly hashed.
    //
    // We check that the checksum varies if the containers are modified
    // slightly, i.e. by changing an element value, permuting elements,
    // etc.
    {
        TIntVec a{-1, 20, 10, 15, 2, 2};
        TIntVec b{-1, 20, 10, 15, 2, 2};
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::common::CChecksum::calculate(seed, a),
                            maths::common::CChecksum::calculate(seed, b));
        b[2] = 3;
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
        b.assign(a.begin(), a.end());
        rng.random_shuffle(b.begin(), b.end());
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
        b.assign(a.begin(), a.end());
        b[b.size() - 1] = 3;
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
    }
    {
        TSizeAnEnumMap a{{-1, E_2}, {20, E_1}, {10, E_3},
                         {15, E_1}, {2, E_2},  {3, E_1}};
        TSizeAnEnumMap b{{-1, E_2}, {20, E_1}, {10, E_3},
                         {15, E_1}, {2, E_2},  {3, E_1}};
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::common::CChecksum::calculate(seed, a),
                            maths::common::CChecksum::calculate(seed, b));
        b[2] = E_1;
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
        b.clear();
        std::copy(a.begin(), a.end(), std::inserter(b, b.end()));
        b.erase(2);
        b[4] = E_2;
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
    }
    {
        TStrSet a{"rain", "in", "spain"};
        std::uint64_t expected = seed;
        core::CHashing::CSafeMurmurHash2String64 hasher;
        for (const auto& str : a) {
            expected = core::CHashing::safeMurmurHash64(
                str.data(), static_cast<int>(str.size()), expected);
        }
        LOG_DEBUG(<< "checksum expected = " << expected);
        LOG_DEBUG(<< "checksum actual   = "
                  << maths::common::CChecksum::calculate(seed, a));
        BOOST_REQUIRE_EQUAL(expected, maths::common::CChecksum::calculate(seed, a));
    }

    // Test that unordered containers are sorted.
    {
        boost::unordered_set<double> a{5.6, 2.1, -3.0, 22.1};
        std::set<double> b{5.6, 2.1, -3.0, 22.1};
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::common::CChecksum::calculate(seed, a),
                            maths::common::CChecksum::calculate(seed, b));
    }
    {
        boost::unordered_map<std::string, double> a{
            {"the", 5.6}, {"quick", 2.1}, {"brown", -3.0}, {"fox", 22.1}};
        std::map<std::string, double> b{
            {"the", 5.6}, {"quick", 2.1}, {"brown", -3.0}, {"fox", 22.1}};
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::common::CChecksum::calculate(seed, a),
                            maths::common::CChecksum::calculate(seed, b));
    }
}

BOOST_AUTO_TEST_CASE(testNullable) {

    // Test optional and pointers.
    //
    // Test null values are hashed and that the hash of non-null values
    // matches the raw object hash.
    {
        BOOST_REQUIRE_NO_THROW(maths::common::CChecksum::calculate(seed, TOptionalDouble()));
        BOOST_REQUIRE_NO_THROW(
            maths::common::CChecksum::calculate(seed, TMeanVarAccumulatorPtr()));
    }
    {
        double value(52.1);
        TOptionalDouble optional(value);
        LOG_DEBUG(<< "checksum expected = "
                  << maths::common::CChecksum::calculate(seed, value));
        LOG_DEBUG(<< "checksum actual   = "
                  << maths::common::CChecksum::calculate(seed, optional));
        BOOST_REQUIRE_EQUAL(maths::common::CChecksum::calculate(seed, value),
                            maths::common::CChecksum::calculate(seed, optional));
    }
    {
        TMeanVarAccumulator value;
        value.add(23489.0);
        value.add(34678.0);
        value.add(67846.0);
        value.add(67469.0);
        TMeanVarAccumulatorPtr pointer(new TMeanVarAccumulator(value));
        LOG_DEBUG(<< "checksum expected = "
                  << maths::common::CChecksum::calculate(seed, value));
        LOG_DEBUG(<< "checksum actual   = "
                  << maths::common::CChecksum::calculate(seed, pointer));
        BOOST_REQUIRE_EQUAL(maths::common::CChecksum::calculate(seed, value),
                            maths::common::CChecksum::calculate(seed, pointer));
    }
}

BOOST_AUTO_TEST_CASE(testAccumulators) {

    // Test accumulators.
    {
        TMeanVarAccumulator value;
        value.add(234.0);
        value.add(378.0);
        value.add(653.0);
        LOG_DEBUG(<< "checksum expected = "
                  << core::CHashing::hashCombine(seed, value.checksum()));
        LOG_DEBUG(<< "checksum actual   = "
                  << maths::common::CChecksum::calculate(seed, value));
        BOOST_REQUIRE_EQUAL(core::CHashing::hashCombine(seed, value.checksum()),
                            maths::common::CChecksum::calculate(seed, value));
    }
}

BOOST_AUTO_TEST_CASE(testPair) {

    // Test pair.
    {
        TDoubleMeanVarAccumulatorPr a;
        a.first = 4789.0;
        a.second.add(647840.0);
        a.second.add(6440.0);
        a.second.add(6440.0);
        a.second.add(64840.0);
        TDoubleMeanVarAccumulatorPr b;
        b.first = 4790.0;
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
        b = a;
        b.second.add(678629.0);
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));

        TDoubleMeanVarAccumulatorPrList collection;
        collection.push_back(a);
        collection.push_back(b);
        std::uint64_t expected = maths::common::CChecksum::calculate(seed, a);
        expected = maths::common::CChecksum::calculate(expected, b);
        LOG_DEBUG(<< "expected checksum = " << expected);
        LOG_DEBUG(<< "actual checksum   = "
                  << maths::common::CChecksum::calculate(seed, collection));
        BOOST_REQUIRE_EQUAL(expected, maths::common::CChecksum::calculate(seed, collection));
    }
}

BOOST_AUTO_TEST_CASE(testArray) {

    LOG_DEBUG(<< "Basic");
    {
        double a[]{1.0, 23.8, 15.2, 14.7};
        double b[]{1.0, 23.8, 15.2, 14.7};
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) ==
                           maths::common::CChecksum::calculate(seed, b));

        b[1] = 23.79;
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
    }

    LOG_DEBUG(<< "Compound");
    {
        double a[][2]{{1.0, 2.0}, {23.8, 3.0}, {15.2, 1.1}};
        std::uint64_t expected = maths::common::CChecksum::calculate(seed, a[0]);
        expected = maths::common::CChecksum::calculate(expected, a[1]);
        expected = maths::common::CChecksum::calculate(expected, a[2]);

        LOG_DEBUG(<< "expected checksum = " << expected);
        LOG_DEBUG(<< "actual checksum   = "
                  << maths::common::CChecksum::calculate(seed, a));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) == expected);
    }
}

BOOST_AUTO_TEST_CASE(testCombinations) {

    // Test that containers of classes with checksum functions are
    // correctly hashed.
    //
    // We check that the checksum varies if the containers are modified
    // slightly, i.e. by changing an element value, permuting elements,
    // etc.

    test::CRandomNumbers rng;
    {
        SFoo values[]{
            SFoo(static_cast<std::uint64_t>(-1)), SFoo(20), SFoo(10), SFoo(15), SFoo(2), SFoo(2)};
        TFooDeque a(std::begin(values), std::end(values));
        TFooDeque b(std::begin(values), std::end(values));
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::common::CChecksum::calculate(seed, a),
                            maths::common::CChecksum::calculate(seed, b));
        b[2] = SFoo(3);
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
        b.assign(std::begin(values), std::end(values));
        rng.random_shuffle(b.begin(), b.end());
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
        b.assign(std::begin(values), std::end(values));
        b[b.size() - 1] = SFoo(3);
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
    }
    {
        SBar values[]{
            SBar(static_cast<std::uint64_t>(-1)), SBar(20), SBar(10), SBar(15), SBar(2), SBar(2)};
        TBarVec a(std::begin(values), std::end(values));
        TBarVec b(std::begin(values), std::end(values));
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::common::CChecksum::calculate(seed, a),
                            maths::common::CChecksum::calculate(seed, b));
        b[2] = SBar(3);
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
        b.assign(std::begin(values), std::end(values));
        rng.random_shuffle(b.begin(), b.end());
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
        b.assign(std::begin(values), std::end(values));
        b[b.size() - 1] = SBar(3);
        LOG_DEBUG(<< "checksum a = " << maths::common::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::common::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::common::CChecksum::calculate(seed, a) !=
                           maths::common::CChecksum::calculate(seed, b));
    }
}

BOOST_AUTO_TEST_SUITE_END()
