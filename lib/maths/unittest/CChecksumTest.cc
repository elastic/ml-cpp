/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>

#include <test/CRandomNumbers.h>

#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <deque>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <vector>

BOOST_AUTO_TEST_SUITE(CChecksumTest)

using namespace ml;

namespace {

enum EAnEnum { E_1, E_2, E_3 };

struct SFoo {
    SFoo(uint64_t key) : s_Key(key) {}
    uint64_t checksum() const { return s_Key; }
    uint64_t s_Key;
};

struct SBar {
    SBar(uint64_t key) : s_Key(key) {}
    uint64_t checksum(uint64_t seed) const {
        return core::CHashing::hashCombine(seed, s_Key);
    }
    uint64_t s_Key;
};

using TIntVec = std::vector<int>;
using TSizeAnEnumMap = std::map<std::size_t, EAnEnum>;
using TStrSet = std::set<std::string>;
using TStrSetCItr = TStrSet::const_iterator;
using TOptionalDouble = boost::optional<double>;
using TOptionalDoubleVec = std::vector<TOptionalDouble>;
using TMeanVarAccumulator =
    maths::CBasicStatistics::SSampleMeanVar<maths::CFloatStorage>::TAccumulator;
using TMeanVarAccumulatorPtr = std::shared_ptr<TMeanVarAccumulator>;
using TDoubleMeanVarAccumulatorPr = std::pair<double, TMeanVarAccumulator>;
using TDoubleMeanVarAccumulatorPrList = std::list<TDoubleMeanVarAccumulatorPr>;
using TFooDeque = std::deque<SFoo>;
using TBarVec = std::vector<SBar>;
}

BOOST_AUTO_TEST_CASE(testMemberChecksum) {
    uint64_t seed = 1679023009937ull;

    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "*** test member functions ***");

    // Test that member functions are invoked.
    SFoo foo(100);
    LOG_DEBUG(<< "checksum foo = " << maths::CChecksum::calculate(seed, foo));
    BOOST_REQUIRE_EQUAL(maths::CChecksum::calculate(seed, foo),
                        core::CHashing::hashCombine(seed, foo.checksum()));
    SBar bar(200);
    LOG_DEBUG(<< "checksum bar = " << maths::CChecksum::calculate(seed, bar));
    BOOST_REQUIRE_EQUAL(maths::CChecksum::calculate(seed, bar), bar.checksum(seed));
}

BOOST_AUTO_TEST_CASE(testContainers) {
    uint64_t seed = 1679023009937ull;

    test::CRandomNumbers rng;

    // Test that containers are correctly hashed.
    //
    // We check that the checksum varies if the containers are modified
    // slightly, i.e. by changing an element value, permuting elements,
    // etc.
    {
        int values[] = {-1, 20, 10, 15, 2, 2};
        TIntVec a(std::begin(values), std::end(values));
        TIntVec b(std::begin(values), std::end(values));
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::CChecksum::calculate(seed, a),
                            maths::CChecksum::calculate(seed, b));
        b[2] = 3;
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
        b.assign(std::begin(values), std::end(values));
        rng.random_shuffle(b.begin(), b.end());
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
        b.assign(std::begin(values), std::end(values));
        b[b.size() - 1] = 3;
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
    }
    {
        TSizeAnEnumMap::value_type values[] = {TSizeAnEnumMap::value_type(-1, E_2),
                                               TSizeAnEnumMap::value_type(20, E_1),
                                               TSizeAnEnumMap::value_type(10, E_3),
                                               TSizeAnEnumMap::value_type(15, E_1),
                                               TSizeAnEnumMap::value_type(2, E_2),
                                               TSizeAnEnumMap::value_type(3, E_1)};
        TSizeAnEnumMap a(std::begin(values), std::end(values));
        TSizeAnEnumMap b(std::begin(values), std::end(values));
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::CChecksum::calculate(seed, a),
                            maths::CChecksum::calculate(seed, b));
        b[2] = E_1;
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
        b.clear();
        std::copy(std::begin(values), std::end(values), std::inserter(b, b.end()));
        b.erase(2);
        b[4] = E_2;
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
    }
    {
        std::string values[] = {"rain", "in", "spain"};
        TStrSet a(std::begin(values), std::end(values));
        uint64_t expected = seed;
        core::CHashing::CSafeMurmurHash2String64 hasher;
        for (TStrSetCItr itr = a.begin(); itr != a.end(); ++itr) {
            expected = core::CHashing::safeMurmurHash64(
                itr->data(), static_cast<int>(itr->size()), expected);
        }
        LOG_DEBUG(<< "checksum expected = " << expected);
        LOG_DEBUG(<< "checksum actual   = " << maths::CChecksum::calculate(seed, a));
        BOOST_REQUIRE_EQUAL(expected, maths::CChecksum::calculate(seed, a));
    }

    // Test that unordered containers are sorted.
    std::string keys[] = {"the", "quick", "brown", "fox"};
    double values[] = {5.6, 2.1, -3.0, 22.1};
    {
        boost::unordered_set<double> a;
        std::set<double> b;
        for (std::size_t i = 0; i < boost::size(values); ++i) {
            a.insert(values[i]);
            b.insert(values[i]);
        }

        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::CChecksum::calculate(seed, a),
                            maths::CChecksum::calculate(seed, b));
    }
    {
        boost::unordered_map<std::string, double> a;
        std::map<std::string, double> b;
        for (std::size_t i = 0; i < boost::size(keys); ++i) {
            a.insert(std::make_pair(keys[i], values[i]));
            b.insert(std::make_pair(keys[i], values[i]));
        }

        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::CChecksum::calculate(seed, a),
                            maths::CChecksum::calculate(seed, b));
    }
}

BOOST_AUTO_TEST_CASE(testNullable) {
    uint64_t seed = 1679023009937ull;

    // Test optional and pointers.
    //
    // Test null values are hashed and that the hash of non-null values
    // matches the raw object hash.
    {
        BOOST_REQUIRE_NO_THROW(maths::CChecksum::calculate(seed, TOptionalDouble()));
        BOOST_REQUIRE_NO_THROW(maths::CChecksum::calculate(seed, TMeanVarAccumulatorPtr()));
    }
    {
        double value(52.1);
        TOptionalDouble optional(value);
        LOG_DEBUG(<< "checksum expected = " << maths::CChecksum::calculate(seed, value));
        LOG_DEBUG(<< "checksum actual   = " << maths::CChecksum::calculate(seed, optional));
        BOOST_REQUIRE_EQUAL(maths::CChecksum::calculate(seed, value),
                            maths::CChecksum::calculate(seed, optional));
    }
    {
        TMeanVarAccumulator value;
        value.add(23489.0);
        value.add(34678.0);
        value.add(67846.0);
        value.add(67469.0);
        TMeanVarAccumulatorPtr pointer(new TMeanVarAccumulator(value));
        LOG_DEBUG(<< "checksum expected = " << maths::CChecksum::calculate(seed, value));
        LOG_DEBUG(<< "checksum actual   = " << maths::CChecksum::calculate(seed, pointer));
        BOOST_REQUIRE_EQUAL(maths::CChecksum::calculate(seed, value),
                            maths::CChecksum::calculate(seed, pointer));
    }
}

BOOST_AUTO_TEST_CASE(testAccumulators) {
    uint64_t seed = 1679023009937ull;

    // Test accumulators.
    {
        TMeanVarAccumulator value;
        value.add(234.0);
        value.add(378.0);
        value.add(653.0);
        LOG_DEBUG(<< "checksum expected = "
                  << core::CHashing::hashCombine(seed, value.checksum()));
        LOG_DEBUG(<< "checksum actual   = " << maths::CChecksum::calculate(seed, value));
        BOOST_REQUIRE_EQUAL(core::CHashing::hashCombine(seed, value.checksum()),
                            maths::CChecksum::calculate(seed, value));
    }
}

BOOST_AUTO_TEST_CASE(testPair) {
    uint64_t seed = 1679023009937ull;

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
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
        b = a;
        b.second.add(678629.0);
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));

        TDoubleMeanVarAccumulatorPrList collection;
        collection.push_back(a);
        collection.push_back(b);
        uint64_t expected = maths::CChecksum::calculate(seed, a);
        expected = maths::CChecksum::calculate(expected, b);
        LOG_DEBUG(<< "expected checksum = " << expected);
        LOG_DEBUG(<< "actual checksum   = " << maths::CChecksum::calculate(seed, collection));
        BOOST_REQUIRE_EQUAL(expected, maths::CChecksum::calculate(seed, collection));
    }
}

BOOST_AUTO_TEST_CASE(testArray) {
    uint64_t seed = 1679023009937ull;

    double a[] = {1.0, 23.8, 15.2, 14.7};
    double b[] = {1.0, 23.8, 15.2, 14.7};

    LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
    LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
    BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) ==
                       maths::CChecksum::calculate(seed, b));

    b[1] = 23.79;
    LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
    LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
    BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
}

BOOST_AUTO_TEST_CASE(testCombinations) {
    uint64_t seed = 1679023009937ull;

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "*** test containers and member functions ***");

    // Test that containers of classes with checksum functions are
    // correctly hashed.
    //
    // We check that the checksum varies if the containers are modified
    // slightly, i.e. by changing an element value, permuting elements,
    // etc.
    {
        SFoo values[] = {
            SFoo(static_cast<uint64_t>(-1)), SFoo(20), SFoo(10), SFoo(15), SFoo(2), SFoo(2)};
        TFooDeque a(std::begin(values), std::end(values));
        TFooDeque b(std::begin(values), std::end(values));
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::CChecksum::calculate(seed, a),
                            maths::CChecksum::calculate(seed, b));
        b[2] = SFoo(3);
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
        b.assign(std::begin(values), std::end(values));
        rng.random_shuffle(b.begin(), b.end());
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
        b.assign(std::begin(values), std::end(values));
        b[b.size() - 1] = 3;
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
    }
    {
        SBar values[] = {
            SBar(static_cast<uint64_t>(-1)), SBar(20), SBar(10), SBar(15), SBar(2), SBar(2)};
        TBarVec a(std::begin(values), std::end(values));
        TBarVec b(std::begin(values), std::end(values));
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_REQUIRE_EQUAL(maths::CChecksum::calculate(seed, a),
                            maths::CChecksum::calculate(seed, b));
        b[2] = SBar(3);
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
        b.assign(std::begin(values), std::end(values));
        rng.random_shuffle(b.begin(), b.end());
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
        b.assign(std::begin(values), std::end(values));
        b[b.size() - 1] = 3;
        LOG_DEBUG(<< "checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG(<< "checksum b = " << maths::CChecksum::calculate(seed, b));
        BOOST_TEST_REQUIRE(maths::CChecksum::calculate(seed, a) !=
                           maths::CChecksum::calculate(seed, b));
    }
}

BOOST_AUTO_TEST_SUITE_END()
