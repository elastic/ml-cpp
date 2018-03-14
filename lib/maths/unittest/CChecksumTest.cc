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

#include "CChecksumTest.h"

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <deque>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <vector>

using namespace ml;

namespace {

enum EAnEnum { E_1, E_2, E_3 };

struct SFoo {
    SFoo(uint64_t key) : s_Key(key) {}
    uint64_t checksum(void) const { return s_Key; }
    uint64_t s_Key;
};

struct SBar {
    SBar(uint64_t key) : s_Key(key) {}
    uint64_t checksum(uint64_t seed) const { return core::CHashing::hashCombine(seed, s_Key); }
    uint64_t s_Key;
};

typedef std::vector<int> TIntVec;
typedef std::map<std::size_t, EAnEnum> TSizeAnEnumMap;
typedef std::set<std::string> TStrSet;
typedef TStrSet::const_iterator TStrSetCItr;
typedef boost::optional<double> TOptionalDouble;
typedef std::vector<TOptionalDouble> TOptionalDoubleVec;
typedef maths::CBasicStatistics::SSampleMeanVar<maths::CFloatStorage>::TAccumulator
    TMeanVarAccumulator;
typedef boost::shared_ptr<TMeanVarAccumulator> TMeanVarAccumulatorPtr;
typedef std::pair<double, TMeanVarAccumulator> TDoubleMeanVarAccumulatorPr;
typedef std::list<TDoubleMeanVarAccumulatorPr> TDoubleMeanVarAccumulatorPrList;
typedef std::deque<SFoo> TFooDeque;
typedef std::vector<SBar> TBarVec;
}

void CChecksumTest::testMemberChecksum(void) {
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CChecksumTest::testMemberChecksum  |");
    LOG_DEBUG("+-------------------------------------+");

    uint64_t seed = 1679023009937ull;

    LOG_DEBUG("");
    LOG_DEBUG("*** test member functions ***");

    // Test that member functions are invoked.
    SFoo foo(100);
    LOG_DEBUG("checksum foo = " << maths::CChecksum::calculate(seed, foo));
    CPPUNIT_ASSERT_EQUAL(maths::CChecksum::calculate(seed, foo),
                         core::CHashing::hashCombine(seed, foo.checksum()));
    SBar bar(200);
    LOG_DEBUG("checksum bar = " << maths::CChecksum::calculate(seed, bar));
    CPPUNIT_ASSERT_EQUAL(maths::CChecksum::calculate(seed, bar), bar.checksum(seed));
}

void CChecksumTest::testContainers(void) {
    LOG_DEBUG("+---------------------------------+");
    LOG_DEBUG("|  CChecksumTest::testContainers  |");
    LOG_DEBUG("+---------------------------------+");

    uint64_t seed = 1679023009937ull;

    test::CRandomNumbers rng;

    // Test that containers are correctly hashed.
    //
    // We check that the checksum varies if the containers are modified
    // slightly, i.e. by changing an element value, permuting elements,
    // etc.
    {
        int values[] = {-1, 20, 10, 15, 2, 2};
        TIntVec a(boost::begin(values), boost::end(values));
        TIntVec b(boost::begin(values), boost::end(values));
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT_EQUAL(maths::CChecksum::calculate(seed, a),
                             maths::CChecksum::calculate(seed, b));
        b[2] = 3;
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
        b.assign(boost::begin(values), boost::end(values));
        rng.random_shuffle(b.begin(), b.end());
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
        b.assign(boost::begin(values), boost::end(values));
        b[b.size() - 1] = 3;
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
    }
    {
        TSizeAnEnumMap::value_type values[] = {TSizeAnEnumMap::value_type(-1, E_2),
                                               TSizeAnEnumMap::value_type(20, E_1),
                                               TSizeAnEnumMap::value_type(10, E_3),
                                               TSizeAnEnumMap::value_type(15, E_1),
                                               TSizeAnEnumMap::value_type(2, E_2),
                                               TSizeAnEnumMap::value_type(3, E_1)};
        TSizeAnEnumMap a(boost::begin(values), boost::end(values));
        TSizeAnEnumMap b(boost::begin(values), boost::end(values));
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT_EQUAL(maths::CChecksum::calculate(seed, a),
                             maths::CChecksum::calculate(seed, b));
        b[2] = E_1;
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
        b.clear();
        std::copy(boost::begin(values), boost::end(values), std::inserter(b, b.end()));
        b.erase(2);
        b[4] = E_2;
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
    }
    {
        std::string values[] = {"rain", "in", "spain"};
        TStrSet a(boost::begin(values), boost::end(values));
        uint64_t expected = seed;
        core::CHashing::CSafeMurmurHash2String64 hasher;
        for (TStrSetCItr itr = a.begin(); itr != a.end(); ++itr) {
            expected = core::CHashing::safeMurmurHash64(itr->data(),
                                                        static_cast<int>(itr->size()),
                                                        expected);
        }
        LOG_DEBUG("checksum expected = " << expected);
        LOG_DEBUG("checksum actual   = " << maths::CChecksum::calculate(seed, a));
        CPPUNIT_ASSERT_EQUAL(expected, maths::CChecksum::calculate(seed, a));
    }

    // Test that unordered containers are sorted.
    std::string keys[] = {"the", "quick", "brown", "fox"};
    double values[] = {5.6, 2.1, -3.0, 22.1};
    {
        boost::unordered_set<double> a;
        std::set<double> b;
        for (std::size_t i = 0u; i < boost::size(values); ++i) {
            a.insert(values[i]);
            b.insert(values[i]);
        }

        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT_EQUAL(maths::CChecksum::calculate(seed, a),
                             maths::CChecksum::calculate(seed, b));
    }
    {
        boost::unordered_map<std::string, double> a;
        std::map<std::string, double> b;
        for (std::size_t i = 0u; i < boost::size(keys); ++i) {
            a.insert(std::make_pair(keys[i], values[i]));
            b.insert(std::make_pair(keys[i], values[i]));
        }

        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT_EQUAL(maths::CChecksum::calculate(seed, a),
                             maths::CChecksum::calculate(seed, b));
    }
}

void CChecksumTest::testNullable(void) {
    LOG_DEBUG("+-------------------------------+");
    LOG_DEBUG("|  CChecksumTest::testNullable  |");
    LOG_DEBUG("+-------------------------------+");

    uint64_t seed = 1679023009937ull;

    // Test optional and pointers.
    //
    // Test null values are hashed and that the hash of non-null values
    // matches the raw object hash.
    {
        CPPUNIT_ASSERT_NO_THROW(maths::CChecksum::calculate(seed, TOptionalDouble()));
        CPPUNIT_ASSERT_NO_THROW(maths::CChecksum::calculate(seed, TMeanVarAccumulatorPtr()));
    }
    {
        double value(52.1);
        TOptionalDouble optional(value);
        LOG_DEBUG("checksum expected = " << maths::CChecksum::calculate(seed, value));
        LOG_DEBUG("checksum actual   = " << maths::CChecksum::calculate(seed, optional));
        CPPUNIT_ASSERT_EQUAL(maths::CChecksum::calculate(seed, value),
                             maths::CChecksum::calculate(seed, optional));
    }
    {
        TMeanVarAccumulator value;
        value.add(23489.0);
        value.add(34678.0);
        value.add(67846.0);
        value.add(67469.0);
        TMeanVarAccumulatorPtr pointer(new TMeanVarAccumulator(value));
        LOG_DEBUG("checksum expected = " << maths::CChecksum::calculate(seed, value));
        LOG_DEBUG("checksum actual   = " << maths::CChecksum::calculate(seed, pointer));
        CPPUNIT_ASSERT_EQUAL(maths::CChecksum::calculate(seed, value),
                             maths::CChecksum::calculate(seed, pointer));
    }
}

void CChecksumTest::testAccumulators(void) {
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CChecksumTest::testAccumulators  |");
    LOG_DEBUG("+-----------------------------------+");

    uint64_t seed = 1679023009937ull;

    // Test accumulators.
    {
        TMeanVarAccumulator value;
        value.add(234.0);
        value.add(378.0);
        value.add(653.0);
        LOG_DEBUG("checksum expected = " << core::CHashing::hashCombine(seed, value.checksum()));
        LOG_DEBUG("checksum actual   = " << maths::CChecksum::calculate(seed, value));
        CPPUNIT_ASSERT_EQUAL(core::CHashing::hashCombine(seed, value.checksum()),
                             maths::CChecksum::calculate(seed, value));
    }
}

void CChecksumTest::testPair(void) {
    LOG_DEBUG("+---------------------------+");
    LOG_DEBUG("|  CChecksumTest::testPair  |");
    LOG_DEBUG("+---------------------------+");

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
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
        b = a;
        b.second.add(678629.0);
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));

        TDoubleMeanVarAccumulatorPrList collection;
        collection.push_back(a);
        collection.push_back(b);
        uint64_t expected = maths::CChecksum::calculate(seed, a);
        expected = maths::CChecksum::calculate(expected, b);
        LOG_DEBUG("expected checksum = " << expected);
        LOG_DEBUG("actual checksum   = " << maths::CChecksum::calculate(seed, collection));
        CPPUNIT_ASSERT_EQUAL(expected, maths::CChecksum::calculate(seed, collection));
    }
}

void CChecksumTest::testArray(void) {
    LOG_DEBUG("+----------------------------+");
    LOG_DEBUG("|  CChecksumTest::testArray  |");
    LOG_DEBUG("+----------------------------+");

    uint64_t seed = 1679023009937ull;

    double a[] = {1.0, 23.8, 15.2, 14.7};
    double b[] = {1.0, 23.8, 15.2, 14.7};

    LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
    LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
    CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) == maths::CChecksum::calculate(seed, b));

    b[1] = 23.79;
    LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
    LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
    CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) != maths::CChecksum::calculate(seed, b));
}

void CChecksumTest::testCombinations(void) {
    LOG_DEBUG("+-----------------------------------+");
    LOG_DEBUG("|  CChecksumTest::testCombinations  |");
    LOG_DEBUG("+-----------------------------------+");

    uint64_t seed = 1679023009937ull;

    test::CRandomNumbers rng;

    LOG_DEBUG("*** test containers and member functions ***");

    // Test that containers of classes with checksum functions are
    // correctly hashed.
    //
    // We check that the checksum varies if the containers are modified
    // slightly, i.e. by changing an element value, permuting elements,
    // etc.
    {
        SFoo values[] =
            {SFoo(static_cast<uint64_t>(-1)), SFoo(20), SFoo(10), SFoo(15), SFoo(2), SFoo(2)};
        TFooDeque a(boost::begin(values), boost::end(values));
        TFooDeque b(boost::begin(values), boost::end(values));
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT_EQUAL(maths::CChecksum::calculate(seed, a),
                             maths::CChecksum::calculate(seed, b));
        b[2] = SFoo(3);
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
        b.assign(boost::begin(values), boost::end(values));
        rng.random_shuffle(b.begin(), b.end());
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
        b.assign(boost::begin(values), boost::end(values));
        b[b.size() - 1] = 3;
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
    }
    {
        SBar values[] =
            {SBar(static_cast<uint64_t>(-1)), SBar(20), SBar(10), SBar(15), SBar(2), SBar(2)};
        TBarVec a(boost::begin(values), boost::end(values));
        TBarVec b(boost::begin(values), boost::end(values));
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT_EQUAL(maths::CChecksum::calculate(seed, a),
                             maths::CChecksum::calculate(seed, b));
        b[2] = SBar(3);
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
        b.assign(boost::begin(values), boost::end(values));
        rng.random_shuffle(b.begin(), b.end());
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
        b.assign(boost::begin(values), boost::end(values));
        b[b.size() - 1] = 3;
        LOG_DEBUG("checksum a = " << maths::CChecksum::calculate(seed, a));
        LOG_DEBUG("checksum b = " << maths::CChecksum::calculate(seed, b));
        CPPUNIT_ASSERT(maths::CChecksum::calculate(seed, a) !=
                       maths::CChecksum::calculate(seed, b));
    }
}

CppUnit::Test* CChecksumTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CChecksumTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CChecksumTest>("CChecksumTest::testMemberChecksum",
                                               &CChecksumTest::testMemberChecksum));
    suiteOfTests->addTest(new CppUnit::TestCaller<CChecksumTest>("CChecksumTest::testContainers",
                                                                 &CChecksumTest::testContainers));
    suiteOfTests->addTest(new CppUnit::TestCaller<CChecksumTest>("CChecksumTest::testNullable",
                                                                 &CChecksumTest::testNullable));
    suiteOfTests->addTest(new CppUnit::TestCaller<CChecksumTest>("CChecksumTest::testAccumulators",
                                                                 &CChecksumTest::testAccumulators));
    suiteOfTests->addTest(new CppUnit::TestCaller<CChecksumTest>("CChecksumTest::testPair",
                                                                 &CChecksumTest::testPair));
    suiteOfTests->addTest(new CppUnit::TestCaller<CChecksumTest>("CChecksumTest::testArray",
                                                                 &CChecksumTest::testArray));
    suiteOfTests->addTest(new CppUnit::TestCaller<CChecksumTest>("CChecksumTest::testCombinations",
                                                                 &CChecksumTest::testCombinations));

    return suiteOfTests;
}
