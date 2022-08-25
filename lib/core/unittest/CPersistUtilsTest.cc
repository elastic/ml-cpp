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

#include <core/CContainerPrinter.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>

#include <test/CRandomNumbers.h>

#include <boost/circular_buffer.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

using TDoubleVec = std::vector<double>;
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDoubleVec::iterator)

BOOST_AUTO_TEST_SUITE(CPersistUtilsTest)

using namespace ml;

using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoubleMap = std::map<std::size_t, double>;
using TIntSet = std::set<int>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TStrIntUMap = boost::unordered_map<std::string, int>;
using TStrDoubleVecVecUMap = boost::unordered_map<std::string, TDoubleVecVec>;
using TSizeUSet = boost::unordered_set<std::size_t>;
using TSizeUSetVec = std::vector<TSizeUSet>;
using TSizeDoublePrBuf = boost::circular_buffer<TSizeDoublePr>;

namespace {

class BasicCompare {};
class ContainerCompare {};

template<typename T, typename = void>
struct compare_container_selector {
    using value = BasicCompare;
};
template<typename T>
struct compare_container_selector<T, std::void_t<typename T::const_iterator>> {
    using value = ContainerCompare;
};

template<typename SELECTOR>
class CCompareImpl {};

//! Convenience function to select implementation.
template<typename T>
bool compare(const T& lhs, const T& rhs) {
    return CCompareImpl<typename compare_container_selector<T>::value>::dispatch(lhs, rhs);
}

struct SFirstLess {
    template<typename U, typename V>
    inline bool operator()(const std::pair<U, V>& lhs, const std::pair<U, V>& rhs) const {
        return lhs.first < rhs.first;
    }
};

struct SEqual {
    bool operator()(double lhs, double rhs) const {
        return std::fabs(lhs - rhs) <= 1e-5 * std::max(std::fabs(lhs), std::fabs(rhs));
    }

    template<typename T>
    bool operator()(T lhs, T rhs) const {
        return this->operator()(static_cast<double>(lhs), static_cast<double>(rhs));
    }

    bool operator()(const TSizeDoublePr& lhs, const TSizeDoublePr& rhs) {
        return lhs.first == rhs.first && this->operator()(lhs.second, rhs.second);
    }

    template<typename U, typename V>
    bool operator()(const std::pair<U, V>& lhs, const std::pair<U, V>& rhs) {
        return compare(lhs.first, rhs.first) && compare(lhs.second, rhs.second);
    }

    template<typename... T>
    bool operator()(const std::tuple<T...>& lhs, const std::tuple<T...>& rhs) {
        return lhs == rhs;
    }
};

template<>
class CCompareImpl<BasicCompare> {
public:
    template<typename T>
    static bool dispatch(const T& lhs, const T& rhs) {
        SEqual eq;
        return eq(lhs, rhs);
    }
};

template<>
class CCompareImpl<ContainerCompare> {
public:
    template<typename T>
    static bool dispatch(const T& lhs, const T& rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (auto i = lhs.begin(), j = rhs.begin(); i != lhs.end(); ++i, ++j) {
            if (compare(*i, *j) == false) {
                return false;
            }
        }
        return true;
    }

    template<typename K, typename V>
    static bool dispatch(const boost::unordered_map<K, V>& lhs,
                         const boost::unordered_map<K, V>& rhs) {
        using TVec = std::vector<std::pair<K, V>>;
        TVec lKeys(lhs.begin(), lhs.end());
        TVec rKeys(rhs.begin(), rhs.end());
        std::sort(lKeys.begin(), lKeys.end(), SFirstLess());
        std::sort(rKeys.begin(), rKeys.end(), SFirstLess());
        return compare(lKeys, rKeys);
    }

    template<typename T>
    static bool dispatch(const boost::unordered_set<T>& lhs,
                         const boost::unordered_set<T>& rhs) {
        using TVec = std::vector<T>;
        TVec lKeys(lhs.begin(), lhs.end());
        TVec rKeys(rhs.begin(), rhs.end());
        std::sort(lKeys.begin(), lKeys.end());
        std::sort(rKeys.begin(), rKeys.end());
        return compare(lKeys, rKeys);
    }
};

template<typename T>
bool equal(const T& lhs, const T& rhs) {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin(), SEqual());
}

template<typename T>
void testPersistRestore(const T& collection, const T& initial = T()) {
    const std::string tag("baseTag");
    std::stringstream origSs;
    {
        core::CJsonStatePersistInserter inserter(origSs);
        core::CPersistUtils::persist(tag, collection, inserter);
    }
    LOG_TRACE(<< "String data is: " << origSs.str());
    LOG_TRACE(<< " - doing restore " << typeid(T).name());
    T restored = initial;
    std::stringstream restoredSs;
    {
        core::CJsonStateRestoreTraverser traverser(origSs);
        BOOST_TEST_REQUIRE(core::CPersistUtils::restore(tag, restored, traverser));
    }
    LOG_TRACE(<< " - doing persist again " << typeid(T).name());
    {
        const T& restoredRef = restored;
        core::CJsonStatePersistInserter inserter(restoredSs);
        core::CPersistUtils::persist(tag, restoredRef, inserter);
    }
    LOG_TRACE(<< "String data is: " << restoredSs.str());

    BOOST_REQUIRE_EQUAL(origSs.str(), restoredSs.str());
    BOOST_TEST_REQUIRE(compare(collection, restored));
}
}

BOOST_AUTO_TEST_CASE(testPersistContainers) {
    // 1) Check that persistence and restoration is idempotent.
    // 2) Check some edge cases.
    // 3) Test failures.
    // 4) Test that vector is correctly reserved on restore.

    {
        TDoubleVec collection;
        testPersistRestore(collection);
        TDoubleVec other;
        LOG_DEBUG(<< "The same: " << compare(collection, other));
    }
    {
        TSizeDoubleMap collection;
        testPersistRestore(collection);
    }

    LOG_DEBUG(<< "*** pair ***");
    {
        std::pair<double, double> pair1{3.1, 5.9};
        testPersistRestore(pair1);

        std::pair<double, int> pair2{3.1, 50};
        testPersistRestore(pair2);
    }
    LOG_DEBUG(<< "*** tuple ***");
    {
        std::tuple<double, double, double> tuple1{1.0, 2.0, 3.4};
        testPersistRestore(tuple1);

        std::tuple<int, double, double> tuple2{99, 2.0, 3.4};
        testPersistRestore(tuple2);
    }
    LOG_DEBUG(<< "*** vector ***");
    {
        TDoubleVec collection;
        testPersistRestore(collection);

        collection.push_back(3.2);
        testPersistRestore(collection);

        collection.push_back(1.0 / 3.0);
        testPersistRestore(collection);

        collection.push_back(5.0);
        collection.push_back(1.678e-45);
        collection.push_back(0.562);
        collection.push_back(0.11);
        collection.push_back(234.768);
        testPersistRestore(collection);
    }
    LOG_DEBUG(<< "*** map ***");
    {
        TSizeDoubleMap collection;
        testPersistRestore(collection);

        collection.emplace(1, 3.2);
        testPersistRestore(collection);

        collection.emplace(4, 1.0 / 3.0);
        testPersistRestore(collection);

        collection.emplace(20, 158.0 / 3.0);
        collection.emplace(0, 1.678e-45);
        testPersistRestore(collection);
    }
    LOG_DEBUG(<< "*** set ***");
    {
        TIntSet collection;
        testPersistRestore(collection);

        collection.insert(1);
        testPersistRestore(collection);

        collection.insert(-1);
        testPersistRestore(collection);

        collection.insert(55);
        collection.insert(-10);
        collection.insert(12);
        collection.insert(1);
        testPersistRestore(collection);
    }
    LOG_DEBUG(<< "*** array ***");
    {
        std::array<double, 5> collection{1.2, 4.6, 5.81, 3.8, 1.0 / 3.0};
        testPersistRestore(collection);
    }
    LOG_DEBUG(<< "*** unordered set ***");
    {
        TSizeUSet set{1, 3, 5, 9, 8, 7, 6, 5, 4};
        testPersistRestore(set);
    }
    LOG_DEBUG(<< "*** nested ***");
    {
        TDoubleVecVec collection1{{22.22, 3456245, 0.0001, 1.3234324, 0.0054, 33.907},
                                  {6.2456, 17.13452345, -99.99},
                                  {0.67}};

        testPersistRestore(collection1);

        TStrVec collection2{
            "y",   "n",   "adsf asdf ", "anything goes", "isn#t this fun?",
            "w w", "yyy", "ttt ttt.t",  "something"};

        testPersistRestore(collection2);

        TStrVecVec collection3(6, collection2);
        collection3[0].push_back("one");
        collection3[1].push_back("two");
        collection3[2].push_back("three");
        collection3[4].push_back("one");

        testPersistRestore(collection3);

        TSizeUSetVec collection4{{1, 3, 5, 7, 9},
                                 {2, 4, 6, 8, 10},
                                 {99, 109, 909, 999, 99999, 91919, 909090909},
                                 {0, 1, 2, 3, 4, 5, 6, 7},
                                 {7, 6, 5, 4, 3, 2, 1, 0}};

        testPersistRestore(collection4);

        std::vector<std::tuple<double, double, int>> collection5{{1.3, 7.1, 0},
                                                                 {17.1, -2.1, 5}};

        testPersistRestore(collection5);
    }
    LOG_DEBUG(<< "*** unordered_map ***");
    {
        TStrIntUMap map{
            {"hello", 66}, {"thing", 999}, {"wanton destruction", 2}, {"0", 1}, {"hola", 43}};
        testPersistRestore(map);
    }
    {
        TStrDoubleVecVecUMap map;
        map.reserve(1);
        TDoubleVecVec vec{{22.22, 3456245, 0.0001, 1.3234324, 0.000000054, 33.907},
                          {6.2456, 17.13452345, -99.99},
                          {0.0000000067}};
        map["shorty"] = vec;

        vec[0].push_back(444);
        vec[1].clear();
        vec[2].push_back(94.94);
        vec[2].push_back(94.95);
        vec[2].push_back(94.96);
        map["charlie"] = vec;

        testPersistRestore(map);
    }
    LOG_DEBUG(<< "*** circular_buffer ***");
    {
        TSizeDoublePrBuf buf(5);
        buf.push_back(std::make_pair(44, 44.4));
        buf.push_back(std::make_pair(32, 0.0000043));
        buf.push_back(std::make_pair(66667, 99.945));
        testPersistRestore(buf, TSizeDoublePrBuf(5));
    }
    LOG_DEBUG(<< "*** pod ***");
    {
        int t = -9956;
        testPersistRestore(t);
        std::size_t s = std::numeric_limits<std::size_t>::max();
        testPersistRestore(s);
        std::string str("This is a string!");
        testPersistRestore(str);
        bool b = true;
        testPersistRestore(b);
        b = false;
        testPersistRestore(b);
    }
    LOG_DEBUG(<< "*** errors ***");
    {
        std::string bad("dejk");
        TDoubleVec collection;
        BOOST_REQUIRE_EQUAL(core::CPersistUtils::fromString(bad, collection), false);
        BOOST_TEST_REQUIRE(collection.empty());

        bad += core::CPersistUtils::DELIMITER;
        bad += "12";
        BOOST_REQUIRE_EQUAL(core::CPersistUtils::fromString(bad, collection), false);
        BOOST_TEST_REQUIRE(collection.empty());

        bad = std::string("1.3") + core::CPersistUtils::DELIMITER + bad;
        BOOST_REQUIRE_EQUAL(core::CPersistUtils::fromString(bad, collection), false);
        BOOST_TEST_REQUIRE(collection.empty());
    }
    {
        std::string bad;
        bad += "14";
        bad += core::CPersistUtils::PAIR_DELIMITER;
        bad += "kdsk";
        TSizeDoubleMap collection;
        BOOST_REQUIRE_EQUAL(core::CPersistUtils::fromString(bad, collection), false);
        BOOST_TEST_REQUIRE(collection.empty());

        bad = std::string("etjdjk") + core::CPersistUtils::PAIR_DELIMITER +
              "2.3" + core::CPersistUtils::DELIMITER + bad;
        BOOST_REQUIRE_EQUAL(core::CPersistUtils::fromString(bad, collection), false);
        BOOST_TEST_REQUIRE(collection.empty());
    }
    {
        std::string state;
        state += "12.3";
        state += core::CPersistUtils::DELIMITER;
        state += "2.4";
        std::array<double, 3> wrongSize;
        BOOST_REQUIRE_EQUAL(core::CPersistUtils::fromString(state, wrongSize), false);
    }
}

BOOST_AUTO_TEST_CASE(testToStringContainers) {

    TDoubleVec collection1{1.1, 2.2, 3.3, 4.4, 9.0};
    std::string state1{core::CPersistUtils::toString(collection1)};
    LOG_DEBUG(<< "state 1 = " << state1);
    TDoubleVec restored1;
    BOOST_TEST_REQUIRE(core::CPersistUtils::fromString(state1, restored1));
    BOOST_TEST_REQUIRE(compare(collection1, restored1));

    std::vector<std::pair<double, int>> collection2{
        {1.1, 1}, {2.2, 2}, {3.3, 3}, {4.4, 4}, {9.0, 0}};
    std::string state2{core::CPersistUtils::toString(collection2)};
    LOG_DEBUG(<< "state 2 = " << state2);
    std::vector<std::pair<double, int>> restored2;
    BOOST_TEST_REQUIRE(core::CPersistUtils::fromString(state2, restored2));
    BOOST_TEST_REQUIRE(compare(collection2, restored2));

    std::vector<std::tuple<double, double, int>> collection3{{1.3, 7.1, 0},
                                                             {17.1, -2.1, 5}};
    std::string state3{core::CPersistUtils::toString(collection3)};
    LOG_DEBUG(<< "state 3 = " << state3);
    std::vector<std::tuple<double, double, int>> restored3;
    BOOST_TEST_REQUIRE(core::CPersistUtils::fromString(state3, restored3));
    BOOST_TEST_REQUIRE(compare(collection3, restored3));
}

BOOST_AUTO_TEST_CASE(testPersistIterators) {
    // Persist only a sub set of a collection
    {
        LOG_DEBUG(<< "*** vector range ***");

        TDoubleVec collection;
        for (int i = 0; i < 20; i++) {
            collection.push_back(i);
        }

        auto begin = collection.begin();
        auto end = collection.begin() + 10;

        std::string state = core::CPersistUtils::toString(begin, end);
        LOG_DEBUG(<< "state = " << state);
        BOOST_TEST_REQUIRE(begin == end);
        TDoubleVec restored;
        core::CPersistUtils::fromString(state, restored);

        TDoubleVec firstTen(10);
        std::iota(firstTen.begin(), firstTen.end(), 0);
        BOOST_TEST_REQUIRE(equal(firstTen, restored));

        auto fifth = collection.begin() + 5;
        auto tenth = collection.begin() + 10;

        state = core::CPersistUtils::toString(fifth, tenth);
        LOG_DEBUG(<< "state = " << state);
        BOOST_TEST_REQUIRE(fifth == tenth);
        restored.clear();
        core::CPersistUtils::fromString(state, restored);

        TDoubleVec fithToTenth;
        for (int i = 5; i < 10; i++) {
            fithToTenth.push_back(i);
        }
        BOOST_TEST_REQUIRE(equal(fithToTenth, restored));
    }
}

BOOST_AUTO_TEST_CASE(testAppend) {
    // Persist only a sub set of a collection
    {
        LOG_DEBUG(<< "*** vector append ***");

        TDoubleVec source(9);
        std::iota(source.begin(), source.end(), 0);

        std::string state = core::CPersistUtils::toString(source);
        LOG_DEBUG(<< "state = " << state);
        TDoubleVec restored;
        core::CPersistUtils::fromString(state, restored);
        BOOST_TEST_REQUIRE(equal(source, restored));

        for (int i = 9; i < 15; i++) {
            source.push_back(i);
        }

        auto begin = source.begin() + 9;
        auto end = source.begin() + 15;

        state = core::CPersistUtils::toString(begin, end);
        BOOST_TEST_REQUIRE(begin == end);
        LOG_DEBUG(<< "state = " << state);

        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        BOOST_TEST_REQUIRE(equal(source, restored));

        for (int i = 15; i < 19; i++) {
            source.push_back(i);
        }

        begin = source.begin() + 15;
        end = source.begin() + 19;

        state = core::CPersistUtils::toString(begin, end);
        BOOST_TEST_REQUIRE(begin == end);
        LOG_DEBUG(<< "state = " << state);

        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        BOOST_TEST_REQUIRE(equal(source, restored));
    }

    {
        LOG_DEBUG(<< "*** map append ***");

        TSizeDoubleMap collection;
        std::string state = core::CPersistUtils::toString(collection);
        LOG_DEBUG(<< "state = " << state);

        TSizeDoubleMap restored;
        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        BOOST_TEST_REQUIRE(equal(collection, restored));

        for (int i = 0; i < 10; i++) {
            collection.emplace(i, 3.2);
        }

        state = core::CPersistUtils::toString(collection);
        LOG_DEBUG(<< "state = " << state);
        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        BOOST_TEST_REQUIRE(equal(collection, restored));

        // add another element
        auto pr = collection.insert(TSizeDoublePr(14, 1.0));

        auto end = collection.end();
        state = core::CPersistUtils::toString(pr.first, end);
        LOG_DEBUG(<< "state = " << state);
        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        BOOST_TEST_REQUIRE(equal(collection, restored));

        pr = collection.insert(TSizeDoublePr(20, 158.0));
        collection.insert(TSizeDoublePr(21, 1.678e-45));

        end = collection.end();
        state = core::CPersistUtils::toString(pr.first, end);
        LOG_DEBUG(<< "state = " << state);
        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        BOOST_TEST_REQUIRE(equal(collection, restored));
    }
}

BOOST_AUTO_TEST_CASE(testFloatingPoint) {

    test::CRandomNumbers rng;

    TDoubleVec doubles;
    TDoubleVec restoredDoubles;
    for (std::size_t i = 0; i < 100; ++i) {
        rng.generateUniformSamples(-1000.0, 1000.0, 10, doubles);
        std::string serialized{core::CPersistUtils::toString(doubles)};
        core::CPersistUtils::fromString(serialized, restoredDoubles);
        for (std::size_t j = 0; j < doubles.size(); ++j) {
            BOOST_REQUIRE_EQUAL(doubles[j], restoredDoubles[j]);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
