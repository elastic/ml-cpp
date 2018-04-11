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

#include "CPersistUtilsTest.h"

#include <core/CContainerPrinter.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>

#include <boost/circular_buffer.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <cmath>
#include <map>
#include <set>
#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
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

template<typename T, typename R = void>
struct enable_if_type {
    using type = R;
};

template<typename T, typename ITR = void>
struct compare_container_selector {
    using value = BasicCompare;
};
template<typename T>
struct compare_container_selector<T, typename enable_if_type<typename T::const_iterator>::type> {
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

    template<typename A, typename B>
    bool operator()(const std::pair<A, B>& lhs, const std::pair<A, B>& rhs) {
        return compare(lhs.first, rhs.first) && compare(lhs.second, rhs.second);
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
        using TCItr = typename T::const_iterator;
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (TCItr i = lhs.begin(), j = rhs.begin(); i != lhs.end(); ++i, ++j) {
            if (!compare(*i, *j)) {
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
    static bool
    dispatch(const boost::unordered_set<T>& lhs, const boost::unordered_set<T>& rhs) {
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
        CPPUNIT_ASSERT(core::CPersistUtils::restore(tag, restored, traverser));
    }
    LOG_TRACE(<< " - doing persist again " << typeid(T).name());
    {
        const T& restoredRef = restored;
        core::CJsonStatePersistInserter inserter(restoredSs);
        core::CPersistUtils::persist(tag, restoredRef, inserter);
    }
    LOG_TRACE(<< "String data is: " << restoredSs.str());

    CPPUNIT_ASSERT_EQUAL(origSs.str(), restoredSs.str());
    CPPUNIT_ASSERT(compare(collection, restored));
}
}

void CPersistUtilsTest::testPersistContainers() {
    // 1) Check that persistence and restoration is idempotent.
    // 2) Check some edge cases.
    // 3) Test failures.
    // 4) Test that vector is correctly reserved on restore.

    {
        TDoubleVec collection;
        testPersistRestore(collection);
        TDoubleVec other;
        LOG_DEBUG(<< "The same: " << ::compare(collection, other));
    }
    {
        TSizeDoubleMap collection;
        testPersistRestore(collection);
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

        collection.insert(TSizeDoublePr(1, 3.2));
        testPersistRestore(collection);

        collection.insert(TSizeDoublePr(4, 1.0 / 3.0));
        testPersistRestore(collection);

        collection.insert(TSizeDoublePr(20, 158.0 / 3.0));
        collection.insert(TSizeDoublePr(0, 1.678e-45));
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
        boost::array<double, 5> collection;
        collection[0] = 1.2;
        collection[1] = 4.6;
        collection[2] = 5.81;
        collection[3] = 3.8;
        collection[4] = 1.0 / 3.0;
        testPersistRestore(collection);
    }
    LOG_DEBUG(<< "*** uset ***");
    {
        TSizeUSet set;
        set.insert(1);
        set.insert(3);
        set.insert(5);
        set.insert(9);
        set.insert(8);
        set.insert(7);
        set.insert(6);
        set.insert(5);
        set.insert(4);
        testPersistRestore(set);
    }
    LOG_DEBUG(<< "*** nested ***") {
        TDoubleVecVec vec(3);
        vec[0].push_back(22.22);
        vec[0].push_back(3456245);
        vec[0].push_back(0.0001);
        vec[0].push_back(1.3234324);
        vec[0].push_back(0.0054);
        vec[0].push_back(33.907);
        vec[1].push_back(6.2456);
        vec[1].push_back(17.13452345);
        vec[1].push_back(-99.99);
        vec[2].push_back(0.67);

        testPersistRestore(vec);

        TStrVec strs;
        strs.push_back("y");
        strs.push_back("n");
        strs.push_back("adsf asdf ");
        strs.push_back("anything goes");
        strs.push_back("isn#t this fun?");
        strs.push_back("w w");
        strs.push_back("yyy");
        strs.push_back("ttt ttt.t");
        strs.push_back("something");

        testPersistRestore(strs);

        TStrVecVec moreStrs;
        moreStrs.push_back(strs);
        moreStrs.push_back(strs);
        moreStrs.push_back(strs);
        moreStrs.push_back(strs);
        moreStrs.push_back(strs);
        moreStrs.push_back(strs);
        moreStrs[0].push_back("one");
        moreStrs[1].push_back("two");
        moreStrs[2].push_back("three");
        moreStrs[4].push_back("one");

        testPersistRestore(moreStrs);

        TSizeUSetVec collection(5);
        collection[0].insert(1);
        collection[0].insert(3);
        collection[0].insert(5);
        collection[0].insert(7);
        collection[0].insert(9);
        collection[1].insert(2);
        collection[1].insert(4);
        collection[1].insert(6);
        collection[1].insert(8);
        collection[1].insert(10);
        collection[2].insert(99);
        collection[2].insert(109);
        collection[2].insert(909);
        collection[2].insert(999);
        collection[2].insert(99999);
        collection[2].insert(91919);
        collection[2].insert(909090909);
        collection[3].insert(0);
        collection[3].insert(1);
        collection[3].insert(2);
        collection[3].insert(3);
        collection[3].insert(4);
        collection[3].insert(5);
        collection[3].insert(6);
        collection[3].insert(7);
        collection[4].insert(7);
        collection[4].insert(6);
        collection[4].insert(5);
        collection[4].insert(4);
        collection[4].insert(3);
        collection[4].insert(2);
        collection[4].insert(1);
        collection[4].insert(0);
        testPersistRestore(collection);
    }
    LOG_DEBUG(<< "*** unordered_map ***");
    {
        TStrIntUMap map;
        map.reserve(5);
        map["hello"] = 66;
        map["thing"] = 999;
        map["wanton destruction"] = 2;
        map["0"] = 1;
        map["hola"] = 43;
        testPersistRestore(map);
    }
    {
        TStrDoubleVecVecUMap map;
        map.reserve(1);
        TDoubleVecVec vec(3);
        vec[0].push_back(22.22);
        vec[0].push_back(3456245);
        vec[0].push_back(0.0001);
        vec[0].push_back(1.3234324);
        vec[0].push_back(0.000000054);
        vec[0].push_back(33.907);
        vec[1].push_back(6.2456);
        vec[1].push_back(17.13452345);
        vec[1].push_back(-99.99);
        vec[2].push_back(0.0000000067);
        map["shorty"] = vec;

        vec[0].push_back(444);
        vec[2].push_back(94.94);
        vec[2].push_back(94.95);
        vec[2].push_back(94.96);
        vec[1].clear();
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
        CPPUNIT_ASSERT(!core::CPersistUtils::fromString(bad, collection));
        CPPUNIT_ASSERT(collection.empty());

        bad += core::CPersistUtils::DELIMITER;
        bad += "12";
        CPPUNIT_ASSERT(!core::CPersistUtils::fromString(bad, collection));
        CPPUNIT_ASSERT(collection.empty());

        bad = std::string("1.3") + core::CPersistUtils::DELIMITER + bad;
        CPPUNIT_ASSERT(!core::CPersistUtils::fromString(bad, collection));
        CPPUNIT_ASSERT(collection.empty());
    }
    {
        std::string bad;
        bad += "14";
        bad += core::CPersistUtils::PAIR_DELIMITER;
        bad += "kdsk";
        TSizeDoubleMap collection;
        CPPUNIT_ASSERT(!core::CPersistUtils::fromString(bad, collection));
        CPPUNIT_ASSERT(collection.empty());

        bad = std::string("etjdjk") + core::CPersistUtils::PAIR_DELIMITER +
              "2.3" + core::CPersistUtils::DELIMITER + bad;
        CPPUNIT_ASSERT(!core::CPersistUtils::fromString(bad, collection));
        CPPUNIT_ASSERT(collection.empty());
    }
    {
        std::string state;
        state += "12.3";
        state += core::CPersistUtils::DELIMITER;
        state += "2.4";
        boost::array<double, 3> wrongSize;
        CPPUNIT_ASSERT(!core::CPersistUtils::fromString(state, wrongSize));
    }
}

void CPersistUtilsTest::testPersistIterators() {
    // Persist only a sub set of a collection
    {
        LOG_DEBUG(<< "*** vector range ***");

        TDoubleVec collection;
        for (int i = 0; i < 20; i++) {
            collection.push_back(i);
        }

        TDoubleVec::iterator begin = collection.begin();
        TDoubleVec::iterator end = collection.begin() + 10;

        std::string state = core::CPersistUtils::toString(begin, end);
        LOG_DEBUG(<< "state = " << state);
        CPPUNIT_ASSERT(begin == end);
        TDoubleVec restored;
        core::CPersistUtils::fromString(state, restored);

        TDoubleVec firstTen;
        for (int i = 0; i < 10; i++) {
            firstTen.push_back(i);
        }
        CPPUNIT_ASSERT(equal(firstTen, restored));

        TDoubleVec::iterator fifth = collection.begin() + 5;
        TDoubleVec::iterator tenth = collection.begin() + 10;

        state = core::CPersistUtils::toString(fifth, tenth);
        LOG_DEBUG(<< "state = " << state);
        CPPUNIT_ASSERT(fifth == tenth);
        restored.clear();
        core::CPersistUtils::fromString(state, restored);

        TDoubleVec fithToTenth;
        for (int i = 5; i < 10; i++) {
            fithToTenth.push_back(i);
        }
        CPPUNIT_ASSERT(equal(fithToTenth, restored));
    }
}

void CPersistUtilsTest::testAppend() {
    // Persist only a sub set of a collection
    {
        LOG_DEBUG(<< "*** vector append ***");

        TDoubleVec source;
        for (int i = 0; i < 9; i++) {
            source.push_back(i);
        }

        std::string state = core::CPersistUtils::toString(source);
        LOG_DEBUG(<< "state = " << state);
        TDoubleVec restored;
        core::CPersistUtils::fromString(state, restored);
        CPPUNIT_ASSERT(equal(source, restored));

        for (int i = 9; i < 15; i++) {
            source.push_back(i);
        }

        TDoubleVec::iterator begin = source.begin() + 9;
        TDoubleVec::iterator end = source.begin() + 15;

        state = core::CPersistUtils::toString(begin, end);
        CPPUNIT_ASSERT(begin == end);
        LOG_DEBUG(<< "state = " << state);

        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        CPPUNIT_ASSERT(equal(source, restored));

        for (int i = 15; i < 19; i++) {
            source.push_back(i);
        }

        begin = source.begin() + 15;
        end = source.begin() + 19;

        state = core::CPersistUtils::toString(begin, end);
        CPPUNIT_ASSERT(begin == end);
        LOG_DEBUG(<< "state = " << state);

        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        CPPUNIT_ASSERT(equal(source, restored));
    }

    {
        LOG_DEBUG(<< "*** map append ***");

        TSizeDoubleMap collection;
        std::string state = core::CPersistUtils::toString(collection);
        LOG_DEBUG(<< "state = " << state);

        TSizeDoubleMap restored;
        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        CPPUNIT_ASSERT(equal(collection, restored));

        for (int i = 0; i < 10; i++) {
            collection.insert(TSizeDoublePr(i, 3.2));
        }

        state = core::CPersistUtils::toString(collection);
        LOG_DEBUG(<< "state = " << state);
        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        CPPUNIT_ASSERT(equal(collection, restored));

        // add another element
        std::pair<TSizeDoubleMap::iterator, bool> pr =
            collection.insert(TSizeDoublePr(14, 1.0));

        TSizeDoubleMap::iterator end = collection.end();
        state = core::CPersistUtils::toString(pr.first, end);
        LOG_DEBUG(<< "state = " << state);
        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        CPPUNIT_ASSERT(equal(collection, restored));

        pr = collection.insert(TSizeDoublePr(20, 158.0));
        collection.insert(TSizeDoublePr(21, 1.678e-45));

        end = collection.end();
        state = core::CPersistUtils::toString(pr.first, end);
        LOG_DEBUG(<< "state = " << state);
        core::CPersistUtils::fromString(state, restored, core::CPersistUtils::DELIMITER,
                                        core::CPersistUtils::PAIR_DELIMITER, true);
        CPPUNIT_ASSERT(equal(collection, restored));
    }
}

CppUnit::Test* CPersistUtilsTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CPersistUtilsTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CPersistUtilsTest>(
        "CPersistUtilsTest::testPersistContainers", &CPersistUtilsTest::testPersistContainers));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPersistUtilsTest>(
        "CPersistUtilsTest::testPersistIterators", &CPersistUtilsTest::testPersistIterators));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPersistUtilsTest>(
        "CPersistUtilsTest::testAppend", &CPersistUtilsTest::testAppend));

    return suiteOfTests;
}
