/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CTriple.h>

#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>

#include <string>

using TStringSizeShortTriple = ml::core::CTriple<std::string, std::size_t, short>;
using TStringSizeShortTripleSizeMap = boost::unordered_map<TStringSizeShortTriple, std::size_t>;

BOOST_TEST_DONT_PRINT_LOG_VALUE(TStringSizeShortTripleSizeMap::iterator);

BOOST_AUTO_TEST_SUITE(CTripleTest)


BOOST_AUTO_TEST_CASE(testOperators) {
    {
        // Assignment
        ml::core::CTriple<std::string, std::size_t, std::size_t> triple1("foo", 10, 8);
        ml::core::CTriple<std::string, std::size_t, std::size_t> triple2("bar", 5, 4);
        triple1 = triple2;
        BOOST_CHECK_EQUAL(std::string("bar"), triple1.first);
        BOOST_CHECK_EQUAL(std::size_t(5), triple1.second);
        BOOST_CHECK_EQUAL(std::size_t(4), triple1.third);
        BOOST_CHECK_EQUAL(std::string("bar"), triple2.first);
        BOOST_CHECK_EQUAL(std::size_t(5), triple2.second);
        BOOST_CHECK_EQUAL(std::size_t(4), triple2.third);
    }
    {
        // Test equality
        ml::core::CTriple<std::string, std::size_t, std::size_t> triple1("foo", 10, 8);
        ml::core::CTriple<std::string, std::size_t, std::size_t> triple2("foo", 10, 8);
        BOOST_TEST(triple1 == triple2);
        BOOST_TEST(triple1.hash() == triple2.hash());
    }
    {
        // Test inequality
        ml::core::CTriple<std::string, std::size_t, std::size_t> triple1("foo", 10, 8);
        ml::core::CTriple<std::string, std::size_t, std::size_t> triple2("foo", 10, 9);
        BOOST_TEST(triple1 != triple2);
    }
    {
        // Test order comparisons
        ml::core::CTriple<std::string, std::size_t, std::size_t> triple1("foo", 10, 8);
        ml::core::CTriple<std::string, std::size_t, std::size_t> triple2("foo", 10, 9);
        BOOST_TEST(triple1 < triple2);
        BOOST_TEST(triple1 <= triple2);
        BOOST_TEST(triple2 > triple1);
        BOOST_TEST(triple2 >= triple1);
    }
}

BOOST_AUTO_TEST_CASE(testBoostHashReady) {

    TStringSizeShortTripleSizeMap map;
    map.emplace(ml::core::make_triple(std::string("foo"), std::size_t(10), short(3)), 1);
    map.emplace(ml::core::make_triple(std::string("bar"), std::size_t(20), short(4)), 2);

    BOOST_CHECK_EQUAL(
        std::size_t(1),
        map[ml::core::make_triple(std::string("foo"), std::size_t(10), short(3))]);
    BOOST_CHECK_EQUAL(
        std::size_t(2),
        map[ml::core::make_triple(std::string("bar"), std::size_t(20), short(4))]);
    BOOST_TEST(map.find(ml::core::make_triple(std::string("bar"), std::size_t(20),
                                                  short(8))) == map.end());
}

BOOST_AUTO_TEST_SUITE_END()
