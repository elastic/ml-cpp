/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>

#include <model/CTokenListCategory.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <string>

BOOST_AUTO_TEST_SUITE(CTokenListCategoryTest)

BOOST_AUTO_TEST_CASE(testCommonTokensSameOrder) {

    std::string baseString{"she sells seashells on the seashore"};
    ml::model::CTokenListCategory::TSizeSizePrVec baseTokenIds{
        {0 /* she */, 2}, {1 /* sells */, 2}, {2 /* seashells */, 2},
        {3 /* on */, 2},  {4 /* the */, 2},   {5 /* seashore */, 2}};
    ml::model::CTokenListCategory::TSizeSizeMap baseUniqueTokenIds(
        baseTokenIds.begin(), baseTokenIds.end());

    ml::model::CTokenListCategory category(false, baseString, baseString.length(),
                                           baseTokenIds, baseTokenIds.size() * 2,
                                           baseUniqueTokenIds);

    std::string newString{"she sells ice cream on the seashore"};
    ml::model::CTokenListCategory::TSizeSizePrVec newTokenIds{
        {0 /* she */, 2},     {1 /* sells */, 2}, {6 /* ice */, 2},
        {7 /* cream */, 2},   {3 /* on */, 2},    {4 /* the */, 2},
        {5 /* seashore */, 2}};
    ml::model::CTokenListCategory::TSizeSizeMap newUniqueTokenIds(
        newTokenIds.begin(), newTokenIds.end());

    BOOST_TEST_REQUIRE(category.addString(false, newString, newString.length(),
                                          newTokenIds, newUniqueTokenIds));

    BOOST_REQUIRE_EQUAL(baseString, category.baseString());
    BOOST_REQUIRE_EQUAL(ml::core::CContainerPrinter::print(baseTokenIds),
                        ml::core::CContainerPrinter::print(category.baseTokenIds()));
    BOOST_REQUIRE_EQUAL(baseTokenIds.size() * 2, category.baseWeight());
    ml::model::CTokenListCategory::TSizeSizeMap expectedCommonUniqueTokenIds{
        {0 /* she */, 2}, {1 /* sells */, 2}, {3 /* on */, 2}, {4 /* the */, 2}, {5 /* seashore */, 2}};
    BOOST_REQUIRE_EQUAL(
        ml::core::CContainerPrinter::print(expectedCommonUniqueTokenIds),
        ml::core::CContainerPrinter::print(category.commonUniqueTokenIds()));
    BOOST_REQUIRE_EQUAL(expectedCommonUniqueTokenIds.size() * 2,
                        category.commonUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(baseUniqueTokenIds.size() * 2, category.origUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(std::max(baseString.length(), newString.length()),
                        category.maxStringLen());
    ml::model::CTokenListCategory::TSizeSizePr expectedOrderedCommonTokenBounds{0, 6};
    BOOST_REQUIRE_EQUAL(
        ml::core::CContainerPrinter::print(expectedOrderedCommonTokenBounds),
        ml::core::CContainerPrinter::print(category.orderedCommonTokenBounds()));
}

BOOST_AUTO_TEST_CASE(testCommonTokensDifferentOrder) {

    std::string baseString{"she sells seashells on the seashore"};
    ml::model::CTokenListCategory::TSizeSizePrVec baseTokenIds{
        {0 /* she */, 2}, {1 /* sells */, 2}, {2 /* seashells */, 2},
        {3 /* on */, 2},  {4 /* the */, 2},   {5 /* seashore */, 2}};
    ml::model::CTokenListCategory::TSizeSizeMap baseUniqueTokenIds(
        baseTokenIds.begin(), baseTokenIds.end());

    ml::model::CTokenListCategory category(false, baseString, baseString.length(),
                                           baseTokenIds, baseTokenIds.size() * 2,
                                           baseUniqueTokenIds);

    std::string newString1{"sells seashells on the seashore, she does"};
    ml::model::CTokenListCategory::TSizeSizePrVec newTokenIds1{
        {1 /* sells */, 2}, {2 /* seashells */, 2}, {3 /* on */, 2},
        {4 /* the */, 2},   {5 /* seashore */, 2},  {0 /* she */, 2},
        {6 /* does */, 2}};
    ml::model::CTokenListCategory::TSizeSizeMap newUniqueTokenIds1(
        newTokenIds1.begin(), newTokenIds1.end());

    BOOST_TEST_REQUIRE(category.addString(false, newString1, newString1.length(),
                                          newTokenIds1, newUniqueTokenIds1));

    BOOST_REQUIRE_EQUAL(baseString, category.baseString());
    BOOST_REQUIRE_EQUAL(ml::core::CContainerPrinter::print(baseTokenIds),
                        ml::core::CContainerPrinter::print(category.baseTokenIds()));
    BOOST_REQUIRE_EQUAL(baseTokenIds.size() * 2, category.baseWeight());
    BOOST_REQUIRE_EQUAL(ml::core::CContainerPrinter::print(baseUniqueTokenIds),
                        ml::core::CContainerPrinter::print(category.commonUniqueTokenIds()));
    BOOST_REQUIRE_EQUAL(baseUniqueTokenIds.size() * 2, category.commonUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(baseUniqueTokenIds.size() * 2, category.origUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(std::max(baseString.length(), newString1.length()),
                        category.maxStringLen());
    ml::model::CTokenListCategory::TSizeSizePr expectedOrderedCommonTokenBounds{1, 6};
    BOOST_REQUIRE_EQUAL(
        ml::core::CContainerPrinter::print(expectedOrderedCommonTokenBounds),
        ml::core::CContainerPrinter::print(category.orderedCommonTokenBounds()));

    std::string newString2{"nice seashells can be found near the seashore"};
    ml::model::CTokenListCategory::TSizeSizePrVec newTokenIds2{
        {7 /* nice */, 2}, {2 /* seashells */, 2}, {8 /* can */, 2},
        {9 /* be */, 2},   {10 /* found */, 2},    {11 /* near */, 2},
        {4 /* the */, 2},  {5 /* seashore */, 2}};
    ml::model::CTokenListCategory::TSizeSizeMap newUniqueTokenIds2(
        newTokenIds2.begin(), newTokenIds2.end());

    BOOST_TEST_REQUIRE(category.addString(false, newString2, newString2.length(),
                                          newTokenIds2, newUniqueTokenIds2));

    BOOST_REQUIRE_EQUAL(baseString, category.baseString());
    BOOST_REQUIRE_EQUAL(ml::core::CContainerPrinter::print(baseTokenIds),
                        ml::core::CContainerPrinter::print(category.baseTokenIds()));
    BOOST_REQUIRE_EQUAL(baseTokenIds.size() * 2, category.baseWeight());
    ml::model::CTokenListCategory::TSizeSizeMap expectedCommonUniqueTokenIds{
        {2 /* seashells */, 2}, {4 /* the */, 2}, {5 /* seashore */, 2}};
    BOOST_REQUIRE_EQUAL(
        ml::core::CContainerPrinter::print(expectedCommonUniqueTokenIds),
        ml::core::CContainerPrinter::print(category.commonUniqueTokenIds()));
    BOOST_REQUIRE_EQUAL(expectedCommonUniqueTokenIds.size() * 2,
                        category.commonUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(baseUniqueTokenIds.size() * 2, category.origUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(std::max(newString1.length(), newString2.length()),
                        category.maxStringLen());
    // The bounds go from {1, 6} to {2, 6} even though there are now only 3
    // common tokens, because the bounds reference the base token indices,
    // and the range needs to be filtered to exclude tokens that are not common.
    // (When the real reverse search is created tokens may also be filtered if
    // their cost is too high for the available budget, so this doesn't create
    // too much complexity outside of the unit test.)
    expectedOrderedCommonTokenBounds = {2, 6};
    BOOST_REQUIRE_EQUAL(
        ml::core::CContainerPrinter::print(expectedOrderedCommonTokenBounds),
        ml::core::CContainerPrinter::print(category.orderedCommonTokenBounds()));

    std::string newString3{"the rock"};
    ml::model::CTokenListCategory::TSizeSizePrVec newTokenIds3{{4 /* the */, 2},
                                                               {12 /* rock */, 2}};
    ml::model::CTokenListCategory::TSizeSizeMap newUniqueTokenIds3(
        newTokenIds3.begin(), newTokenIds3.end());

    BOOST_TEST_REQUIRE(category.addString(false, newString3, newString3.length(),
                                          newTokenIds3, newUniqueTokenIds3));

    BOOST_REQUIRE_EQUAL(baseString, category.baseString());
    BOOST_REQUIRE_EQUAL(ml::core::CContainerPrinter::print(baseTokenIds),
                        ml::core::CContainerPrinter::print(category.baseTokenIds()));
    BOOST_REQUIRE_EQUAL(baseTokenIds.size() * 2, category.baseWeight());
    expectedCommonUniqueTokenIds = {{4 /* the */, 2}};
    BOOST_REQUIRE_EQUAL(
        ml::core::CContainerPrinter::print(expectedCommonUniqueTokenIds),
        ml::core::CContainerPrinter::print(category.commonUniqueTokenIds()));
    BOOST_REQUIRE_EQUAL(expectedCommonUniqueTokenIds.size() * 2,
                        category.commonUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(baseUniqueTokenIds.size() * 2, category.origUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(std::max(newString2.length(), newString3.length()),
                        category.maxStringLen());
    // The bounds go from {2, 6} to {4, 5} as there's now only one common token
    // and it's in position 4.
    expectedOrderedCommonTokenBounds = {4, 5};
    BOOST_REQUIRE_EQUAL(
        ml::core::CContainerPrinter::print(expectedOrderedCommonTokenBounds),
        ml::core::CContainerPrinter::print(category.orderedCommonTokenBounds()));
}

BOOST_AUTO_TEST_SUITE_END()
