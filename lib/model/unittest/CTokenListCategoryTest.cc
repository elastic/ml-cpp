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
        {0 /* she */, 1}, {1 /* sells */, 1}, {2 /* seashells */, 1},
        {3 /* on */, 1},  {4 /* the */, 1},   {5 /* seashore */, 1}};
    ml::model::CTokenListCategory::TSizeSizeMap baseUniqueTokenIds(
        baseTokenIds.begin(), baseTokenIds.end());

    ml::model::CTokenListCategory category(false, baseString, baseString.length(), baseTokenIds,
                                           baseTokenIds.size(), baseUniqueTokenIds);

    std::string newString{"she sells ice cream on the seashore"};
    ml::model::CTokenListCategory::TSizeSizePrVec newTokenIds{
        {0 /* she */, 1},     {1 /* sells */, 1}, {6 /* ice */, 1},
        {7 /* cream */, 1},   {3 /* on */, 1},    {4 /* the */, 1},
        {5 /* seashore */, 1}};
    ml::model::CTokenListCategory::TSizeSizeMap newUniqueTokenIds(
        newTokenIds.begin(), newTokenIds.end());

    BOOST_TEST_REQUIRE(category.addString(false, newString, newString.length(),
                                          newTokenIds, newUniqueTokenIds));

    BOOST_REQUIRE_EQUAL(baseString, category.baseString());
    BOOST_REQUIRE_EQUAL(ml::core::CContainerPrinter::print(baseTokenIds),
                        ml::core::CContainerPrinter::print(category.baseTokenIds()));
    BOOST_REQUIRE_EQUAL(baseTokenIds.size(), category.baseWeight());
    ml::model::CTokenListCategory::TSizeSizeMap expectedCommonUniqueTokenIds{
        {0 /* she */, 1}, {1 /* sells */, 1}, {3 /* on */, 1}, {4 /* the */, 1}, {5 /* seashore */, 1}};
    BOOST_REQUIRE_EQUAL(
        ml::core::CContainerPrinter::print(expectedCommonUniqueTokenIds),
        ml::core::CContainerPrinter::print(category.commonUniqueTokenIds()));
    BOOST_REQUIRE_EQUAL(expectedCommonUniqueTokenIds.size(),
                        category.commonUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(baseUniqueTokenIds.size(), category.origUniqueTokenWeight());
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
        {0 /* she */, 1}, {1 /* sells */, 1}, {2 /* seashells */, 1},
        {3 /* on */, 1},  {4 /* the */, 1},   {5 /* seashore */, 1}};
    ml::model::CTokenListCategory::TSizeSizeMap baseUniqueTokenIds(
        baseTokenIds.begin(), baseTokenIds.end());

    ml::model::CTokenListCategory category(false, baseString, baseString.length(), baseTokenIds,
                                           baseTokenIds.size(), baseUniqueTokenIds);

    std::string newString{"sells seashells on the seashore, she does"};
    ml::model::CTokenListCategory::TSizeSizePrVec newTokenIds{
        {1 /* sells */, 1}, {2 /* seashells */, 1}, {3 /* on */, 1},
        {4 /* the */, 1},   {5 /* seashore */, 1},  {0 /* she */, 1},
        {6 /* does */, 1}};
    ml::model::CTokenListCategory::TSizeSizeMap newUniqueTokenIds(
        newTokenIds.begin(), newTokenIds.end());

    BOOST_TEST_REQUIRE(category.addString(false, newString, newString.length(),
                                          newTokenIds, newUniqueTokenIds));

    BOOST_REQUIRE_EQUAL(baseString, category.baseString());
    BOOST_REQUIRE_EQUAL(ml::core::CContainerPrinter::print(baseTokenIds),
                        ml::core::CContainerPrinter::print(category.baseTokenIds()));
    BOOST_REQUIRE_EQUAL(baseTokenIds.size(), category.baseWeight());
    BOOST_REQUIRE_EQUAL(ml::core::CContainerPrinter::print(baseUniqueTokenIds),
                        ml::core::CContainerPrinter::print(category.commonUniqueTokenIds()));
    BOOST_REQUIRE_EQUAL(baseUniqueTokenIds.size(), category.commonUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(baseUniqueTokenIds.size(), category.origUniqueTokenWeight());
    BOOST_REQUIRE_EQUAL(std::max(baseString.length(), newString.length()),
                        category.maxStringLen());
    // FIXME: {0, 1} is clearly sub-optimal here; {1, 6} would be much better
    ml::model::CTokenListCategory::TSizeSizePr expectedOrderedCommonTokenBounds{0, 1};
    BOOST_REQUIRE_EQUAL(
        ml::core::CContainerPrinter::print(expectedOrderedCommonTokenBounds),
        ml::core::CContainerPrinter::print(category.orderedCommonTokenBounds()));
}

BOOST_AUTO_TEST_SUITE_END()
