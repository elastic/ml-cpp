/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CGathererTools.h>
#include <model/CModelParams.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CGathererToolsTest)

using namespace ml;
using namespace model;

namespace {
const CGathererTools::CSumGatherer::TStrVec EMPTY_STR_VEC;
const CGathererTools::CSumGatherer::TStoredStringPtrVec EMPTY_STR_PTR_VEC;
}


BOOST_AUTO_TEST_CASE(testSumGathererIsRedundant) {
    using TDouble1Vec = CGathererTools::CSumGatherer::TDouble1Vec;

    core_t::TTime bucketLength(100);
    SModelParams modelParams(bucketLength);
    modelParams.s_LatencyBuckets = 3;
    CGathererTools::CSumGatherer sumGatherer(
        modelParams, 0, 100, bucketLength, EMPTY_STR_VEC.begin(), EMPTY_STR_VEC.end());

    sumGatherer.add(100, TDouble1Vec{1.0}, 1, 0, EMPTY_STR_PTR_VEC);
    sumGatherer.startNewBucket(100);
    sumGatherer.add(200, TDouble1Vec{1.0}, 1, 0, EMPTY_STR_PTR_VEC);
    sumGatherer.startNewBucket(200);
    sumGatherer.add(300, TDouble1Vec{1.0}, 1, 0, EMPTY_STR_PTR_VEC);
    sumGatherer.startNewBucket(300);
    sumGatherer.add(400, TDouble1Vec{1.0}, 1, 0, EMPTY_STR_PTR_VEC);
    sumGatherer.startNewBucket(400);

    BOOST_TEST(sumGatherer.isRedundant(400) == false);

    sumGatherer.startNewBucket(500);
    BOOST_TEST(sumGatherer.isRedundant(500) == false);
    sumGatherer.startNewBucket(600);
    BOOST_TEST(sumGatherer.isRedundant(600) == false);
    sumGatherer.startNewBucket(700);
    BOOST_TEST(sumGatherer.isRedundant(700));
}

BOOST_AUTO_TEST_SUITE_END()
