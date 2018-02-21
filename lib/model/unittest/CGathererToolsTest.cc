/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
#include "CGathererToolsTest.h"

#include <model/CGathererTools.h>
#include <model/CModelParams.h>

using namespace ml;
using namespace model;

namespace
{
const CGathererTools::CSumGatherer::TStrVec EMPTY_STR_VEC;
const CGathererTools::CSumGatherer::TStoredStringPtrVec EMPTY_STR_PTR_VEC;
}

CppUnit::Test *CGathererToolsTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CGathererToolsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CGathererToolsTest>(
                                   "CGathererToolsTest::testSumGathererIsRedundant",
                                   &CGathererToolsTest::testSumGathererIsRedundant) );

    return suiteOfTests;
}

void CGathererToolsTest::testSumGathererIsRedundant(void)
{
    using TDouble1Vec = CGathererTools::CSumGatherer::TDouble1Vec;

    core_t::TTime bucketLength(100);
    SModelParams modelParams(bucketLength);
    modelParams.s_LatencyBuckets = 3;
    CGathererTools::CSumGatherer sumGatherer(modelParams, 0, 100, bucketLength, EMPTY_STR_VEC.begin(), EMPTY_STR_VEC.end());

    sumGatherer.add(100, TDouble1Vec{1.0}, 1, 0, EMPTY_STR_PTR_VEC);
    sumGatherer.startNewBucket(100);
    sumGatherer.add(200, TDouble1Vec{1.0}, 1, 0, EMPTY_STR_PTR_VEC);
    sumGatherer.startNewBucket(200);
    sumGatherer.add(300, TDouble1Vec{1.0}, 1, 0, EMPTY_STR_PTR_VEC);
    sumGatherer.startNewBucket(300);
    sumGatherer.add(400, TDouble1Vec{1.0}, 1, 0, EMPTY_STR_PTR_VEC);
    sumGatherer.startNewBucket(400);

    CPPUNIT_ASSERT(sumGatherer.isRedundant(400) == false);

    sumGatherer.startNewBucket(500);
    CPPUNIT_ASSERT(sumGatherer.isRedundant(500) == false);
    sumGatherer.startNewBucket(600);
    CPPUNIT_ASSERT(sumGatherer.isRedundant(600) == false);
    sumGatherer.startNewBucket(700);
    CPPUNIT_ASSERT(sumGatherer.isRedundant(700));
}
