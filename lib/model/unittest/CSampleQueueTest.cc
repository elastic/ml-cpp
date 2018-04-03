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

#include "CSampleQueueTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CoreTypes.h>

#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CIntegerTools.h>

#include <model/ModelTypes.h>

#include <boost/math/constants/constants.hpp>

#include <test/CRandomNumbers.h>

using namespace ml;
using namespace model;

typedef std::vector<double>                                        TDoubleVec;
typedef std::vector<CSample>                                       TSampleVec;
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
typedef CSampleQueue<TMeanAccumulator>                             TTestSampleQueue;

void CSampleQueueTest::testSampleToString(void) {
    CSample sample(10, {3.0}, 0.8, 1.0);

    CPPUNIT_ASSERT_EQUAL(std::string("10;8e-1;1;3"), CSample::SToString()(sample));
}

void CSampleQueueTest::testSampleFromString(void) {
    CSample sample;

    CPPUNIT_ASSERT(CSample::SFromString()("15;7e-1;3;2.0", sample));

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(15), sample.time());
    CPPUNIT_ASSERT_EQUAL(2.0, sample.value()[0]);
    CPPUNIT_ASSERT_EQUAL(0.7, sample.varianceScale());
    CPPUNIT_ASSERT_EQUAL(3.0, sample.count());
}

void CSampleQueueTest::testAddGivenQueueIsEmptyShouldCreateNewSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(1, {1.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), queue.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(1), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(1), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(1), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, queue[0].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenQueueIsFullShouldResize(void) {
    std::size_t sampleCountFactor(1);
    std::size_t latencyBuckets(1);
    double growthFactor(0.5);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(1);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(1, {1.0}, 1, sampleCount);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), queue.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), queue.capacity());

    queue.add(2, {2.0}, 1, sampleCount);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.capacity());

    queue.add(3, {3.0}, 1, sampleCount);
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.capacity());

    queue.add(4, {4.0}, 1, sampleCount);
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), queue.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), queue.capacity());

    queue.add(5, {5.0}, 1, sampleCount);
    CPPUNIT_ASSERT_EQUAL(std::size_t(5), queue.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(6), queue.capacity());

    queue.add(6, {6.0}, 1, sampleCount);
    CPPUNIT_ASSERT_EQUAL(std::size_t(6), queue.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(6), queue.capacity());

    queue.add(7, {7.0}, 1, sampleCount);
    CPPUNIT_ASSERT_EQUAL(std::size_t(7), queue.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(9), queue.capacity());
}

void CSampleQueueTest::testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(0, {1.0}, 1, sampleCount);

    queue.add(3, {2.5}, 2, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), queue.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(2.0, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(2), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(3.0, queue[0].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSampleButDifferentBucket(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue.latestEnd());

    queue.add(9, {1.0}, 1, sampleCount);
    queue.add(10, {2.5}, 2, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(10), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(10), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(2.5, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(10), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(2.0, queue[0].s_Statistic.count());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(9), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(9), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(1.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(9), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[1].s_Statistic.count());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(10), queue.latestEnd());
}

void CSampleQueueTest::testAddGivenTimeIsInOrderAndCloseToFullLatestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(0, {1.0}, 5, sampleCount);

    queue.add(3, {2.5}, 2, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(2.5, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(2.0, queue[0].s_Statistic.count());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(1.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(5.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsInOrderAndFarFromLatestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(0, {1.0}, 1, sampleCount);

    queue.add(5, {2.5}, 2, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(5), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(5), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(2.5, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(5), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(2.0, queue[0].s_Statistic.count());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(1.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsWithinFullLatestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(0, {1.0}, 2, sampleCount);
    queue.add(4, {1.0}, 3, sampleCount);

    queue.add(3, {7.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(4), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(2.0, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(6.0, queue[0].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndFarBeforeEarliestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(8, {1.0}, 5, sampleCount);
    queue.add(15, {1.0}, 3, sampleCount);

    queue.add(3, {7.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[2].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[2].s_End);
    CPPUNIT_ASSERT_EQUAL(7.0, queue[2].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[2].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[2].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloseBeforeFullEarliestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(8, {1.0}, 5, sampleCount);
    queue.add(15, {1.0}, 3, sampleCount);

    queue.add(5, {7.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(5), queue[2].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(5), queue[2].s_End);
    CPPUNIT_ASSERT_EQUAL(7.0, queue[2].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(5), queue[2].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[2].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(9, {1.0}, 4, sampleCount);
    queue.add(15, {1.0}, 3, sampleCount);

    queue.add(6, {6.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(6), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(9), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(2.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(8), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(5.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSampleButDifferentBucket(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(11, {1.0}, 4, sampleCount);

    queue.add(9, {6.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(11), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(11), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(1.0, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(11), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(4.0, queue[0].s_Statistic.count());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(9), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(9), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(6.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(9), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndWithinSomeSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 1, sampleCount);
    queue.add(6, {2.0}, 1, sampleCount);
    queue.add(8, {4.0}, 1, sampleCount);
    queue.add(12, {1.0}, 1, sampleCount);

    queue.add(7, {6.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(6), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(8), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(4.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(7), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(3.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatest(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 1, sampleCount);
    queue.add(5, {1.0}, 1, sampleCount);
    queue.add(10, {4.0}, 1, sampleCount);
    queue.add(15, {1.0}, 1, sampleCount);

    queue.add(12, {6.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(4), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(10), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(12), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(5.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(11), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(2.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatestButDifferentBucket(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 1, sampleCount);
    queue.add(8, {4.0}, 1, sampleCount);
    queue.add(15, {1.0}, 1, sampleCount);

    queue.add(10, {6.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(4), queue.size());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToPreviousOfNonFullSubSamples(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 1, sampleCount);
    queue.add(8, {2.0}, 1, sampleCount);
    queue.add(15, {3.0}, 1, sampleCount);

    queue.add(3, {7.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[2].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[2].s_End);
    CPPUNIT_ASSERT_EQUAL(4.0, queue[2].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(2), queue[2].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(2.0, queue[2].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToNextOfNonFullSubSamples(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 1, sampleCount);
    queue.add(8, {2.0}, 1, sampleCount);
    queue.add(15, {3.0}, 1, sampleCount);

    queue.add(5, {7.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(5), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(8), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(4.5, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(7), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(2.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToPreviousOfFullSubSamples(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 5, sampleCount);
    queue.add(8, {2.0}, 5, sampleCount);
    queue.add(15, {3.0}, 1, sampleCount);

    queue.add(3, {7.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[2].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[2].s_End);
    CPPUNIT_ASSERT_EQUAL(2.0, queue[2].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(1), queue[2].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(6.0, queue[2].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToNextOfFullSubSamples(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 5, sampleCount);
    queue.add(8, {2.0}, 5, sampleCount);
    queue.add(15, {3.0}, 5, sampleCount);

    queue.add(5, {8.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(5), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(8), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(3.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(8), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(6.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToPreviousSubSampleButOnlyNextHasSpace(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 5, sampleCount);
    queue.add(5, {2.0}, 1, sampleCount);
    queue.add(10, {3.0}, 1, sampleCount);

    queue.add(2, {7.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(2), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(5), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(4.5, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(4), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(2.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToNextSubSampleButOnlyPreviousHasSpace(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 1, sampleCount);
    queue.add(5, {2.0}, 5, sampleCount);
    queue.add(10, {3.0}, 5, sampleCount);

    queue.add(3, {8.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[2].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[2].s_End);
    CPPUNIT_ASSERT_EQUAL(4.5, queue[2].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(2), queue[2].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(2.0, queue[2].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndFallsInBigEnoughGap(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 1, sampleCount);
    queue.add(15, {2.0}, 1, sampleCount);

    queue.add(6, {8.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(6), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(6), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(8.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(6), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testAddGivenTimeIsHistoricalAndFallsInTooSmallGap(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 1, sampleCount);
    queue.add(4, {1.0}, 1, sampleCount);
    queue.add(9, {2.0}, 1, sampleCount);

    queue.add(6, {7.0}, 1, sampleCount);

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(6), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(3.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(3.0, queue[1].s_Statistic.count());
}

void CSampleQueueTest::testCanSampleGivenEmptyQueue(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    CPPUNIT_ASSERT(queue.canSample(42) == false);
}

void CSampleQueueTest::testCanSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(24, {1.0}, 1, sampleCount);
    queue.add(26, {1.0}, 1, sampleCount);
    queue.add(45, {1.0}, 5, sampleCount);

    CPPUNIT_ASSERT(queue.canSample(0) == false);
    CPPUNIT_ASSERT(queue.canSample(16) == false);

    CPPUNIT_ASSERT(queue.canSample(17));
    CPPUNIT_ASSERT(queue.canSample(40));
}

void CSampleQueueTest::testSampleGivenExactlyOneSampleOfExactCountToBeCreated(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    TTestSampleQueue::TSampleVec samples;
    queue.add(0, {1.0}, 5, sampleCount);
    queue.add(6, {3.0}, 5, sampleCount);
    queue.add(30, {5.0}, 1, sampleCount);
    CPPUNIT_ASSERT(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), samples.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), samples[0].time());
    CPPUNIT_ASSERT_EQUAL(2.0, samples[0].value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, samples[0].varianceScale());

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), queue.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(5.0, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[0].s_Statistic.count());
}

void CSampleQueueTest::testSampleGivenExactlyOneSampleOfOverCountToBeCreated(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    TTestSampleQueue::TSampleVec samples;
    queue.add(0, {0.0}, 3, sampleCount);
    queue.add(1, {1.0}, 1, sampleCount);
    queue.add(6, {3.0}, 7, sampleCount);
    queue.add(30, {5.0}, 1, sampleCount);
    CPPUNIT_ASSERT(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), samples.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(4), samples[0].time());
    CPPUNIT_ASSERT_EQUAL(2.0, samples[0].value()[0]);
    CPPUNIT_ASSERT(samples[0].varianceScale() < 1.0);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), queue.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(5.0, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[0].s_Statistic.count());
}

void CSampleQueueTest::testSampleGivenOneSampleToBeCreatedAndRemainder(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    TTestSampleQueue::TSampleVec samples;
    queue.add(0, {1.0}, 5, sampleCount);
    queue.add(6, {3.0}, 5, sampleCount);
    queue.add(7, {3.0}, 1, sampleCount);
    queue.add(8, {5.0}, 1, sampleCount);
    queue.add(40, {8.0}, 1, sampleCount);
    CPPUNIT_ASSERT(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), samples.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(3), samples[0].time());
    CPPUNIT_ASSERT_EQUAL(2.0, samples[0].value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, samples[0].varianceScale());

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(7), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(8), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(4.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(8), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(2.0, queue[1].s_Statistic.count());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(40), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(40), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(8.0, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(40), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[0].s_Statistic.count());
}

void CSampleQueueTest::testSampleGivenTwoSamplesToBeCreatedAndRemainder(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    TTestSampleQueue::TSampleVec samples;
    queue.add(0, {1.0}, 5, sampleCount);
    queue.add(2, {4.0}, 5, sampleCount);
    queue.add(7, {2.0}, 5, sampleCount);
    queue.add(8, {5.0}, 5, sampleCount);
    queue.add(9, {0.0}, 1, sampleCount);
    queue.add(30, {8.0}, 1, sampleCount);
    CPPUNIT_ASSERT(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), samples.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(1), samples[0].time());
    CPPUNIT_ASSERT_EQUAL(2.5, samples[0].value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, samples[0].varianceScale());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(8), samples[1].time());
    CPPUNIT_ASSERT_EQUAL(3.5, samples[1].value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, samples[1].varianceScale());

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(9), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(9), queue[1].s_End);
    CPPUNIT_ASSERT_EQUAL(0.0, queue[1].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(9), queue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[1].s_Statistic.count());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_End);
    CPPUNIT_ASSERT_EQUAL(8.0, queue[0].s_Statistic.value()[0]);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, queue[0].s_Statistic.count());
}

void CSampleQueueTest::testSampleGivenNoSampleToBeCreated(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    TTestSampleQueue::TSampleVec samples;
    queue.add(0, {1.0}, 4, sampleCount);
    queue.add(30, {5.0}, 1, sampleCount);
    CPPUNIT_ASSERT(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    CPPUNIT_ASSERT(samples.empty());

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), queue.size());
}

void CSampleQueueTest::testSampleGivenUsingSubSamplesUpToCountExceedItMoreThanUsingOneLess(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    TTestSampleQueue::TSampleVec samples;
    queue.add(0, {2.0}, 5, sampleCount);
    queue.add(2, {2.0}, 3, sampleCount);
    queue.add(10, {6.0}, 6, sampleCount);
    queue.add(30, {8.0}, 1, sampleCount);
    CPPUNIT_ASSERT(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), samples.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(1), samples[0].time());
    CPPUNIT_ASSERT_EQUAL(2.0, samples[0].value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.25, samples[0].varianceScale());
}

void CSampleQueueTest::testResetBucketGivenEmptyQueue(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.resetBucket(10);

    CPPUNIT_ASSERT(queue.empty());
}

void CSampleQueueTest::testResetBucketGivenBucketBeforeEarliestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(10, {1.0}, 5, sampleCount);
    queue.add(15, {1.0}, 5, sampleCount);
    queue.add(20, {1.0}, 5, sampleCount);
    queue.add(24, {1.0}, 5, sampleCount);
    queue.add(29, {1.0}, 5, sampleCount);
    queue.add(30, {1.0}, 5, sampleCount);

    queue.resetBucket(0);

    CPPUNIT_ASSERT_EQUAL(std::size_t(6), queue.size());
}

void CSampleQueueTest::testResetBucketGivenBucketAtEarliestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(10, {1.0}, 3, sampleCount);
    queue.add(11, {1.0}, 2, sampleCount);
    queue.add(15, {1.0}, 5, sampleCount);
    queue.add(20, {1.0}, 5, sampleCount);
    queue.add(24, {1.0}, 5, sampleCount);
    queue.add(29, {1.0}, 5, sampleCount);
    queue.add(30, {1.0}, 5, sampleCount);

    queue.resetBucket(10);

    CPPUNIT_ASSERT_EQUAL(std::size_t(4), queue.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(29), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(24), queue[2].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(20), queue[3].s_Start);
}

void CSampleQueueTest::testResetBucketGivenBucketInBetweenWithoutAnySubSamples(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(10, {1.0}, 5, sampleCount);
    queue.add(15, {1.0}, 5, sampleCount);
    queue.add(30, {1.0}, 5, sampleCount);

    queue.resetBucket(20);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());
}

void CSampleQueueTest::testResetBucketGivenBucketAtInBetweenSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(10, {1.0}, 5, sampleCount);
    queue.add(15, {1.0}, 5, sampleCount);
    queue.add(20, {1.0}, 5, sampleCount);
    queue.add(24, {1.0}, 5, sampleCount);
    queue.add(29, {1.0}, 5, sampleCount);
    queue.add(30, {1.0}, 5, sampleCount);

    queue.resetBucket(20);

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), queue.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(15), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(10), queue[2].s_Start);
}

void CSampleQueueTest::testResetBucketGivenBucketAtLatestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(10, {1.0}, 5, sampleCount);
    queue.add(15, {1.0}, 5, sampleCount);
    queue.add(20, {1.0}, 5, sampleCount);
    queue.add(24, {1.0}, 5, sampleCount);
    queue.add(29, {1.0}, 5, sampleCount);
    queue.add(30, {1.0}, 5, sampleCount);

    queue.resetBucket(30);

    CPPUNIT_ASSERT_EQUAL(std::size_t(5), queue.size());
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(29), queue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(24), queue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(20), queue[2].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(15), queue[3].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(10), queue[4].s_Start);
}

void CSampleQueueTest::testResetBucketGivenBucketAfterLatestSubSample(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(10, {1.0}, 5, sampleCount);
    queue.add(15, {1.0}, 5, sampleCount);
    queue.add(20, {1.0}, 5, sampleCount);
    queue.add(24, {1.0}, 5, sampleCount);
    queue.add(29, {1.0}, 5, sampleCount);
    queue.add(30, {1.0}, 5, sampleCount);

    queue.resetBucket(40);

    CPPUNIT_ASSERT_EQUAL(std::size_t(6), queue.size());
}

void CSampleQueueTest::testSubSamplesNeverSpanOverDifferentBuckets(void) {
    std::size_t sampleCountFactor(10);
    std::size_t latencyBuckets(3);
    double growthFactor(0.1);
    core_t::TTime bucketLength(600);
    unsigned int sampleCount(45);

    core_t::TTime latency = (latencyBuckets + 1) * bucketLength;
    std::size_t numberOfMeasurements = 5000;

    test::CRandomNumbers rng;

    core_t::TTime latestTime = bucketLength * (latencyBuckets + 1);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    for (std::size_t measurementId = 0; measurementId < numberOfMeasurements; ++measurementId) {
        TDoubleVec testData;
        rng.generateUniformSamples(static_cast<double>(latestTime - latency),
                                   static_cast<double>(latestTime), 1, testData);
        latestTime += 60 + static_cast<core_t::TTime>(40.0 * std::sin(
                                                          boost::math::constants::two_pi<double>()
                                                          * static_cast<double>(latestTime % 86400) / 86400.0));
        core_t::TTime measurementTime = static_cast<core_t::TTime>(testData[0]);
        queue.add(measurementTime, {1.0}, 1u, sampleCount);
    }

    for (std::size_t i = 0; i < queue.size(); ++i) {
        core_t::TTime startBucket = maths::CIntegerTools::floor(queue[i].s_Start, bucketLength);
        core_t::TTime endBucket = maths::CIntegerTools::floor(queue[i].s_End, bucketLength);
        CPPUNIT_ASSERT_EQUAL(startBucket, endBucket);
    }
}

void CSampleQueueTest::testPersistence(void) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(0, {1.0}, 3, sampleCount);
    queue.add(2, {3.5}, 2, sampleCount);
    queue.add(30, {8.0}, 1, sampleCount);

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        queue.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_DEBUG("XML:\n" << origXml);

    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    TTestSampleQueue restoredQueue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    traverser.traverseSubLevel(boost::bind(&TTestSampleQueue::acceptRestoreTraverser,
                                           &restoredQueue,
                                           _1));

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), restoredQueue.size());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), restoredQueue[1].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(2), restoredQueue[1].s_End);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, restoredQueue[1].s_Statistic.value()[0], 0.0001);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(1), restoredQueue[1].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(5.0, restoredQueue[1].s_Statistic.count());

    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), restoredQueue[0].s_Start);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), restoredQueue[0].s_End);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, restoredQueue[0].s_Statistic.value()[0], 0.0001);
    CPPUNIT_ASSERT_EQUAL(core_t::TTime(30), restoredQueue[0].s_Statistic.time());
    CPPUNIT_ASSERT_EQUAL(1.0, restoredQueue[0].s_Statistic.count());
}

void CSampleQueueTest::testQualityOfSamplesGivenConstantRate(void) {
    std::size_t sampleCountFactor(5);
    std::size_t latencyBuckets(3);
    double growthFactor(0.1);
    core_t::TTime bucketLength(600);
    unsigned int sampleCount(30);

    core_t::TTime latency = (latencyBuckets + 1) * bucketLength;
    std::size_t numberOfMeasurements = 5000;
    std::size_t numberOfRuns = 100;

    test::CRandomNumbers rng;

    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanQueueSize;
    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanMinVariance;
    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanMaxVariance;

    for (std::size_t runId = 0; runId < numberOfRuns; ++runId) {
        TSampleVec samples;
        core_t::TTime latestTime = bucketLength * (latencyBuckets + 1);
        TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
        for (std::size_t measurementId = 0; measurementId < numberOfMeasurements; ++measurementId) {
            TDoubleVec testData;
            rng.generateUniformSamples(static_cast<double>(latestTime - latency),
                                       static_cast<double>(latestTime), 1, testData);
            latestTime += 60;
            core_t::TTime measurementTime = static_cast<core_t::TTime>(testData[0]);
            queue.add(measurementTime, {1.0}, 1u, sampleCount);
        }
        meanQueueSize.add(queue.size());
        queue.sample(latestTime, sampleCount, model_t::E_IndividualMeanByPerson, samples);

        maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator varianceStat;
        maths::CBasicStatistics::COrderStatisticsStack<double, 5u> varianceMin;
        maths::CBasicStatistics::COrderStatisticsStack<double, 5u, std::greater<double> > varianceMax;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            varianceStat.add(samples[i].varianceScale());
            varianceMin.add(samples[i].varianceScale());
            varianceMax.add(samples[i].varianceScale());
        }
        varianceMin.sort();
        varianceMax.sort();
        meanMinVariance.add(varianceMin[0]);
        meanMaxVariance.add(varianceMax[0]);

        LOG_DEBUG("Results for run: " << runId);
        LOG_DEBUG("Mean variance scale = " << maths::CBasicStatistics::mean(varianceStat));
        LOG_DEBUG("Variance of variance scale = " << maths::CBasicStatistics::variance(varianceStat));
        LOG_DEBUG("Top min variance scale = " << varianceMin.print());
        LOG_DEBUG("Top max variance scale = " << varianceMax.print());
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(varianceStat) > 0.98);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(varianceStat) < 1.01);
        CPPUNIT_ASSERT(maths::CBasicStatistics::variance(varianceStat) < 0.0025);
        CPPUNIT_ASSERT(varianceMin[0] > 0.85);
        CPPUNIT_ASSERT(varianceMax[0] < 1.12);
    }
    LOG_DEBUG("Mean queue size = " << maths::CBasicStatistics::mean(meanQueueSize));
    LOG_DEBUG("Mean min variance = " << maths::CBasicStatistics::mean(meanMinVariance));
    LOG_DEBUG("Mean max variance = " << maths::CBasicStatistics::mean(meanMaxVariance));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMinVariance) > 0.90);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMaxVariance) < 1.1);
}

void CSampleQueueTest::testQualityOfSamplesGivenVariableRate(void) {
    std::size_t sampleCountFactor(5);
    std::size_t latencyBuckets(3);
    double growthFactor(0.1);
    core_t::TTime bucketLength(600);
    unsigned int sampleCount(30);

    core_t::TTime latency = (latencyBuckets + 1) * bucketLength;
    std::size_t numberOfMeasurements = 5000;
    std::size_t numberOfRuns = 100;

    test::CRandomNumbers rng;

    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanQueueSize;
    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanMinVariance;
    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanMaxVariance;

    for (std::size_t runId = 0; runId < numberOfRuns; ++runId) {
        TSampleVec samples;
        core_t::TTime latestTime = bucketLength * (latencyBuckets + 1);
        TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
        for (std::size_t measurementId = 0; measurementId < numberOfMeasurements; ++measurementId) {
            TDoubleVec testData;
            rng.generateUniformSamples(static_cast<double>(latestTime - latency),
                                       static_cast<double>(latestTime), 1, testData);
            latestTime += 60 + static_cast<core_t::TTime>(40.0 * std::sin(
                                                              boost::math::constants::two_pi<double>()
                                                              * static_cast<double>(latestTime % 86400) / 86400.0));
            core_t::TTime measurementTime = static_cast<core_t::TTime>(testData[0]);
            queue.add(measurementTime, {1.0}, 1u, sampleCount);
        }
        meanQueueSize.add(queue.size());
        queue.sample(latestTime, sampleCount, model_t::E_IndividualMeanByPerson, samples);

        maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator varianceStat;
        maths::CBasicStatistics::COrderStatisticsStack<double, 5u> varianceMin;
        maths::CBasicStatistics::COrderStatisticsStack<double, 5u, std::greater<double> > varianceMax;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            varianceStat.add(samples[i].varianceScale());
            varianceMin.add(samples[i].varianceScale());
            varianceMax.add(samples[i].varianceScale());
        }
        varianceMin.sort();
        varianceMax.sort();
        meanMinVariance.add(varianceMin[0]);
        meanMaxVariance.add(varianceMax[0]);

        LOG_DEBUG("Results for run: " << runId);
        LOG_DEBUG("Mean variance scale = " << maths::CBasicStatistics::mean(varianceStat));
        LOG_DEBUG("Variance of variance scale = " << maths::CBasicStatistics::variance(varianceStat));
        LOG_DEBUG("Top min variance scale = " << varianceMin.print());
        LOG_DEBUG("Top max variance scale = " << varianceMax.print());
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(varianceStat) > 0.97);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(varianceStat) < 1.01);
        CPPUNIT_ASSERT(maths::CBasicStatistics::variance(varianceStat) < 0.0065);
        CPPUNIT_ASSERT(varianceMin[0] > 0.74);
        CPPUNIT_ASSERT(varianceMax[0] < 1.26);
    }
    LOG_DEBUG("Mean queue size = " << maths::CBasicStatistics::mean(meanQueueSize));
    LOG_DEBUG("Mean min variance = " << maths::CBasicStatistics::mean(meanMinVariance));
    LOG_DEBUG("Mean max variance = " << maths::CBasicStatistics::mean(meanMaxVariance));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMinVariance) > 0.82);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanMaxVariance) < 1.16);
}

void CSampleQueueTest::testQualityOfSamplesGivenHighLatencyAndDataInReverseOrder(void) {
    std::size_t sampleCountFactor(5);
    std::size_t latencyBuckets(500);
    double growthFactor(0.1);
    core_t::TTime bucketLength(600);
    unsigned int sampleCount(30);

    std::size_t numberOfMeasurements = 5000;

    test::CRandomNumbers rng;

    TSampleVec samples;
    core_t::TTime latestTime = 60 * numberOfMeasurements;
    core_t::TTime time = latestTime;
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    for (std::size_t measurementId = 0; measurementId < numberOfMeasurements; ++measurementId) {
        queue.add(time, {1.0}, 1u, sampleCount);
        time -= 60;
    }
    queue.add(360000, {1.0}, 1u, sampleCount);
    queue.sample(latestTime, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator varianceStat;
    maths::CBasicStatistics::COrderStatisticsStack<double, 1u> varianceMin;
    maths::CBasicStatistics::COrderStatisticsStack<double, 1u, std::greater<double> > varianceMax;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        varianceStat.add(samples[i].varianceScale());
        varianceMin.add(samples[i].varianceScale());
        varianceMax.add(samples[i].varianceScale());
    }

    LOG_DEBUG("Mean variance scale = " << maths::CBasicStatistics::mean(varianceStat));
    LOG_DEBUG("Variance of variance scale = " << maths::CBasicStatistics::variance(varianceStat));
    LOG_DEBUG("Min variance scale = " << varianceMin[0]);
    LOG_DEBUG("Max variance scale = " << varianceMax[0]);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(varianceStat) >= 0.999);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(varianceStat) <= 1.0);
    CPPUNIT_ASSERT(maths::CBasicStatistics::variance(varianceStat) <= 0.0001);
    CPPUNIT_ASSERT(varianceMin[0] > 0.96);
    CPPUNIT_ASSERT(varianceMax[0] <= 1.0);
}

CppUnit::Test *CSampleQueueTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CSampleQueueTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testSampleToString",
                               &CSampleQueueTest::testSampleToString));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testSampleFromString",
                               &CSampleQueueTest::testSampleFromString));

    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenQueueIsEmptyShouldCreateNewSubSample",
                               &CSampleQueueTest::testAddGivenQueueIsEmptyShouldCreateNewSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenQueueIsFullShouldResize",
                               &CSampleQueueTest::testAddGivenQueueIsFullShouldResize));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSample",
                               &CSampleQueueTest::testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSampleButDifferentBucket",
                               &CSampleQueueTest::testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSampleButDifferentBucket));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsInOrderAndCloseToFullLatestSubSample",
                               &CSampleQueueTest::testAddGivenTimeIsInOrderAndCloseToFullLatestSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsInOrderAndFarFromLatestSubSample",
                               &CSampleQueueTest::testAddGivenTimeIsInOrderAndFarFromLatestSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsWithinFullLatestSubSample",
                               &CSampleQueueTest::testAddGivenTimeIsWithinFullLatestSubSample));

    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndFarBeforeEarliestSubSample",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndFarBeforeEarliestSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloseBeforeFullEarliestSubSample",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloseBeforeFullEarliestSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSample",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSampleButDifferentBucket",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSampleButDifferentBucket));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndWithinSomeSubSample",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndWithinSomeSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatest",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatest));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatestButDifferentBucket",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatestButDifferentBucket));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToPreviousOfNonFullSubSamples",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToPreviousOfNonFullSubSamples));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToNextOfNonFullSubSamples",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToNextOfNonFullSubSamples));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToPreviousOfFullSubSamples",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToPreviousOfFullSubSamples));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToNextOfFullSubSamples",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToNextOfFullSubSamples));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToPreviousSubSampleButOnlyNextHasSpace",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToPreviousSubSampleButOnlyNextHasSpace));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToNextSubSampleButOnlyPreviousHasSpace",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndCloserToNextSubSampleButOnlyPreviousHasSpace));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndFallsInBigEnoughGap",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndFallsInBigEnoughGap));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testAddGivenTimeIsHistoricalAndFallsInTooSmallGap",
                               &CSampleQueueTest::testAddGivenTimeIsHistoricalAndFallsInTooSmallGap));

    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testCanSampleGivenEmptyQueue",
                               &CSampleQueueTest::testCanSampleGivenEmptyQueue));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testCanSample",
                               &CSampleQueueTest::testCanSample));

    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testSampleGivenExactlyOneSampleOfExactCountToBeCreated",
                               &CSampleQueueTest::testSampleGivenExactlyOneSampleOfExactCountToBeCreated));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testSampleGivenExactlyOneSampleOfOverCountToBeCreated",
                               &CSampleQueueTest::testSampleGivenExactlyOneSampleOfOverCountToBeCreated));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testSampleGivenOneSampleToBeCreatedAndRemainder",
                               &CSampleQueueTest::testSampleGivenOneSampleToBeCreatedAndRemainder));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testSampleGivenTwoSamplesToBeCreatedAndRemainder",
                               &CSampleQueueTest::testSampleGivenTwoSamplesToBeCreatedAndRemainder));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testSampleGivenNoSampleToBeCreated",
                               &CSampleQueueTest::testSampleGivenNoSampleToBeCreated));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testSampleGivenUsingSubSamplesUpToCountExceedItMoreThanUsingOneLess",
                               &CSampleQueueTest::testSampleGivenUsingSubSamplesUpToCountExceedItMoreThanUsingOneLess));

    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testResetBucketGivenEmptyQueue",
                               &CSampleQueueTest::testResetBucketGivenEmptyQueue));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testResetBucketGivenBucketBeforeEarliestSubSample",
                               &CSampleQueueTest::testResetBucketGivenBucketBeforeEarliestSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testResetBucketGivenBucketAtEarliestSubSample",
                               &CSampleQueueTest::testResetBucketGivenBucketAtEarliestSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testResetBucketGivenBucketInBetweenWithoutAnySubSamples",
                               &CSampleQueueTest::testResetBucketGivenBucketInBetweenWithoutAnySubSamples));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testResetBucketGivenBucketAtInBetweenSubSample",
                               &CSampleQueueTest::testResetBucketGivenBucketAtInBetweenSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testResetBucketGivenBucketAtLatestSubSample",
                               &CSampleQueueTest::testResetBucketGivenBucketAtLatestSubSample));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testResetBucketGivenBucketAfterLatestSubSample",
                               &CSampleQueueTest::testResetBucketGivenBucketAfterLatestSubSample));

    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testSubSamplesNeverSpanOverDifferentBuckets",
                               &CSampleQueueTest::testSubSamplesNeverSpanOverDifferentBuckets));

    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testPersistence",
                               &CSampleQueueTest::testPersistence));

    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testQualityOfSamplesGivenConstantRate",
                               &CSampleQueueTest::testQualityOfSamplesGivenConstantRate));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testQualityOfSamplesGivenVariableRate",
                               &CSampleQueueTest::testQualityOfSamplesGivenVariableRate));
    suiteOfTests->addTest( new CppUnit::TestCaller<CSampleQueueTest>(
                               "CSampleQueueTest::testQualityOfSamplesGivenHighLatencyAndDataInReverseOrder",
                               &CSampleQueueTest::testQualityOfSamplesGivenHighLatencyAndDataInReverseOrder));

    return suiteOfTests;
}
