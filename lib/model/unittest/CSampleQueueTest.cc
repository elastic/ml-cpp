/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

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

#include <test/BoostTestCloseAbsolute.h>

#include <boost/math/constants/constants.hpp>
#include <boost/test/unit_test.hpp>

#include <test/CRandomNumbers.h>

BOOST_AUTO_TEST_SUITE(CSampleQueueTest)

using namespace ml;
using namespace model;

using TDoubleVec = std::vector<double>;
using TSampleVec = std::vector<CSample>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TTestSampleQueue = CSampleQueue<TMeanAccumulator>;

BOOST_AUTO_TEST_CASE(testSampleToString) {
    CSample sample(10, {3.0}, 0.8, 1.0);

    BOOST_REQUIRE_EQUAL(std::string("10;8e-1;1;3"), CSample::SToString()(sample));
}

BOOST_AUTO_TEST_CASE(testSampleFromString) {
    CSample sample;

    BOOST_TEST_REQUIRE(CSample::SFromString()("15;7e-1;3;2.0", sample));

    BOOST_REQUIRE_EQUAL(core_t::TTime(15), sample.time());
    BOOST_REQUIRE_EQUAL(2.0, sample.value()[0]);
    BOOST_REQUIRE_EQUAL(0.7, sample.varianceScale());
    BOOST_REQUIRE_EQUAL(3.0, sample.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenQueueIsEmptyShouldCreateNewSubSample) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(1, {1.0}, 1, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(1), queue.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(1), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(1), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(core_t::TTime(1), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, queue[0].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenQueueIsFullShouldResize) {
    std::size_t sampleCountFactor(1);
    std::size_t latencyBuckets(1);
    double growthFactor(0.5);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(1);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(1, {1.0}, 1, sampleCount);
    BOOST_REQUIRE_EQUAL(std::size_t(1), queue.size());
    BOOST_REQUIRE_EQUAL(std::size_t(1), queue.capacity());

    queue.add(2, {2.0}, 1, sampleCount);
    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.size());
    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.capacity());

    queue.add(3, {3.0}, 1, sampleCount);
    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());
    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.capacity());

    queue.add(4, {4.0}, 1, sampleCount);
    BOOST_REQUIRE_EQUAL(std::size_t(4), queue.size());
    BOOST_REQUIRE_EQUAL(std::size_t(4), queue.capacity());

    queue.add(5, {5.0}, 1, sampleCount);
    BOOST_REQUIRE_EQUAL(std::size_t(5), queue.size());
    BOOST_REQUIRE_EQUAL(std::size_t(6), queue.capacity());

    queue.add(6, {6.0}, 1, sampleCount);
    BOOST_REQUIRE_EQUAL(std::size_t(6), queue.size());
    BOOST_REQUIRE_EQUAL(std::size_t(6), queue.capacity());

    queue.add(7, {7.0}, 1, sampleCount);
    BOOST_REQUIRE_EQUAL(std::size_t(7), queue.size());
    BOOST_REQUIRE_EQUAL(std::size_t(9), queue.capacity());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSample) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(0, {1.0}, 1, sampleCount);

    queue.add(3, {2.5}, 2, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(1), queue.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(2.0, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(2), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(3.0, queue[0].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSampleButDifferentBucket) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue.latestEnd());

    queue.add(9, {1.0}, 1, sampleCount);
    queue.add(10, {2.5}, 2, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(10), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(10), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(2.5, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(10), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(2.0, queue[0].s_Statistic.count());

    BOOST_REQUIRE_EQUAL(core_t::TTime(9), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(9), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(1.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(9), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[1].s_Statistic.count());

    BOOST_REQUIRE_EQUAL(core_t::TTime(10), queue.latestEnd());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsInOrderAndCloseToFullLatestSubSample) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(0, {1.0}, 5, sampleCount);

    queue.add(3, {2.5}, 2, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(2.5, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(2.0, queue[0].s_Statistic.count());

    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(1.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(5.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsInOrderAndFarFromLatestSubSample) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(0, {1.0}, 1, sampleCount);

    queue.add(5, {2.5}, 2, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(5), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(5), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(2.5, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(5), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(2.0, queue[0].s_Statistic.count());

    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(1.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsWithinFullLatestSubSample) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(0, {1.0}, 2, sampleCount);
    queue.add(4, {1.0}, 3, sampleCount);

    queue.add(3, {7.0}, 1, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(1), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(4), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(2.0, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(6.0, queue[0].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndFarBeforeEarliestSubSample) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(8, {1.0}, 5, sampleCount);
    queue.add(15, {1.0}, 3, sampleCount);

    queue.add(3, {7.0}, 1, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[2].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[2].s_End);
    BOOST_REQUIRE_EQUAL(7.0, queue[2].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[2].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[2].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloseBeforeFullEarliestSubSample) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(8, {1.0}, 5, sampleCount);
    queue.add(15, {1.0}, 3, sampleCount);

    queue.add(5, {7.0}, 1, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(5), queue[2].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(5), queue[2].s_End);
    BOOST_REQUIRE_EQUAL(7.0, queue[2].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(5), queue[2].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[2].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSample) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(9, {1.0}, 4, sampleCount);
    queue.add(15, {1.0}, 3, sampleCount);

    queue.add(6, {6.0}, 1, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(6), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(9), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(2.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(8), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(5.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSampleButDifferentBucket) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(11, {1.0}, 4, sampleCount);

    queue.add(9, {6.0}, 1, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(11), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(11), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(1.0, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(11), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(4.0, queue[0].s_Statistic.count());

    BOOST_REQUIRE_EQUAL(core_t::TTime(9), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(9), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(6.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(9), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndWithinSomeSubSample) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(6), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(8), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(4.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(7), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(3.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatest) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(4), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(10), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(12), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(5.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(11), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(2.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatestButDifferentBucket) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(4), queue.size());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloserToPreviousOfNonFullSubSamples) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[2].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[2].s_End);
    BOOST_REQUIRE_EQUAL(4.0, queue[2].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(2), queue[2].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(2.0, queue[2].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloserToNextOfNonFullSubSamples) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(5), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(8), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(4.5, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(7), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(2.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloserToPreviousOfFullSubSamples) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[2].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[2].s_End);
    BOOST_REQUIRE_EQUAL(2.0, queue[2].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(1), queue[2].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(6.0, queue[2].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloserToNextOfFullSubSamples) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(5), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(8), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(3.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(8), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(6.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloserToPreviousSubSampleButOnlyNextHasSpace) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(2), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(5), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(4.5, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(4), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(2.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndCloserToNextSubSampleButOnlyPreviousHasSpace) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[2].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[2].s_End);
    BOOST_REQUIRE_EQUAL(4.5, queue[2].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(2), queue[2].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(2.0, queue[2].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndFallsInBigEnoughGap) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(5);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.add(0, {1.0}, 1, sampleCount);
    queue.add(15, {2.0}, 1, sampleCount);

    queue.add(6, {8.0}, 1, sampleCount);

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(6), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(6), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(8.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(6), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testAddGivenTimeIsHistoricalAndFallsInTooSmallGap) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(0), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(6), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(3.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(3.0, queue[1].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testCanSampleGivenEmptyQueue) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    BOOST_TEST_REQUIRE(queue.canSample(42) == false);
}

BOOST_AUTO_TEST_CASE(testCanSample) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    queue.add(24, {1.0}, 1, sampleCount);
    queue.add(26, {1.0}, 1, sampleCount);
    queue.add(45, {1.0}, 5, sampleCount);

    BOOST_TEST_REQUIRE(queue.canSample(0) == false);
    BOOST_TEST_REQUIRE(queue.canSample(16) == false);

    BOOST_TEST_REQUIRE(queue.canSample(17));
    BOOST_TEST_REQUIRE(queue.canSample(40));
}

BOOST_AUTO_TEST_CASE(testSampleGivenExactlyOneSampleOfExactCountToBeCreated) {
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
    BOOST_TEST_REQUIRE(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    BOOST_REQUIRE_EQUAL(std::size_t(1), samples.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), samples[0].time());
    BOOST_REQUIRE_EQUAL(2.0, samples[0].value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, samples[0].varianceScale());

    BOOST_REQUIRE_EQUAL(std::size_t(1), queue.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(5.0, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[0].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testSampleGivenExactlyOneSampleOfOverCountToBeCreated) {
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
    BOOST_TEST_REQUIRE(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    BOOST_REQUIRE_EQUAL(std::size_t(1), samples.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(4), samples[0].time());
    BOOST_REQUIRE_EQUAL(2.0, samples[0].value()[0]);
    BOOST_TEST_REQUIRE(samples[0].varianceScale() < 1.0);

    BOOST_REQUIRE_EQUAL(std::size_t(1), queue.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(5.0, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[0].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testSampleGivenOneSampleToBeCreatedAndRemainder) {
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
    BOOST_TEST_REQUIRE(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    BOOST_REQUIRE_EQUAL(std::size_t(1), samples.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(3), samples[0].time());
    BOOST_REQUIRE_EQUAL(2.0, samples[0].value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, samples[0].varianceScale());

    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(7), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(8), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(4.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(8), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(2.0, queue[1].s_Statistic.count());

    BOOST_REQUIRE_EQUAL(core_t::TTime(40), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(40), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(8.0, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(40), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[0].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testSampleGivenTwoSamplesToBeCreatedAndRemainder) {
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
    BOOST_TEST_REQUIRE(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    BOOST_REQUIRE_EQUAL(std::size_t(2), samples.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(1), samples[0].time());
    BOOST_REQUIRE_EQUAL(2.5, samples[0].value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, samples[0].varianceScale());

    BOOST_REQUIRE_EQUAL(core_t::TTime(8), samples[1].time());
    BOOST_REQUIRE_EQUAL(3.5, samples[1].value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, samples[1].varianceScale());

    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(9), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(9), queue[1].s_End);
    BOOST_REQUIRE_EQUAL(0.0, queue[1].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(9), queue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[1].s_Statistic.count());

    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_End);
    BOOST_REQUIRE_EQUAL(8.0, queue[0].s_Statistic.value()[0]);
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, queue[0].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testSampleGivenNoSampleToBeCreated) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    unsigned int sampleCount(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);
    TTestSampleQueue::TSampleVec samples;
    queue.add(0, {1.0}, 4, sampleCount);
    queue.add(30, {5.0}, 1, sampleCount);
    BOOST_TEST_REQUIRE(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    BOOST_TEST_REQUIRE(samples.empty());

    BOOST_REQUIRE_EQUAL(std::size_t(2), queue.size());
}

BOOST_AUTO_TEST_CASE(testSampleGivenUsingSubSamplesUpToCountExceedItMoreThanUsingOneLess) {
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
    BOOST_TEST_REQUIRE(queue.canSample(0));

    queue.sample(0, sampleCount, model_t::E_IndividualMeanByPerson, samples);

    BOOST_REQUIRE_EQUAL(std::size_t(1), samples.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(1), samples[0].time());
    BOOST_REQUIRE_EQUAL(2.0, samples[0].value()[0]);
    BOOST_REQUIRE_EQUAL(1.25, samples[0].varianceScale());
}

BOOST_AUTO_TEST_CASE(testResetBucketGivenEmptyQueue) {
    std::size_t sampleCountFactor(2);
    std::size_t latencyBuckets(2);
    double growthFactor(0.1);
    core_t::TTime bucketLength(10);
    TTestSampleQueue queue(1, sampleCountFactor, latencyBuckets, growthFactor, bucketLength);

    queue.resetBucket(10);

    BOOST_TEST_REQUIRE(queue.empty());
}

BOOST_AUTO_TEST_CASE(testResetBucketGivenBucketBeforeEarliestSubSample) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(6), queue.size());
}

BOOST_AUTO_TEST_CASE(testResetBucketGivenBucketAtEarliestSubSample) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(4), queue.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(29), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(24), queue[2].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(20), queue[3].s_Start);
}

BOOST_AUTO_TEST_CASE(testResetBucketGivenBucketInBetweenWithoutAnySubSamples) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());
}

BOOST_AUTO_TEST_CASE(testResetBucketGivenBucketAtInBetweenSubSample) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(3), queue.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(15), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(10), queue[2].s_Start);
}

BOOST_AUTO_TEST_CASE(testResetBucketGivenBucketAtLatestSubSample) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(5), queue.size());
    BOOST_REQUIRE_EQUAL(core_t::TTime(29), queue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(24), queue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(20), queue[2].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(15), queue[3].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(10), queue[4].s_Start);
}

BOOST_AUTO_TEST_CASE(testResetBucketGivenBucketAfterLatestSubSample) {
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

    BOOST_REQUIRE_EQUAL(std::size_t(6), queue.size());
}

BOOST_AUTO_TEST_CASE(testSubSamplesNeverSpanOverDifferentBuckets) {
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
        latestTime +=
            60 + static_cast<core_t::TTime>(
                     40.0 * std::sin(boost::math::constants::two_pi<double>() *
                                     static_cast<double>(latestTime % 86400) / 86400.0));
        core_t::TTime measurementTime = static_cast<core_t::TTime>(testData[0]);
        queue.add(measurementTime, {1.0}, 1u, sampleCount);
    }

    for (std::size_t i = 0; i < queue.size(); ++i) {
        core_t::TTime startBucket = maths::CIntegerTools::floor(queue[i].s_Start, bucketLength);
        core_t::TTime endBucket = maths::CIntegerTools::floor(queue[i].s_End, bucketLength);
        BOOST_REQUIRE_EQUAL(startBucket, endBucket);
    }
}

BOOST_AUTO_TEST_CASE(testPersistence) {
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
    LOG_DEBUG(<< "XML:\n" << origXml);

    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    TTestSampleQueue restoredQueue(1, sampleCountFactor, latencyBuckets,
                                   growthFactor, bucketLength);
    traverser.traverseSubLevel(std::bind(&TTestSampleQueue::acceptRestoreTraverser,
                                         &restoredQueue, std::placeholders::_1));

    BOOST_REQUIRE_EQUAL(std::size_t(2), restoredQueue.size());

    BOOST_REQUIRE_EQUAL(core_t::TTime(0), restoredQueue[1].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(2), restoredQueue[1].s_End);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(2.0, restoredQueue[1].s_Statistic.value()[0], 0.0001);
    BOOST_REQUIRE_EQUAL(core_t::TTime(1), restoredQueue[1].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(5.0, restoredQueue[1].s_Statistic.count());

    BOOST_REQUIRE_EQUAL(core_t::TTime(30), restoredQueue[0].s_Start);
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), restoredQueue[0].s_End);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(8.0, restoredQueue[0].s_Statistic.value()[0], 0.0001);
    BOOST_REQUIRE_EQUAL(core_t::TTime(30), restoredQueue[0].s_Statistic.time());
    BOOST_REQUIRE_EQUAL(1.0, restoredQueue[0].s_Statistic.count());
}

BOOST_AUTO_TEST_CASE(testQualityOfSamplesGivenConstantRate) {
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
        for (std::size_t measurementId = 0;
             measurementId < numberOfMeasurements; ++measurementId) {
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
        maths::CBasicStatistics::COrderStatisticsStack<double, 5u, std::greater<double>> varianceMax;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            varianceStat.add(samples[i].varianceScale());
            varianceMin.add(samples[i].varianceScale());
            varianceMax.add(samples[i].varianceScale());
        }
        varianceMin.sort();
        varianceMax.sort();
        meanMinVariance.add(varianceMin[0]);
        meanMaxVariance.add(varianceMax[0]);

        LOG_DEBUG(<< "Results for run: " << runId);
        LOG_DEBUG(<< "Mean variance scale = " << maths::CBasicStatistics::mean(varianceStat));
        LOG_DEBUG(<< "Variance of variance scale = "
                  << maths::CBasicStatistics::variance(varianceStat));
        LOG_DEBUG(<< "Top min variance scale = " << varianceMin.print());
        LOG_DEBUG(<< "Top max variance scale = " << varianceMax.print());
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(varianceStat) > 0.98);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(varianceStat) < 1.01);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(varianceStat) < 0.0025);
        BOOST_TEST_REQUIRE(varianceMin[0] > 0.85);
        BOOST_TEST_REQUIRE(varianceMax[0] < 1.12);
    }
    LOG_DEBUG(<< "Mean queue size = " << maths::CBasicStatistics::mean(meanQueueSize));
    LOG_DEBUG(<< "Mean min variance = " << maths::CBasicStatistics::mean(meanMinVariance));
    LOG_DEBUG(<< "Mean max variance = " << maths::CBasicStatistics::mean(meanMaxVariance));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanMinVariance) > 0.90);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanMaxVariance) < 1.1);
}

BOOST_AUTO_TEST_CASE(testQualityOfSamplesGivenVariableRate) {
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
        for (std::size_t measurementId = 0;
             measurementId < numberOfMeasurements; ++measurementId) {
            TDoubleVec testData;
            rng.generateUniformSamples(static_cast<double>(latestTime - latency),
                                       static_cast<double>(latestTime), 1, testData);
            latestTime +=
                60 + static_cast<core_t::TTime>(
                         40.0 * std::sin(boost::math::constants::two_pi<double>() *
                                         static_cast<double>(latestTime % 86400) / 86400.0));
            core_t::TTime measurementTime = static_cast<core_t::TTime>(testData[0]);
            queue.add(measurementTime, {1.0}, 1u, sampleCount);
        }
        meanQueueSize.add(queue.size());
        queue.sample(latestTime, sampleCount, model_t::E_IndividualMeanByPerson, samples);

        maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator varianceStat;
        maths::CBasicStatistics::COrderStatisticsStack<double, 5u> varianceMin;
        maths::CBasicStatistics::COrderStatisticsStack<double, 5u, std::greater<double>> varianceMax;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            varianceStat.add(samples[i].varianceScale());
            varianceMin.add(samples[i].varianceScale());
            varianceMax.add(samples[i].varianceScale());
        }
        varianceMin.sort();
        varianceMax.sort();
        meanMinVariance.add(varianceMin[0]);
        meanMaxVariance.add(varianceMax[0]);

        LOG_DEBUG(<< "Results for run: " << runId);
        LOG_DEBUG(<< "Mean variance scale = " << maths::CBasicStatistics::mean(varianceStat));
        LOG_DEBUG(<< "Variance of variance scale = "
                  << maths::CBasicStatistics::variance(varianceStat));
        LOG_DEBUG(<< "Top min variance scale = " << varianceMin.print());
        LOG_DEBUG(<< "Top max variance scale = " << varianceMax.print());
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(varianceStat) > 0.97);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(varianceStat) < 1.01);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(varianceStat) < 0.0065);
        BOOST_TEST_REQUIRE(varianceMin[0] > 0.74);
        BOOST_TEST_REQUIRE(varianceMax[0] < 1.26);
    }
    LOG_DEBUG(<< "Mean queue size = " << maths::CBasicStatistics::mean(meanQueueSize));
    LOG_DEBUG(<< "Mean min variance = " << maths::CBasicStatistics::mean(meanMinVariance));
    LOG_DEBUG(<< "Mean max variance = " << maths::CBasicStatistics::mean(meanMaxVariance));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanMinVariance) > 0.82);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanMaxVariance) < 1.16);
}

BOOST_AUTO_TEST_CASE(testQualityOfSamplesGivenHighLatencyAndDataInReverseOrder) {
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
    maths::CBasicStatistics::COrderStatisticsStack<double, 1u, std::greater<double>> varianceMax;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        varianceStat.add(samples[i].varianceScale());
        varianceMin.add(samples[i].varianceScale());
        varianceMax.add(samples[i].varianceScale());
    }

    LOG_DEBUG(<< "Mean variance scale = " << maths::CBasicStatistics::mean(varianceStat));
    LOG_DEBUG(<< "Variance of variance scale = "
              << maths::CBasicStatistics::variance(varianceStat));
    LOG_DEBUG(<< "Min variance scale = " << varianceMin[0]);
    LOG_DEBUG(<< "Max variance scale = " << varianceMax[0]);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(varianceStat) >= 0.999);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(varianceStat) <= 1.0);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(varianceStat) <= 0.0001);
    BOOST_TEST_REQUIRE(varianceMin[0] > 0.96);
    BOOST_TEST_REQUIRE(varianceMax[0] <= 1.0);
}

BOOST_AUTO_TEST_SUITE_END()
