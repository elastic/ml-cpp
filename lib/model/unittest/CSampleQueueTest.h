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
#ifndef INCLUDED_CSampleQueueTest_h
#define INCLUDED_CSampleQueueTest_h

#include <cppunit/extensions/HelperMacros.h>

#include "../../../include/model/CSampleQueue.h"

class CSampleQueueTest : public CppUnit::TestFixture
{
    public:
        void testSampleToString(void);
        void testSampleFromString(void);

        void testAddGivenQueueIsEmptyShouldCreateNewSubSample(void);
        void testAddGivenQueueIsFullShouldResize(void);
        void testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSample(void);
        void testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSampleButDifferentBucket(void);
        void testAddGivenTimeIsInOrderAndCloseToFullLatestSubSample(void);
        void testAddGivenTimeIsInOrderAndFarFromLatestSubSample(void);
        void testAddGivenTimeIsWithinFullLatestSubSample(void);

        void testAddGivenTimeIsHistoricalAndFarBeforeEarliestSubSample(void);
        void testAddGivenTimeIsHistoricalAndCloseBeforeFullEarliestSubSample(void);
        void testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSample(void);
        void testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSampleButDifferentBucket(void);
        void testAddGivenTimeIsHistoricalAndWithinSomeSubSample(void);
        void testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatest(void);
        void testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatestButDifferentBucket(void);
        void testAddGivenTimeIsHistoricalAndCloserToPreviousOfNonFullSubSamples(void);
        void testAddGivenTimeIsHistoricalAndCloserToNextOfNonFullSubSamples(void);
        void testAddGivenTimeIsHistoricalAndCloserToPreviousOfFullSubSamples(void);
        void testAddGivenTimeIsHistoricalAndCloserToNextOfFullSubSamples(void);
        void testAddGivenTimeIsHistoricalAndCloserToPreviousSubSampleButOnlyNextHasSpace(void);
        void testAddGivenTimeIsHistoricalAndCloserToNextSubSampleButOnlyPreviousHasSpace(void);
        void testAddGivenTimeIsHistoricalAndFallsInBigEnoughGap(void);
        void testAddGivenTimeIsHistoricalAndFallsInTooSmallGap(void);

        void testCanSampleGivenEmptyQueue(void);
        void testCanSample(void);

        void testSampleGivenExactlyOneSampleOfExactCountToBeCreated(void);
        void testSampleGivenExactlyOneSampleOfOverCountToBeCreated(void);
        void testSampleGivenOneSampleToBeCreatedAndRemainder(void);
        void testSampleGivenTwoSamplesToBeCreatedAndRemainder(void);
        void testSampleGivenNoSampleToBeCreated(void);
        void testSampleGivenUsingSubSamplesUpToCountExceedItMoreThanUsingOneLess(void);

        void testResetBucketGivenEmptyQueue(void);
        void testResetBucketGivenBucketBeforeEarliestSubSample(void);
        void testResetBucketGivenBucketAtEarliestSubSample(void);
        void testResetBucketGivenBucketInBetweenWithoutAnySubSamples(void);
        void testResetBucketGivenBucketAtInBetweenSubSample(void);
        void testResetBucketGivenBucketAtLatestSubSample(void);
        void testResetBucketGivenBucketAfterLatestSubSample(void);

        void testSubSamplesNeverSpanOverDifferentBuckets(void);

        void testPersistence(void);

        void testQualityOfSamplesGivenConstantRate(void);
        void testQualityOfSamplesGivenVariableRate(void);
        void testQualityOfSamplesGivenHighLatencyAndDataInReverseOrder(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CSampleQueueTest_h
