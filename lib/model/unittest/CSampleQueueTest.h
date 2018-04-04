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

class CSampleQueueTest : public CppUnit::TestFixture {
public:
    void testSampleToString();
    void testSampleFromString();

    void testAddGivenQueueIsEmptyShouldCreateNewSubSample();
    void testAddGivenQueueIsFullShouldResize();
    void testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSample();
    void testAddGivenTimeIsInOrderAndCloseToNonFullLatestSubSampleButDifferentBucket();
    void testAddGivenTimeIsInOrderAndCloseToFullLatestSubSample();
    void testAddGivenTimeIsInOrderAndFarFromLatestSubSample();
    void testAddGivenTimeIsWithinFullLatestSubSample();

    void testAddGivenTimeIsHistoricalAndFarBeforeEarliestSubSample();
    void testAddGivenTimeIsHistoricalAndCloseBeforeFullEarliestSubSample();
    void testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSample();
    void testAddGivenTimeIsHistoricalAndCloseBeforeNonFullEarliestSubSampleButDifferentBucket();
    void testAddGivenTimeIsHistoricalAndWithinSomeSubSample();
    void testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatest();
    void testAddGivenTimeIsHistoricalAndCloserToSubSampleBeforeLatestButDifferentBucket();
    void testAddGivenTimeIsHistoricalAndCloserToPreviousOfNonFullSubSamples();
    void testAddGivenTimeIsHistoricalAndCloserToNextOfNonFullSubSamples();
    void testAddGivenTimeIsHistoricalAndCloserToPreviousOfFullSubSamples();
    void testAddGivenTimeIsHistoricalAndCloserToNextOfFullSubSamples();
    void testAddGivenTimeIsHistoricalAndCloserToPreviousSubSampleButOnlyNextHasSpace();
    void testAddGivenTimeIsHistoricalAndCloserToNextSubSampleButOnlyPreviousHasSpace();
    void testAddGivenTimeIsHistoricalAndFallsInBigEnoughGap();
    void testAddGivenTimeIsHistoricalAndFallsInTooSmallGap();

    void testCanSampleGivenEmptyQueue();
    void testCanSample();

    void testSampleGivenExactlyOneSampleOfExactCountToBeCreated();
    void testSampleGivenExactlyOneSampleOfOverCountToBeCreated();
    void testSampleGivenOneSampleToBeCreatedAndRemainder();
    void testSampleGivenTwoSamplesToBeCreatedAndRemainder();
    void testSampleGivenNoSampleToBeCreated();
    void testSampleGivenUsingSubSamplesUpToCountExceedItMoreThanUsingOneLess();

    void testResetBucketGivenEmptyQueue();
    void testResetBucketGivenBucketBeforeEarliestSubSample();
    void testResetBucketGivenBucketAtEarliestSubSample();
    void testResetBucketGivenBucketInBetweenWithoutAnySubSamples();
    void testResetBucketGivenBucketAtInBetweenSubSample();
    void testResetBucketGivenBucketAtLatestSubSample();
    void testResetBucketGivenBucketAfterLatestSubSample();

    void testSubSamplesNeverSpanOverDifferentBuckets();

    void testPersistence();

    void testQualityOfSamplesGivenConstantRate();
    void testQualityOfSamplesGivenVariableRate();
    void testQualityOfSamplesGivenHighLatencyAndDataInReverseOrder();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CSampleQueueTest_h
