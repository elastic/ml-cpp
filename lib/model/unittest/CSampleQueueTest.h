/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CSampleQueueTest_h
#define INCLUDED_CSampleQueueTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <model/CSampleQueue.h>

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
