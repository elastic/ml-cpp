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
#ifndef INCLUDED_CJsonOutputWriterTest_h
#define INCLUDED_CJsonOutputWriterTest_h

#include <cppunit/extensions/HelperMacros.h>


class CJsonOutputWriterTest : public CppUnit::TestFixture
{
    public:
        void testSimpleWrite();
        void testWriteNonAnomalousBucket();
        void testBucketWrite();
        void testBucketWriteInterim();
        void testLimitedRecordsWrite();
        void testLimitedRecordsWriteInterim();
        void testFlush();
        void testWriteCategoryDefinition();
        void testWriteWithInfluences();
        void testWriteInfluencers();
        void testWriteInfluencersWithLimit();
        void testPersistNormalizer();
        void testPartitionScores();
        void testReportMemoryUsage();
        void testWriteScheduledEvent();
        void testThroughputWithScopedAllocator();
        void testThroughputWithoutScopedAllocator();

        static CppUnit::Test *suite();

    private:
        void testBucketWriteHelper(bool isInterim);
        void testLimitedRecordsWriteHelper(bool isInterim);
        void testThroughputHelper(bool useScopedAllocator);
};

#endif // INCLUDED_CJsonOutputWriterTest_h

