/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CJsonOutputWriterTest_h
#define INCLUDED_CJsonOutputWriterTest_h

#include <cppunit/extensions/HelperMacros.h>


class CJsonOutputWriterTest : public CppUnit::TestFixture
{
    public:
        void testSimpleWrite(void);
        void testWriteNonAnomalousBucket(void);
        void testBucketWrite(void);
        void testBucketWriteInterim(void);
        void testLimitedRecordsWrite(void);
        void testLimitedRecordsWriteInterim(void);
        void testFlush(void);
        void testWriteCategoryDefinition(void);
        void testWriteWithInfluences(void);
        void testWriteInfluencers(void);
        void testWriteInfluencersWithLimit(void);
        void testPersistNormalizer(void);
        void testPartitionScores(void);
        void testReportMemoryUsage(void);
        void testWriteScheduledEvent(void);
        void testThroughputWithScopedAllocator(void);
        void testThroughputWithoutScopedAllocator(void);

        static CppUnit::Test *suite();

    private:
        void testBucketWriteHelper(bool isInterim);
        void testLimitedRecordsWriteHelper(bool isInterim);
        void testThroughputHelper(bool useScopedAllocator);
};

#endif // INCLUDED_CJsonOutputWriterTest_h

