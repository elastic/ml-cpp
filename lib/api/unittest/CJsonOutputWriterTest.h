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

