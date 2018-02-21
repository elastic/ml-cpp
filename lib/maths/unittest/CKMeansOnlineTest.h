/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CKMeansOnlineTest_h
#define INCLUDED_CKMeansOnlineTest_h

#include <cppunit/extensions/HelperMacros.h>

class CKMeansOnlineTest : public CppUnit::TestFixture
{
    public:
        void testVariance(void);
        void testAdd(void);
        void testReduce(void);
        void testClustering(void);
        void testSplit(void);
        void testMerge(void);
        void testPropagateForwardsByTime(void);
        void testSample(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CKMeansOnlineTest_h
