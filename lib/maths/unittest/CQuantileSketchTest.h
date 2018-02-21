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

#ifndef INCLUDED_CQuantileSketchTest_h
#define INCLUDED_CQuantileSketchTest_h

#include <cppunit/extensions/HelperMacros.h>

class CQuantileSketchTest : public CppUnit::TestFixture
{
    public:
        void testAdd(void);
        void testReduce(void);
        void testMerge(void);
        void testMedian(void);
        void testPropagateForwardByTime(void);
        void testQuantileAccuracy(void);
        void testCdf(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CQuantileSketchTest_h
