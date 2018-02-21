/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
        void testMad(void);
        void testPropagateForwardByTime(void);
        void testQuantileAccuracy(void);
        void testCdf(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CQuantileSketchTest_h
