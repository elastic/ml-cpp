/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CKMostCorrelatedTest_h
#define INCLUDED_CKMostCorrelatedTest_h

#include <cppunit/extensions/HelperMacros.h>

class CKMostCorrelatedTest : public CppUnit::TestFixture
{
    public:
        void testCorrelation(void);
        void testNextProjection(void);
        void testMostCorrelated(void);
        void testRemoveVariables(void);
        void testAccuracy(void);
        void testStability(void);
        void testChangingCorrelation(void);
        void testMissingData(void);
        void testPersistence(void);
        void testScale(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CKMostCorrelatedTest_h
