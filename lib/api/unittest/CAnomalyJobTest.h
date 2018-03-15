/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CAnomalyJobTest_h
#define INCLUDED_CAnomalyJobTest_h

#include <core/CoreTypes.h>

#include <cppunit/extensions/HelperMacros.h>

class CAnomalyJobTest : public CppUnit::TestFixture
{
    public:
        void testLicense(void);
        void testBadTimes(void);
        void testOutOfSequence(void);
        void testControlMessages(void);
        void testSkipTimeControlMessage(void);
        void testOutOfPhase(void);
        void testBucketSelection(void);
        void testModelPlot(void);
        void testInterimResultEdgeCases(void);
        void testRestoreFailsWithEmptyStream(void);

        static CppUnit::Test *suite(void);

};

#endif // INCLUDED_CAnomalyJobTest_h

