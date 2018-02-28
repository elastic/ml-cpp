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

