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

#ifndef INCLUDED_CBjkstUniqueValuesTest_h
#define INCLUDED_CBjkstUniqueValuesTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBjkstUniqueValuesTest : public CppUnit::TestFixture
{
    public:
        void testTrailingZeros(void);
        void testNumber(void);
        void testRemove(void);
        void testSwap(void);
        void testSmall(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CBjkstUniqueValuesTest_h
