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
#ifndef INCLUDED_CJsonStateRestoreTraverserTest_h
#define INCLUDED_CJsonStateRestoreTraverserTest_h

#include <cppunit/extensions/HelperMacros.h>


class CJsonStateRestoreTraverserTest : public CppUnit::TestFixture
{
    public:
        void testRestore1(void);
        void testRestore2(void);
        void testRestore3(void);
        void testRestore4(void);
        void testParsingBooleanFields(void);
        void testRestore1IgnoreArrays(void);
        void testRestore1IgnoreArraysNested(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CJsonStateRestoreTraverserTest_h

