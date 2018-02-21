/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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

