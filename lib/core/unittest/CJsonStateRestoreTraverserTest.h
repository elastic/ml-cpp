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
        void testRestore1();
        void testRestore2();
        void testRestore3();
        void testRestore4();
        void testParsingBooleanFields();
        void testRestore1IgnoreArrays();
        void testRestore1IgnoreArraysNested();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CJsonStateRestoreTraverserTest_h

