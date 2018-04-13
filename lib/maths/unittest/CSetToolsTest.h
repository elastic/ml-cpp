/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CSetToolsTest_h
#define INCLUDED_CSetToolsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSetToolsTest : public CppUnit::TestFixture
{
    public:
        void testInplaceSetDifference();
        void testSetSizes();
        void testJaccard();
        void testOverlap();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CSetToolsTest_h
