/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CProgNameTest_h
#define INCLUDED_CProgNameTest_h

#include <cppunit/extensions/HelperMacros.h>


class CProgNameTest : public CppUnit::TestFixture
{
    public:
        void testProgName();
        void testProgDir();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CProgNameTest_h

