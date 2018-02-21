/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CWindowsErrorTest_h
#define INCLUDED_CWindowsErrorTest_h

#include <cppunit/extensions/HelperMacros.h>


class CWindowsErrorTest : public CppUnit::TestFixture
{
    public:
        void testErrors(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CWindowsErrorTest_h

