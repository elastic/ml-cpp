/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CProcessTest_h
#define INCLUDED_CProcessTest_h

#include <cppunit/extensions/HelperMacros.h>


class CProcessTest : public CppUnit::TestFixture
{
    public:
        void testPids(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CProcessTest_h

