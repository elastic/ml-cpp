/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CSolversTest_h
#define INCLUDED_CSolversTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSolversTest : public CppUnit::TestFixture
{
    public:
        void testBracket(void);
        void testBisection(void);
        void testBrent(void);
        void testSublevelSet(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CSolversTest_h
