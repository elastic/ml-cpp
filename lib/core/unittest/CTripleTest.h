/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CTripleTest_h
#define INCLUDED_CTripleTest_h

#include <cppunit/extensions/HelperMacros.h>


class CTripleTest : public CppUnit::TestFixture
{
    public:
        void testOperators(void);
        void testBoostHashReady(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CTripleTest_h

