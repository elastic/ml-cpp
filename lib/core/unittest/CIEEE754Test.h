/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CIEEE754Test_h
#define INCLUDED_CIEEE754Test_h

#include <cppunit/extensions/HelperMacros.h>

class CIEEE754Test : public CppUnit::TestFixture
{
    public:
        void testRound();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CIEEE754Test_h
