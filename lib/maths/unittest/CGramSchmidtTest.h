/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CGramSchmidtTest_h
#define INCLUDED_CGramSchmidtTest_h

#include <cppunit/extensions/HelperMacros.h>

class CGramSchmidtTest : public CppUnit::TestFixture
{
    public:
        void testOrthogonality();
        void testNormalisation();
        void testSpan();
        void testEdgeCases();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CGramSchmidtTest_h
