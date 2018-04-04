/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CFunctionalTest_h
#define INCLUDED_CFunctionalTest_h

#include <cppunit/extensions/HelperMacros.h>

class CFunctionalTest : public CppUnit::TestFixture
{
    public:
        void testIsNull();
        void testDereference();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CFunctionalTest_h
