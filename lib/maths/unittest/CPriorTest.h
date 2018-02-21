/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CPriorTest_h
#define INCLUDED_CPriorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CPriorTest : public CppUnit::TestFixture
{
    public:
        void testExpectation(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CPriorTest_h
