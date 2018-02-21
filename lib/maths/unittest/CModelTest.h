/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDE_CModelTest_h
#define INCLUDE_CModelTest_h

#include <cppunit/extensions/HelperMacros.h>

class CModelTest : public CppUnit::TestFixture
{
    public:
        void testAll(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDE_CModelTest_h
