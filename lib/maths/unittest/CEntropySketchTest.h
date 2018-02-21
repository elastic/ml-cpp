/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CEntropySketchTest_h
#define INCLUDED_CEntropySketchTest_h

#include <cppunit/extensions/HelperMacros.h>

class CEntropySketchTest : public CppUnit::TestFixture
{
    public:
        void testAll(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CEntropySketchTest_h
