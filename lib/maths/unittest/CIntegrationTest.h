/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CIntegration_h
#define INCLUDED_CIntegration_h

#include <cppunit/extensions/HelperMacros.h>

class CIntegrationTest : public CppUnit::TestFixture
{
    public:
        void testAllSingleVariate(void);
        void testAdaptive(void);
        void testSparseGrid(void);
        void testMultivariateSmooth(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CIntegration_h
