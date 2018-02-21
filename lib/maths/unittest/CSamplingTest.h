/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CSamplingTest_h
#define INCLUDED_CSamplingTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSamplingTest : public CppUnit::TestFixture
{
    public:
        void testMultinomialSample(void);
        void testMultivariateNormalSample(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CSamplingTest_h
