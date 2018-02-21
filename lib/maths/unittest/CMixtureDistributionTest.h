/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMixtureDistributionTest_h
#define INCLUDED_CMixtureDistributionTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMixtureDistributionTest : public CppUnit::TestFixture
{
    public:
        void testSupport(void);
        void testMode(void);
        void testPdf(void);
        void testCdf(void);
        void testQuantile(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CMixtureDistributionTest_h
