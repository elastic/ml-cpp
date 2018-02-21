/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CStatisticalTestsTest_h
#define INCLUDED_CStatisticalTestsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CStatisticalTestsTest : public CppUnit::TestFixture
{
    public:
        void testCramerVonMises(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CStatisticalTestsTest_h
