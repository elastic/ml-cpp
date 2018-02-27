/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CTrendTestsTest_h
#define INCLUDED_CTrendTestsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CTrendTestsTest : public CppUnit::TestFixture
{
    public:
        void testRandomizedPeriodicity(void);
        void testCalendarCyclic(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CTrendTestsTest_h
