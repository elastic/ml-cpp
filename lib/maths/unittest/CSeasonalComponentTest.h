/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CSeasonalComponentTest_h
#define INCLUDED_CSeasonalComponentTest_h

#include <cppunit/extensions/HelperMacros.h>

class CSeasonalComponentTest : public CppUnit::TestFixture
{
    public:
        void testNoPeriodicity(void);
        void testConstantPeriodic(void);
        void testTimeVaryingPeriodic(void);
        void testVeryLowVariation(void);
        void testVariance(void);
        void testPersist(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CSeasonalComponentTest_h
