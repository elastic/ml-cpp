/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CForecastTest_h
#define INCLUDED_CForecastTest_h

#include <cppunit/extensions/HelperMacros.h>

class CForecastTest : public CppUnit::TestFixture
{
    public:
        void testDailyNoLongTermTrend(void);
        void testDailyConstantLongTermTrend(void);
        void testDailyVaryingLongTermTrend(void);
        void testComplexNoLongTermTrend(void);
        void testComplexConstantLongTermTrend(void);
        void testComplexVaryingLongTermTrend(void);
        void testNonNegative(void);
        void testFinancialIndex(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CForecastTest_h
