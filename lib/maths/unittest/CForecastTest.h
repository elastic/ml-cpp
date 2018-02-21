/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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
