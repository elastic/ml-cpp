/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CForecastTest_h
#define INCLUDED_CForecastTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <core/CoreTypes.h>

#include <functional>

class CForecastTest : public CppUnit::TestFixture
{
    public:
        void testDailyNoLongTermTrend();
        void testDailyConstantLongTermTrend();
        void testDailyVaryingLongTermTrend();
        void testComplexNoLongTermTrend();
        void testComplexConstantLongTermTrend();
        void testComplexVaryingLongTermTrend();
        void testNonNegative();
        void testFinancialIndex();

        static CppUnit::Test *suite();

    private:
        using TTrend = std::function<double (ml::core_t::TTime, double)>;

    private:
        void test(TTrend trend,
                  ml::core_t::TTime bucketLength,
                  std::size_t daysToLearn,
                  double noiseVariance,
                  double maximumPercentageOutOfBounds,
                  double maximumError);
};

#endif // INCLUDED_CForecastTest_h
