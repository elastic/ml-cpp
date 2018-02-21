/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataSummaryStatisticsTest_h
#define INCLUDED_CDataSummaryStatisticsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataSummaryStatisticsTest : public CppUnit::TestFixture
{
    public:
        void testRate(void);
        void testCategoricalDistinctCount(void);
        void testCategoricalTopN(void);
        void testNumericBasicStatistics(void);
        void testNumericDistribution(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CDataSummaryStatisticsTest_h
