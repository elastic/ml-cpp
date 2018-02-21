/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CBasicStatisticsTest_h
#define INCLUDED_CBasicStatisticsTest_h

#include <cppunit/extensions/HelperMacros.h>


class CBasicStatisticsTest : public CppUnit::TestFixture
{
    public:
        void testMean(void);
        void testCentralMoments(void);
        void testVectorCentralMoments(void);
        void testCovariances(void);
        void testCovariancesLedoitWolf(void);
        void testMedian(void);
        void testOrderStatistics(void);
        void testMinMax(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CBasicStatisticsTest_h

