/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CBasicStatisticsTest_h
#define INCLUDED_CBasicStatisticsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBasicStatisticsTest : public CppUnit::TestFixture {
public:
    void testMean();
    void testCentralMoments();
    void testVectorCentralMoments();
    void testCovariances();
    void testCovariancesLedoitWolf();
    void testMedian();
    void testOrderStatistics();
    void testMinMax();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CBasicStatisticsTest_h
