/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataSummaryStatisticsTest_h
#define INCLUDED_CDataSummaryStatisticsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataSummaryStatisticsTest : public CppUnit::TestFixture {
public:
    void testRate();
    void testCategoricalDistinctCount();
    void testCategoricalTopN();
    void testNumericBasicStatistics();
    void testNumericDistribution();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataSummaryStatisticsTest_h
