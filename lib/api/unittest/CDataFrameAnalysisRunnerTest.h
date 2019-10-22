/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameAnalysisRunnerTest_h
#define INCLUDED_CDataFrameAnalysisRunnerTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameAnalysisRunnerTest : public CppUnit::TestFixture {
public:
    void testComputeExecutionStrategyForOutliers();
    void testComputeAndSaveExecutionStrategyDiskUsageFlag();
    void testEstimateMemoryUsage_0();
    void testEstimateMemoryUsage_1();
    void testEstimateMemoryUsage_10();
    void testEstimateMemoryUsage_100();
    void testEstimateMemoryUsage_1000();
    void testColumnsForWhichEmptyIsMissingClassification();
    void testColumnsForWhichEmptyIsMissingRegression();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataFrameAnalysisRunnerTest_h
