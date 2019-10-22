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
    void testEstimateMemoryUsageFor0Rows();
    void testEstimateMemoryUsageFor1Row();
    void testEstimateMemoryUsageFor10Rows();
    void testEstimateMemoryUsageFor100Rows();
    void testEstimateMemoryUsageFor1000Rows();
    void testColumnsForWhichEmptyIsMissingClassification();
    void testColumnsForWhichEmptyIsMissingRegression();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataFrameAnalysisRunnerTest_h
