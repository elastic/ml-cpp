/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CDataFrameAnalyzerTrainingTest_h
#define INCLUDED_CDataFrameAnalyzerTrainingTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameAnalyzerTrainingTest : public CppUnit::TestFixture {
public:
    void testRunBoostedTreeRegressionTraining();
    void testRunBoostedTreeRegressionTrainingWithParams();
    void testRunBoostedTreeRegressionTrainingWithRowsMissingTargetValue();
    void testRunBoostedTreeRegressionTrainingWithStateRecovery();
    void testRunBoostedTreeClassifierTraining();

    static CppUnit::Test* suite();
};
#endif // INCLUDED_CDataFrameAnalyzerTrainingTest_h
