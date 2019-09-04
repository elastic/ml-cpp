/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameAnalyzerTest_h
#define INCLUDED_CDataFrameAnalyzerTest_h

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameAnalyzerTest : public CppUnit::TestFixture {
public:
    void testWithoutControlMessages();
    void testRunOutlierDetection();
    void testRunOutlierDetectionPartitioned();
    void testRunOutlierFeatureInfluences();
    void testRunOutlierDetectionWithParams();
    void testRunBoostedTreeTraining();
    void testRunBoostedTreeTrainingWithStateRecovery();
    void testRunBoostedTreeTrainingWithParams();
    void testRunBoostedTreeTrainingWithRowsMissingTargetValue();
    void testFlushMessage();
    void testErrors();
    void testRoundTripDocHashes();
    void testCategoricalFields();

    static CppUnit::Test* suite();

private:

    void
    testRunBoostedTreeTrainingWithStateRecoverySubroutine(size_t numberHyperparameters,
                                                          double lambda,
                                                          double gamma,
                                                          double eta,
                                                          size_t maximumNumberTrees,
                                                          double featureBagFraction,
                                                          size_t numberRoundsPerHyperparameter,
                                                          size_t intermediateIteration,
                                                          size_t finalIteration) const;
};

#endif // INCLUDED_CDataFrameAnalyzerTest_h
