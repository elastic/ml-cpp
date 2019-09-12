/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameAnalyzerTest_h
#define INCLUDED_CDataFrameAnalyzerTest_h

#include <core/CDataFrame.h>
#include <core/CDataSearcher.h>

#include <maths/CLinearAlgebraEigen.h>

#include <api/CDataFrameAnalyzer.h>

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameAnalyzerTest : public CppUnit::TestFixture {
public:
    using TDataAdderUPtr = std::unique_ptr<ml::core::CDataAdder>;
    using TPersisterSupplier = std::function<TDataAdderUPtr()>;
    using TDataSearcherUPtr = std::unique_ptr<ml::core::CDataSearcher>;
    using TRestoreSearcherSupplier = std::function<TDataSearcherUPtr()>;

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
    using TDoubleVec = std::vector<double>;
    using TStrVec = std::vector<std::string>;

private:
    void testRunBoostedTreeTrainingWithStateRecoverySubroutine(double lambda,
                                                               double gamma,
                                                               double eta,
                                                               size_t maximumNumberTrees,
                                                               double featureBagFraction,
                                                               size_t numberRoundsPerHyperparameter,
                                                               size_t intermediateIteration,
                                                               size_t finalIteration) const;

    std::unique_ptr<ml::core::CDataFrame>
    passDataToAnalyzer(const TDoubleVec& weights,
                       const TDoubleVec& values,
                       ml::api::CDataFrameAnalyzer& analyzer,
                       const TStrVec& fieldNames) const;
};

#endif // INCLUDED_CDataFrameAnalyzerTest_h
