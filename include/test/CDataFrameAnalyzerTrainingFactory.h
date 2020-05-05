/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_test_CDataFrameAnalyzerTrainingFactory_h
#define INCLUDED_ml_test_CDataFrameAnalyzerTrainingFactory_h

#include <core/CDataFrame.h>
#include <core/CSmallVector.h>

#include <maths/CBoostedTreeFactory.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CTools.h>

#include <api/CDataFrameAnalysisInstrumentation.h>
#include <api/CDataFrameAnalyzer.h>

#include <test/CRandomNumbers.h>
#include <test/ImportExport.h>

#include <string>
#include <vector>

namespace ml {
namespace test {
//! \brief Collection of helping methods to create regression and classification data for tests.
class TEST_EXPORT CDataFrameAnalyzerTrainingFactory {
public:
    using TStrVec = std::vector<std::string>;
    using TDoubleVec = std::vector<double>;
    using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
    using TLossUPtr = std::unique_ptr<maths::boosted_tree::CLoss>;
    using TTargetTransformer = std::function<double(double)>;
    using TLossFunctionType = maths::boosted_tree::ELossType;

public:
    static void addPredictionTestData(TLossFunctionType type,
                                      const TStrVec& fieldNames,
                                      TStrVec fieldValues,
                                      api::CDataFrameAnalyzer& analyzer,
                                      std::size_t numberExamples = 100) {

        test::CRandomNumbers rng;

        TDoubleVec weights;
        rng.generateUniformSamples(-1.0, 1.0, fieldNames.size() - 3, weights);
        TDoubleVec regressors;
        rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, regressors);

        TStrVec targets;
        switch (type) {
        case TLossFunctionType::E_MseRegression:
        case TLossFunctionType::E_HuberRegression:
            setupLinearRegressionData(fieldNames, fieldValues, analyzer,
                                      weights, regressors, targets);
            break;
        case TLossFunctionType::E_MsleRegression:
            setupLinearRegressionData(fieldNames, fieldValues, analyzer, weights, regressors,
                                      targets, [](double x) { return x * x; });
            break;
        case TLossFunctionType::E_BinaryClassification:
            setupBinaryClassificationData(fieldNames, fieldValues, analyzer,
                                          weights, regressors, targets);
            break;
        case TLossFunctionType::E_MulticlassClassification:
            // TODO
            break;
        }
    }

    template<typename T>
    static void addPredictionTestData(TLossFunctionType type,
                                      const TStrVec& fieldNames,
                                      TStrVec fieldValues,
                                      api::CDataFrameAnalyzer& analyzer,
                                      std::vector<T>& expectedPredictions,
                                      std::size_t numberExamples = 100,
                                      double alpha = -1.0,
                                      double lambda = -1.0,
                                      double gamma = -1.0,
                                      double softTreeDepthLimit = -1.0,
                                      double softTreeDepthTolerance = -1.0,
                                      double eta = 0.0,
                                      std::size_t maximumNumberTrees = 0,
                                      double featureBagFraction = 0.0,
                                      double lossFunctionParameter = 1.0) {

        test::CRandomNumbers rng;

        TDoubleVec weights;
        rng.generateUniformSamples(-1.0, 1.0, fieldNames.size() - 3, weights);
        TDoubleVec regressors;
        rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, regressors);

        TStrVec targets;
        auto frame = [&] {
            switch (type) {
            case TLossFunctionType::E_MseRegression:
            case TLossFunctionType::E_HuberRegression:
                return setupLinearRegressionData(fieldNames, fieldValues, analyzer,
                                                 weights, regressors, targets);
            case TLossFunctionType::E_MsleRegression:
                return setupLinearRegressionData(fieldNames, fieldValues, analyzer,
                                                 weights, regressors, targets,
                                                 [](double x) { return x * x; });
            case TLossFunctionType::E_BinaryClassification:
                return setupBinaryClassificationData(fieldNames, fieldValues, analyzer,
                                                     weights, regressors, targets);
            case TLossFunctionType::E_MulticlassClassification:
                // TODO
                return TDataFrameUPtr{};
            }
            return TDataFrameUPtr{};
        }();

        TLossUPtr loss;
        switch (type) {
        case TLossFunctionType::E_MseRegression:
            loss = std::make_unique<maths::boosted_tree::CMse>();
            break;
        case TLossFunctionType::E_MsleRegression:
            loss = std::make_unique<maths::boosted_tree::CMsle>(lossFunctionParameter);
            break;
        case TLossFunctionType::E_HuberRegression:
            loss = std::make_unique<maths::boosted_tree::CPseudoHuber>(lossFunctionParameter);
            break;
        case TLossFunctionType::E_BinaryClassification:
            loss = std::make_unique<maths::boosted_tree::CBinomialLogisticLoss>();
            break;
        case TLossFunctionType::E_MulticlassClassification:
            // TODO
            loss = TLossUPtr{};
            break;
        }

        maths::CBoostedTreeFactory treeFactory{
            maths::CBoostedTreeFactory::constructFromParameters(1, std::move(loss))};
        if (alpha >= 0.0) {
            treeFactory.depthPenaltyMultiplier(alpha);
        }
        if (lambda >= 0.0) {
            treeFactory.leafWeightPenaltyMultiplier(lambda);
        }
        if (gamma >= 0.0) {
            treeFactory.treeSizePenaltyMultiplier(gamma);
        }
        if (softTreeDepthLimit >= 0.0) {
            treeFactory.softTreeDepthLimit(softTreeDepthLimit);
        }
        if (softTreeDepthTolerance >= 0.0) {
            treeFactory.softTreeDepthTolerance(softTreeDepthTolerance);
        }
        if (eta > 0.0) {
            treeFactory.eta(eta);
        }
        if (maximumNumberTrees > 0) {
            treeFactory.maximumNumberTrees(maximumNumberTrees);
        }
        if (featureBagFraction > 0.0) {
            treeFactory.featureBagFraction(featureBagFraction);
        }

        ml::api::CDataFrameTrainBoostedTreeInstrumentation instrumentation("testJob");
        treeFactory.analysisInstrumentation(instrumentation);

        auto tree = treeFactory.buildFor(*frame, weights.size());

        tree->train();
        tree->predict();

        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                auto prediction = tree->readAndAdjustPrediction(*row);
                appendPrediction(*frame, weights.size(), prediction, expectedPredictions);
            }
        });
    }

    static TDataFrameUPtr setupBinaryClassificationData(const TStrVec& fieldNames,
                                                        TStrVec& fieldValues,
                                                        api::CDataFrameAnalyzer& analyzer,
                                                        const TDoubleVec& weights,
                                                        const TDoubleVec& regressors,
                                                        TStrVec& targets);
    static TDataFrameUPtr
    setupLinearRegressionData(const TStrVec& fieldNames,
                              TStrVec& fieldValues,
                              api::CDataFrameAnalyzer& analyzer,
                              const TDoubleVec& weights,
                              const TDoubleVec& regressors,
                              TStrVec& targets,
                              TTargetTransformer targetTransformer = [](double x) {
                                  return x;
                              });

private:
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TBoolVec = std::vector<bool>;
    using TRowItr = core::CDataFrame::TRowItr;

private:
    static void appendPrediction(core::CDataFrame&,
                                 std::size_t,
                                 const TDouble2Vec& prediction,
                                 TDoubleVec& predictions);

    static void appendPrediction(core::CDataFrame& frame,
                                 std::size_t target,
                                 const TDouble2Vec& class1Score,
                                 TStrVec& predictions);
};
}
}

#endif // INCLUDED_ml_test_CDataFrameAnalyzerTrainingFactory_h
