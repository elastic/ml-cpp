/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_test_CDataFrameAnalyzerTrainingFactory_h
#define INCLUDED_ml_test_CDataFrameAnalyzerTrainingFactory_h

#include <core/CDataFrame.h>

#include <maths/CBoostedTreeFactory.h>
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
    enum EPredictionType { E_Regression, E_BinaryClassification };
    using TStrVec = std::vector<std::string>;
    using TDoubleVec = std::vector<double>;
    using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;

public:
    template<typename T>
    static void addPredictionTestData(EPredictionType type,
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
                                      double featureBagFraction = 0.0) {

        test::CRandomNumbers rng;

        TDoubleVec weights;
        rng.generateUniformSamples(-1.0, 1.0, fieldNames.size() - 3, weights);
        TDoubleVec regressors;
        rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, regressors);

        TStrVec targets;
        auto frame = type == E_Regression
                         ? setupLinearRegressionData(fieldNames, fieldValues, analyzer,
                                                     weights, regressors, targets)
                         : setupBinaryClassificationData(fieldNames, fieldValues, analyzer,
                                                         weights, regressors, targets);

        std::unique_ptr<maths::boosted_tree::CLoss> loss;
        if (type == E_Regression) {
            loss = std::make_unique<maths::boosted_tree::CMse>();
        } else {
            loss = std::make_unique<maths::boosted_tree::CBinomialLogistic>();
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
                double prediction{(*row)[tree->columnHoldingPrediction()]};
                appendPrediction(*frame, weights.size(), prediction,
                                 tree->probabilityAtWhichToAssignClassOne(),
                                 expectedPredictions);
            }
        });
    }

    static TDataFrameUPtr setupBinaryClassificationData(const TStrVec& fieldNames,
                                                        TStrVec& fieldValues,
                                                        api::CDataFrameAnalyzer& analyzer,
                                                        const TDoubleVec& weights,
                                                        const TDoubleVec& regressors,
                                                        TStrVec& targets);
    static TDataFrameUPtr setupLinearRegressionData(const TStrVec& fieldNames,
                                                    TStrVec& fieldValues,
                                                    api::CDataFrameAnalyzer& analyzer,
                                                    const TDoubleVec& weights,
                                                    const TDoubleVec& regressors,
                                                    TStrVec& targets);

private:
    using TBoolVec = std::vector<bool>;
    using TRowItr = core::CDataFrame::TRowItr;

private:
    static void
    appendPrediction(core::CDataFrame&, std::size_t, double prediction, double, TDoubleVec& predictions);

    static void appendPrediction(core::CDataFrame& frame,
                                 std::size_t columnHoldingPrediction,
                                 double logOddsClass1,
                                 double threshold,
                                 TStrVec& predictions);
};
}
}

#endif // INCLUDED_ml_test_CDataFrameAnalyzerTrainingFactory_h
