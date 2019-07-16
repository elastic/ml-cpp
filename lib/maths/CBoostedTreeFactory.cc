/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeFactory.h>


namespace ml {
namespace maths {



void CBoostedTreeFactory::constructBoostedTree(core::CDataFrame& frame) {
    if (m_Loss == nullptr) {
        HANDLE_FATAL(<< "Internal error: no loss function defined for regression."
                     << " Please report this problem.");
        return;
    }
    if (m_DependentVariable >= frame.numberColumns()) {
        HANDLE_FATAL(<< "Input error: invalid dependent variable for regression");
        return;
    }

    this->initializeMissingFeatureMasks(frame);

    TPackedBitVectorVec trainingRowMasks;
    TPackedBitVectorVec testingRowMasks;
    std::tie(trainingRowMasks, testingRowMasks) = this->crossValidationRowMasks();

    // We store the gradient and curvature of the loss function and the predicted
    // value for the dependent variable of the regression.
    frame.resizeColumns(m_NumberThreads, frame.numberColumns() + 3);

    this->initializeFeatureSampleDistribution(frame);
    this->initializeHyperparameters(frame, recordProgress);

    LOG_TRACE(<< "Main training loop...");

    CBayesianOptimisation bopt{this->hyperparameterBoundingBox()};

    std::size_t numberRounds{this->numberHyperparameterTuningRounds()};
}

//! Setup the missing feature row masks.
void CBoostedTreeFactory::initializeMissingFeatureMasks(const core::CDataFrame& frame) {

    m_MissingFeatureRowMasks.resize(frame.numberColumns());

    auto result = frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                double value{(*row)[i]};
                if (CDataFrameUtils::isMissing(value)) {
                    m_MissingFeatureRowMasks[i].extend(
                        false, row->index() - m_MissingFeatureRowMasks[i].size());
                    m_MissingFeatureRowMasks[i].extend(true);
                }
            }
        }
    });

    for (auto& mask : m_MissingFeatureRowMasks) {
        mask.extend(false, frame.numberRows() - mask.size());
        LOG_TRACE(<< "# missing = " << mask.manhattan());
    }
}

//! Get the row masks to use for the training and testing sets for k-fold
//! cross validation estimates of the generalisation error.
std::pair<TPackedBitVectorVec, TPackedBitVectorVec>
CBoostedTreeFactory::crossValidationRowMasks() const {

    core::CPackedBitVector mask{~m_MissingFeatureRowMasks[m_DependentVariable]};

    TPackedBitVectorVec trainingRowMasks(m_NumberFolds);

    for (auto row = mask.beginOneBits(); row != mask.endOneBits(); ++row) {
        std::size_t fold{CSampling::uniformSample(m_Rng, 0, m_NumberFolds)};
        trainingRowMasks[fold].extend(true, *row - trainingRowMasks[fold].size());
        trainingRowMasks[fold].extend(false);
    }

    for (auto& fold : trainingRowMasks) {
        fold.extend(true, mask.size() - fold.size());
        LOG_TRACE(<< "# training = " << fold.manhattan());
    }

    TPackedBitVectorVec testingRowMasks(m_NumberFolds,
                                        core::CPackedBitVector{mask.size(), true});
    for (std::size_t i = 0; i < m_NumberFolds; ++i) {
        testingRowMasks[i] ^= trainingRowMasks[i];
        LOG_TRACE(<< "# testing = " << testingRowMasks[i].manhattan());
    }

    return {trainingRowMasks, testingRowMasks};
}

//! Initialize the regressors sample distribution.
void initializeFeatureSampleDistribution(const core::CDataFrame& frame) {

    // Exclude all constant features by zeroing their probabilities.

    std::size_t n{numberFeatures(frame)};

    TSizeVec regressors(n);
    std::iota(regressors.begin(), regressors.end(), 0);
    regressors.erase(regressors.begin() + m_DependentVariable);

    TDoubleVecVec distinct(n);
    frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (std::size_t i = 0; i < regressors.size(); ++i) {
                double value{(*row)[regressors[i]]};
                if (distinct[i].size() == 2) {
                    continue;
                }
                if (distinct[i].empty()) {
                    distinct[i].push_back(value);
                }
                if (value != distinct[i][0]) {
                    distinct[i].push_back(value);
                }
            }
        }
    });

    regressors.erase(
        std::remove_if(regressors.begin(), regressors.end(),
                       [&](std::size_t i) { return distinct[i].size() < 2; }),
        regressors.end());
    LOG_TRACE(<< "regressors = " << core::CContainerPrinter::print(regressors));

    // TODO consider "correlation" with target variable.

    m_FeatureSampleProbabilities.assign(n, 0.0);
    if (regressors.empty()) {
        HANDLE_FATAL(<< "Input error: all features constant.");
    } else {
        double p{1.0 / static_cast<double>(regressors.size())};
        for (auto feature : regressors) {
            m_FeatureSampleProbabilities[feature] = p;
        }
    }
    LOG_TRACE(<< "P(sample) = "
              << core::CContainerPrinter::print(m_FeatureSampleProbabilities));
}

//! Read overrides for hyperparameters and if necessary estimate the initial
//! values for \f$\lambda\f$ and \f$\gamma\f$ which match the gain from an
//! overfit tree.
void CBoostedTreeFactory::initializeHyperparameters(core::CDataFrame& frame,
                                                    TProgressCallback recordProgress) {

    m_Lambda = m_LambdaOverride.value_or(0.0);
    m_Gamma = m_GammaOverride.value_or(0.0);
    if (m_EtaOverride) {
        m_Eta = *m_EtaOverride;
    } else {
        // Eta is the learning rate. There is a lot of empirical evidence that
        // this should not be much larger than 0.1. Conceptually, we're making
        // random choices regarding which features we'll use to split when
        // fitting a single tree and we only observe a random sample from the
        // function we're trying to learn. Using more trees with a smaller learning
        // rate reduces the variance that the decisions or particular sample we
        // train with introduces to predictions. The scope for variation increases
        // with the number of features so we use a lower learning rate with more
        // features. Furthermore, the leaf weights naturally decrease as we add
        // more trees, since the prediction errors decrease, so we slowly increase
        // the learning rate to maintain more equal tree weights. This tends to
        // produce forests which generalise as well but are much smaller and so
        // train faster.
        m_Eta = 1.0 /
                std::max(10.0, std::sqrt(static_cast<double>(frame.numberColumns() - 4)));
        m_EtaGrowthRatePerTree = 1.0 + m_Eta / 2.0;
    }
    if (m_MaximumNumberTreesOverride) {
        m_MaximumNumberTrees = *m_MaximumNumberTreesOverride;
    } else {
        // This needs to be tied to the learn rate to avoid bias.
        m_MaximumNumberTrees = static_cast<std::size_t>(2.0 / m_Eta + 0.5);
    }
    if (m_FeatureBagFractionOverride) {
        m_FeatureBagFraction = *m_FeatureBagFractionOverride;
    }

    if (m_LambdaOverride && m_GammaOverride) {
        // Fall through.
    } else {
        core::CPackedBitVector trainingRowMask{frame.numberRows(), true};

        auto tree = this->initializePredictionsAndLossDerivatives(frame, trainingRowMask);

        double L[2];
        double T[2];
        double W[2];

        std::tie(L[0], T[0], W[0]) =
            this->regularisedLoss(frame, trainingRowMask, {std::move(tree)});
        LOG_TRACE(<< "loss = " << L[0] << ", # leaves = " << T[0]
                  << ", sum square weights = " << W[0]);

        auto forest = this->trainForest(frame, trainingRowMask, recordProgress);

        std::tie(L[1], T[1], W[1]) = this->regularisedLoss(frame, trainingRowMask, forest);
        LOG_TRACE(<< "loss = " << L[1] << ", # leaves = " << T[1]
                  << ", sum square weights = " << W[1]);

        double scale{static_cast<double>(m_NumberFolds - 1) / static_cast<double>(m_NumberFolds)};
        double lambda{scale * std::max((L[0] - L[1]) / (W[1] - W[0]), 0.0) / 5.0};
        double gamma{scale * std::max((L[0] - L[1]) / (T[1] - T[0]), 0.0) / 5.0};

        if (m_LambdaOverride == boost::none) {
            m_Lambda = m_GammaOverride ? lambda : 0.5 * lambda;
        }
        if (m_GammaOverride == boost::none) {
            m_Gamma = m_LambdaOverride ? gamma : 0.5 * gamma;
        }
        LOG_TRACE(<< "lambda(initial) = " << m_Lambda << " gamma(initial) = " << m_Gamma);
    }

    m_MaximumTreeSizeFraction = 10.0;

    if (m_MaximumNumberTreesOverride == boost::none) {
        // We allow a large number of trees by default in the main parameter
        // optimisation loop. In practice, we should use many fewer if they
        // don't significantly improve test error.
        m_MaximumNumberTrees *= 10;
    }
}

//! Initialize the predictions and loss function derivatives for the masked
//! rows in \p frame.
TNodeVec CBoostedTreeFactory::initializePredictionsAndLossDerivatives(
    core::CDataFrame& frame,
    const core::CPackedBitVector& trainingRowMask) const {

    frame.writeColumns(m_NumberThreads, 0, frame.numberRows(),
                       [](TRowItr beginRows, TRowItr endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               std::size_t numberColumns{row->numberColumns()};
                               row->writeColumn(predictionColumn(numberColumns), 0.0);
                               row->writeColumn(lossGradientColumn(numberColumns), 0.0);
                               row->writeColumn(lossCurvatureColumn(numberColumns), 0.0);
                           }
                       },
                       &trainingRowMask);

    TNodeVec tree(1);
    this->refreshPredictionsAndLossDerivatives(frame, trainingRowMask, m_Eta, tree);

    return tree;
}
void CBoostedTreeFactory::constructBoostedTree(std::size_t numberThreads,
                                               std::size_t dependentVariable,
                                               CBoostedTree::TLossFunctionUPtr loss) {
    treeImpl = std::make_unique<CBoostedTree::CImpl>(numberThreads, dependentVariable, std::move(loss));
}

}
}
