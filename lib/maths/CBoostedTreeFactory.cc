/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeFactory.h>

#include <maths/CBayesianOptimisation.h>
#include <maths/CBoostedTreeImpl.h>

namespace ml {
namespace maths {

CBoostedTreeFactory::TBoostedTreeUPtr CBoostedTreeFactory::build() {
    this->initializeMissingFeatureMasks(*m_Frame);
    std::tie(m_Tree->m_Impl->m_TrainingRowMasks,
             m_Tree->m_Impl->m_TestingRowMasks) = this->crossValidationRowMasks();

    // We store the gradient and curvature of the loss function and the predicted
    // value for the dependent variable of the regression.
    m_Frame->resizeColumns(m_Tree->m_Impl->m_NumberThreads, m_Frame->numberColumns() + 3);

    this->initializeFeatureSampleDistribution(*m_Frame);
    this->initializeHyperparameters(*m_Frame, m_ProgressCallback);
    this->m_Tree->m_Impl->m_BayesianOptimization =
        std::make_unique<CBayesianOptimisation>(this->hyperparameterBoundingBox());
    this->m_Tree->m_Impl->m_NumberRounds = this->numberHyperparameterTuningRounds();

    this->m_Tree->m_Impl->m_CurrentRound = 0; // for first start
    return std::move(m_Tree);
}

CBoostedTreeFactory::operator TBoostedTreeUPtr() {
    return std::move(this->build());
}

std::size_t CBoostedTreeFactory::numberHyperparameterTuningRounds() const {
    return std::max(this->m_Tree->m_Impl->m_MaximumOptimisationRoundsPerHyperparameter *
                        this->m_Tree->m_Impl->numberHyperparametersToTune(),
                    std::size_t{1});
}

CBayesianOptimisation::TDoubleDoublePrVec CBoostedTreeFactory::hyperparameterBoundingBox() const {

    // We need sensible bounds for the region we'll search for optimal values.
    // For all parameters where we have initial estimates we use bounds of the
    // form a * initial and b * initial for a < 1 < b. For other parameters we
    // use a fixed range. Ideally, we'd use the smallest intervals that have a
    // high probability of containing good parameter values. We also parameterise
    // so the probability any subinterval contains a good value is proportional
    // to its length. For parameters whose difference is naturally measured as
    // a ratio, i.e. roughly speaking difference(p_1, p_0) = p_1 / p_0 for p_0
    // less than p_1, this translates to using log parameter values.

    CBayesianOptimisation::TDoubleDoublePrVec result;
    if (m_Tree->m_Impl->m_LambdaOverride == boost::none) {
        result.emplace_back(std::log(m_Tree->m_Impl->m_Lambda / 10.0),
                            std::log(10.0 * m_Tree->m_Impl->m_Lambda));
    }
    if (m_Tree->m_Impl->m_GammaOverride == boost::none) {
        result.emplace_back(std::log(m_Tree->m_Impl->m_Gamma / 10.0),
                            std::log(10.0 * m_Tree->m_Impl->m_Gamma));
    }
    if (m_Tree->m_Impl->m_EtaOverride == boost::none) {
        double rate{m_Tree->m_Impl->m_EtaGrowthRatePerTree - 1.0};
        result.emplace_back(std::log(0.3 * m_Tree->m_Impl->m_Eta),
                            std::log(3.0 * m_Tree->m_Impl->m_Eta));
        result.emplace_back(1.0 + rate / 2.0, 1.0 + 1.5 * rate);
    }
    if (m_Tree->m_Impl->m_FeatureBagFractionOverride == boost::none) {
        result.emplace_back(0.2, 0.8);
    }
    return result;
}

void CBoostedTreeFactory::initializeMissingFeatureMasks(const core::CDataFrame& frame) {

    m_Tree->m_Impl->m_MissingFeatureRowMasks.resize(frame.numberColumns());

    auto result = frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                double value{(*row)[i]};
                if (CDataFrameUtils::isMissing(value)) {
                    m_Tree->m_Impl->m_MissingFeatureRowMasks[i].extend(
                        false, row->index() -
                                   m_Tree->m_Impl->m_MissingFeatureRowMasks[i].size());
                    m_Tree->m_Impl->m_MissingFeatureRowMasks[i].extend(true);
                }
            }
        }
    });

    for (auto& mask : m_Tree->m_Impl->m_MissingFeatureRowMasks) {
        mask.extend(false, frame.numberRows() - mask.size());
        LOG_TRACE(<< "# missing = " << mask.manhattan());
    }
}

std::pair<CBoostedTreeImpl::TPackedBitVectorVec, CBoostedTreeImpl::TPackedBitVectorVec>
CBoostedTreeFactory::crossValidationRowMasks() const {

    core::CPackedBitVector mask{
        ~m_Tree->m_Impl->m_MissingFeatureRowMasks[m_Tree->m_Impl->m_DependentVariable]};

    TPackedBitVectorVec trainingRowMasks(m_Tree->m_Impl->m_NumberFolds);

    for (auto row = mask.beginOneBits(); row != mask.endOneBits(); ++row) {
        std::size_t fold{CSampling::uniformSample(m_Tree->m_Impl->m_Rng, 0,
                                                  m_Tree->m_Impl->m_NumberFolds)};
        trainingRowMasks[fold].extend(true, *row - trainingRowMasks[fold].size());
        trainingRowMasks[fold].extend(false);
    }

    for (auto& fold : trainingRowMasks) {
        fold.extend(true, mask.size() - fold.size());
        LOG_TRACE(<< "# training = " << fold.manhattan());
    }

    TPackedBitVectorVec testingRowMasks(m_Tree->m_Impl->m_NumberFolds,
                                        core::CPackedBitVector{mask.size(), true});
    for (std::size_t i = 0; i < m_Tree->m_Impl->m_NumberFolds; ++i) {
        testingRowMasks[i] ^= trainingRowMasks[i];
        LOG_TRACE(<< "# testing = " << testingRowMasks[i].manhattan());
    }

    return {trainingRowMasks, testingRowMasks};
}

void CBoostedTreeFactory::initializeFeatureSampleDistribution(const core::CDataFrame& frame) {

    // Exclude all constant features by zeroing their probabilities.

    std::size_t n{m_Tree->m_Impl->numberFeatures(frame)};

    TSizeVec regressors(n);
    std::iota(regressors.begin(), regressors.end(), 0);
    regressors.erase(regressors.begin() + m_Tree->m_Impl->m_DependentVariable);

    TDoubleVec mics(CDataFrameUtils::micWithColumn(
        frame, regressors, m_Tree->m_Impl->m_DependentVariable));

    regressors.erase(std::remove_if(regressors.begin(), regressors.end(),
                                    [&](std::size_t i) { return mics[i] == 0.0; }),
                     regressors.end());
    LOG_TRACE(<< "candidate regressors = " << core::CContainerPrinter::print(regressors));

    m_Tree->m_Impl->m_FeatureSampleProbabilities.assign(n, 0.0);
    if (regressors.empty()) {
        HANDLE_FATAL(<< "Input error: all features constant.");
    } else {
        std::stable_sort(regressors.begin(), regressors.end(),
                         [&mics](std::size_t lhs, std::size_t rhs) {
                             return mics[lhs] > mics[rhs];
                         });

        std::size_t maximumNumberFeatures{frame.numberRows() / m_Tree->m_Impl->m_RowsPerFeature};
        LOG_TRACE(<< "Using up to " << maximumNumberFeatures << " out of "
                  << regressors.size() << " features");

        regressors.resize(std::min(maximumNumberFeatures, regressors.size()));

        double Z{std::accumulate(
            regressors.begin(), regressors.end(), 0.0,
            [&mics](double z, std::size_t i) { return z + mics[i]; })};
        LOG_TRACE(<< "Z = " << Z);
        for (auto i : regressors) {
            m_Tree->m_Impl->m_FeatureSampleProbabilities[i] = mics[i] / Z;
        }
    }
    LOG_TRACE(<< "P(sample) = "
              << core::CContainerPrinter::print(m_Tree->m_Impl->m_FeatureSampleProbabilities));
}

void CBoostedTreeFactory::initializeHyperparameters(core::CDataFrame& frame,
                                                    CBoostedTree::TProgressCallback recordProgress) {

    m_Tree->m_Impl->m_Lambda = m_Tree->m_Impl->m_LambdaOverride.value_or(0.0);
    m_Tree->m_Impl->m_Gamma = m_Tree->m_Impl->m_GammaOverride.value_or(0.0);
    if (m_Tree->m_Impl->m_EtaOverride) {
        m_Tree->m_Impl->m_Eta = *(m_Tree->m_Impl->m_EtaOverride);
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
        m_Tree->m_Impl->m_Eta =
            1.0 / std::max(10.0, std::sqrt(static_cast<double>(frame.numberColumns() - 4)));
        m_Tree->m_Impl->m_EtaGrowthRatePerTree = 1.0 + m_Tree->m_Impl->m_Eta / 2.0;
    }
    if (m_Tree->m_Impl->m_MaximumNumberTreesOverride) {
        m_Tree->m_Impl->m_MaximumNumberTrees = *(m_Tree->m_Impl->m_MaximumNumberTreesOverride);
    } else {
        // This needs to be tied to the learn rate to avoid bias.
        m_Tree->m_Impl->m_MaximumNumberTrees =
            static_cast<std::size_t>(2.0 / m_Tree->m_Impl->m_Eta + 0.5);
    }
    if (m_Tree->m_Impl->m_FeatureBagFractionOverride) {
        m_Tree->m_Impl->m_FeatureBagFraction = *(m_Tree->m_Impl->m_FeatureBagFractionOverride);
    }

    if (m_Tree->m_Impl->m_LambdaOverride && m_Tree->m_Impl->m_GammaOverride) {
        // Fall through.
    } else {
        core::CPackedBitVector trainingRowMask{frame.numberRows(), true};

        auto tree = m_Tree->m_Impl->initializePredictionsAndLossDerivatives(frame, trainingRowMask);

        double L[2];
        double T[2];
        double W[2];

        std::tie(L[0], T[0], W[0]) = this->m_Tree->m_Impl->regularisedLoss(
            frame, trainingRowMask, {std::move(tree)});
        LOG_TRACE(<< "loss = " << L[0] << ", # leaves = " << T[0]
                  << ", sum square weights = " << W[0]);

        auto forest = m_Tree->m_Impl->trainForest(frame, trainingRowMask, recordProgress);

        std::tie(L[1], T[1], W[1]) =
            m_Tree->m_Impl->regularisedLoss(frame, trainingRowMask, forest);
        LOG_TRACE(<< "loss = " << L[1] << ", # leaves = " << T[1]
                  << ", sum square weights = " << W[1]);

        double scale{static_cast<double>(m_Tree->m_Impl->m_NumberFolds - 1) /
                     static_cast<double>(m_Tree->m_Impl->m_NumberFolds)};
        double lambda{scale * std::max((L[0] - L[1]) / (W[1] - W[0]), 0.0) / 5.0};
        double gamma{scale * std::max((L[0] - L[1]) / (T[1] - T[0]), 0.0) / 5.0};

        if (m_Tree->m_Impl->m_LambdaOverride == boost::none) {
            m_Tree->m_Impl->m_Lambda = m_Tree->m_Impl->m_GammaOverride ? lambda : 0.5 * lambda;
        }
        if (m_Tree->m_Impl->m_GammaOverride == boost::none) {
            m_Tree->m_Impl->m_Gamma = m_Tree->m_Impl->m_LambdaOverride ? gamma : 0.5 * gamma;
        }
        LOG_TRACE(<< "lambda(initial) = " << m_Tree->m_Impl->m_Lambda
                  << " gamma(initial) = " << m_Tree->m_Impl->m_Gamma);
    }

    m_Tree->m_Impl->m_MaximumTreeSizeFraction = 10.0;

    if (m_Tree->m_Impl->m_MaximumNumberTreesOverride == boost::none) {
        // We allow a large number of trees by default in the main parameter
        // optimisation loop. In practice, we should use many fewer if they
        // don't significantly improve test error.
        m_Tree->m_Impl->m_MaximumNumberTrees *= 10;
    }
}

CBoostedTreeFactory
CBoostedTreeFactory::constructFromParameters(std::size_t numberThreads,
                                             std::size_t dependentVariable,
                                             CBoostedTree::TLossFunctionUPtr loss) {
    return {numberThreads, dependentVariable, std::move(loss)};
}

CBoostedTreeFactory::CBoostedTreeFactory(std::size_t numberThreads,
                                         std::size_t dependentVariable,
                                         CBoostedTree::TLossFunctionUPtr loss)
    : m_Tree{new CBoostedTree(numberThreads, dependentVariable, std::move(loss))} {
}

CBoostedTreeFactory& CBoostedTreeFactory::numberFolds(std::size_t folds) {
    this->m_Tree->m_Impl->numberFolds(folds);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::lambda(double lambda) {
    this->m_Tree->m_Impl->lambda(lambda);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::gamma(double gamma) {
    this->m_Tree->m_Impl->gamma(gamma);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::eta(double eta) {
    this->m_Tree->m_Impl->eta(eta);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::maximumNumberTrees(std::size_t maximumNumberTrees) {
    this->m_Tree->m_Impl->maximumNumberTrees(maximumNumberTrees);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::featureBagFraction(double featureBagFraction) {
    this->m_Tree->m_Impl->featureBagFraction(featureBagFraction);
    return *this;
}

CBoostedTreeFactory&
CBoostedTreeFactory::maximumOptimisationRoundsPerHyperparameter(std::size_t rounds) {
    this->m_Tree->m_Impl->maximumOptimisationRoundsPerHyperparameter(rounds);
    return *this;
}

CBoostedTreeFactory&
CBoostedTreeFactory::progressCallback(CBoostedTree::TProgressCallback callback) {
    this->m_ProgressCallback = callback;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::frame(core::CDataFrame& frame) {
    this->m_Frame = &frame;
    this->m_Tree->frame(&frame);
    return *this;
}

const CBoostedTree& CBoostedTreeFactory::incompleteTreeObject() const {
    return *m_Tree;
}
}
}
