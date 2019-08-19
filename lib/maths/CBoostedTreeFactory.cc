/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeFactory.h>

#include <core/CJsonStateRestoreTraverser.h>

#include <maths/CBayesianOptimisation.h>
#include <maths/CBoostedTreeImpl.h>
#include <maths/CSampling.h>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TRowItr = core::CDataFrame::TRowItr;

CBoostedTreeFactory::TBoostedTreeUPtr
CBoostedTreeFactory::buildFor(core::CDataFrame& frame, std::size_t dependentVariable) {

    m_TreeImpl->m_DependentVariable = dependentVariable;

    this->initializeMissingFeatureMasks(frame);
    std::tie(m_TreeImpl->m_TrainingRowMasks, m_TreeImpl->m_TestingRowMasks) =
        this->crossValidationRowMasks();

    // We store the gradient and curvature of the loss function and the predicted
    // value for the dependent variable of the regression.
    frame.resizeColumns(m_TreeImpl->m_NumberThreads, frame.numberColumns() + 3);

    // TODO we should do feature selection per fold.
    if (this->initializeFeatureSampleDistribution(frame)) {
        this->initializeHyperparameters(frame);
        this->initializeHyperparameterOptimisation();
    }

    // TODO can only use factory to create one object since this is moved. This seems trappy.
    return TBoostedTreeUPtr{new CBoostedTree(frame, std::move(m_TreeImpl), m_RecordProgress)};
}

std::size_t CBoostedTreeFactory::numberHyperparameterTuningRounds() const {
    return std::max(m_TreeImpl->m_MaximumOptimisationRoundsPerHyperparameter *
                        m_TreeImpl->numberHyperparametersToTune(),
                    std::size_t{1});
}

void CBoostedTreeFactory::initializeHyperparameterOptimisation() const {

    // We need sensible bounds for the region we'll search for optimal values.
    // For all parameters where we have initial estimates we use bounds of the
    // form a * initial and b * initial for a < 1 < b. For other parameters we
    // use a fixed range. Ideally, we'd use the smallest intervals that have a
    // high probability of containing good parameter values. We also parameterise
    // so the probability any subinterval contains a good value is proportional
    // to its length. For parameters whose difference is naturally measured as
    // a ratio, i.e. roughly speaking difference(p_1, p_0) = p_1 / p_0 for p_0
    // less than p_1, this translates to using log parameter values.

    CBayesianOptimisation::TDoubleDoublePrVec boundingBox;
    if (m_TreeImpl->m_LambdaOverride == boost::none) {
        boundingBox.emplace_back(std::log(m_TreeImpl->m_Lambda / 10.0),
                                 std::log(10.0 * m_TreeImpl->m_Lambda));
    }
    if (m_TreeImpl->m_GammaOverride == boost::none) {
        boundingBox.emplace_back(std::log(m_TreeImpl->m_Gamma / 10.0),
                                 std::log(10.0 * m_TreeImpl->m_Gamma));
    }
    if (m_TreeImpl->m_EtaOverride == boost::none) {
        double rate{m_TreeImpl->m_EtaGrowthRatePerTree - 1.0};
        boundingBox.emplace_back(std::log(0.3 * m_TreeImpl->m_Eta),
                                 std::log(3.0 * m_TreeImpl->m_Eta));
        boundingBox.emplace_back(1.0 + rate / 2.0, 1.0 + 1.5 * rate);
    }
    if (m_TreeImpl->m_FeatureBagFractionOverride == boost::none) {
        boundingBox.emplace_back(0.2, 0.8);
    }

    m_TreeImpl->m_BayesianOptimization =
        std::make_unique<CBayesianOptimisation>(std::move(boundingBox));
    m_TreeImpl->m_NumberRounds = this->numberHyperparameterTuningRounds();
    m_TreeImpl->m_CurrentRound = 0; // for first start

    m_RecordProgress(1.0 / static_cast<double>(m_TreeImpl->m_NumberRounds + 3));
}

void CBoostedTreeFactory::initializeMissingFeatureMasks(const core::CDataFrame& frame) const {

    m_TreeImpl->m_MissingFeatureRowMasks.resize(frame.numberColumns());

    auto result = frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                double value{(*row)[i]};
                if (CDataFrameUtils::isMissing(value)) {
                    m_TreeImpl->m_MissingFeatureRowMasks[i].extend(
                        false, row->index() -
                                   m_TreeImpl->m_MissingFeatureRowMasks[i].size());
                    m_TreeImpl->m_MissingFeatureRowMasks[i].extend(true);
                }
            }
        }
    });

    for (auto& mask : m_TreeImpl->m_MissingFeatureRowMasks) {
        mask.extend(false, frame.numberRows() - mask.size());
        LOG_TRACE(<< "# missing = " << mask.manhattan());
    }
}

std::pair<CBoostedTreeImpl::TPackedBitVectorVec, CBoostedTreeImpl::TPackedBitVectorVec>
CBoostedTreeFactory::crossValidationRowMasks() const {

    core::CPackedBitVector allTrainingRowsMask{m_TreeImpl->allTrainingRowsMask()};

    TPackedBitVectorVec trainingRowMasks(m_TreeImpl->m_NumberFolds);

    for (auto row = allTrainingRowsMask.beginOneBits();
         row != allTrainingRowsMask.endOneBits(); ++row) {
        std::size_t fold{CSampling::uniformSample(m_TreeImpl->m_Rng, 0,
                                                  m_TreeImpl->m_NumberFolds)};
        trainingRowMasks[fold].extend(true, *row - trainingRowMasks[fold].size());
        trainingRowMasks[fold].extend(false);
    }

    for (auto& fold : trainingRowMasks) {
        fold.extend(true, allTrainingRowsMask.size() - fold.size());
        fold &= allTrainingRowsMask;
        LOG_TRACE(<< "# training = " << fold.manhattan());
    }

    TPackedBitVectorVec testingRowMasks(m_TreeImpl->m_NumberFolds,
                                        std::move(allTrainingRowsMask));
    for (std::size_t i = 0; i < m_TreeImpl->m_NumberFolds; ++i) {
        testingRowMasks[i] ^= trainingRowMasks[i];
        LOG_TRACE(<< "# testing = " << testingRowMasks[i].manhattan());
    }

    return {trainingRowMasks, testingRowMasks};
}

bool CBoostedTreeFactory::initializeFeatureSampleDistribution(const core::CDataFrame& frame) const {

    // Exclude all constant features by zeroing their probabilities.

    std::size_t n{numberFeatures(frame)};

    TSizeVec regressors(n);
    std::iota(regressors.begin(), regressors.end(), 0);
    regressors.erase(regressors.begin() + m_TreeImpl->m_DependentVariable);

    TDoubleVec mics(CDataFrameUtils::micWithColumn(
        CDataFrameUtils::CMetricColumnValue{m_TreeImpl->m_DependentVariable}, frame, regressors));

    regressors.erase(std::remove_if(regressors.begin(), regressors.end(),
                                    [&](std::size_t i) { return mics[i] == 0.0; }),
                     regressors.end());
    LOG_TRACE(<< "candidate regressors = " << core::CContainerPrinter::print(regressors));

    m_TreeImpl->m_FeatureSampleProbabilities.assign(n, 0.0);
    if (regressors.size() > 0) {
        std::stable_sort(regressors.begin(), regressors.end(),
                         [&mics](std::size_t lhs, std::size_t rhs) {
                             return mics[lhs] > mics[rhs];
                         });

        std::size_t maximumNumberFeatures{frame.numberRows() / m_TreeImpl->m_RowsPerFeature};
        LOG_TRACE(<< "Using up to " << maximumNumberFeatures << " out of "
                  << regressors.size() << " features");

        regressors.resize(std::min(maximumNumberFeatures, regressors.size()));

        double Z{std::accumulate(
            regressors.begin(), regressors.end(), 0.0,
            [&mics](double z, std::size_t i) { return z + mics[i]; })};
        LOG_TRACE(<< "Z = " << Z);
        for (auto i : regressors) {
            m_TreeImpl->m_FeatureSampleProbabilities[i] = mics[i] / Z;
        }
        LOG_TRACE(<< "P(sample) = "
                  << core::CContainerPrinter::print(m_TreeImpl->m_FeatureSampleProbabilities));
        return true;
    }
    return false;
}

void CBoostedTreeFactory::initializeHyperparameters(core::CDataFrame& frame) const {

    m_TreeImpl->m_Lambda = m_TreeImpl->m_LambdaOverride.value_or(0.0);
    m_TreeImpl->m_Gamma = m_TreeImpl->m_GammaOverride.value_or(0.0);
    if (m_TreeImpl->m_EtaOverride) {
        m_TreeImpl->m_Eta = *(m_TreeImpl->m_EtaOverride);
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
        m_TreeImpl->m_Eta = 1.0 / std::max(10.0, std::sqrt(static_cast<double>(
                                                     frame.numberColumns() - 4)));
        m_TreeImpl->m_EtaGrowthRatePerTree = 1.0 + m_TreeImpl->m_Eta / 2.0;
    }
    if (m_TreeImpl->m_MaximumNumberTreesOverride) {
        m_TreeImpl->m_MaximumNumberTrees = *(m_TreeImpl->m_MaximumNumberTreesOverride);
    } else {
        // This needs to be tied to the learn rate to avoid bias.
        m_TreeImpl->m_MaximumNumberTrees =
            static_cast<std::size_t>(2.0 / m_TreeImpl->m_Eta + 0.5);
    }
    if (m_TreeImpl->m_FeatureBagFractionOverride) {
        m_TreeImpl->m_FeatureBagFraction = *(m_TreeImpl->m_FeatureBagFractionOverride);
    }

    if (m_TreeImpl->m_LambdaOverride && m_TreeImpl->m_GammaOverride) {
        // Fall through.
    } else {
        core::CPackedBitVector trainingRowMask{m_TreeImpl->allTrainingRowsMask()};

        auto vector = m_TreeImpl->initializePredictionsAndLossDerivatives(frame, trainingRowMask);

        double L[2];
        double T[2];
        double W[2];

        std::tie(L[0], T[0], W[0]) =
            m_TreeImpl->regularisedLoss(frame, trainingRowMask, {std::move(vector)});
        LOG_TRACE(<< "loss = " << L[0] << ", # leaves = " << T[0]
                  << ", sum square weights = " << W[0]);

        auto forest = m_TreeImpl->trainForest(frame, trainingRowMask, m_RecordProgress);

        std::tie(L[1], T[1], W[1]) =
            m_TreeImpl->regularisedLoss(frame, trainingRowMask, forest);
        LOG_TRACE(<< "loss = " << L[1] << ", # leaves = " << T[1]
                  << ", sum square weights = " << W[1]);

        double scale{static_cast<double>(m_TreeImpl->m_NumberFolds - 1) /
                     static_cast<double>(m_TreeImpl->m_NumberFolds)};
        double lambda{scale * std::max((L[0] - L[1]) / (W[1] - W[0]), 0.0) / 5.0};
        double gamma{scale * std::max((L[0] - L[1]) / (T[1] - T[0]), 0.0) / 5.0};

        if (m_TreeImpl->m_LambdaOverride == boost::none) {
            m_TreeImpl->m_Lambda = m_TreeImpl->m_GammaOverride ? lambda : 0.5 * lambda;
        }
        if (m_TreeImpl->m_GammaOverride == boost::none) {
            m_TreeImpl->m_Gamma = m_TreeImpl->m_LambdaOverride ? gamma : 0.5 * gamma;
        }
        LOG_TRACE(<< "lambda(initial) = " << m_TreeImpl->m_Lambda
                  << " gamma(initial) = " << m_TreeImpl->m_Gamma);
    }

    m_TreeImpl->m_MaximumTreeSizeFraction = 10.0;

    if (m_TreeImpl->m_MaximumNumberTreesOverride == boost::none) {
        // We allow a large number of trees by default in the main parameter
        // optimisation loop. In practice, we should use many fewer if they
        // don't significantly improve test error.
        m_TreeImpl->m_MaximumNumberTrees *= 10;
    }
}

CBoostedTreeFactory
CBoostedTreeFactory::constructFromParameters(std::size_t numberThreads,
                                             CBoostedTree::TLossFunctionUPtr loss) {
    return {numberThreads, std::move(loss)};
}

CBoostedTreeFactory::TBoostedTreeUPtr
CBoostedTreeFactory::constructFromString(std::stringstream& jsonStringStream,
                                         core::CDataFrame& frame) {
    try {
        TBoostedTreeUPtr treePtr{
            new CBoostedTree{frame, TBoostedTreeImplUPtr{new CBoostedTreeImpl{}}}};
        core::CJsonStateRestoreTraverser traverser(jsonStringStream);
        if (treePtr->acceptRestoreTraverser(traverser) == false || traverser.haveBadState()) {
            throw std::runtime_error{"failed to restore boosted tree"};
        }
        return treePtr;
    } catch (const std::exception& e) {
        HANDLE_FATAL(<< "Input error: '" << e.what() << "'. Check logs for more details.");
    }
    return nullptr;
}

CBoostedTreeFactory::CBoostedTreeFactory(std::size_t numberThreads,
                                         CBoostedTree::TLossFunctionUPtr loss)
    : m_TreeImpl{std::make_unique<CBoostedTreeImpl>(numberThreads, std::move(loss))} {
}

CBoostedTreeFactory::CBoostedTreeFactory(CBoostedTreeFactory&&) = default;

CBoostedTreeFactory& CBoostedTreeFactory::operator=(CBoostedTreeFactory&&) = default;

CBoostedTreeFactory::~CBoostedTreeFactory() = default;

CBoostedTreeFactory& CBoostedTreeFactory::numberFolds(std::size_t numberFolds) {
    if (numberFolds < 2) {
        LOG_WARN(<< "Must use at least two-folds for cross validation");
        numberFolds = 2;
    }
    m_TreeImpl->m_NumberFolds = numberFolds;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::lambda(double lambda) {
    if (lambda < 0.0) {
        LOG_WARN(<< "Lambda must be non-negative");
        lambda = 0.0;
    }
    m_TreeImpl->m_LambdaOverride = lambda;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::gamma(double gamma) {
    if (gamma < 0.0) {
        LOG_WARN(<< "Gamma must be non-negative");
        gamma = 0.0;
    }
    m_TreeImpl->m_GammaOverride = gamma;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::eta(double eta) {
    if (eta < MINIMUM_ETA) {
        LOG_WARN(<< "Truncating supplied learning rate " << eta
                 << " which must be no smaller than " << MINIMUM_ETA);
        eta = std::max(eta, MINIMUM_ETA);
    }
    if (eta > 1.0) {
        LOG_WARN(<< "Using a learning rate greater than one doesn't make sense");
        eta = 1.0;
    }
    m_TreeImpl->m_EtaOverride = eta;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::maximumNumberTrees(std::size_t maximumNumberTrees) {
    if (maximumNumberTrees == 0) {
        LOG_WARN(<< "Forest must have at least one tree");
        maximumNumberTrees = 1;
    }
    if (maximumNumberTrees > MAXIMUM_NUMBER_TREES) {
        LOG_WARN(<< "Truncating supplied maximum number of trees " << maximumNumberTrees
                 << " which must be no larger than " << MAXIMUM_NUMBER_TREES);
        maximumNumberTrees = std::min(maximumNumberTrees, MAXIMUM_NUMBER_TREES);
    }
    m_TreeImpl->m_MaximumNumberTreesOverride = maximumNumberTrees;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::featureBagFraction(double featureBagFraction) {
    if (featureBagFraction < 0.0 || featureBagFraction > 1.0) {
        LOG_WARN(<< "Truncating supplied feature bag fraction " << featureBagFraction
                 << " which must be positive and not more than one");
        featureBagFraction = CTools::truncate(featureBagFraction, 0.0, 1.0);
    }
    m_TreeImpl->m_FeatureBagFractionOverride = featureBagFraction;
    return *this;
}

CBoostedTreeFactory&
CBoostedTreeFactory::maximumOptimisationRoundsPerHyperparameter(std::size_t rounds) {
    m_TreeImpl->m_MaximumOptimisationRoundsPerHyperparameter = rounds;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::rowsPerFeature(std::size_t rowsPerFeature) {
    if (m_TreeImpl->m_RowsPerFeature == 0) {
        LOG_WARN(<< "Must have at least one training example per feature");
        rowsPerFeature = 1;
    }
    m_TreeImpl->m_RowsPerFeature = rowsPerFeature;
    return *this;
}

CBoostedTreeFactory&
CBoostedTreeFactory::progressCallback(CBoostedTree::TProgressCallback callback) {
    m_RecordProgress = callback;
    return *this;
}

std::size_t CBoostedTreeFactory::estimateMemoryUsage(std::size_t numberRows,
                                                     std::size_t numberColumns) const {
    return m_TreeImpl->estimateMemoryUsage(numberRows, numberColumns);
}

std::size_t CBoostedTreeFactory::numberExtraColumnsForTrain() const {
    return m_TreeImpl->numberExtraColumnsForTrain();
}

const double CBoostedTreeFactory::MINIMUM_ETA{1e-3};
const std::size_t CBoostedTreeFactory::MAXIMUM_NUMBER_TREES{
    static_cast<std::size_t>(2.0 / MINIMUM_ETA + 0.5)};
}
}
