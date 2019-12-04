/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeFactory.h>

#include <core/CJsonStateRestoreTraverser.h>

#include <maths/CBayesianOptimisation.h>
#include <maths/CBoostedTreeImpl.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CSampling.h>

#include <cmath>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TRowItr = core::CDataFrame::TRowItr;

namespace {
const std::size_t MIN_REGULARIZER_INDEX{0};
const std::size_t BEST_REGULARIZER_INDEX{1};
const std::size_t MAX_REGULARIZER_INDEX{2};
const std::size_t LINE_SEARCH_ITERATIONS{8};
const double MIN_SOFT_DEPTH_LIMIT{2.0};
const double MIN_SOFT_DEPTH_LIMIT_TOLERANCE{0.05};
const double MAX_SOFT_DEPTH_LIMIT_TOLERANCE{0.25};
const double MIN_ETA{1e-3};
const double MIN_ETA_SCALE{0.3};
const double MAX_ETA_SCALE{3.0};
const double MIN_ETA_GROWTH_RATE_SCALE{0.5};
const double MAX_ETA_GROWTH_RATE_SCALE{1.5};
const double MIN_FEATURE_BAG_FRACTION{0.2};
const double MAX_FEATURE_BAG_FRACTION{0.8};
const double MIN_DOWNSAMPLE_LINE_SEARCH_RANGE{2.0};
const double MAX_DOWNSAMPLE_LINE_SEARCH_RANGE{144.0};
const double MIN_DOWNSAMPLE_FACTOR_SCALE{0.3};
const double MAX_DOWNSAMPLE_FACTOR_SCALE{3.0};
const std::size_t MAX_NUMBER_TREES{static_cast<std::size_t>(2.0 / MIN_ETA + 0.5)};

std::size_t computeMaximumNumberTrees(double eta) {
    return static_cast<std::size_t>(2.0 / eta / MIN_DOWNSAMPLE_FACTOR_SCALE + 0.5);
}

std::size_t scaleMaximumNumberTrees(std::size_t maximumNumberTrees) {
    // We actively optimise for eta and allow it to be up to MIN_ETA_SCALE
    // smaller than the initial value. We need to allow the number of trees
    // to increase proportionally to avoid bias. In practice, we should use
    // fewer if they don't significantly improve the validation error.
    return static_cast<std::size_t>(
        static_cast<double>(maximumNumberTrees) / MIN_ETA_SCALE + 0.5);
}
}

CBoostedTreeFactory::TBoostedTreeUPtr
CBoostedTreeFactory::buildFor(core::CDataFrame& frame,
                              TLossFunctionUPtr loss,
                              std::size_t dependentVariable) {

    if (loss == nullptr) {
        HANDLE_FATAL(<< "Internal error: must supply a loss function");
        return nullptr;
    }

    this->initializeTrainingProgressMonitoring();

    m_TreeImpl->m_DependentVariable = dependentVariable;
    m_TreeImpl->m_Loss = std::move(loss);

    this->initializeMissingFeatureMasks(frame);

    frame.resizeColumns(m_TreeImpl->m_NumberThreads,
                        frame.numberColumns() + this->numberExtraColumnsForTrain());

    this->initializeCrossValidation(frame);
    this->selectFeaturesAndEncodeCategories(frame);
    this->determineFeatureDataTypes(frame);

    if (this->initializeFeatureSampleDistribution()) {
        this->initializeHyperparameters(frame);
        this->initializeHyperparameterOptimisation();
    }

    auto treeImpl = std::make_unique<CBoostedTreeImpl>(m_NumberThreads,
                                                       m_TreeImpl->m_Loss->clone());
    std::swap(m_TreeImpl, treeImpl);
    return TBoostedTreeUPtr{new CBoostedTree{frame, m_RecordProgress, m_RecordMemoryUsage,
                                             m_RecordTrainingState, std::move(treeImpl)}};
}

CBoostedTreeFactory::TBoostedTreeUPtr
CBoostedTreeFactory::restoreFor(core::CDataFrame& frame, std::size_t dependentVariable) {

    if (dependentVariable != m_TreeImpl->m_DependentVariable) {
        HANDLE_FATAL(<< "Internal error: expected dependent variable "
                     << m_TreeImpl->m_DependentVariable << " got " << dependentVariable);
        return nullptr;
    }

    this->resumeRestoredTrainingProgressMonitoring();

    frame.resizeColumns(m_TreeImpl->m_NumberThreads,
                        frame.numberColumns() + this->numberExtraColumnsForTrain());

    return TBoostedTreeUPtr{new CBoostedTree{frame, m_RecordProgress, m_RecordMemoryUsage,
                                             m_RecordTrainingState, std::move(m_TreeImpl)}};
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
    if (m_TreeImpl->m_DownsampleFactorOverride == boost::none) {
        boundingBox.emplace_back(
            m_LogDownsampleFactorSearchInterval(MIN_REGULARIZER_INDEX),
            m_LogDownsampleFactorSearchInterval(MAX_REGULARIZER_INDEX));
    }
    if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        boundingBox.emplace_back(
            m_LogDepthPenaltyMultiplierSearchInterval(MIN_REGULARIZER_INDEX),
            m_LogDepthPenaltyMultiplierSearchInterval(MAX_REGULARIZER_INDEX));
    }
    if (m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        boundingBox.emplace_back(
            m_LogLeafWeightPenaltyMultiplierSearchInterval(MIN_REGULARIZER_INDEX),
            m_LogLeafWeightPenaltyMultiplierSearchInterval(MAX_REGULARIZER_INDEX));
    }
    if (m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        boundingBox.emplace_back(
            m_LogTreeSizePenaltyMultiplierSearchInterval(MIN_REGULARIZER_INDEX),
            m_LogTreeSizePenaltyMultiplierSearchInterval(MAX_REGULARIZER_INDEX));
    }
    if (m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit() == boost::none) {
        boundingBox.emplace_back(m_SoftDepthLimitSearchInterval(MIN_REGULARIZER_INDEX),
                                 m_SoftDepthLimitSearchInterval(MAX_REGULARIZER_INDEX));
    }
    if (m_TreeImpl->m_RegularizationOverride.softTreeDepthTolerance() == boost::none) {
        boundingBox.emplace_back(MIN_SOFT_DEPTH_LIMIT_TOLERANCE, MAX_SOFT_DEPTH_LIMIT_TOLERANCE);
    }
    if (m_TreeImpl->m_EtaOverride == boost::none) {
        double rate{m_TreeImpl->m_EtaGrowthRatePerTree - 1.0};
        boundingBox.emplace_back(std::log(MIN_ETA_SCALE * m_TreeImpl->m_Eta),
                                 std::log(MAX_ETA_SCALE * m_TreeImpl->m_Eta));
        boundingBox.emplace_back(1.0 + MIN_ETA_GROWTH_RATE_SCALE * rate,
                                 1.0 + MAX_ETA_GROWTH_RATE_SCALE * rate);
    }
    if (m_TreeImpl->m_FeatureBagFractionOverride == boost::none) {
        boundingBox.emplace_back(MIN_FEATURE_BAG_FRACTION, MAX_FEATURE_BAG_FRACTION);
    }
    LOG_TRACE(<< "hyperparameter search bounding box = "
              << core::CContainerPrinter::print(boundingBox));

    m_TreeImpl->m_BayesianOptimization = std::make_unique<CBayesianOptimisation>(
        std::move(boundingBox),
        m_BayesianOptimisationRestarts.value_or(CBayesianOptimisation::RESTARTS));
    m_TreeImpl->m_NumberRounds = this->numberHyperparameterTuningRounds();
    m_TreeImpl->m_CurrentRound = 0; // for first start
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

void CBoostedTreeFactory::initializeCrossValidation(core::CDataFrame& frame) const {

    core::CPackedBitVector allTrainingRowsMask{m_TreeImpl->allTrainingRowsMask()};

    TDoubleVec frequencies;
    core::CDataFrame::TRowFunc writeRowWeight;

    std::size_t numberBuckets(m_StratifyRegressionCrossValidation ? 10 : 1);
    std::tie(m_TreeImpl->m_TrainingRowMasks, m_TreeImpl->m_TestingRowMasks, frequencies) =
        CDataFrameUtils::stratifiedCrossValidationRowMasks(
            m_TreeImpl->m_NumberThreads, frame, m_TreeImpl->m_DependentVariable,
            m_TreeImpl->m_Rng, m_TreeImpl->m_NumberFolds, numberBuckets, allTrainingRowsMask);

    if (frame.columnIsCategorical()[m_TreeImpl->m_DependentVariable]) {
        writeRowWeight = [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                if (m_BalanceClassTrainingLoss) {
                    // Weight by inverse category frequency.
                    std::size_t category{static_cast<std::size_t>(
                        (*row)[m_TreeImpl->m_DependentVariable])};
                    row->writeColumn(exampleWeightColumn(row->numberColumns()),
                                     1.0 / frequencies[category]);
                } else {
                    row->writeColumn(exampleWeightColumn(row->numberColumns()), 1.0);
                }
            }
        };
    } else {
        writeRowWeight = [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                row->writeColumn(exampleWeightColumn(row->numberColumns()), 1.0);
            }
        };
    }

    frame.writeColumns(m_NumberThreads, 0, frame.numberRows(), writeRowWeight,
                       &allTrainingRowsMask);
}

void CBoostedTreeFactory::selectFeaturesAndEncodeCategories(const core::CDataFrame& frame) const {

    // TODO we should do feature selection per fold.

    TSizeVec regressors(frame.numberColumns() - this->numberExtraColumnsForTrain());
    std::iota(regressors.begin(), regressors.end(), 0);
    regressors.erase(regressors.begin() + m_TreeImpl->m_DependentVariable);
    LOG_TRACE(<< "candidate regressors = " << core::CContainerPrinter::print(regressors));

    m_TreeImpl->m_Encoder = std::make_unique<CDataFrameCategoryEncoder>(
        CMakeDataFrameCategoryEncoder{m_TreeImpl->m_NumberThreads, frame,
                                      m_TreeImpl->m_DependentVariable}
            .minimumRowsPerFeature(m_TreeImpl->m_RowsPerFeature)
            .minimumFrequencyToOneHotEncode(m_MinimumFrequencyToOneHotEncode)
            .rowMask(m_TreeImpl->allTrainingRowsMask())
            .columnMask(std::move(regressors)));
    m_TreeImpl->m_TrainingProgress.increment(1);
}

void CBoostedTreeFactory::determineFeatureDataTypes(const core::CDataFrame& frame) const {

    TSizeVec columnMask(m_TreeImpl->m_Encoder->numberEncodedColumns());
    std::iota(columnMask.begin(), columnMask.end(), 0);
    columnMask.erase(std::remove_if(columnMask.begin(), columnMask.end(),
                                    [this](std::size_t index) {
                                        return m_TreeImpl->m_Encoder->isBinary(index);
                                    }),
                     columnMask.end());

    m_TreeImpl->m_FeatureDataTypes = CDataFrameUtils::columnDataTypes(
        m_TreeImpl->m_NumberThreads, frame, m_TreeImpl->allTrainingRowsMask(),
        columnMask, m_TreeImpl->m_Encoder.get());
}

bool CBoostedTreeFactory::initializeFeatureSampleDistribution() const {

    // Compute feature sample probabilities.

    TDoubleVec mics(m_TreeImpl->m_Encoder->encodedColumnMics());
    LOG_TRACE(<< "candidate regressors MICe = " << core::CContainerPrinter::print(mics));

    if (mics.size() > 0) {
        double Z{std::accumulate(mics.begin(), mics.end(), 0.0,
                                 [](double z, double mic) { return z + mic; })};
        LOG_TRACE(<< "Z = " << Z);
        for (auto& mic : mics) {
            mic /= Z;
        }
        m_TreeImpl->m_FeatureSampleProbabilities = std::move(mics);
        LOG_TRACE(<< "P(sample) = "
                  << core::CContainerPrinter::print(m_TreeImpl->m_FeatureSampleProbabilities));
        return true;
    }
    return false;
}

void CBoostedTreeFactory::initializeHyperparameters(core::CDataFrame& frame) {

    if (m_TreeImpl->m_EtaOverride != boost::none) {
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

    if (m_TreeImpl->m_MaximumNumberTreesOverride != boost::none) {
        m_TreeImpl->m_MaximumNumberTrees = *(m_TreeImpl->m_MaximumNumberTreesOverride);
    } else {
        // This needs to be tied to the learn rate to avoid bias.
        m_TreeImpl->m_MaximumNumberTrees = computeMaximumNumberTrees(m_TreeImpl->m_Eta);
    }

    if (m_TreeImpl->m_FeatureBagFractionOverride != boost::none) {
        m_TreeImpl->m_FeatureBagFraction = *(m_TreeImpl->m_FeatureBagFractionOverride);
    }

    double numberFeatures{static_cast<double>(m_TreeImpl->m_Encoder->numberEncodedColumns())};
    double downSampleFactor{m_InitialDownsampleRowsPerFeature * numberFeatures /
                            m_TreeImpl->m_TrainingRowMasks[0].manhattan()};
    m_TreeImpl->m_DownsampleFactor = m_TreeImpl->m_DownsampleFactorOverride.value_or(
        CTools::truncate(downSampleFactor, 0.05, 0.5));

    m_TreeImpl->m_Regularization
        .depthPenaltyMultiplier(
            m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier().value_or(0.0))
        .treeSizePenaltyMultiplier(
            m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier().value_or(0.0))
        .leafWeightPenaltyMultiplier(
            m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier().value_or(0.0))
        .softTreeDepthLimit(
            m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit().value_or(0.0))
        .softTreeDepthTolerance(
            m_TreeImpl->m_RegularizationOverride.softTreeDepthTolerance().value_or(0.0));

    if (m_TreeImpl->m_RegularizationOverride.countNotSet() > 0) {
        this->initializeUnsetRegularizationHyperparameters(frame);
    }

    this->initializeUnsetDownsampleFactor(frame);

    if (m_TreeImpl->m_MaximumNumberTreesOverride == boost::none) {
        m_TreeImpl->m_MaximumNumberTrees =
            scaleMaximumNumberTrees(m_TreeImpl->m_MaximumNumberTrees);
    }
}

void CBoostedTreeFactory::initializeUnsetRegularizationHyperparameters(core::CDataFrame& frame) {

    // The strategy here is to:
    //   1) Get percentile estimates of the gain in the loss function and its sum
    //      curvature from the splits selected in a single tree with regularisers
    //      zeroed,
    //   2) Use these to extract reasonable intervals to search for the multipliers
    //      for the various regularisation penalties,
    //   3) Line search these intervales for a turning point in the test loss for
    //      the base learner, i.e. the point at which transition to overfit occurs.
    //
    // We'll search intervals in the vicinity of these values in the hyperparameter
    // optimisation loop.

    double log2MaxTreeSize{std::log2(static_cast<double>(m_TreeImpl->maximumTreeSize(
                               m_TreeImpl->m_TrainingRowMasks[0]))) +
                           1.0};
    m_TreeImpl->m_Regularization.softTreeDepthLimit(
        m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit().value_or(log2MaxTreeSize));
    m_TreeImpl->m_Regularization.softTreeDepthTolerance(
        m_TreeImpl->m_RegularizationOverride.softTreeDepthTolerance().value_or(
            0.5 * (MIN_SOFT_DEPTH_LIMIT_TOLERANCE + MAX_SOFT_DEPTH_LIMIT_TOLERANCE)));
    LOG_TRACE(<< "max depth = " << m_TreeImpl->m_Regularization.softTreeDepthLimit()
              << ", tolerance = " << m_TreeImpl->m_Regularization.softTreeDepthTolerance());

    auto gainAndTotalCurvaturePerNode =
        this->estimateTreeGainAndCurvature(frame, {1.0, 50.0, 90.0});
    LOG_TRACE(<< "gains and total curvatures per node = "
              << core::CContainerPrinter::print(gainAndTotalCurvaturePerNode));

    double gainPerNode1stPercentile{gainAndTotalCurvaturePerNode[0].first};
    double gainPerNode50thPercentile{gainAndTotalCurvaturePerNode[1].first};
    double gainPerNode90thPercentile{gainAndTotalCurvaturePerNode[2].first};
    double totalCurvaturePerNode1stPercentile{gainAndTotalCurvaturePerNode[0].second};
    double totalCurvaturePerNode90thPercentile{gainAndTotalCurvaturePerNode[2].second};

    // Make sure all line search intervals are not empty.
    gainPerNode1stPercentile = std::min(gainPerNode1stPercentile, 0.1 * gainPerNode90thPercentile);
    totalCurvaturePerNode1stPercentile = std::min(
        totalCurvaturePerNode1stPercentile, 0.1 * totalCurvaturePerNode90thPercentile);

    // Search for depth limit at which the tree starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit() == boost::none) {
        if (gainPerNode90thPercentile > 0.0) {
            if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
                m_TreeImpl->m_Regularization.depthPenaltyMultiplier(gainPerNode50thPercentile);
            }
            double initialSoftDepthLimit{MIN_SOFT_DEPTH_LIMIT + log2MaxTreeSize};
            double stepSize{-log2MaxTreeSize /
                            static_cast<double>(LINE_SEARCH_ITERATIONS - 1)};
            TVector initial{initialSoftDepthLimit};
            TVector fallback{{-log2MaxTreeSize, -log2MaxTreeSize / 2.0, 0.0}};
            LOG_TRACE(<< "soft depth limit step size = " << stepSize);

            auto applySoftDepthLimitStep = [&](CBoostedTreeImpl& tree,
                                               double stepSize_, std::size_t step) {
                tree.m_Regularization.softTreeDepthLimit(
                    initialSoftDepthLimit + static_cast<double>(step) * stepSize_);
                return true;
            };

            m_SoftDepthLimitSearchInterval =
                initial + this->testLossLineSearch(frame, applySoftDepthLimitStep,
                                                   2.0 * stepSize, -2.0 * stepSize, stepSize)
                              .value_or(fallback);
            LOG_TRACE(<< "soft depth limit search interval = ["
                      << m_SoftDepthLimitSearchInterval.toDelimited() << "]");

            m_TreeImpl->m_Regularization.softTreeDepthLimit(
                m_SoftDepthLimitSearchInterval(BEST_REGULARIZER_INDEX));
        } else {
            m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit(
                m_TreeImpl->m_Regularization.softTreeDepthLimit());
        }
    }

    // Set the depth limit to its smallest value and search for the value of the
    // penalty multiplier at which the tree starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        if (gainPerNode90thPercentile > 0.0) {
            double logInitialDepthPenaltyMultiplier{std::log(gainPerNode90thPercentile)};
            double searchIntervalSize{gainPerNode90thPercentile / gainPerNode1stPercentile};
            double stepSize{-std::log(searchIntervalSize) /
                            static_cast<double>(LINE_SEARCH_ITERATIONS - 1)};
            TVector initial{logInitialDepthPenaltyMultiplier};
            TVector fallback{{std::log(gainPerNode1stPercentile),
                              0.5 * std::log(gainPerNode1stPercentile * gainPerNode90thPercentile),
                              std::log(gainPerNode90thPercentile)}};
            LOG_TRACE(<< "depth penalty multiplier step size = " << stepSize);

            auto applyDepthPenaltyMultiplierStep =
                [logInitialDepthPenaltyMultiplier](
                    CBoostedTreeImpl& tree, double stepSize_, std::size_t step) {
                    tree.m_Regularization.depthPenaltyMultiplier(
                        std::exp(logInitialDepthPenaltyMultiplier +
                                 static_cast<double>(step) * stepSize_));
                    return true;
                };

            m_LogDepthPenaltyMultiplierSearchInterval =
                initial + this->testLossLineSearch(frame, applyDepthPenaltyMultiplierStep,
                                                   2.0 * stepSize, -2.0 * stepSize, stepSize)
                              .value_or(fallback - initial);
            LOG_TRACE(<< "log depth penalty multiplier search interval = ["
                      << m_LogDepthPenaltyMultiplierSearchInterval.toDelimited() << "]");

            m_TreeImpl->m_Regularization.depthPenaltyMultiplier(std::exp(
                m_LogDepthPenaltyMultiplierSearchInterval(MIN_REGULARIZER_INDEX)));
        } else {
            m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier(0.0);
        }
    }

    // Search for the value of the tree size penalty multiplier at which the tree
    // starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        if (gainPerNode90thPercentile > 0.0) {
            double logInitialTreeSizePenaltyMultiplier{std::log(gainPerNode90thPercentile)};
            double searchIntervalSize{gainPerNode90thPercentile / gainPerNode1stPercentile};
            double stepSize{-std::log(searchIntervalSize) /
                            static_cast<double>(LINE_SEARCH_ITERATIONS - 1)};
            TVector initial{logInitialTreeSizePenaltyMultiplier};
            TVector fallback{{std::log(gainPerNode1stPercentile),
                              0.5 * std::log(gainPerNode1stPercentile * gainPerNode90thPercentile),
                              std::log(gainPerNode90thPercentile)}};
            LOG_TRACE(<< "log tree size penalty multiplier step size = " << stepSize);

            auto applyTreeSizePenaltyMultiplierStep =
                [logInitialTreeSizePenaltyMultiplier](
                    CBoostedTreeImpl& tree, double stepSize_, std::size_t step) {
                    tree.m_Regularization.treeSizePenaltyMultiplier(
                        std::exp(logInitialTreeSizePenaltyMultiplier +
                                 static_cast<double>(step) * stepSize_));
                    return true;
                };

            m_LogTreeSizePenaltyMultiplierSearchInterval =
                initial + this->testLossLineSearch(frame, applyTreeSizePenaltyMultiplierStep,
                                                   2.0 * stepSize, -2.0 * stepSize, stepSize)
                              .value_or(fallback - initial);
            LOG_TRACE(<< "log tree size penalty multiplier search interval = ["
                      << m_LogTreeSizePenaltyMultiplierSearchInterval.toDelimited() << "]");

            m_TreeImpl->m_Regularization.treeSizePenaltyMultiplier(std::exp(
                m_LogTreeSizePenaltyMultiplierSearchInterval(MIN_REGULARIZER_INDEX)));
        } else {
            m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier(0.0);
        }
    }

    // Search for the value of the leaf weight penalty multiplier at which the
    // tree starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        if (totalCurvaturePerNode90thPercentile > 0.0) {
            double logInitialLeafWeightPenaltyMultiplier{
                std::log(totalCurvaturePerNode90thPercentile)};
            double searchIntervalSize{totalCurvaturePerNode90thPercentile /
                                      totalCurvaturePerNode1stPercentile};
            double stepSize{-std::log(searchIntervalSize) /
                            static_cast<double>(LINE_SEARCH_ITERATIONS - 1)};
            TVector initial{logInitialLeafWeightPenaltyMultiplier};
            TVector fallback{{std::log(totalCurvaturePerNode1stPercentile),
                              0.5 * std::log(totalCurvaturePerNode1stPercentile *
                                             totalCurvaturePerNode90thPercentile),
                              std::log(totalCurvaturePerNode90thPercentile)}};
            LOG_TRACE(<< "log leaf weight penalty multiplier step size = " << stepSize);

            auto applyLeafWeightPenaltyMultiplierStep =
                [logInitialLeafWeightPenaltyMultiplier](
                    CBoostedTreeImpl& tree, double stepSize_, std::size_t step) {
                    tree.m_Regularization.leafWeightPenaltyMultiplier(
                        std::exp(logInitialLeafWeightPenaltyMultiplier +
                                 static_cast<double>(step) * stepSize_));
                    return true;
                };

            m_LogLeafWeightPenaltyMultiplierSearchInterval =
                initial + this->testLossLineSearch(frame, applyLeafWeightPenaltyMultiplierStep,
                                                   2.0 * stepSize, -2.0 * stepSize, stepSize)
                              .value_or(fallback - initial);
            LOG_TRACE(<< "log leaf weight penalty multiplier search interval = ["
                      << m_LogLeafWeightPenaltyMultiplierSearchInterval.toDelimited()
                      << "]");
        } else {
            m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier(0.0);
        }
    }

    if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        m_TreeImpl->m_Regularization.depthPenaltyMultiplier(std::exp(
            m_LogDepthPenaltyMultiplierSearchInterval(BEST_REGULARIZER_INDEX)));
    }
    if (m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        m_TreeImpl->m_Regularization.treeSizePenaltyMultiplier(std::exp(
            m_LogTreeSizePenaltyMultiplierSearchInterval(BEST_REGULARIZER_INDEX)));
    }
    if (m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        m_TreeImpl->m_Regularization.leafWeightPenaltyMultiplier(std::exp(
            m_LogLeafWeightPenaltyMultiplierSearchInterval(BEST_REGULARIZER_INDEX)));
    }

    if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() != boost::none &&
        m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() == 0.0) {
        m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit(MIN_SOFT_DEPTH_LIMIT);
        m_TreeImpl->m_RegularizationOverride.softTreeDepthTolerance(MIN_SOFT_DEPTH_LIMIT_TOLERANCE);
    }
    LOG_TRACE(<< "regularization(initial) = " << m_TreeImpl->m_Regularization.print());
}

void CBoostedTreeFactory::initializeUnsetDownsampleFactor(core::CDataFrame& frame) {

    if (m_TreeImpl->m_DownsampleFactorOverride == boost::none) {
        double searchIntervalSize{std::log(CTools::truncate(
            m_TreeImpl->m_TrainingRowMasks[0].manhattan() / 100.0,
            MIN_DOWNSAMPLE_LINE_SEARCH_RANGE, MAX_DOWNSAMPLE_LINE_SEARCH_RANGE))};
        double logInitialDownsampleFactor{std::log(std::min(
            std::sqrt(searchIntervalSize) * m_TreeImpl->m_DownsampleFactor, 1.0))};
        double stepSize{-std::log(searchIntervalSize) /
                        static_cast<double>(LINE_SEARCH_ITERATIONS - 1)};
        TVector initial{logInitialDownsampleFactor};
        TVector fallback{{std::log(MIN_DOWNSAMPLE_FACTOR_SCALE / MAX_DOWNSAMPLE_FACTOR_SCALE),
                          std::log(1.0 / MAX_DOWNSAMPLE_FACTOR_SCALE), 0.0}};
        LOG_TRACE(<< "log down sample factor step size = " << stepSize);

        double previousDownsampleFactor{m_TreeImpl->m_DownsampleFactor};
        double previousDepthPenaltyMultiplier{
            m_TreeImpl->m_Regularization.depthPenaltyMultiplier()};
        double previousTreeSizePenaltyMultiplier{
            m_TreeImpl->m_Regularization.treeSizePenaltyMultiplier()};
        double previousLeafWeightPenaltyMultiplier{
            m_TreeImpl->m_Regularization.leafWeightPenaltyMultiplier()};

        // We need to scale the regularisation terms to account for the difference
        // in the down sample factor compared to the value used in the line search.
        auto scaleRegularizers = [&](CBoostedTreeImpl& tree, double downsampleFactor) {
            double scale{previousDownsampleFactor / downsampleFactor};
            if (tree.m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
                tree.m_Regularization.depthPenaltyMultiplier(scale * previousDepthPenaltyMultiplier);
            }
            if (tree.m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
                tree.m_Regularization.treeSizePenaltyMultiplier(
                    scale * previousTreeSizePenaltyMultiplier);
            }
            if (tree.m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
                tree.m_Regularization.leafWeightPenaltyMultiplier(
                    scale * previousLeafWeightPenaltyMultiplier);
            }
            return scale;
        };

        auto applyDownsampleFactorStep = [&](CBoostedTreeImpl& tree,
                                             double stepSize_, std::size_t step) {
            double downsampleFactor{std::exp(logInitialDownsampleFactor +
                                             static_cast<double>(step) * stepSize_)};
            tree.m_DownsampleFactor = downsampleFactor;
            scaleRegularizers(tree, downsampleFactor);
            return tree.m_DownsampleFactor * tree.m_TrainingRowMasks[0].manhattan() > 10.0;
        };

        m_LogDownsampleFactorSearchInterval =
            initial + this->testLossLineSearch(frame, applyDownsampleFactorStep,
                                               std::log(MIN_DOWNSAMPLE_FACTOR_SCALE),
                                               std::log(MAX_DOWNSAMPLE_FACTOR_SCALE), stepSize)
                          .value_or(fallback - initial);

        // Truncate the log(scale) to be less than or equal to log(1.0).
        m_LogDownsampleFactorSearchInterval =
            min(m_LogDownsampleFactorSearchInterval, TVector{0.0});
        LOG_TRACE(<< "log down sample factor search interval = ["
                  << m_LogDownsampleFactorSearchInterval.toDelimited() << "]");

        m_TreeImpl->m_DownsampleFactor =
            std::exp(m_LogDownsampleFactorSearchInterval(BEST_REGULARIZER_INDEX));

        TVector logScale{std::log(scaleRegularizers(*m_TreeImpl, m_TreeImpl->m_DownsampleFactor))};
        m_LogTreeSizePenaltyMultiplierSearchInterval += logScale;
        m_LogLeafWeightPenaltyMultiplierSearchInterval += logScale;
    }
}

CBoostedTreeFactory::TDoubleDoublePrVec
CBoostedTreeFactory::estimateTreeGainAndCurvature(core::CDataFrame& frame,
                                                  const TDoubleVec& percentiles) const {

    std::size_t maximumNumberOfTrees{1};
    std::swap(maximumNumberOfTrees, m_TreeImpl->m_MaximumNumberTrees);
    auto forest = m_TreeImpl->trainForest(frame, m_TreeImpl->m_TrainingRowMasks[0],
                                          m_RecordMemoryUsage);
    std::swap(maximumNumberOfTrees, m_TreeImpl->m_MaximumNumberTrees);

    TDoubleDoublePrVec result;
    result.reserve(percentiles.size());

    for (auto percentile : percentiles) {
        double gain;
        double curvature;
        std::tie(gain, curvature) =
            m_TreeImpl->gainAndCurvatureAtPercentile(percentile, forest);
        LOG_TRACE(<< "gain = " << gain << ", curvature = " << curvature);

        result.emplace_back(gain, curvature);
    }

    return result;
}

CBoostedTreeFactory::TOptionalVector
CBoostedTreeFactory::testLossLineSearch(core::CDataFrame& frame,
                                        const TApplyRegularizerStep& applyRegularizerStep,
                                        double returnedIntervalLeftEndOffset,
                                        double returnedIntervalRightEndOffset,
                                        double stepSize) const {

    // This uses a quadratic approximation to the test loss function w.r.t.
    // the scaled regularization hyperparameter from which it estimates the
    // minimum error point in the interval we search here. Separately, it
    // examines size of the residual errors w.r.t. to the variation in the
    // best fit curve over the interval. We truncate the interval the main
    // hyperparameter optimisation loop searches if we determine there is a
    // low chance of missing the best solution by doing so.

    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    CLeastSquaresOnlineRegression<2, double> leastSquaresQuadraticTestLoss;
    TDoubleVec testLosses;
    testLosses.reserve(LINE_SEARCH_ITERATIONS);

    for (std::size_t i = 0; i < LINE_SEARCH_ITERATIONS; ++i) {
        if (applyRegularizerStep(*m_TreeImpl, stepSize, i) == false) {
            m_TreeImpl->m_TrainingProgress.increment(
                (LINE_SEARCH_ITERATIONS - i - 1) * m_TreeImpl->m_MaximumNumberTrees);
            break;
        }
        auto forest = m_TreeImpl->trainForest(
            frame, m_TreeImpl->m_TrainingRowMasks[0], m_RecordMemoryUsage);
        double testLoss{m_TreeImpl->meanLoss(frame, m_TreeImpl->m_TestingRowMasks[0], forest)};
        leastSquaresQuadraticTestLoss.add(static_cast<double>(i) * stepSize, testLoss);
        testLosses.push_back(testLoss);
    }
    LOG_TRACE(<< "test losses = " << core::CContainerPrinter::print(testLosses));

    CLeastSquaresOnlineRegression<2, double>::TArray params;
    if (leastSquaresQuadraticTestLoss.parameters(params) == false) {
        return TOptionalVector{};
    }

    double gradient{params[1]};
    double curvature{params[2]};
    LOG_TRACE(<< "[intercept, slope, curvature] = "
              << core::CContainerPrinter::print(params));

    // Find the minimizer of the least squares quadratic fit to the test loss
    // in the search interval. (Note step size is negative.)
    double leftEndpoint{static_cast<double>(LINE_SEARCH_ITERATIONS - 1) * stepSize};
    double rightEndpoint{0.0};
    double stationaryPoint{-(gradient == curvature ? 0.5 : gradient / 2.0 / curvature)};
    double bestRegularizer{[&] {
        if (curvature < 0.0) {
            // Stationary point is a maximum so use furthest point in interval.
            double distanceToLeftEndpoint{std::fabs(leftEndpoint - stationaryPoint)};
            double distanceToRightEndpoint{std::fabs(rightEndpoint - stationaryPoint)};
            return distanceToLeftEndpoint > distanceToRightEndpoint ? leftEndpoint : rightEndpoint;
        }
        // Stationary point is a minimum so use nearest point in the interval.
        return CTools::truncate(stationaryPoint, leftEndpoint, rightEndpoint);
    }()};
    LOG_TRACE(<< "best regularizer = " << bestRegularizer);

    TVector interval{{returnedIntervalLeftEndOffset, 0.0, returnedIntervalRightEndOffset}};
    if (curvature > 0.0) {
        // Find a short interval with a high probability of containing the optimal
        // regularisation parameter if we found a minimum. In particular, we solve
        // curvature * (x - best)^2 = 3 sigma where sigma is the standard deviation
        // of the test loss residuals to get the interval endpoints. We don't
        // extrapolate the loss function outside the line segment we searched so
        // don't truncate if an endpoint lies outside the searched interval.
        TMeanVarAccumulator residualMoments;
        for (std::size_t i = 0; i < LINE_SEARCH_ITERATIONS; ++i) {
            residualMoments.add(testLosses[i] - leastSquaresQuadraticTestLoss.predict(
                                                    static_cast<double>(i) * stepSize));
        }
        double sigma{std::sqrt(CBasicStatistics::variance(residualMoments))};
        double threeSigmaInterval{std::sqrt(3.0 * sigma / curvature)};
        if (bestRegularizer - threeSigmaInterval >= leftEndpoint) {
            interval(MIN_REGULARIZER_INDEX) =
                std::max(-threeSigmaInterval, returnedIntervalLeftEndOffset);
        }
        if (bestRegularizer + threeSigmaInterval <= rightEndpoint) {
            interval(MAX_REGULARIZER_INDEX) =
                std::min(threeSigmaInterval, returnedIntervalRightEndOffset);
        }
    }
    interval += TVector{bestRegularizer};

    return TOptionalVector{interval};
}

CBoostedTreeFactory CBoostedTreeFactory::constructFromParameters(std::size_t numberThreads) {
    return {numberThreads};
}

CBoostedTreeFactory CBoostedTreeFactory::constructFromString(std::istream& jsonStringStream) {
    CBoostedTreeFactory result{1};
    try {
        core::CJsonStateRestoreTraverser traverser(jsonStringStream);
        if (result.m_TreeImpl->acceptRestoreTraverser(traverser) == false ||
            traverser.haveBadState()) {
            throw std::runtime_error{"failed to restore boosted tree"};
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string{"Input error: '"} + e.what() + "'"};
    }
    return result;
}

CBoostedTreeFactory::CBoostedTreeFactory(std::size_t numberThreads)
    : m_NumberThreads{numberThreads},
      m_TreeImpl{std::make_unique<CBoostedTreeImpl>(numberThreads, nullptr)},
      m_LogDepthPenaltyMultiplierSearchInterval{0.0}, m_LogTreeSizePenaltyMultiplierSearchInterval{0.0},
      m_LogLeafWeightPenaltyMultiplierSearchInterval{0.0}, m_TopShapValues{0} {
}

CBoostedTreeFactory::CBoostedTreeFactory(CBoostedTreeFactory&&) = default;

CBoostedTreeFactory& CBoostedTreeFactory::operator=(CBoostedTreeFactory&&) = default;

CBoostedTreeFactory::~CBoostedTreeFactory() = default;

CBoostedTreeFactory& CBoostedTreeFactory::minimumFrequencyToOneHotEncode(double frequency) {
    if (frequency >= 1.0) {
        LOG_WARN(<< "Frequency to one-hot encode must be less than one");
        frequency = 1.0 - std::numeric_limits<double>::epsilon();
    }
    m_MinimumFrequencyToOneHotEncode = frequency;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::numberFolds(std::size_t numberFolds) {
    if (numberFolds < 2) {
        LOG_WARN(<< "Must use at least two-folds for cross validation");
        numberFolds = 2;
    }
    m_TreeImpl->m_NumberFolds = numberFolds;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::stratifyRegressionCrossValidation(bool stratify) {
    m_StratifyRegressionCrossValidation = stratify;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::initialDownsampleRowsPerFeature(double rowsPerFeature) {
    m_InitialDownsampleRowsPerFeature = rowsPerFeature;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::depthPenaltyMultiplier(double depthPenaltyMultiplier) {
    if (depthPenaltyMultiplier < 0.0) {
        LOG_WARN(<< "Depth penalty multiplier must be non-negative");
        depthPenaltyMultiplier = 0.0;
    }
    m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier(depthPenaltyMultiplier);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::treeSizePenaltyMultiplier(double treeSizePenaltyMultiplier) {
    if (treeSizePenaltyMultiplier < 0.0) {
        LOG_WARN(<< "Tree size penalty multiplier must be non-negative");
        treeSizePenaltyMultiplier = 0.0;
    }
    m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier(treeSizePenaltyMultiplier);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::leafWeightPenaltyMultiplier(double leafWeightPenaltyMultiplier) {
    if (leafWeightPenaltyMultiplier < 0.0) {
        LOG_WARN(<< "Leaf weight penalty multiplier must be non-negative");
        leafWeightPenaltyMultiplier = 0.0;
    }
    m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier(leafWeightPenaltyMultiplier);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::softTreeDepthLimit(double softTreeDepthLimit) {
    if (softTreeDepthLimit < MIN_SOFT_DEPTH_LIMIT) {
        LOG_WARN(<< "Minimum tree depth must be at least two");
        softTreeDepthLimit = MIN_SOFT_DEPTH_LIMIT;
    }
    m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit(softTreeDepthLimit);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::softTreeDepthTolerance(double softTreeDepthTolerance) {
    if (softTreeDepthTolerance < 0.01) {
        LOG_WARN(<< "Minimum tree depth tolerance must be at least 0.01");
        softTreeDepthTolerance = 0.01;
    }
    m_TreeImpl->m_RegularizationOverride.softTreeDepthTolerance(softTreeDepthTolerance);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::eta(double eta) {
    if (eta < MIN_ETA) {
        LOG_WARN(<< "Truncating supplied learning rate " << eta
                 << " which must be no smaller than " << MIN_ETA);
        eta = std::max(eta, MIN_ETA);
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
    if (maximumNumberTrees > MAX_NUMBER_TREES) {
        LOG_WARN(<< "Truncating supplied maximum number of trees " << maximumNumberTrees
                 << " which must be no larger than " << MAX_NUMBER_TREES);
        maximumNumberTrees = std::min(maximumNumberTrees, MAX_NUMBER_TREES);
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

CBoostedTreeFactory& CBoostedTreeFactory::bayesianOptimisationRestarts(std::size_t restarts) {
    m_BayesianOptimisationRestarts = std::max(restarts, std::size_t{1});
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

CBoostedTreeFactory& CBoostedTreeFactory::balanceClassTrainingLoss(bool balance) {
    m_BalanceClassTrainingLoss = balance;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::progressCallback(TProgressCallback callback) {
    m_RecordProgress = std::move(callback);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::memoryUsageCallback(TMemoryUsageCallback callback) {
    m_RecordMemoryUsage = std::move(callback);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::trainingStateCallback(TTrainingStateCallback callback) {
    m_RecordTrainingState = std::move(callback);
    return *this;
}

std::size_t CBoostedTreeFactory::estimateMemoryUsage(std::size_t numberRows,
                                                     std::size_t numberColumns) const {
    std::size_t shapValuesExtraColumns =
        (m_TopShapValues > 0) ? numberRows * numberColumns * sizeof(CFloatStorage) : 0;
    std::size_t maximumNumberTrees{this->mainLoopMaximumNumberTrees()};
    std::swap(maximumNumberTrees, m_TreeImpl->m_MaximumNumberTrees);
    std::size_t result{m_TreeImpl->estimateMemoryUsage(numberRows, numberColumns) +
                       shapValuesExtraColumns};
    std::swap(maximumNumberTrees, m_TreeImpl->m_MaximumNumberTrees);
    return result;
}

std::size_t CBoostedTreeFactory::numberExtraColumnsForTrain() const {
    return CBoostedTreeImpl::numberExtraColumnsForTrain();
}

void CBoostedTreeFactory::initializeTrainingProgressMonitoring() {

    // The base unit is the cost of training on one tree.
    //
    // This comprises:
    //  - The cost of category encoding and feature selection which we count as
    //    one unit,
    //  - One unit for estimating the expected gain and sum curvature per node,
    //  - LINE_SEARCH_ITERATIONS * "maximum number trees" units per regularization
    //    parameter which isn't user defined,
    //  - LINE_SEARCH_ITERATIONS * "maximum number trees" per forest for training
    //    the downsampling factor if it isn't user defined,
    //  - The main optimisation loop which costs number folds * maximum number
    //    trees per forest units per iteration,
    //  - The cost of the final train which we count as an extra loop.

    std::size_t totalNumberSteps{2};
    std::size_t lineSearchMaximumNumberTrees{
        computeMaximumNumberTrees(m_TreeImpl->m_Eta)};
    if (m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit() == boost::none) {
        totalNumberSteps += LINE_SEARCH_ITERATIONS * lineSearchMaximumNumberTrees;
    }
    if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        totalNumberSteps += LINE_SEARCH_ITERATIONS * lineSearchMaximumNumberTrees;
    }
    if (m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        totalNumberSteps += LINE_SEARCH_ITERATIONS * lineSearchMaximumNumberTrees;
    }
    if (m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        totalNumberSteps += LINE_SEARCH_ITERATIONS * lineSearchMaximumNumberTrees;
    }
    if (m_TreeImpl->m_DownsampleFactorOverride == boost::none) {
        totalNumberSteps += LINE_SEARCH_ITERATIONS * lineSearchMaximumNumberTrees;
    }
    totalNumberSteps += (this->numberHyperparameterTuningRounds() + 1) *
                        this->mainLoopMaximumNumberTrees() * m_TreeImpl->m_NumberFolds;
    LOG_TRACE(<< "total number steps = " << totalNumberSteps);
    m_TreeImpl->m_TrainingProgress =
        core::CLoopProgress{totalNumberSteps, m_RecordProgress, 1.0, 1024};
}

void CBoostedTreeFactory::resumeRestoredTrainingProgressMonitoring() {
    m_TreeImpl->m_TrainingProgress.progressCallback(m_RecordProgress);
    m_TreeImpl->m_TrainingProgress.resumeRestored();
}

std::size_t CBoostedTreeFactory::mainLoopMaximumNumberTrees() const {
    if (m_TreeImpl->m_MaximumNumberTreesOverride == boost::none) {
        std::size_t maximumNumberTrees{computeMaximumNumberTrees(m_TreeImpl->m_Eta)};
        maximumNumberTrees = scaleMaximumNumberTrees(maximumNumberTrees);
        return maximumNumberTrees;
    }
    return *m_TreeImpl->m_MaximumNumberTreesOverride;
}

void CBoostedTreeFactory::noopRecordTrainingState(std::function<void(core::CStatePersistInserter&)>) {
}

void CBoostedTreeFactory::noopRecordProgress(double) {
}

void CBoostedTreeFactory::noopRecordMemoryUsage(std::int64_t) {
}

CBoostedTreeFactory& CBoostedTreeFactory::topShapValues(std::size_t topShapValues) {
    m_TopShapValues = topShapValues;
    m_TreeImpl->m_TopShapValues = topShapValues;
    return *this;
}
}
}
