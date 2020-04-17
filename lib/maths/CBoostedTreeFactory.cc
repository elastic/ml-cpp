/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeFactory.h>

#include <core/CJsonStateRestoreTraverser.h>

#include <maths/CBayesianOptimisation.h>
#include <maths/CBoostedTreeImpl.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CSampling.h>

#include <boost/optional/optional_io.hpp>

#include <cmath>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;
using TDoubleVec = std::vector<double>;
using TRowItr = core::CDataFrame::TRowItr;

namespace {
const std::size_t MIN_REGULARIZER_INDEX{0};
const std::size_t BEST_REGULARIZER_INDEX{1};
const std::size_t MAX_REGULARIZER_INDEX{2};
const std::size_t MAX_LINE_SEARCH_ITERATIONS{10};
const double LINE_SEARCH_MINIMUM_RELATIVE_EI_TO_CONTINUE{0.01};
const double MIN_SOFT_DEPTH_LIMIT{2.0};
const double MIN_SOFT_DEPTH_LIMIT_TOLERANCE{0.05};
const double MAX_SOFT_DEPTH_LIMIT_TOLERANCE{0.25};
const double MIN_ETA{1e-3};
const double MIN_ETA_SCALE{0.5};
const double MAX_ETA_SCALE{2.0};
const double MIN_ETA_GROWTH_RATE_SCALE{0.5};
const double MAX_ETA_GROWTH_RATE_SCALE{1.5};
const double MIN_FEATURE_BAG_FRACTION{0.2};
const double MAX_FEATURE_BAG_FRACTION{0.8};
const double MIN_DOWNSAMPLE_LINE_SEARCH_RANGE{2.0};
const double MAX_DOWNSAMPLE_LINE_SEARCH_RANGE{144.0};
const double MIN_DOWNSAMPLE_FACTOR{1e-3};
const double MIN_DOWNSAMPLE_FACTOR_SCALE{0.3};
const double MAX_DOWNSAMPLE_FACTOR_SCALE{3.0};
// This isn't a hard limit but we increase the number of default training folds
// if the initial downsample fraction would be larger than this.
const double MAX_DESIRED_INITIAL_DOWNSAMPLE_FRACTION{0.5};
const double MAX_NUMBER_FOLDS{5.0};
const std::size_t MAX_NUMBER_TREES{static_cast<std::size_t>(2.0 / MIN_ETA + 0.5)};
// We scale eta in the upfront calculation of the total number of steps we expect
// for progress monitoring because we don't know what value we'll choose in the
// line search. Assuming it is less than one avoids a large pause in progress if
// it is reduced in the line search.
const double MAIN_LOOP_ETA_SCALE_FOR_PROGRESS{0.5};

double computeEta(std::size_t numberRegressors) {
    // eta is the learning rate. There is a lot of empirical evidence that
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
    return 1.0 / std::max(10.0, std::sqrt(static_cast<double>(numberRegressors)));
}

std::size_t computeMaximumNumberTrees(double eta) {
    return static_cast<std::size_t>(3.0 / eta / MIN_DOWNSAMPLE_FACTOR_SCALE + 0.5);
}

bool intervalIsEmpty(const CBoostedTreeFactory::TVector& interval) {
    return interval(MAX_REGULARIZER_INDEX) - interval(MIN_REGULARIZER_INDEX) == 0.0;
}
}

CBoostedTreeFactory::TBoostedTreeUPtr
CBoostedTreeFactory::buildFor(core::CDataFrame& frame, std::size_t dependentVariable) {

    m_TreeImpl->m_DependentVariable = dependentVariable;

    this->initializeNumberFolds(frame);
    this->initializeTrainingProgressMonitoring(frame);
    this->initializeMissingFeatureMasks(frame);

    this->resizeDataFrame(frame);

    this->initializeCrossValidation(frame);
    this->selectFeaturesAndEncodeCategories(frame);
    this->determineFeatureDataTypes(frame);
    m_TreeImpl->m_Instrumentation->updateMemoryUsage(core::CMemory::dynamicSize(m_TreeImpl));
    m_TreeImpl->m_Instrumentation->lossType(m_TreeImpl->m_Loss->name());

    if (this->initializeFeatureSampleDistribution()) {
        this->initializeHyperparameters(frame);
        this->initializeHyperparameterOptimisation();
    }

    auto treeImpl = std::make_unique<CBoostedTreeImpl>(m_NumberThreads,
                                                       m_TreeImpl->m_Loss->clone());
    std::swap(m_TreeImpl, treeImpl);
    return TBoostedTreeUPtr{
        new CBoostedTree{frame, m_RecordTrainingState, std::move(treeImpl)}};
}

CBoostedTreeFactory::TBoostedTreeUPtr
CBoostedTreeFactory::restoreFor(core::CDataFrame& frame, std::size_t dependentVariable) {

    if (dependentVariable != m_TreeImpl->m_DependentVariable) {
        HANDLE_FATAL(<< "Internal error: expected dependent variable "
                     << m_TreeImpl->m_DependentVariable << " got " << dependentVariable);
        return nullptr;
    }

    this->resumeRestoredTrainingProgressMonitoring();
    this->resizeDataFrame(frame);
    m_TreeImpl->m_Instrumentation->updateMemoryUsage(core::CMemory::dynamicSize(m_TreeImpl));
    m_TreeImpl->m_Instrumentation->lossType(m_TreeImpl->m_Loss->name());

    return TBoostedTreeUPtr{
        new CBoostedTree{frame, m_RecordTrainingState, std::move(m_TreeImpl)}};
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
        boundingBox.emplace_back(m_LogEtaSearchInterval(MIN_REGULARIZER_INDEX),
                                 m_LogEtaSearchInterval(MAX_REGULARIZER_INDEX));
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
                    auto& mask = m_TreeImpl->m_MissingFeatureRowMasks[i];
                    mask.extend(false, row->index() - mask.size());
                    mask.extend(true);
                }
            }
        }
    });

    for (auto& mask : m_TreeImpl->m_MissingFeatureRowMasks) {
        mask.extend(false, frame.numberRows() - mask.size());
        LOG_TRACE(<< "# missing = " << mask.manhattan());
    }
}

void CBoostedTreeFactory::initializeNumberFolds(core::CDataFrame& frame) const {
    if (m_TreeImpl->m_NumberFoldsOverride == boost::none) {
        auto result = frame.readRows(
            m_NumberThreads,
            core::bindRetrievableState(
                [this](std::size_t& numberTrainingRows, TRowItr beginRows, TRowItr endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        double target{(*row)[m_TreeImpl->m_DependentVariable]};
                        if (CDataFrameUtils::isMissing(target) == false) {
                            ++numberTrainingRows;
                        }
                    }
                },
                std::size_t{0}));
        std::size_t totalNumberTrainingRows{0};
        for (const auto& numberTrainingRows : result.first) {
            totalNumberTrainingRows += numberTrainingRows.s_FunctionState;
        }
        LOG_TRACE(<< "total number training rows = " << totalNumberTrainingRows);

        // We want to choose the number of folds so we'll have enough training data
        // after leaving out one fold. We choose the initial downsample size based
        // on the same sort of criterion. So we require that leaving out one fold
        // shouldn't mean than we have fewer rows than constant * desired downsample
        // # rows if possible. We choose the constant to be two for no particularly
        // good reason except that:
        //   1. it isn't too large
        //   2. it still means we'll have plenty of variation between random bags.
        //
        // In order to estimate this we use the number of input features as a proxy
        // for the number of features we'll actually use after feature selection.
        //
        // So how does the following work: we'd like "c * f * # rows" training rows.
        // For k folds we'll have "(1 - 1 / k) * # rows" training rows. So we want
        // to find the smallest integer k s.t. c * f * # rows <= (1 - 1 / k) * # rows.
        // This gives k = ceil(1 / (1 - c * f)). However, we also upper bound this
        // by MAX_NUMBER_FOLDS.

        double initialDownsampleFraction{(m_InitialDownsampleRowsPerFeature *
                                          static_cast<double>(frame.numberColumns() - 1)) /
                                         static_cast<double>(totalNumberTrainingRows)};

        m_TreeImpl->m_NumberFolds = static_cast<std::size_t>(
            std::ceil(1.0 / std::max(1.0 - initialDownsampleFraction / MAX_DESIRED_INITIAL_DOWNSAMPLE_FRACTION,
                                     1.0 / MAX_NUMBER_FOLDS)));
        LOG_TRACE(<< "initial downsample fraction = " << initialDownsampleFraction
                  << " # folds = " << m_TreeImpl->m_NumberFolds);
    } else {
        m_TreeImpl->m_NumberFolds = *m_TreeImpl->m_NumberFoldsOverride;
    }
}

void CBoostedTreeFactory::resizeDataFrame(core::CDataFrame& frame) const {
    std::size_t numberLossParameters{m_TreeImpl->m_Loss->numberParameters()};
    m_TreeImpl->m_ExtraColumns = frame.resizeColumns(
        m_TreeImpl->m_NumberThreads, extraColumns(numberLossParameters));
    m_TreeImpl->m_Instrumentation->updateMemoryUsage(core::CMemory::dynamicSize(frame));
}

void CBoostedTreeFactory::initializeCrossValidation(core::CDataFrame& frame) const {

    core::CPackedBitVector allTrainingRowsMask{m_TreeImpl->allTrainingRowsMask()};
    std::size_t dependentVariable{m_TreeImpl->m_DependentVariable};

    std::size_t numberBuckets(m_StratifyRegressionCrossValidation ? 10 : 1);
    std::tie(m_TreeImpl->m_TrainingRowMasks, m_TreeImpl->m_TestingRowMasks, std::ignore) =
        CDataFrameUtils::stratifiedCrossValidationRowMasks(
            m_TreeImpl->m_NumberThreads, frame, dependentVariable, m_TreeImpl->m_Rng,
            m_TreeImpl->m_NumberFolds, numberBuckets, allTrainingRowsMask);

    frame.writeColumns(m_NumberThreads, 0, frame.numberRows(),
                       [&](TRowItr beginRows, TRowItr endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               writeExampleWeight(*row, m_TreeImpl->m_ExtraColumns, 1.0);
                           }
                       },
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
    m_TreeImpl->m_TrainingProgress.increment(100);
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
        m_TreeImpl->m_Eta =
            computeEta(frame.numberColumns() - this->numberExtraColumnsForTrain());
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
    double downsampleFactor{m_InitialDownsampleRowsPerFeature * numberFeatures /
                            m_TreeImpl->m_TrainingRowMasks[0].manhattan()};
    m_TreeImpl->m_DownsampleFactor = m_TreeImpl->m_DownsampleFactorOverride.value_or(
        CTools::truncate(downsampleFactor, 0.05, 0.5));

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
    this->initializeUnsetEta(frame);
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
            double minSoftDepthLimit{MIN_SOFT_DEPTH_LIMIT};
            double maxSoftDepthLimit{MIN_SOFT_DEPTH_LIMIT + log2MaxTreeSize};
            double meanSoftDepthLimit{(minSoftDepthLimit + maxSoftDepthLimit) / 2.0};
            double mainLoopSearchInterval{log2MaxTreeSize / 2.0};
            LOG_TRACE(<< "mean soft depth limit = " << meanSoftDepthLimit);

            auto applySoftDepthLimit = [](CBoostedTreeImpl& tree, double softDepthLimit) {
                tree.m_Regularization.softTreeDepthLimit(softDepthLimit);
                return true;
            };

            TVector fallback{{minSoftDepthLimit, meanSoftDepthLimit, maxSoftDepthLimit}};
            m_SoftDepthLimitSearchInterval =
                this->testLossLineSearch(frame, applySoftDepthLimit, minSoftDepthLimit,
                                         maxSoftDepthLimit, -mainLoopSearchInterval / 2.0,
                                         mainLoopSearchInterval / 2.0)
                    .value_or(fallback);
            m_SoftDepthLimitSearchInterval =
                max(m_SoftDepthLimitSearchInterval, TVector{1.0});
            LOG_TRACE(<< "soft depth limit search interval = ["
                      << m_SoftDepthLimitSearchInterval.toDelimited() << "]");
            m_TreeImpl->m_Regularization.softTreeDepthLimit(
                m_SoftDepthLimitSearchInterval(BEST_REGULARIZER_INDEX));
        }
        if (gainPerNode90thPercentile <= 0.0 ||
            intervalIsEmpty(m_SoftDepthLimitSearchInterval)) {
            m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit(
                m_TreeImpl->m_Regularization.softTreeDepthLimit());
        }
    }

    // Set the depth limit to its smallest value and search for the value of the
    // penalty multiplier at which the tree starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        if (gainPerNode90thPercentile > 0.0) {
            double searchIntervalSize{2.0 * gainPerNode90thPercentile / gainPerNode1stPercentile};
            double logMaxDepthPenaltyMultiplier{CTools::stableLog(gainPerNode90thPercentile)};
            double logMinDepthPenaltyMultiplier{logMaxDepthPenaltyMultiplier -
                                                CTools::stableLog(searchIntervalSize)};
            double meanLogDepthPenaltyMultiplier{
                (logMinDepthPenaltyMultiplier + logMaxDepthPenaltyMultiplier) / 2.0};
            double mainLoopSearchInterval{CTools::stableLog(searchIntervalSize) / 2.0};
            LOG_TRACE(<< "mean log depth penalty multiplier = " << meanLogDepthPenaltyMultiplier);

            auto applyDepthPenaltyMultiplier = [](CBoostedTreeImpl& tree, double logDepthPenalty) {
                tree.m_Regularization.depthPenaltyMultiplier(CTools::stableExp(logDepthPenalty));
                return true;
            };

            TVector fallback;
            fallback(MIN_REGULARIZER_INDEX) = logMinDepthPenaltyMultiplier;
            fallback(BEST_REGULARIZER_INDEX) = meanLogDepthPenaltyMultiplier;
            fallback(MAX_REGULARIZER_INDEX) = logMaxDepthPenaltyMultiplier;

            m_LogDepthPenaltyMultiplierSearchInterval =
                this->testLossLineSearch(frame, applyDepthPenaltyMultiplier,
                                         logMinDepthPenaltyMultiplier, logMaxDepthPenaltyMultiplier,
                                         -mainLoopSearchInterval / 2.0,
                                         mainLoopSearchInterval / 2.0)
                    .value_or(fallback);
            LOG_TRACE(<< "log depth penalty multiplier search interval = ["
                      << m_LogDepthPenaltyMultiplierSearchInterval.toDelimited() << "]");

            m_TreeImpl->m_Regularization.depthPenaltyMultiplier(CTools::stableExp(
                m_LogDepthPenaltyMultiplierSearchInterval(BEST_REGULARIZER_INDEX)));
        }
        if (gainPerNode90thPercentile <= 0.0 ||
            intervalIsEmpty(m_LogDepthPenaltyMultiplierSearchInterval)) {
            m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier(
                m_TreeImpl->m_Regularization.depthPenaltyMultiplier());
        }
    }

    // Search for the value of the tree size penalty multiplier at which the tree
    // starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        if (gainPerNode90thPercentile > 0.0) {
            double searchIntervalSize{2.0 * gainPerNode90thPercentile / gainPerNode1stPercentile};
            double logMaxTreeSizePenaltyMultiplier{CTools::stableLog(gainPerNode90thPercentile)};
            double logMinTreeSizePenaltyMultiplier{
                logMaxTreeSizePenaltyMultiplier - CTools::stableLog(searchIntervalSize)};
            double meanLogTreeSizePenaltyMultiplier{
                (logMinTreeSizePenaltyMultiplier + logMaxTreeSizePenaltyMultiplier) / 2.0};
            double mainLoopSearchInterval{0.5 * CTools::stableLog(searchIntervalSize)};
            LOG_TRACE(<< "mean log tree size penalty multiplier = "
                      << meanLogTreeSizePenaltyMultiplier);

            auto applyTreeSizePenaltyMultiplier = [](CBoostedTreeImpl& tree,
                                                     double logTreeSizePenalty) {
                tree.m_Regularization.treeSizePenaltyMultiplier(
                    CTools::stableExp(logTreeSizePenalty));
                return true;
            };

            TVector fallback;
            fallback(MIN_REGULARIZER_INDEX) = logMinTreeSizePenaltyMultiplier;
            fallback(BEST_REGULARIZER_INDEX) = meanLogTreeSizePenaltyMultiplier;
            fallback(MAX_REGULARIZER_INDEX) = logMaxTreeSizePenaltyMultiplier;

            m_LogTreeSizePenaltyMultiplierSearchInterval =
                this->testLossLineSearch(frame, applyTreeSizePenaltyMultiplier,
                                         logMinTreeSizePenaltyMultiplier,
                                         logMaxTreeSizePenaltyMultiplier,
                                         -mainLoopSearchInterval / 2.0,
                                         mainLoopSearchInterval / 2.0)
                    .value_or(fallback);
            LOG_TRACE(<< "log tree size penalty multiplier search interval = ["
                      << m_LogTreeSizePenaltyMultiplierSearchInterval.toDelimited() << "]");

            m_TreeImpl->m_Regularization.treeSizePenaltyMultiplier(CTools::stableExp(
                m_LogTreeSizePenaltyMultiplierSearchInterval(BEST_REGULARIZER_INDEX)));
        }
        if (gainPerNode90thPercentile <= 0.0 ||
            intervalIsEmpty(m_LogTreeSizePenaltyMultiplierSearchInterval)) {
            m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier(
                m_TreeImpl->m_Regularization.treeSizePenaltyMultiplier());
        }
    }

    // Search for the value of the leaf weight penalty multiplier at which the
    // tree starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        if (totalCurvaturePerNode90thPercentile > 0.0) {
            double searchIntervalSize{2.0 * totalCurvaturePerNode90thPercentile /
                                      totalCurvaturePerNode1stPercentile};
            double logMaxLeafWeightPenaltyMultiplier{
                CTools::stableLog(totalCurvaturePerNode90thPercentile)};
            double logMinLeafWeightPenaltyMultiplier{
                logMaxLeafWeightPenaltyMultiplier - CTools::stableLog(searchIntervalSize)};
            double meanLogLeafWeightPenaltyMultiplier{
                (logMinLeafWeightPenaltyMultiplier + logMaxLeafWeightPenaltyMultiplier) / 2.0};
            double mainLoopSearchInterval{0.5 * CTools::stableLog(searchIntervalSize)};
            LOG_TRACE(<< "mean log leaf weight penalty multiplier = "
                      << meanLogLeafWeightPenaltyMultiplier);

            auto applyLeafWeightPenaltyMultiplier = [](CBoostedTreeImpl& tree,
                                                       double logLeafWeightPenalty) {
                tree.m_Regularization.leafWeightPenaltyMultiplier(
                    CTools::stableExp(logLeafWeightPenalty));
                return true;
            };

            TVector fallback;
            fallback(MIN_REGULARIZER_INDEX) = logMinLeafWeightPenaltyMultiplier;
            fallback(BEST_REGULARIZER_INDEX) = meanLogLeafWeightPenaltyMultiplier;
            fallback(MAX_REGULARIZER_INDEX) = logMaxLeafWeightPenaltyMultiplier;

            m_LogLeafWeightPenaltyMultiplierSearchInterval =
                this->testLossLineSearch(frame, applyLeafWeightPenaltyMultiplier,
                                         logMinLeafWeightPenaltyMultiplier,
                                         logMaxLeafWeightPenaltyMultiplier,
                                         -mainLoopSearchInterval / 2.0,
                                         mainLoopSearchInterval / 2.0)
                    .value_or(fallback);
            LOG_TRACE(<< "log leaf weight penalty multiplier search interval = ["
                      << m_LogLeafWeightPenaltyMultiplierSearchInterval.toDelimited()
                      << "]");
            m_TreeImpl->m_Regularization.leafWeightPenaltyMultiplier(CTools::stableExp(
                m_LogLeafWeightPenaltyMultiplierSearchInterval(BEST_REGULARIZER_INDEX)));
        }
        if (totalCurvaturePerNode90thPercentile <= 0.0 ||
            intervalIsEmpty(m_LogLeafWeightPenaltyMultiplierSearchInterval)) {
            m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier(
                m_TreeImpl->m_Regularization.leafWeightPenaltyMultiplier());
        }
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
        double searchIntervalSize{CTools::truncate(
            m_TreeImpl->m_TrainingRowMasks[0].manhattan() / 100.0,
            MIN_DOWNSAMPLE_LINE_SEARCH_RANGE, MAX_DOWNSAMPLE_LINE_SEARCH_RANGE)};
        double logMaxDownsampleFactor{CTools::stableLog(std::min(
            std::sqrt(searchIntervalSize) * m_TreeImpl->m_DownsampleFactor, 1.0))};
        double logMinDownsampleFactor{logMaxDownsampleFactor -
                                      CTools::stableLog(searchIntervalSize)};
        double meanLogDownSampleFactor{(logMinDownsampleFactor + logMaxDownsampleFactor) / 2.0};
        LOG_TRACE(<< "mean log down sample factor = " << meanLogDownSampleFactor);

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

        double numberTrainingRows{m_TreeImpl->m_TrainingRowMasks[0].manhattan()};

        auto applyDownsampleFactor = [&](CBoostedTreeImpl& tree, double logDownsampleFactor) {
            double downsampleFactor{CTools::stableExp(logDownsampleFactor)};
            tree.m_DownsampleFactor = downsampleFactor;
            scaleRegularizers(tree, downsampleFactor);
            return tree.m_DownsampleFactor * numberTrainingRows > 10.0;
        };

        TVector fallback{{logMinDownsampleFactor, meanLogDownSampleFactor, logMaxDownsampleFactor}};
        m_LogDownsampleFactorSearchInterval =
            this->testLossLineSearch(frame, applyDownsampleFactor,
                                     logMinDownsampleFactor, logMaxDownsampleFactor,
                                     CTools::stableLog(MIN_DOWNSAMPLE_FACTOR_SCALE),
                                     CTools::stableLog(MAX_DOWNSAMPLE_FACTOR_SCALE))
                .value_or(fallback);

        // Truncate the log(scale) to be less than or equal to log(1.0) and the down
        // sampled set contains at least ten examples on average.
        m_LogDownsampleFactorSearchInterval =
            min(max(m_LogDownsampleFactorSearchInterval,
                    TVector{CTools::stableLog(10.0 / numberTrainingRows)}),
                TVector{0.0});
        LOG_TRACE(<< "log down sample factor search interval = ["
                  << m_LogDownsampleFactorSearchInterval.toDelimited() << "]");

        m_TreeImpl->m_DownsampleFactor = CTools::stableExp(
            m_LogDownsampleFactorSearchInterval(BEST_REGULARIZER_INDEX));

        TVector logScale{CTools::stableLog(
            scaleRegularizers(*m_TreeImpl, m_TreeImpl->m_DownsampleFactor))};
        m_LogTreeSizePenaltyMultiplierSearchInterval += logScale;
        m_LogLeafWeightPenaltyMultiplierSearchInterval += logScale;

        if (intervalIsEmpty(m_LogDownsampleFactorSearchInterval)) {
            m_TreeImpl->m_DownsampleFactorOverride = m_TreeImpl->m_DownsampleFactor;
        }
    }
}

void CBoostedTreeFactory::initializeUnsetEta(core::CDataFrame& frame) {

    if (m_TreeImpl->m_EtaOverride == boost::none) {
        double searchIntervalSize{5.0 * MAX_ETA_SCALE / MIN_ETA_SCALE};
        double logMaxEta{
            CTools::stableLog(std::sqrt(searchIntervalSize) * m_TreeImpl->m_Eta)};
        double logMinEta{logMaxEta - CTools::stableLog(searchIntervalSize)};
        double meanLogEta{(logMaxEta + logMinEta) / 2.0};
        double mainLoopSearchInterval{CTools::stableLog(0.2 * searchIntervalSize)};
        LOG_TRACE(<< "mean log eta = " << meanLogEta);

        auto applyEta = [](CBoostedTreeImpl& tree, double eta) {
            tree.m_Eta = CTools::stableExp(eta);
            tree.m_EtaGrowthRatePerTree = 1.0 + tree.m_Eta / 2.0;
            tree.m_MaximumNumberTrees = computeMaximumNumberTrees(tree.m_Eta);
            return true;
        };

        double eta{m_TreeImpl->m_Eta};

        TVector fallback;
        fallback(MIN_REGULARIZER_INDEX) = logMinEta;
        fallback(BEST_REGULARIZER_INDEX) = meanLogEta;
        fallback(MAX_REGULARIZER_INDEX) = logMaxEta;

        m_LogEtaSearchInterval =
            this->testLossLineSearch(frame, applyEta, logMinEta, logMaxEta,
                                     -mainLoopSearchInterval / 2.0,
                                     mainLoopSearchInterval / 2.0)
                .value_or(fallback);
        m_LogEtaSearchInterval = min(m_LogEtaSearchInterval, TVector{0.0});
        LOG_TRACE(<< "log eta search interval = ["
                  << m_LogEtaSearchInterval.toDelimited() << "]");
        applyEta(*m_TreeImpl, m_LogEtaSearchInterval(BEST_REGULARIZER_INDEX));

        if (intervalIsEmpty(m_LogEtaSearchInterval)) {
            m_TreeImpl->m_EtaOverride = m_TreeImpl->m_Eta;
        } else {
            m_TreeImpl->m_MaximumNumberTrees =
                computeMaximumNumberTrees(MIN_ETA_SCALE * m_TreeImpl->m_Eta);
        }

        m_TreeImpl->m_TrainingProgress.incrementRange(
            static_cast<int>(this->mainLoopNumberSteps(m_TreeImpl->m_Eta)) -
            static_cast<int>(this->mainLoopNumberSteps(MAIN_LOOP_ETA_SCALE_FOR_PROGRESS * eta)));
    }
}

CBoostedTreeFactory::TDoubleDoublePrVec
CBoostedTreeFactory::estimateTreeGainAndCurvature(core::CDataFrame& frame,
                                                  const TDoubleVec& percentiles) const {

    std::size_t maximumNumberOfTrees{1};
    std::swap(maximumNumberOfTrees, m_TreeImpl->m_MaximumNumberTrees);
    CBoostedTreeImpl::TNodeVecVec forest;
    std::tie(forest, std::ignore, std::ignore) = m_TreeImpl->trainForest(
        frame, m_TreeImpl->m_TrainingRowMasks[0],
        m_TreeImpl->m_TestingRowMasks[0], m_TreeImpl->m_TrainingProgress);
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
                                        const TApplyRegularizer& applyRegularizer,
                                        double intervalLeftEnd,
                                        double intervalRightEnd,
                                        double returnedIntervalLeftEndOffset,
                                        double returnedIntervalRightEndOffset) const {

    // This has the following steps:
    //   1. Coarse search the interval [intervalLeftEnd, intervalRightEnd] using
    //      fixed steps,
    //   2. Fine tune, via Bayesian Optimisation targeting expected improvement,
    //      and stop if the expected improvement small compared to the current
    //      minimum test loss,
    //   3. Calculate the parameter interval which gives the lowest test losses,
    //   4. Fit an OLS quadratic approximation to the test losses in the interval
    //      from step 3 and use it to estimate the best parameter value,
    //   5. Compare the size of the residual errors w.r.t. to the OLS curve from
    //      step 4 with its variation over the interval from step 3 and truncate
    //      the returned interval if we can determine there is a low chance of
    //      missing the best solution by doing so.

    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;

    auto boptVector = [](double regularizer) {
        return SConstant<CBayesianOptimisation::TVector>::get(1, regularizer);
    };

    TMinAccumulator minTestLoss;
    TDoubleDoublePrVec testLosses;
    testLosses.reserve(MAX_LINE_SEARCH_ITERATIONS);
    // Ensure we choose one value based on expected improvement.
    std::size_t minNumberTestLosses{5};

    CBayesianOptimisation bopt{{{intervalLeftEnd, intervalRightEnd}}};
    for (auto regularizer :
         {intervalLeftEnd, (2.0 * intervalLeftEnd + intervalRightEnd) / 3.0,
          (intervalLeftEnd + 2.0 * intervalRightEnd) / 3.0, intervalRightEnd}) {
        if (applyRegularizer(*m_TreeImpl, regularizer) == false) {
            m_TreeImpl->m_TrainingProgress.increment(
                (MAX_LINE_SEARCH_ITERATIONS - testLosses.size()) * m_TreeImpl->m_MaximumNumberTrees);
            break;
        }

        CBoostedTreeImpl::TNodeVecVec forest;
        double testLoss;
        std::tie(forest, testLoss, std::ignore) = m_TreeImpl->trainForest(
            frame, m_TreeImpl->m_TrainingRowMasks[0],
            m_TreeImpl->m_TestingRowMasks[0], m_TreeImpl->m_TrainingProgress);
        bopt.add(boptVector(regularizer), testLoss, 0.0);
        minTestLoss.add(testLoss);
        testLosses.emplace_back(regularizer, testLoss);
    }
    while (testLosses.size() > 0 && testLosses.size() < MAX_LINE_SEARCH_ITERATIONS) {
        CBayesianOptimisation::TVector regularizer;
        TOptionalDouble EI;
        std::tie(regularizer, EI) = bopt.maximumExpectedImprovement();
        double threshold{LINE_SEARCH_MINIMUM_RELATIVE_EI_TO_CONTINUE * minTestLoss[0]};
        LOG_TRACE(<< "EI = " << EI << " threshold to continue = " << threshold);
        if ((testLosses.size() >= minNumberTestLosses && EI != boost::none && *EI < threshold) ||
            applyRegularizer(*m_TreeImpl, regularizer(0)) == false) {
            m_TreeImpl->m_TrainingProgress.increment(
                (MAX_LINE_SEARCH_ITERATIONS - testLosses.size()) * m_TreeImpl->m_MaximumNumberTrees);
            break;
        }
        CBoostedTreeImpl::TNodeVecVec forest;
        double testLoss;
        std::tie(forest, testLoss, std::ignore) = m_TreeImpl->trainForest(
            frame, m_TreeImpl->m_TrainingRowMasks[0],
            m_TreeImpl->m_TestingRowMasks[0], m_TreeImpl->m_TrainingProgress);
        bopt.add(regularizer, testLoss, 0.0);
        minTestLoss.add(testLoss);
        testLosses.emplace_back(regularizer(0), testLoss);
    }

    std::sort(testLosses.begin(), testLosses.end());
    LOG_TRACE(<< "test losses = " << core::CContainerPrinter::print(testLosses));

    if (testLosses.empty()) {
        return TOptionalVector{};
    }

    // Find the smallest test losses and the corresponding regularizer interval.
    auto minimumTestLosses = CBasicStatistics::orderStatisticsAccumulator<TDoubleDoublePr>(
        minNumberTestLosses - 1, COrderings::SSecondLess{});
    minimumTestLosses.add(testLosses);
    double minGoodRegularizer{std::min_element(minimumTestLosses.begin(),
                                               minimumTestLosses.end(),
                                               COrderings::SFirstLess{})
                                  ->first};
    double maxGoodRegularizer{std::max_element(minimumTestLosses.begin(),
                                               minimumTestLosses.end(),
                                               COrderings::SFirstLess{})
                                  ->first};
    auto beginGoodRegularizerLosses =
        std::find_if(testLosses.begin(), testLosses.end(),
                     [minGoodRegularizer](const TDoubleDoublePr& loss) {
                         return loss.first == minGoodRegularizer;
                     });
    auto endGoodRegularizerLosses =
        std::find_if(testLosses.begin(), testLosses.end(),
                     [maxGoodRegularizer](const TDoubleDoublePr& loss) {
                         return loss.first == maxGoodRegularizer;
                     }) +
        1;
    LOG_TRACE(<< "good regularizer range = [" << minGoodRegularizer << ","
              << maxGoodRegularizer << "]");

    CLeastSquaresOnlineRegression<2, double> leastSquaresQuadraticTestLoss;
    for (auto loss = beginGoodRegularizerLosses; loss != endGoodRegularizerLosses; ++loss) {
        leastSquaresQuadraticTestLoss.add(loss->first, loss->second);
    }
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
    double stationaryPoint{-(gradient == curvature ? 0.5 : gradient / 2.0 / curvature)};
    double bestRegularizer{[&] {
        if (curvature < 0.0) {
            // Stationary point is a maximum so use furthest point in interval.
            double distanceToLeftEndpoint{std::fabs(minGoodRegularizer - stationaryPoint)};
            double distanceToRightEndpoint{std::fabs(maxGoodRegularizer - stationaryPoint)};
            return distanceToLeftEndpoint > distanceToRightEndpoint
                       ? minGoodRegularizer
                       : maxGoodRegularizer;
        }
        // Stationary point is a minimum so use nearest point in the interval.
        return CTools::truncate(stationaryPoint, minGoodRegularizer, maxGoodRegularizer);
    }()};
    LOG_TRACE(<< "best regularizer = " << bestRegularizer);

    TVector interval{{returnedIntervalLeftEndOffset, 0.0, returnedIntervalRightEndOffset}};
    if (minGoodRegularizer > intervalLeftEnd) {
        interval(MIN_REGULARIZER_INDEX) = std::max(
            minGoodRegularizer - bestRegularizer, interval(MIN_REGULARIZER_INDEX));
    }
    if (maxGoodRegularizer < intervalRightEnd) {
        interval(MAX_REGULARIZER_INDEX) = std::min(
            maxGoodRegularizer - bestRegularizer, interval(MAX_REGULARIZER_INDEX));
    }
    if (curvature > 0.0) {
        // Find a short interval with a high probability of containing the optimal
        // regularisation parameter if we found a minimum. In particular, we solve
        // curvature * (x - best)^2 = 3 sigma where sigma is the standard deviation
        // of the test loss residuals to get the interval endpoints. We don't
        // extrapolate the loss function outside the line segment we searched so
        // don't truncate if an endpoint lies outside the searched interval.
        TMeanVarAccumulator residualMoments;
        for (auto loss = beginGoodRegularizerLosses;
             loss != endGoodRegularizerLosses; ++loss) {
            residualMoments.add(loss->second -
                                leastSquaresQuadraticTestLoss.predict(loss->first));
        }
        double sigma{std::sqrt(CBasicStatistics::variance(residualMoments))};
        double threeSigmaInterval{std::sqrt(3.0 * sigma / curvature)};
        if (bestRegularizer - threeSigmaInterval >= minGoodRegularizer) {
            interval(MIN_REGULARIZER_INDEX) =
                std::max(-threeSigmaInterval, returnedIntervalLeftEndOffset);
        }
        if (bestRegularizer + threeSigmaInterval <= maxGoodRegularizer) {
            interval(MAX_REGULARIZER_INDEX) =
                std::min(threeSigmaInterval, returnedIntervalRightEndOffset);
        }
    }
    interval += TVector{bestRegularizer};

    return TOptionalVector{interval};
}

CBoostedTreeFactory CBoostedTreeFactory::constructFromParameters(std::size_t numberThreads,
                                                                 TLossFunctionUPtr loss) {
    return {numberThreads, std::move(loss)};
}

CBoostedTreeFactory CBoostedTreeFactory::constructFromString(std::istream& jsonStream) {
    CBoostedTreeFactory result{1, nullptr};
    try {
        core::CJsonStateRestoreTraverser traverser(jsonStream);
        if (result.m_TreeImpl->acceptRestoreTraverser(traverser) == false ||
            traverser.haveBadState()) {
            throw std::runtime_error{"failed to restore boosted tree"};
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string{"Input error: '"} + e.what() + "'"};
    }
    return result;
}

CBoostedTreeFactory::CBoostedTreeFactory(std::size_t numberThreads, TLossFunctionUPtr loss)
    : m_NumberThreads{numberThreads},
      m_TreeImpl{std::make_unique<CBoostedTreeImpl>(numberThreads, std::move(loss))},
      m_LogDepthPenaltyMultiplierSearchInterval{0.0}, m_LogTreeSizePenaltyMultiplierSearchInterval{0.0},
      m_LogLeafWeightPenaltyMultiplierSearchInterval{0.0} {
}

CBoostedTreeFactory::CBoostedTreeFactory(CBoostedTreeFactory&&) = default;

CBoostedTreeFactory& CBoostedTreeFactory::operator=(CBoostedTreeFactory&&) = default;

CBoostedTreeFactory::~CBoostedTreeFactory() = default;

CBoostedTreeFactory&
CBoostedTreeFactory::classAssignmentObjective(CBoostedTree::EClassAssignmentObjective objective) {
    m_TreeImpl->m_ClassAssignmentObjective = objective;
    return *this;
}

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
    m_TreeImpl->m_NumberFoldsOverride = numberFolds;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::stratifyRegressionCrossValidation(bool stratify) {
    m_StratifyRegressionCrossValidation = stratify;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::stopCrossValidationEarly(bool stopEarly) {
    m_TreeImpl->m_StopCrossValidationEarly = stopEarly;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::initialDownsampleRowsPerFeature(double rowsPerFeature) {
    m_InitialDownsampleRowsPerFeature = rowsPerFeature;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::downsampleFactor(double factor) {
    if (factor <= MIN_DOWNSAMPLE_FACTOR) {
        LOG_WARN(<< "Downsample factor must be non-negative");
        factor = MIN_DOWNSAMPLE_FACTOR;
    } else if (factor > 1.0) {
        LOG_WARN(<< "Downsample factor must be no larger than one");
        factor = 1.0;
    }
    m_TreeImpl->m_DownsampleFactorOverride = factor;
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

CBoostedTreeFactory& CBoostedTreeFactory::numberTopShapValues(std::size_t numberTopShapValues) {
    m_TreeImpl->m_NumberTopShapValues = numberTopShapValues;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::analysisInstrumentation(
    CDataFrameTrainBoostedTreeInstrumentationInterface& instrumentation) {
    m_TreeImpl->m_Instrumentation = &instrumentation;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::trainingStateCallback(TTrainingStateCallback callback) {
    m_RecordTrainingState = std::move(callback);
    return *this;
}

std::size_t CBoostedTreeFactory::estimateMemoryUsage(std::size_t numberRows,
                                                     std::size_t numberColumns) const {
    std::size_t maximumNumberTrees{this->mainLoopMaximumNumberTrees(
        m_TreeImpl->m_EtaOverride != boost::none ? *m_TreeImpl->m_EtaOverride
                                                 : computeEta(numberColumns))};
    std::swap(maximumNumberTrees, m_TreeImpl->m_MaximumNumberTrees);
    std::size_t result{m_TreeImpl->estimateMemoryUsage(numberRows, numberColumns)};
    std::swap(maximumNumberTrees, m_TreeImpl->m_MaximumNumberTrees);
    return result;
}

std::size_t CBoostedTreeFactory::numberExtraColumnsForTrain() const {
    return CBoostedTreeImpl::numberExtraColumnsForTrain(m_TreeImpl->m_Loss->numberParameters());
}

void CBoostedTreeFactory::initializeTrainingProgressMonitoring(const core::CDataFrame& frame) {

    // The base unit is the cost of training on one tree.
    //
    // This comprises:
    //  - The cost of category encoding and feature selection which we count as
    //    one hundred units,
    //  - One unit for estimating the expected gain and sum curvature per node,
    //  - LINE_SEARCH_ITERATIONS * "maximum number trees" units per regularization
    //    parameter which isn't user defined,
    //  - LINE_SEARCH_ITERATIONS * "maximum number trees" per forest for training
    //    the downsampling factor if it isn't user defined,
    //  - LINE_SEARCH_ITERATIONS * "maximum number trees" per forest for the learn
    //    learn rate if it isn't user defined,
    //  - The main optimisation loop which costs number folds * maximum number
    //    trees per forest units per iteration,
    //  - The cost of the final train which we count as an extra loop.

    double eta{m_TreeImpl->m_EtaOverride != boost::none
                   ? *m_TreeImpl->m_EtaOverride
                   : computeEta(frame.numberColumns())};

    std::size_t totalNumberSteps{101};
    std::size_t lineSearchMaximumNumberTrees{computeMaximumNumberTrees(eta)};
    if (m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit() == boost::none) {
        totalNumberSteps += MAX_LINE_SEARCH_ITERATIONS * lineSearchMaximumNumberTrees;
    }
    if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        totalNumberSteps += MAX_LINE_SEARCH_ITERATIONS * lineSearchMaximumNumberTrees;
    }
    if (m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        totalNumberSteps += MAX_LINE_SEARCH_ITERATIONS * lineSearchMaximumNumberTrees;
    }
    if (m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        totalNumberSteps += MAX_LINE_SEARCH_ITERATIONS * lineSearchMaximumNumberTrees;
    }
    if (m_TreeImpl->m_DownsampleFactorOverride == boost::none) {
        totalNumberSteps += MAX_LINE_SEARCH_ITERATIONS * lineSearchMaximumNumberTrees;
    }
    if (m_TreeImpl->m_EtaOverride == boost::none) {
        totalNumberSteps += MAX_LINE_SEARCH_ITERATIONS *
                            computeMaximumNumberTrees(MAIN_LOOP_ETA_SCALE_FOR_PROGRESS * eta);
    }
    totalNumberSteps += this->mainLoopNumberSteps(MAIN_LOOP_ETA_SCALE_FOR_PROGRESS * eta);
    LOG_TRACE(<< "total number steps = " << totalNumberSteps);
    m_TreeImpl->m_TrainingProgress = core::CLoopProgress{
        totalNumberSteps, m_TreeImpl->m_Instrumentation->progressCallback(), 1.0, 1024};
}

void CBoostedTreeFactory::resumeRestoredTrainingProgressMonitoring() {
    m_TreeImpl->m_TrainingProgress.progressCallback(
        m_TreeImpl->m_Instrumentation->progressCallback());
    m_TreeImpl->m_TrainingProgress.resumeRestored();
}

std::size_t CBoostedTreeFactory::mainLoopNumberSteps(double eta) const {
    return (this->numberHyperparameterTuningRounds() + 1) *
           this->mainLoopMaximumNumberTrees(eta) * m_TreeImpl->m_NumberFolds;
}

std::size_t CBoostedTreeFactory::mainLoopMaximumNumberTrees(double eta) const {
    if (m_TreeImpl->m_MaximumNumberTreesOverride == boost::none) {
        return computeMaximumNumberTrees(MIN_ETA_SCALE * eta);
    }
    return *m_TreeImpl->m_MaximumNumberTreesOverride;
}

void CBoostedTreeFactory::noopRecordTrainingState(CBoostedTree::TPersistFunc) {
}
}
}
