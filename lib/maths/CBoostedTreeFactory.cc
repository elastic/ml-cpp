/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeFactory.h>

#include <core/CDataFrame.h>
#include <core/CIEEE754.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBayesianOptimisation.h>
#include <maths/CBoostedTreeImpl.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/COrderings.h>
#include <maths/CQuantileSketch.h>
#include <maths/CSampling.h>

#include <boost/optional/optional_io.hpp>

#include <cmath>
#include <memory>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;
using TRowItr = core::CDataFrame::TRowItr;

namespace {
const std::size_t MIN_PARAMETER_INDEX{0};
const std::size_t BEST_PARAMETER_INDEX{1};
const std::size_t MAX_PARAMETER_INDEX{2};
const std::size_t MAX_LINE_SEARCH_ITERATIONS{10};
const double LINE_SEARCH_MINIMUM_RELATIVE_EI_TO_CONTINUE{0.01};
const double MIN_ROWS_PER_FEATURE{20.0};
const double MIN_SOFT_DEPTH_LIMIT{2.0};
const double MIN_SOFT_DEPTH_LIMIT_TOLERANCE{0.05};
const double MAX_SOFT_DEPTH_LIMIT_TOLERANCE{0.25};
const double MIN_ETA{1e-3};
const double MIN_ETA_SCALE{0.5};
const double MAX_ETA_SCALE{2.0};
const double MIN_ETA_GROWTH_RATE_SCALE{0.5};
const double MAX_ETA_GROWTH_RATE_SCALE{1.5};
const double FEATURE_BAG_FRACTION_LINE_SEARCH_RANGE{8.0};
const double MAX_FEATURE_BAG_FRACTION{0.8};
const double MIN_DOWNSAMPLE_LINE_SEARCH_RANGE{2.0};
const double MAX_DOWNSAMPLE_LINE_SEARCH_RANGE{144.0};
const double MIN_DOWNSAMPLE_FACTOR{1e-3};
const double MIN_INITIAL_DOWNSAMPLE_FACTOR{0.05};
const double MAX_INITIAL_DOWNSAMPLE_FACTOR{0.5};
const double MIN_DOWNSAMPLE_FACTOR_SCALE{0.3};
const double MAX_DOWNSAMPLE_FACTOR_SCALE{3.0};
// This isn't a hard limit but we increase the number of default training folds
// if the initial downsample fraction would be larger than this.
const double MAX_DESIRED_INITIAL_DOWNSAMPLE_FRACTION{0.5};
const double MAX_NUMBER_FOLDS{5.0};
const std::size_t MAX_NUMBER_TREES{static_cast<std::size_t>(2.0 / MIN_ETA + 0.5)};
const double EPS{0.01};

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
    return interval(MAX_PARAMETER_INDEX) - interval(MIN_PARAMETER_INDEX) == 0.0;
}

auto validInputStream(core::CDataSearcher& restoreSearcher) {
    try {
        // Note that the search arguments are ignored here.
        auto inputStream = restoreSearcher.search(1, 1);
        if (inputStream == nullptr) {
            LOG_ERROR(<< "Unable to connect to data store.");
            return decltype(restoreSearcher.search(1, 1)){};
        }

        if (inputStream->bad()) {
            LOG_ERROR(<< "State restoration search returned bad stream.");
            return decltype(restoreSearcher.search(1, 1)){};
        }

        if (inputStream->fail()) {
            // If the stream exists and has failed then state is missing.
            LOG_ERROR(<< "State restoration search returned failed stream.");
            return decltype(restoreSearcher.search(1, 1)){};
        }
        return inputStream;

    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
    }
    return decltype(restoreSearcher.search(1, 1)){};
}
}

CBoostedTreeFactory::TBoostedTreeUPtr
CBoostedTreeFactory::buildForTrain(core::CDataFrame& frame, std::size_t dependentVariable) {

    m_TreeImpl->m_DependentVariable = dependentVariable;

    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->initializeNumberFolds(frame); });
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->initializeMissingFeatureMasks(frame); });

    this->prepareDataFrameForTrain(frame);

    m_TreeImpl->m_InitializationStage != CBoostedTreeImpl::E_NotInitialized
        ? this->skipProgressMonitoringFeatureSelection()
        : this->startProgressMonitoringFeatureSelection();

    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->initializeCrossValidation(frame); });
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->selectFeaturesAndEncodeCategories(frame); });
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->determineFeatureDataTypes(frame); });

    m_TreeImpl->m_Instrumentation->updateMemoryUsage(core::CMemory::dynamicSize(m_TreeImpl));
    m_TreeImpl->m_Instrumentation->lossType(m_TreeImpl->m_Loss->name());
    m_TreeImpl->m_Instrumentation->flush();

    this->startProgressMonitoringInitializeHyperparameters(frame);

    if (this->initializeFeatureSampleDistribution()) {
        this->initializeHyperparameters(frame);
        this->initializeHyperparameterOptimisation();
    }

    auto treeImpl = std::make_unique<CBoostedTreeImpl>(m_NumberThreads,
                                                       m_TreeImpl->m_Loss->clone());
    std::swap(m_TreeImpl, treeImpl);
    treeImpl->m_InitializationStage = CBoostedTreeImpl::E_FullyInitialized;

    return TBoostedTreeUPtr{
        new CBoostedTree{frame, m_RecordTrainingState, std::move(treeImpl)}};
}

CBoostedTreeFactory::TBoostedTreeUPtr
CBoostedTreeFactory::buildForTrainIncremental(core::CDataFrame& frame,
                                              std::size_t dependentVariable) {

    m_TreeImpl->m_DependentVariable = dependentVariable;
    m_TreeImpl->m_IncrementalTraining = true;

    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->initializeNumberFolds(frame); });
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->initializeMissingFeatureMasks(frame); });
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized, [&] {
        if (frame.numberRows() > m_TreeImpl->m_NewTrainingRowMask.size()) {
            // We assume any additional rows are new examples.
            m_TreeImpl->m_NewTrainingRowMask.extend(
                true, frame.numberRows() - m_TreeImpl->m_NewTrainingRowMask.size());
        }
    });

    this->prepareDataFrameForIncrementalTrain(frame);

    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->initializeCrossValidation(frame); });

    m_TreeImpl->m_Instrumentation->updateMemoryUsage(core::CMemory::dynamicSize(m_TreeImpl));
    m_TreeImpl->m_Instrumentation->lossType(m_TreeImpl->m_Loss->name());
    m_TreeImpl->m_Instrumentation->flush();

    this->initializeHyperparameters(frame);
    this->initializeHyperparameterOptimisation();

    auto treeImpl = std::make_unique<CBoostedTreeImpl>(m_NumberThreads,
                                                       m_TreeImpl->m_Loss->clone());
    std::swap(m_TreeImpl, treeImpl);
    treeImpl->m_InitializationStage = CBoostedTreeImpl::E_FullyInitialized;

    return TBoostedTreeUPtr{
        new CBoostedTree{frame, m_RecordTrainingState, std::move(treeImpl)}};
}

CBoostedTreeFactory::TBoostedTreeUPtr
CBoostedTreeFactory::buildForPredict(core::CDataFrame& frame, std::size_t dependentVariable) {

    m_TreeImpl->m_DependentVariable = dependentVariable;

    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->initializeMissingFeatureMasks(frame); });

    this->prepareDataFrameForTrain(frame);

    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->determineFeatureDataTypes(frame); });

    m_TreeImpl->m_Instrumentation->updateMemoryUsage(core::CMemory::dynamicSize(m_TreeImpl));
    m_TreeImpl->m_Instrumentation->lossType(m_TreeImpl->m_Loss->name());
    m_TreeImpl->m_Instrumentation->flush();

    auto treeImpl = std::make_unique<CBoostedTreeImpl>(m_NumberThreads,
                                                       m_TreeImpl->m_Loss->clone());
    std::swap(m_TreeImpl, treeImpl);
    treeImpl->m_InitializationStage = CBoostedTreeImpl::E_FullyInitialized;

    return TBoostedTreeUPtr{
        new CBoostedTree{frame, m_RecordTrainingState, std::move(treeImpl)}};
}

CBoostedTreeFactory::TBoostedTreeUPtr
CBoostedTreeFactory::restoreFor(core::CDataFrame& frame, std::size_t dependentVariable) {

    if (dependentVariable != m_TreeImpl->m_DependentVariable) {
        HANDLE_FATAL(<< "Internal error: expected dependent variable "
                     << m_TreeImpl->m_DependentVariable << " got "
                     << dependentVariable << ".");
        return nullptr;
    }

    switch (m_TreeImpl->m_InitializationStage) {
    case CBoostedTreeImpl::E_FullyInitialized:
        break;
    case CBoostedTreeImpl::E_NotInitialized:
    case CBoostedTreeImpl::E_SoftTreeDepthLimitInitialized:
    case CBoostedTreeImpl::E_DepthPenaltyMultiplierInitialized:
    case CBoostedTreeImpl::E_TreeSizePenaltyMultiplierInitialized:
    case CBoostedTreeImpl::E_LeafWeightPenaltyMultiplierInitialized:
    case CBoostedTreeImpl::E_DownsampleFactorInitialized:
    case CBoostedTreeImpl::E_FeatureBagFractionInitialized:
    case CBoostedTreeImpl::E_EtaInitialized:
        // We only ever checkpoint after fully initialising for incremental train.
        return this->buildForTrain(frame, dependentVariable);
    }

    if (m_TreeImpl->m_IncrementalTraining == false) {
        this->prepareDataFrameForTrain(frame);
    } else {
        this->prepareDataFrameForIncrementalTrain(frame);
    }

    m_TreeImpl->m_Instrumentation->updateMemoryUsage(core::CMemory::dynamicSize(m_TreeImpl));
    m_TreeImpl->m_Instrumentation->lossType(m_TreeImpl->m_Loss->name());
    m_TreeImpl->m_Instrumentation->flush();

    if (m_TreeImpl->m_IncrementalTraining == false) {
        this->skipProgressMonitoringFeatureSelection();
        this->skipProgressMonitoringInitializeHyperparameters();
    }

    return TBoostedTreeUPtr{
        new CBoostedTree{frame, m_RecordTrainingState, std::move(m_TreeImpl)}};
}

std::size_t CBoostedTreeFactory::numberHyperparameterTuningRounds() const {
    return m_TreeImpl->m_MaximumOptimisationRoundsPerHyperparameter *
           m_TreeImpl->numberHyperparametersToTune();
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

    m_TreeImpl->initializeTunableHyperparameters();

    CBayesianOptimisation::TDoubleDoublePrVec boundingBox;
    for (const auto& parameter : m_TreeImpl->m_TunableHyperparameters) {
        switch (parameter) {
        case E_DownsampleFactor:
            boundingBox.emplace_back(
                m_LogDownsampleFactorSearchInterval(MIN_PARAMETER_INDEX),
                m_LogDownsampleFactorSearchInterval(MAX_PARAMETER_INDEX));
            break;
        case E_Alpha:
            boundingBox.emplace_back(
                m_LogDepthPenaltyMultiplierSearchInterval(MIN_PARAMETER_INDEX),
                m_LogDepthPenaltyMultiplierSearchInterval(MAX_PARAMETER_INDEX));
            break;
        case E_Lambda:
            boundingBox.emplace_back(
                m_LogLeafWeightPenaltyMultiplierSearchInterval(MIN_PARAMETER_INDEX),
                m_LogLeafWeightPenaltyMultiplierSearchInterval(MAX_PARAMETER_INDEX));
            break;
        case E_Gamma:
            boundingBox.emplace_back(
                m_LogTreeSizePenaltyMultiplierSearchInterval(MIN_PARAMETER_INDEX),
                m_LogTreeSizePenaltyMultiplierSearchInterval(MAX_PARAMETER_INDEX));
            break;
        case E_SoftTreeDepthLimit:
            boundingBox.emplace_back(m_SoftDepthLimitSearchInterval(MIN_PARAMETER_INDEX),
                                     m_SoftDepthLimitSearchInterval(MAX_PARAMETER_INDEX));
            break;
        case E_SoftTreeDepthTolerance:
            boundingBox.emplace_back(MIN_SOFT_DEPTH_LIMIT_TOLERANCE,
                                     MAX_SOFT_DEPTH_LIMIT_TOLERANCE);
            break;
        case E_Eta:
            boundingBox.emplace_back(m_LogEtaSearchInterval(MIN_PARAMETER_INDEX),
                                     m_LogEtaSearchInterval(MAX_PARAMETER_INDEX));
            break;
        case E_EtaGrowthRatePerTree: {
            double rate{m_TreeImpl->m_EtaGrowthRatePerTree - 1.0};
            boundingBox.emplace_back(1.0 + MIN_ETA_GROWTH_RATE_SCALE * rate,
                                     1.0 + MAX_ETA_GROWTH_RATE_SCALE * rate);
            break;
        }
        case E_FeatureBagFraction:
            boundingBox.emplace_back(
                CTools::stableExp(m_LogFeatureBagFractionInterval(MIN_PARAMETER_INDEX)),
                CTools::stableExp(m_LogFeatureBagFractionInterval(MAX_PARAMETER_INDEX)));
            break;
        case E_PredictionChangeCost:
            boundingBox.emplace_back(CTools::stableLog(0.01), CTools::stableLog(2.0));
            break;
        case E_TreeTopologyChangePenalty:
            boundingBox.emplace_back(
                m_LogTreeTopologyChangePenaltySearchInterval(MIN_PARAMETER_INDEX),
                m_LogTreeTopologyChangePenaltySearchInterval(MAX_PARAMETER_INDEX));
            break;
        case E_MaximumNumberTrees:
            // maximum number trees is not a tunable parameter
            break;
        }
    }
    LOG_TRACE(<< "hyperparameter search bounding box = "
              << core::CContainerPrinter::print(boundingBox));

    m_TreeImpl->m_BayesianOptimization = std::make_unique<CBayesianOptimisation>(
        std::move(boundingBox),
        m_BayesianOptimisationRestarts.value_or(CBayesianOptimisation::RESTARTS));

    m_TreeImpl->m_CurrentRound = 0;
    m_TreeImpl->m_BestHyperparameters = CBoostedTreeHyperparameters(
        m_TreeImpl->m_Regularization, m_TreeImpl->m_DownsampleFactor, m_TreeImpl->m_Eta,
        m_TreeImpl->m_EtaGrowthRatePerTree, m_TreeImpl->m_MaximumNumberTrees,
        m_TreeImpl->m_FeatureBagFraction, m_TreeImpl->m_PredictionChangeCost);
    m_TreeImpl->m_NumberRounds = this->numberHyperparameterTuningRounds();
}

void CBoostedTreeFactory::initializeMissingFeatureMasks(const core::CDataFrame& frame) const {

    m_TreeImpl->m_MissingFeatureRowMasks.clear();
    m_TreeImpl->m_MissingFeatureRowMasks.resize(frame.numberColumns());

    auto result = frame.readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
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
                [this](std::size_t& numberTrainingRows,
                       const TRowItr& beginRows, const TRowItr& endRows) {
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

void CBoostedTreeFactory::prepareDataFrameForTrain(core::CDataFrame& frame) const {

    // Extend the frame with the bookkeeping columns used in train.
    std::size_t numberLossParameters{m_TreeImpl->m_Loss->numberParameters()};
    std::size_t frameMemory{core::CMemory::dynamicSize(frame)};
    std::tie(m_TreeImpl->m_ExtraColumns, m_TreeImpl->m_PaddedExtraColumns) =
        frame.resizeColumns(m_TreeImpl->m_NumberThreads,
                            extraColumnsForTrain(numberLossParameters));
    m_TreeImpl->m_Instrumentation->updateMemoryUsage(
        core::CMemory::dynamicSize(frame) - frameMemory);
    m_TreeImpl->m_Instrumentation->flush();

    core::CPackedBitVector allTrainingRowsMask{m_TreeImpl->allTrainingRowsMask()};
    frame.writeColumns(m_NumberThreads, 0, frame.numberRows(),
                       [&](const TRowItr& beginRows, const TRowItr& endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               writeExampleWeight(*row, m_TreeImpl->m_ExtraColumns, 1.0);
                           }
                       },
                       &allTrainingRowsMask);
}

void CBoostedTreeFactory::prepareDataFrameForIncrementalTrain(core::CDataFrame& frame) const {

    this->prepareDataFrameForTrain(frame);

    // Extend the frame with the bookkeeping columns used in incremental train.
    std::size_t frameMemory{core::CMemory::dynamicSize(frame)};
    TSizeVec extraColumns;
    std::size_t paddedExtraColumns;
    std::size_t numberLossParameters{m_TreeImpl->m_Loss->numberParameters()};
    std::tie(extraColumns, paddedExtraColumns) = frame.resizeColumns(
        m_TreeImpl->m_NumberThreads, extraColumnsForIncrementalTrain(numberLossParameters));
    m_TreeImpl->m_ExtraColumns.insert(m_TreeImpl->m_ExtraColumns.end(),
                                      extraColumns.begin(), extraColumns.end());
    m_TreeImpl->m_PaddedExtraColumns = m_TreeImpl->m_PaddedExtraColumns
                                           ? *m_TreeImpl->m_PaddedExtraColumns + paddedExtraColumns
                                           : paddedExtraColumns;
    m_TreeImpl->m_Instrumentation->updateMemoryUsage(
        core::CMemory::dynamicSize(frame) - frameMemory);
    m_TreeImpl->m_Instrumentation->flush();

    // Compute predictions from the old model on the new training data.
    m_TreeImpl->predict(m_TreeImpl->m_NewTrainingRowMask, frame);

    // Copy all predictions to previous prediction column(s) in frame.
    frame.writeColumns(m_NumberThreads, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            readPreviousPrediction(*row, m_TreeImpl->m_ExtraColumns, numberLossParameters) =
                readPrediction(*row, m_TreeImpl->m_ExtraColumns, numberLossParameters);
        }
    });
}

void CBoostedTreeFactory::initializeCrossValidation(core::CDataFrame& frame) const {

    core::CPackedBitVector allTrainingRowsMask{m_TreeImpl->allTrainingRowsMask()};
    std::size_t dependentVariable{m_TreeImpl->m_DependentVariable};

    std::size_t numberThreads{m_TreeImpl->m_NumberThreads};
    std::size_t numberFolds{m_TreeImpl->m_NumberFolds};
    std::size_t numberBuckets(m_StratifyRegressionCrossValidation ? 10 : 1);
    auto& rng = m_TreeImpl->m_Rng;

    if (m_TreeImpl->m_IncrementalTraining == false) {
        std::tie(m_TreeImpl->m_TrainingRowMasks, m_TreeImpl->m_TestingRowMasks, std::ignore) =
            CDataFrameUtils::stratifiedCrossValidationRowMasks(
                numberThreads, frame, dependentVariable, rng, numberFolds,
                numberBuckets, allTrainingRowsMask);
    } else {

        // Use separate stratified samples on old and new training data to ensure
        // we have even splits of old and new data across all folds.

        std::tie(m_TreeImpl->m_TrainingRowMasks, m_TreeImpl->m_TestingRowMasks, std::ignore) =
            CDataFrameUtils::stratifiedCrossValidationRowMasks(
                numberThreads, frame, dependentVariable, rng, numberFolds, numberBuckets,
                allTrainingRowsMask & ~m_TreeImpl->m_NewTrainingRowMask);

        if (m_TreeImpl->m_NewTrainingRowMask.manhattan() > 0.0) {
            TPackedBitVectorVec newTrainingRowMasks;
            TPackedBitVectorVec newTestingRowMasks;
            std::tie(newTrainingRowMasks, newTestingRowMasks, std::ignore) =
                CDataFrameUtils::stratifiedCrossValidationRowMasks(
                    numberThreads, frame, dependentVariable, rng, numberFolds, numberBuckets,
                    allTrainingRowsMask & m_TreeImpl->m_NewTrainingRowMask);
            for (std::size_t i = 0; i < numberFolds; ++i) {
                m_TreeImpl->m_TrainingRowMasks[i] |= newTrainingRowMasks[i];
                m_TreeImpl->m_TestingRowMasks[i] |= newTestingRowMasks[i];
            }
        }
    }
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
            .columnMask(std::move(regressors))
            .progressCallback(m_TreeImpl->m_Instrumentation->progressCallback()));
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
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->initializeHyperparametersSetup(frame); });

    if (m_TreeImpl->m_IncrementalTraining == false) {
        if (m_TreeImpl->m_RegularizationOverride.countNotSetForTrain() > 0) {
            this->initializeUnsetRegularizationHyperparameters(frame);
        }
        this->initializeUnsetDownsampleFactor(frame);
        this->initializeUnsetFeatureBagFraction(frame);
        this->initializeUnsetEta(frame);
    } else {
        skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                    [this] { this->initializeUnsetTreeTopologyPenalty(); });
    }
}

void CBoostedTreeFactory::initializeHyperparametersSetup(core::CDataFrame& frame) {
    if (m_TreeImpl->m_EtaOverride != boost::none) {
        m_TreeImpl->m_Eta = *(m_TreeImpl->m_EtaOverride);
    } else {
        m_TreeImpl->m_Eta =
            computeEta(frame.numberColumns() - this->numberExtraColumnsForTrain());
        m_TreeImpl->m_EtaGrowthRatePerTree = 1.0 + m_TreeImpl->m_Eta / 2.0;
    }

    if (m_TreeImpl->m_EtaGrowthRatePerTreeOverride != boost::none) {
        m_TreeImpl->m_EtaGrowthRatePerTree = *(m_TreeImpl->m_EtaGrowthRatePerTreeOverride);
    }

    if (m_TreeImpl->m_MaximumNumberTreesOverride != boost::none) {
        m_TreeImpl->m_MaximumNumberTrees = *(m_TreeImpl->m_MaximumNumberTreesOverride);
    } else {
        // This needs to be tied to the learn rate to avoid bias.
        m_TreeImpl->m_MaximumNumberTrees = computeMaximumNumberTrees(m_TreeImpl->m_Eta);
    }

    double numberFeatures{static_cast<double>(m_TreeImpl->m_Encoder->numberEncodedColumns())};

    if (m_TreeImpl->m_FeatureBagFractionOverride != boost::none) {
        m_TreeImpl->m_FeatureBagFraction = *(m_TreeImpl->m_FeatureBagFractionOverride);
    } else {
        m_TreeImpl->m_FeatureBagFraction =
            std::min(m_TreeImpl->m_FeatureBagFraction,
                     m_TreeImpl->m_TrainingRowMasks[0].manhattan() /
                         MIN_ROWS_PER_FEATURE / numberFeatures);
    }

    m_TreeImpl->m_PredictionChangeCost =
        m_TreeImpl->m_PredictionChangeCostOverride.value_or(0.5);

    double downsampleFactor{m_InitialDownsampleRowsPerFeature * numberFeatures /
                            m_TreeImpl->m_TrainingRowMasks[0].manhattan()};
    m_TreeImpl->m_DownsampleFactor = m_TreeImpl->m_DownsampleFactorOverride.value_or(
        CTools::truncate(downsampleFactor, MIN_INITIAL_DOWNSAMPLE_FACTOR,
                         MAX_INITIAL_DOWNSAMPLE_FACTOR));

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
            m_TreeImpl->m_RegularizationOverride.softTreeDepthTolerance().value_or(0.0))
        .treeTopologyChangePenalty(
            m_TreeImpl->m_RegularizationOverride.treeTopologyChangePenalty().value_or(0.0));
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
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized, [&] {
        m_TreeImpl->m_Regularization.softTreeDepthLimit(
            m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit().value_or(log2MaxTreeSize));
        m_TreeImpl->m_Regularization.softTreeDepthTolerance(
            m_TreeImpl->m_RegularizationOverride.softTreeDepthTolerance().value_or(
                0.5 * (MIN_SOFT_DEPTH_LIMIT_TOLERANCE + MAX_SOFT_DEPTH_LIMIT_TOLERANCE)));

        auto gainAndTotalCurvaturePerNode =
            this->estimateTreeGainAndCurvature(frame, {1.0, 50.0, 90.0});

        m_GainPerNode1stPercentile = gainAndTotalCurvaturePerNode[0].first;
        m_GainPerNode50thPercentile = gainAndTotalCurvaturePerNode[1].first;
        m_GainPerNode90thPercentile = gainAndTotalCurvaturePerNode[2].first;
        m_TotalCurvaturePerNode1stPercentile = gainAndTotalCurvaturePerNode[0].second;
        m_TotalCurvaturePerNode90thPercentile = gainAndTotalCurvaturePerNode[2].second;

        // Make sure all line search intervals are not empty.
        m_GainPerNode1stPercentile = std::min(m_GainPerNode1stPercentile,
                                              0.1 * m_GainPerNode90thPercentile);
        m_TotalCurvaturePerNode1stPercentile =
            std::min(m_TotalCurvaturePerNode1stPercentile,
                     0.1 * m_TotalCurvaturePerNode90thPercentile);

        LOG_TRACE(<< "max depth = " << m_TreeImpl->m_Regularization.softTreeDepthLimit()
                  << ", tolerance = " << m_TreeImpl->m_Regularization.softTreeDepthTolerance()
                  << ", gains and total curvatures per node = "
                  << core::CContainerPrinter::print(gainAndTotalCurvaturePerNode));
    });

    // Search for depth limit at which the tree starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit() == boost::none) {
        if (this->skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_SoftTreeDepthLimitInitialized, [&] {
                if (m_GainPerNode90thPercentile > 0.0) {
                    if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() ==
                        boost::none) {
                        m_TreeImpl->m_Regularization.depthPenaltyMultiplier(m_GainPerNode50thPercentile);
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
                        this->testLossLineSearch(frame, applySoftDepthLimit,
                                                 minSoftDepthLimit, maxSoftDepthLimit,
                                                 -mainLoopSearchInterval / 2.0,
                                                 mainLoopSearchInterval / 2.0)
                            .value_or(fallback);
                    m_SoftDepthLimitSearchInterval = max(
                        m_SoftDepthLimitSearchInterval, TVector{MIN_SOFT_DEPTH_LIMIT});
                    LOG_TRACE(<< "soft depth limit search interval = ["
                              << m_SoftDepthLimitSearchInterval.toDelimited() << "]");
                    m_TreeImpl->m_Regularization.softTreeDepthLimit(
                        m_SoftDepthLimitSearchInterval(BEST_PARAMETER_INDEX));
                }
                if (m_GainPerNode90thPercentile <= 0.0 ||
                    intervalIsEmpty(m_SoftDepthLimitSearchInterval)) {
                    m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit(
                        m_TreeImpl->m_Regularization.softTreeDepthLimit());
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }

    // Set the depth limit to its smallest value and search for the value of the
    // penalty multiplier at which the tree starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        if (this->skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_DepthPenaltyMultiplierInitialized, [&] {
                if (m_GainPerNode90thPercentile > 0.0) {
                    double searchIntervalSize{2.0 * m_GainPerNode90thPercentile /
                                              m_GainPerNode1stPercentile};
                    double logMaxDepthPenaltyMultiplier{
                        CTools::stableLog(m_GainPerNode90thPercentile)};
                    double logMinDepthPenaltyMultiplier{
                        logMaxDepthPenaltyMultiplier - CTools::stableLog(searchIntervalSize)};
                    double meanLogDepthPenaltyMultiplier{
                        (logMinDepthPenaltyMultiplier + logMaxDepthPenaltyMultiplier) / 2.0};
                    double mainLoopSearchInterval{CTools::stableLog(searchIntervalSize) / 2.0};
                    LOG_TRACE(<< "mean log depth penalty multiplier = "
                              << meanLogDepthPenaltyMultiplier);

                    auto applyDepthPenaltyMultiplier = [](CBoostedTreeImpl& tree,
                                                          double logDepthPenalty) {
                        tree.m_Regularization.depthPenaltyMultiplier(
                            CTools::stableExp(logDepthPenalty));
                        return true;
                    };

                    TVector fallback;
                    fallback(MIN_PARAMETER_INDEX) = logMinDepthPenaltyMultiplier;
                    fallback(BEST_PARAMETER_INDEX) = meanLogDepthPenaltyMultiplier;
                    fallback(MAX_PARAMETER_INDEX) = logMaxDepthPenaltyMultiplier;

                    m_LogDepthPenaltyMultiplierSearchInterval =
                        this->testLossLineSearch(frame, applyDepthPenaltyMultiplier,
                                                 logMinDepthPenaltyMultiplier,
                                                 logMaxDepthPenaltyMultiplier,
                                                 -mainLoopSearchInterval / 2.0,
                                                 mainLoopSearchInterval / 2.0)
                            .value_or(fallback);
                    LOG_TRACE(<< "log depth penalty multiplier search interval = ["
                              << m_LogDepthPenaltyMultiplierSearchInterval.toDelimited()
                              << "]");

                    m_TreeImpl->m_Regularization.depthPenaltyMultiplier(CTools::stableExp(
                        m_LogDepthPenaltyMultiplierSearchInterval(BEST_PARAMETER_INDEX)));
                }
                if (m_GainPerNode90thPercentile <= 0.0 ||
                    intervalIsEmpty(m_LogDepthPenaltyMultiplierSearchInterval)) {
                    m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier(
                        m_TreeImpl->m_Regularization.depthPenaltyMultiplier());
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }

    // Search for the value of the tree size penalty multiplier at which the tree
    // starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        if (this->skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_TreeSizePenaltyMultiplierInitialized, [&] {
                if (m_GainPerNode90thPercentile > 0.0) {
                    double searchIntervalSize{2.0 * m_GainPerNode90thPercentile /
                                              m_GainPerNode1stPercentile};
                    double logMaxTreeSizePenaltyMultiplier{
                        CTools::stableLog(m_GainPerNode90thPercentile)};
                    double logMinTreeSizePenaltyMultiplier{
                        logMaxTreeSizePenaltyMultiplier - CTools::stableLog(searchIntervalSize)};
                    double meanLogTreeSizePenaltyMultiplier{
                        (logMinTreeSizePenaltyMultiplier + logMaxTreeSizePenaltyMultiplier) / 2.0};
                    double mainLoopSearchInterval{0.5 * CTools::stableLog(searchIntervalSize)};
                    LOG_TRACE(<< "mean log tree size penalty multiplier = "
                              << meanLogTreeSizePenaltyMultiplier);

                    auto applyTreeSizePenaltyMultiplier =
                        [](CBoostedTreeImpl& tree, double logTreeSizePenalty) {
                            tree.m_Regularization.treeSizePenaltyMultiplier(
                                CTools::stableExp(logTreeSizePenalty));
                            return true;
                        };

                    TVector fallback;
                    fallback(MIN_PARAMETER_INDEX) = logMinTreeSizePenaltyMultiplier;
                    fallback(BEST_PARAMETER_INDEX) = meanLogTreeSizePenaltyMultiplier;
                    fallback(MAX_PARAMETER_INDEX) = logMaxTreeSizePenaltyMultiplier;

                    m_LogTreeSizePenaltyMultiplierSearchInterval =
                        this->testLossLineSearch(frame, applyTreeSizePenaltyMultiplier,
                                                 logMinTreeSizePenaltyMultiplier,
                                                 logMaxTreeSizePenaltyMultiplier,
                                                 -mainLoopSearchInterval / 2.0,
                                                 mainLoopSearchInterval / 2.0)
                            .value_or(fallback);
                    LOG_TRACE(<< "log tree size penalty multiplier search interval = ["
                              << m_LogTreeSizePenaltyMultiplierSearchInterval.toDelimited()
                              << "]");

                    m_TreeImpl->m_Regularization.treeSizePenaltyMultiplier(CTools::stableExp(
                        m_LogTreeSizePenaltyMultiplierSearchInterval(BEST_PARAMETER_INDEX)));
                }
                if (m_GainPerNode90thPercentile <= 0.0 ||
                    intervalIsEmpty(m_LogTreeSizePenaltyMultiplierSearchInterval)) {
                    m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier(
                        m_TreeImpl->m_Regularization.treeSizePenaltyMultiplier());
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }

    // Search for the value of the leaf weight penalty multiplier at which the
    // tree starts to overfit.
    if (m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        if (this->skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_LeafWeightPenaltyMultiplierInitialized, [&] {
                if (m_TotalCurvaturePerNode90thPercentile > 0.0) {
                    double searchIntervalSize{2.0 * m_TotalCurvaturePerNode90thPercentile /
                                              m_TotalCurvaturePerNode1stPercentile};
                    double logMaxLeafWeightPenaltyMultiplier{
                        CTools::stableLog(m_TotalCurvaturePerNode90thPercentile)};
                    double logMinLeafWeightPenaltyMultiplier{
                        logMaxLeafWeightPenaltyMultiplier -
                        CTools::stableLog(searchIntervalSize)};
                    double meanLogLeafWeightPenaltyMultiplier{
                        (logMinLeafWeightPenaltyMultiplier + logMaxLeafWeightPenaltyMultiplier) / 2.0};
                    double mainLoopSearchInterval{0.5 * CTools::stableLog(searchIntervalSize)};
                    LOG_TRACE(<< "mean log leaf weight penalty multiplier = "
                              << meanLogLeafWeightPenaltyMultiplier);

                    auto applyLeafWeightPenaltyMultiplier =
                        [](CBoostedTreeImpl& tree, double logLeafWeightPenalty) {
                            tree.m_Regularization.leafWeightPenaltyMultiplier(
                                CTools::stableExp(logLeafWeightPenalty));
                            return true;
                        };

                    TVector fallback;
                    fallback(MIN_PARAMETER_INDEX) = logMinLeafWeightPenaltyMultiplier;
                    fallback(BEST_PARAMETER_INDEX) = meanLogLeafWeightPenaltyMultiplier;
                    fallback(MAX_PARAMETER_INDEX) = logMaxLeafWeightPenaltyMultiplier;

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
                        m_LogLeafWeightPenaltyMultiplierSearchInterval(BEST_PARAMETER_INDEX)));
                }
                if (m_TotalCurvaturePerNode90thPercentile <= 0.0 ||
                    intervalIsEmpty(m_LogLeafWeightPenaltyMultiplierSearchInterval)) {
                    m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier(
                        m_TreeImpl->m_Regularization.leafWeightPenaltyMultiplier());
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
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
        if (this->skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_DownsampleFactorInitialized, [&] {
                double searchIntervalSize{CTools::truncate(
                    m_TreeImpl->m_TrainingRowMasks[0].manhattan() / 100.0,
                    MIN_DOWNSAMPLE_LINE_SEARCH_RANGE, MAX_DOWNSAMPLE_LINE_SEARCH_RANGE)};
                double logMaxDownsampleFactor{CTools::stableLog(std::min(
                    std::sqrt(searchIntervalSize) * m_TreeImpl->m_DownsampleFactor, 1.0))};
                double logMinDownsampleFactor{logMaxDownsampleFactor -
                                              CTools::stableLog(searchIntervalSize)};
                double meanLogDownSampleFactor{
                    (logMinDownsampleFactor + logMaxDownsampleFactor) / 2.0};
                LOG_TRACE(<< "mean log downsample factor = " << meanLogDownSampleFactor);

                double initialDownsampleFactor{m_TreeImpl->m_DownsampleFactor};
                double initialDepthPenaltyMultiplier{
                    m_TreeImpl->m_Regularization.depthPenaltyMultiplier()};
                double initialTreeSizePenaltyMultiplier{
                    m_TreeImpl->m_Regularization.treeSizePenaltyMultiplier()};
                double initialLeafWeightPenaltyMultiplier{
                    m_TreeImpl->m_Regularization.leafWeightPenaltyMultiplier()};

                // We need to scale the regularisation terms to account for the difference
                // in the downsample factor compared to the value used in the line search.
                auto scaleRegularizers = [&](CBoostedTreeImpl& tree, double downsampleFactor) {
                    double scale{initialDownsampleFactor / downsampleFactor};
                    tree.m_Regularization.depthPenaltyMultiplier(initialDepthPenaltyMultiplier);
                    tree.m_Regularization.treeSizePenaltyMultiplier(initialTreeSizePenaltyMultiplier);
                    tree.m_Regularization.leafWeightPenaltyMultiplier(
                        initialLeafWeightPenaltyMultiplier);
                    tree.scaleRegularizers(scale);
                    return scale;
                };

                double numberTrainingRows{m_TreeImpl->m_TrainingRowMasks[0].manhattan()};

                auto applyDownsampleFactor = [&](CBoostedTreeImpl& tree,
                                                 double logDownsampleFactor) {
                    double downsampleFactor{CTools::stableExp(logDownsampleFactor)};
                    tree.m_DownsampleFactor = downsampleFactor;
                    scaleRegularizers(tree, downsampleFactor);
                    return tree.m_DownsampleFactor * numberTrainingRows > 10.0;
                };

                // If there is very little relative difference in the loss prefer smaller
                // downsample factors because they train faster. We add a penalty which is
                // eps * lmin * (x - xmin) / (xmax - xmin) for x the downsample factor,
                // [xmin, xmax] the search interval and lmin the minimum test loss. This
                // means we'll never use a parameter whose loss is more than 1 + eps times
                // larger than the minimum.
                auto adjustTestLoss = [=](double logDownsampleFactor,
                                          double minTestLoss, double testLoss) {
                    return testLoss + CTools::linearlyInterpolate(
                                          logMinDownsampleFactor, logMaxDownsampleFactor,
                                          0.0, EPS * minTestLoss, logDownsampleFactor);
                };

                TVector fallback;
                fallback(MIN_PARAMETER_INDEX) = logMinDownsampleFactor;
                fallback(BEST_PARAMETER_INDEX) = meanLogDownSampleFactor;
                fallback(MAX_PARAMETER_INDEX) = logMaxDownsampleFactor;

                m_LogDownsampleFactorSearchInterval =
                    this->testLossLineSearch(
                            frame, applyDownsampleFactor,
                            logMinDownsampleFactor, logMaxDownsampleFactor,
                            CTools::stableLog(MIN_DOWNSAMPLE_FACTOR_SCALE),
                            CTools::stableLog(MAX_DOWNSAMPLE_FACTOR_SCALE), adjustTestLoss)
                        .value_or(fallback);

                // Truncate the log(factor) to be less than or equal to log(1.0) and the
                // downsampled set contains at least ten examples on average.
                m_LogDownsampleFactorSearchInterval =
                    min(max(m_LogDownsampleFactorSearchInterval,
                            TVector{CTools::stableLog(10.0 / numberTrainingRows)}),
                        TVector{0.0});
                LOG_TRACE(<< "log downsample factor search interval = ["
                          << m_LogDownsampleFactorSearchInterval.toDelimited() << "]");

                m_TreeImpl->m_DownsampleFactor = CTools::stableExp(
                    m_LogDownsampleFactorSearchInterval(BEST_PARAMETER_INDEX));

                TVector logScale{CTools::stableLog(
                    scaleRegularizers(*m_TreeImpl, m_TreeImpl->m_DownsampleFactor))};
                m_LogTreeSizePenaltyMultiplierSearchInterval += logScale;
                m_LogLeafWeightPenaltyMultiplierSearchInterval += logScale;

                if (intervalIsEmpty(m_LogDownsampleFactorSearchInterval)) {
                    m_TreeImpl->m_DownsampleFactorOverride = m_TreeImpl->m_DownsampleFactor;
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }
}

void CBoostedTreeFactory::initializeUnsetFeatureBagFraction(core::CDataFrame& frame) {

    if (m_TreeImpl->m_FeatureBagFractionOverride == boost::none) {
        if (this->skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_FeatureBagFractionInitialized, [&] {
                double searchIntervalSize{FEATURE_BAG_FRACTION_LINE_SEARCH_RANGE};
                double logMaxFeatureBagFraction{CTools::stableLog(std::min(
                    2.0 * m_TreeImpl->m_FeatureBagFraction, MAX_FEATURE_BAG_FRACTION))};
                double logMinFeatureBagFraction{logMaxFeatureBagFraction -
                                                CTools::stableLog(searchIntervalSize)};
                double mainLoopSearchInterval{CTools::stableLog(0.2 * searchIntervalSize)};

                auto applyFeatureBagFraction = [&](CBoostedTreeImpl& tree,
                                                   double logFeatureBagFraction) {
                    tree.m_FeatureBagFraction = CTools::stableExp(logFeatureBagFraction);
                    return tree.featureBagSize(tree.m_FeatureBagFraction) > 1;
                };

                // If there is very little relative difference in the loss prefer smaller
                // feature bag fractions because they train faster. We add a penalty which
                // is eps * lmin * (x - xmin) / (xmax - xmin) for x the feature bag fraction,
                // [xmin, xmax] the search interval and lmin the minimum test loss. This
                // means we'll never use a parameter whose loss is more than 1 + eps times
                // larger than the minimum.
                auto adjustTestLoss = [=](double logFeatureBagFraction,
                                          double minTestLoss, double testLoss) {
                    return testLoss + CTools::linearlyInterpolate(
                                          logMinFeatureBagFraction, logMaxFeatureBagFraction,
                                          0.0, EPS * minTestLoss, logFeatureBagFraction);
                };

                TVector fallback;
                fallback(MIN_PARAMETER_INDEX) = logMinFeatureBagFraction;
                fallback(BEST_PARAMETER_INDEX) = logMaxFeatureBagFraction;
                fallback(MAX_PARAMETER_INDEX) = logMaxFeatureBagFraction;
                m_LogFeatureBagFractionInterval =
                    this->testLossLineSearch(
                            frame, applyFeatureBagFraction, logMinFeatureBagFraction,
                            logMaxFeatureBagFraction, -mainLoopSearchInterval / 2.0,
                            mainLoopSearchInterval / 2.0, adjustTestLoss)
                        .value_or(fallback);

                // Truncate the log(fraction) to be less than or equal to log(MAX_FEATURE_BAG_FRACTION).
                m_LogFeatureBagFractionInterval =
                    min(m_LogFeatureBagFractionInterval,
                        TVector{CTools::stableLog(MAX_FEATURE_BAG_FRACTION)});
                LOG_TRACE(<< "log feature bag fraction search interval = ["
                          << m_LogFeatureBagFractionInterval.toDelimited() << "]");

                m_TreeImpl->m_FeatureBagFraction = CTools::stableExp(
                    m_LogFeatureBagFractionInterval(BEST_PARAMETER_INDEX));

                if (intervalIsEmpty(m_LogFeatureBagFractionInterval)) {
                    m_TreeImpl->m_FeatureBagFractionOverride = m_TreeImpl->m_FeatureBagFraction;
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }
}

void CBoostedTreeFactory::initializeUnsetEta(core::CDataFrame& frame) {

    if (m_TreeImpl->m_EtaOverride == boost::none) {
        if (skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_EtaInitialized, [&] {
                double searchIntervalSize{5.0 * MAX_ETA_SCALE / MIN_ETA_SCALE};
                double logMaxEta{CTools::stableLog(std::sqrt(searchIntervalSize) *
                                                   m_TreeImpl->m_Eta)};
                double logMinEta{logMaxEta - CTools::stableLog(searchIntervalSize)};
                double meanLogEta{(logMaxEta + logMinEta) / 2.0};
                double mainLoopSearchInterval{CTools::stableLog(0.2 * searchIntervalSize)};
                LOG_TRACE(<< "mean log eta = " << meanLogEta);

                auto applyEta = [](CBoostedTreeImpl& tree, double eta) {
                    tree.m_Eta = CTools::stableExp(eta);
                    if (tree.m_EtaGrowthRatePerTreeOverride == boost::none) {
                        tree.m_EtaGrowthRatePerTree = 1.0 + tree.m_Eta / 2.0;
                    }
                    if (tree.m_MaximumNumberTreesOverride == boost::none) {
                        tree.m_MaximumNumberTrees = computeMaximumNumberTrees(tree.m_Eta);
                    }
                    return true;
                };

                TVector fallback;
                fallback(MIN_PARAMETER_INDEX) = logMinEta;
                fallback(BEST_PARAMETER_INDEX) = meanLogEta;
                fallback(MAX_PARAMETER_INDEX) = logMaxEta;

                m_LogEtaSearchInterval =
                    this->testLossLineSearch(frame, applyEta, logMinEta, logMaxEta,
                                             -mainLoopSearchInterval / 2.0,
                                             mainLoopSearchInterval / 2.0)
                        .value_or(fallback);
                m_LogEtaSearchInterval = min(m_LogEtaSearchInterval, TVector{0.0});
                LOG_TRACE(<< "log eta search interval = ["
                          << m_LogEtaSearchInterval.toDelimited() << "]");
                applyEta(*m_TreeImpl, m_LogEtaSearchInterval(BEST_PARAMETER_INDEX));

                if (intervalIsEmpty(m_LogEtaSearchInterval)) {
                    m_TreeImpl->m_EtaOverride = m_TreeImpl->m_Eta;
                } else if (m_TreeImpl->m_MaximumNumberTreesOverride == boost::none) {
                    m_TreeImpl->m_MaximumNumberTrees =
                        computeMaximumNumberTrees(MIN_ETA_SCALE * m_TreeImpl->m_Eta);
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame, 0.5));
        }
    }
}

void CBoostedTreeFactory::initializeUnsetTreeTopologyPenalty() {

    if (m_TreeImpl->m_RegularizationOverride.treeTopologyChangePenalty() == boost::none) {

        CFastQuantileSketch quantiles{CQuantileSketch::E_Linear, 50};
        for (const auto& tree : m_TreeImpl->m_BestForest) {
            for (const auto& node : tree) {
                if (node.isLeaf() == false) {
                    quantiles.add(node.gain());
                }
            }
        }

        if (quantiles.count() > 0) {
            // We use the best forest internal gain percentiles to bound the range to search
            // for the penalty. This ensures we search a range which encompasses the penalty
            // having little impact on split selected to strongly resisting changing the tree.

            double gainPercentiles[3];
            quantiles.quantile(1.0, gainPercentiles[0]);
            quantiles.quantile(20.0, gainPercentiles[1]);
            quantiles.quantile(90.0, gainPercentiles[2]);

            m_TreeImpl->m_Regularization.treeTopologyChangePenalty(gainPercentiles[1]);

            gainPercentiles[0] = CTools::stableLog(gainPercentiles[0]);
            gainPercentiles[1] = CTools::stableLog(gainPercentiles[1]);
            gainPercentiles[2] = CTools::stableLog(gainPercentiles[2]);
            m_LogTreeTopologyChangePenaltySearchInterval = TVector{gainPercentiles};
        }
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
            CBoostedTreeImpl::gainAndCurvatureAtPercentile(percentile, forest);
        LOG_TRACE(<< "gain = " << gain << ", curvature = " << curvature);

        result.emplace_back(gain, curvature);
    }

    return result;
}

CBoostedTreeFactory::TOptionalVector
CBoostedTreeFactory::testLossLineSearch(core::CDataFrame& frame,
                                        const TApplyParameter& applyParameter,
                                        double intervalLeftEnd,
                                        double intervalRightEnd,
                                        double returnedIntervalLeftEndOffset,
                                        double returnedIntervalRightEndOffset,
                                        const TAdjustTestLoss& adjustTestLoss_) const {

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

    TMinAccumulator minTestLoss;
    TDoubleDoublePrVec testLosses;
    testLosses.reserve(MAX_LINE_SEARCH_ITERATIONS);
    // Ensure we choose one value based on expected improvement.
    std::size_t minNumberTestLosses{5};

    for (auto parameter :
         {intervalLeftEnd, (2.0 * intervalLeftEnd + intervalRightEnd) / 3.0,
          (intervalLeftEnd + 2.0 * intervalRightEnd) / 3.0, intervalRightEnd}) {
        if (applyParameter(*m_TreeImpl, parameter) == false) {
            m_TreeImpl->m_TrainingProgress.increment(
                (MAX_LINE_SEARCH_ITERATIONS - testLosses.size()) * m_TreeImpl->m_MaximumNumberTrees);
            break;
        }

        CBoostedTreeImpl::TNodeVecVec forest;
        double testLoss;
        std::tie(forest, testLoss, std::ignore) = m_TreeImpl->trainForest(
            frame, m_TreeImpl->m_TrainingRowMasks[0],
            m_TreeImpl->m_TestingRowMasks[0], m_TreeImpl->m_TrainingProgress);
        minTestLoss.add(testLoss);
        testLosses.emplace_back(parameter, testLoss);
    }

    if (testLosses.empty()) {
        return TOptionalVector{};
    }

    auto boptVector = [](double parameter) {
        return SConstant<CBayesianOptimisation::TVector>::get(1, parameter);
    };
    auto adjustTestLoss = [=](double parameter, double testLoss) {
        auto min = std::min_element(testLosses.begin(), testLosses.end(),
                                    COrderings::SSecondLess{});
        return adjustTestLoss_(parameter, min->second, testLoss);
    };

    CBayesianOptimisation bopt{{{intervalLeftEnd, intervalRightEnd}}};
    for (auto& parameterAndTestLoss : testLosses) {
        double parameter;
        double testLoss;
        std::tie(parameter, testLoss) = parameterAndTestLoss;
        double adjustedTestLoss{adjustTestLoss(parameter, testLoss)};
        bopt.add(boptVector(parameter), adjustedTestLoss, 0.0);
        parameterAndTestLoss.second = adjustedTestLoss;
    }

    while (testLosses.size() > 0 && testLosses.size() < MAX_LINE_SEARCH_ITERATIONS) {
        CBayesianOptimisation::TVector parameter;
        TOptionalDouble EI;
        std::tie(parameter, EI) = bopt.maximumExpectedImprovement();
        double threshold{LINE_SEARCH_MINIMUM_RELATIVE_EI_TO_CONTINUE * minTestLoss[0]};
        LOG_TRACE(<< "EI = " << EI << " threshold to continue = " << threshold);
        if ((testLosses.size() >= minNumberTestLosses && EI != boost::none && *EI < threshold) ||
            applyParameter(*m_TreeImpl, parameter(0)) == false) {
            m_TreeImpl->m_TrainingProgress.increment(
                (MAX_LINE_SEARCH_ITERATIONS - testLosses.size()) * m_TreeImpl->m_MaximumNumberTrees);
            break;
        }

        CBoostedTreeImpl::TNodeVecVec forest;
        double testLoss;
        std::tie(forest, testLoss, std::ignore) = m_TreeImpl->trainForest(
            frame, m_TreeImpl->m_TrainingRowMasks[0],
            m_TreeImpl->m_TestingRowMasks[0], m_TreeImpl->m_TrainingProgress);

        minTestLoss.add(testLoss);

        double adjustedTestLoss{adjustTestLoss(parameter(0), testLoss)};
        bopt.add(parameter, adjustedTestLoss, 0.0);
        testLosses.emplace_back(parameter(0), adjustedTestLoss);
    }

    std::sort(testLosses.begin(), testLosses.end());
    LOG_TRACE(<< "test losses = " << core::CContainerPrinter::print(testLosses));

    // Find the smallest test losses and the corresponding parameter interval.
    auto minimumTestLosses = CBasicStatistics::orderStatisticsAccumulator<TDoubleDoublePr>(
        minNumberTestLosses - 1, COrderings::SSecondLess{});
    minimumTestLosses.add(testLosses);
    double minGoodParameter{std::min_element(minimumTestLosses.begin(),
                                             minimumTestLosses.end(), COrderings::SFirstLess{})
                                ->first};
    double maxGoodParameter{std::max_element(minimumTestLosses.begin(),
                                             minimumTestLosses.end(), COrderings::SFirstLess{})
                                ->first};
    auto beginGoodParameterLosses =
        std::find_if(testLosses.begin(), testLosses.end(),
                     [minGoodParameter](const TDoubleDoublePr& loss) {
                         return loss.first == minGoodParameter;
                     });
    auto endGoodParameterLosses =
        std::find_if(testLosses.begin(), testLosses.end(),
                     [maxGoodParameter](const TDoubleDoublePr& loss) {
                         return loss.first == maxGoodParameter;
                     }) +
        1;
    LOG_TRACE(<< "good parameter range = [" << minGoodParameter << ","
              << maxGoodParameter << "]");

    CLeastSquaresOnlineRegression<2, double> leastSquaresQuadraticTestLoss;
    for (auto loss = beginGoodParameterLosses; loss != endGoodParameterLosses; ++loss) {
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
    double bestParameter{[&] {
        if (curvature < 0.0) {
            // Stationary point is a maximum so use furthest point in interval.
            double distanceToLeftEndpoint{std::fabs(minGoodParameter - stationaryPoint)};
            double distanceToRightEndpoint{std::fabs(maxGoodParameter - stationaryPoint)};
            return distanceToLeftEndpoint > distanceToRightEndpoint ? minGoodParameter
                                                                    : maxGoodParameter;
        }
        // Stationary point is a minimum so use nearest point in the interval.
        return CTools::truncate(stationaryPoint, minGoodParameter, maxGoodParameter);
    }()};
    LOG_TRACE(<< "best parameter = " << bestParameter);

    TVector interval{{returnedIntervalLeftEndOffset, 0.0, returnedIntervalRightEndOffset}};
    if (minGoodParameter > intervalLeftEnd) {
        interval(MIN_PARAMETER_INDEX) = std::max(minGoodParameter - bestParameter,
                                                 interval(MIN_PARAMETER_INDEX));
    }
    if (maxGoodParameter < intervalRightEnd) {
        interval(MAX_PARAMETER_INDEX) = std::min(maxGoodParameter - bestParameter,
                                                 interval(MAX_PARAMETER_INDEX));
    }
    if (curvature > 0.0) {
        // Find a short interval with a high probability of containing the optimal
        // regularisation parameter if we found a minimum. In particular, we solve
        // curvature * (x - best)^2 = 3 sigma where sigma is the standard deviation
        // of the test loss residuals to get the interval endpoints. We don't
        // extrapolate the loss function outside the line segment we searched so
        // don't truncate if an endpoint lies outside the searched interval.
        TMeanVarAccumulator residualMoments;
        for (auto loss = beginGoodParameterLosses; loss != endGoodParameterLosses; ++loss) {
            residualMoments.add(loss->second -
                                leastSquaresQuadraticTestLoss.predict(loss->first));
        }
        double sigma{std::sqrt(CBasicStatistics::variance(residualMoments))};
        double threeSigmaInterval{std::sqrt(3.0 * sigma / curvature)};
        if (bestParameter - threeSigmaInterval >= minGoodParameter) {
            interval(MIN_PARAMETER_INDEX) =
                std::max(-threeSigmaInterval, returnedIntervalLeftEndOffset);
        }
        if (bestParameter + threeSigmaInterval <= maxGoodParameter) {
            interval(MAX_PARAMETER_INDEX) =
                std::min(threeSigmaInterval, returnedIntervalRightEndOffset);
        }
    }
    interval += TVector{bestParameter};

    return TOptionalVector{interval};
}

CBoostedTreeFactory CBoostedTreeFactory::constructFromParameters(std::size_t numberThreads,
                                                                 TLossFunctionUPtr loss) {
    return {numberThreads, std::move(loss)};
}

CBoostedTreeFactory CBoostedTreeFactory::constructFromString(std::istream& jsonStream) {
    CBoostedTreeFactory result{1, nullptr};
    try {
        core::CJsonStateRestoreTraverser traverser{jsonStream};
        if (result.acceptRestoreTraverser(traverser) == false || traverser.haveBadState()) {
            throw std::runtime_error{"failed to restore boosted tree"};
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string{"Input error: '"} + e.what() + "'"};
    }
    return result;
}

CBoostedTreeFactory CBoostedTreeFactory::constructFromDefinition(
    std::size_t numberThreads,
    TLossFunctionUPtr loss,
    core::CDataSearcher& dataSearcher,
    core::CDataFrame& frame,
    const TRestoreDataSummarizationFunc& dataSummarizationRestoreCallback,
    const TRestoreBestForestFunc& bestForestRestoreCallback) {

    CBoostedTreeFactory factory{constructFromParameters(numberThreads, std::move(loss))};

    // Read data summarization from the stream.
    TEncoderUPtr encoder;
    TStrSizeUMap encodingsIndices;
    std::tie(encoder, encodingsIndices) =
        dataSummarizationRestoreCallback(validInputStream(dataSearcher), frame);
    if (encoder != nullptr) {
        factory.featureEncoder(std::move(encoder));
    } else {
        HANDLE_FATAL(<< "Failed restoring data summarization.");
    }

    // Read best forest from the stream.
    auto bestForest = bestForestRestoreCallback(validInputStream(dataSearcher), encodingsIndices);
    if (bestForest != nullptr) {
        factory.bestForest(std::move(*bestForest.release()));
    } else {
        HANDLE_FATAL(<< "Failed restoring best forest from the model definition.");
    }

    return factory;
}

CBoostedTreeFactory CBoostedTreeFactory::constructFromModel(TBoostedTreeUPtr model) {
    CBoostedTreeFactory result{1, nullptr};
    result.m_TreeImpl = std::move(model->m_Impl);
    result.m_TreeImpl->m_InitializationStage = CBoostedTreeImpl::E_NotInitialized;
    result.m_TreeImpl->m_MeanForestSizeAccumulator = CBoostedTreeImpl::TMeanAccumulator{};
    result.m_TreeImpl->m_MeanLossAccumulator = CBoostedTreeImpl::TMeanAccumulator{};
    return result;
}

std::size_t CBoostedTreeFactory::maximumNumberRows() {
    return core::CPackedBitVector::maximumSize();
}

CBoostedTreeFactory::CBoostedTreeFactory(std::size_t numberThreads, TLossFunctionUPtr loss)
    : m_NumberThreads{numberThreads},
      m_TreeImpl{std::make_unique<CBoostedTreeImpl>(numberThreads, std::move(loss))} {
}

CBoostedTreeFactory::CBoostedTreeFactory(CBoostedTreeFactory&&) noexcept = default;

CBoostedTreeFactory& CBoostedTreeFactory::operator=(CBoostedTreeFactory&&) noexcept = default;

CBoostedTreeFactory::~CBoostedTreeFactory() = default;

CBoostedTreeFactory&
CBoostedTreeFactory::classAssignmentObjective(CBoostedTree::EClassAssignmentObjective objective) {
    m_TreeImpl->m_ClassAssignmentObjective = objective;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::classificationWeights(TStrDoublePrVec weights) {
    m_TreeImpl->m_ClassificationWeightsOverride = std::move(weights);
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

CBoostedTreeFactory& CBoostedTreeFactory::treeTopologyChangePenalty(double treeTopologyChangePenalty) {
    if (treeTopologyChangePenalty < 0.0) {
        LOG_WARN(<< "tree topology change penalty must be non-negative");
        treeTopologyChangePenalty = 0.0;
    }
    m_TreeImpl->m_RegularizationOverride.treeTopologyChangePenalty(treeTopologyChangePenalty);
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

CBoostedTreeFactory& CBoostedTreeFactory::etaGrowthRatePerTree(double etaGrowthRatePerTree) {
    if (etaGrowthRatePerTree < MIN_ETA) {
        LOG_WARN(<< "Truncating supplied learning rate growth rate " << etaGrowthRatePerTree
                 << " which must be no smaller than " << MIN_ETA);
        etaGrowthRatePerTree = std::max(etaGrowthRatePerTree, MIN_ETA);
    }
    m_TreeImpl->m_EtaGrowthRatePerTreeOverride = etaGrowthRatePerTree;
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

CBoostedTreeFactory& CBoostedTreeFactory::predictionChangeCost(double predictionChangeCost) {
    if (predictionChangeCost < 0.0) {
        LOG_WARN(<< "tree topology change penalty must be non-negative");
        predictionChangeCost = 0.0;
    }
    m_TreeImpl->m_PredictionChangeCostOverride = predictionChangeCost;
    return *this;
}

CBoostedTreeFactory&
CBoostedTreeFactory::maximumOptimisationRoundsPerHyperparameterForTrain(std::size_t rounds) {
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

CBoostedTreeFactory& CBoostedTreeFactory::earlyStoppingEnabled(bool enable) {
    m_TreeImpl->m_StopHyperparameterOptimizationEarly = enable;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::newTrainingRowMask(core::CPackedBitVector rowMask) {
    m_TreeImpl->m_NewTrainingRowMask = std::move(rowMask);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::retrainFraction(double fraction) {
    m_TreeImpl->m_RetrainFraction = fraction;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::featureEncoder(TEncoderUPtr encoder) {
    m_TreeImpl->m_Encoder = std::move(encoder);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::bestForest(TNodeVecVec forest) {
    m_TreeImpl->m_BestForest = std::move(forest);
    return *this;
}

std::size_t CBoostedTreeFactory::estimateMemoryUsageTrain(std::size_t numberRows,
                                                          std::size_t numberColumns) const {
    std::size_t maximumNumberTrees{this->mainLoopMaximumNumberTrees(
        m_TreeImpl->m_EtaOverride != boost::none ? *m_TreeImpl->m_EtaOverride
                                                 : computeEta(numberColumns))};
    std::swap(maximumNumberTrees, m_TreeImpl->m_MaximumNumberTrees);
    std::size_t result{m_TreeImpl->estimateMemoryUsageTrain(numberRows, numberColumns)};
    std::swap(maximumNumberTrees, m_TreeImpl->m_MaximumNumberTrees);
    return result;
}

std::size_t
CBoostedTreeFactory::estimateMemoryUsageTrainIncremental(std::size_t numberRows,
                                                         std::size_t numberColumns) const {
    return m_TreeImpl->estimateMemoryUsageTrainIncremental(numberRows, numberColumns);
}

std::size_t CBoostedTreeFactory::numberExtraColumnsForTrain() const {
    return m_TreeImpl->m_PaddedExtraColumns == boost::none
               ? numberExtraColumnsForTrain(m_TreeImpl->m_Loss->numberParameters())
               : *m_TreeImpl->m_PaddedExtraColumns;
}

std::size_t CBoostedTreeFactory::numberExtraColumnsForTrain(std::size_t numberParameters) {
    return CBoostedTreeImpl::numberExtraColumnsForTrain(numberParameters);
}

void CBoostedTreeFactory::startProgressMonitoringFeatureSelection() {
    m_TreeImpl->m_Instrumentation->startNewProgressMonitoredTask(FEATURE_SELECTION);
}

void CBoostedTreeFactory::startProgressMonitoringInitializeHyperparameters(const core::CDataFrame& frame) {

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

    m_TreeImpl->m_Instrumentation->startNewProgressMonitoredTask(COARSE_PARAMETER_SEARCH);

    std::size_t totalNumberSteps{0};
    if (m_TreeImpl->m_RegularizationOverride.softTreeDepthLimit() == boost::none) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (m_TreeImpl->m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (m_TreeImpl->m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (m_TreeImpl->m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (m_TreeImpl->m_FeatureBagFractionOverride == boost::none) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (m_TreeImpl->m_DownsampleFactorOverride == boost::none) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (m_TreeImpl->m_EtaOverride == boost::none) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame, 0.5);
    }

    LOG_TRACE(<< "initial search total number steps = " << totalNumberSteps);
    m_TreeImpl->m_TrainingProgress = core::CLoopProgress{
        totalNumberSteps, m_TreeImpl->m_Instrumentation->progressCallback(), 1.0, 1024};
}

std::size_t
CBoostedTreeFactory::lineSearchMaximumNumberIterations(const core::CDataFrame& frame,
                                                       double etaScale) const {
    double eta{m_TreeImpl->m_EtaOverride != boost::none
                   ? *m_TreeImpl->m_EtaOverride
                   : computeEta(frame.numberColumns() - this->numberExtraColumnsForTrain())};
    return MAX_LINE_SEARCH_ITERATIONS * computeMaximumNumberTrees(etaScale * eta);
}

std::size_t CBoostedTreeFactory::mainLoopMaximumNumberTrees(double eta) const {
    if (m_TreeImpl->m_MaximumNumberTreesOverride == boost::none) {
        return computeMaximumNumberTrees(MIN_ETA_SCALE * eta);
    }
    return *m_TreeImpl->m_MaximumNumberTreesOverride;
}

template<typename F>
bool CBoostedTreeFactory::skipIfAfter(int stage, const F& f) {
    if (m_TreeImpl->m_InitializationStage <= stage) {
        f();
        return false;
    }
    return true;
}

template<typename F>
bool CBoostedTreeFactory::skipCheckpointIfAtOrAfter(int stage, const F& f) {
    if (m_TreeImpl->m_InitializationStage < stage) {
        f();
        m_TreeImpl->m_InitializationStage =
            static_cast<CBoostedTreeImpl::EInitializationStage>(stage);
        m_RecordTrainingState([this](core::CStatePersistInserter& inserter) {
            this->acceptPersistInserter(inserter);
        });
        return false;
    }
    return true;
}

void CBoostedTreeFactory::skipProgressMonitoringFeatureSelection() {
    m_TreeImpl->m_Instrumentation->startNewProgressMonitoredTask(FEATURE_SELECTION);
}

void CBoostedTreeFactory::skipProgressMonitoringInitializeHyperparameters() {
    m_TreeImpl->m_Instrumentation->startNewProgressMonitoredTask(COARSE_PARAMETER_SEARCH);
}

void CBoostedTreeFactory::noopRecordTrainingState(CBoostedTree::TPersistFunc) {
}

double CBoostedTreeFactory::noopAdjustTestLoss(double, double, double testLoss) {
    return testLoss;
}

namespace {
const std::string VERSION_7_9_TAG{"7.9"};

// clang-format off
const std::string FACTORY_TAG{"factory"};
const std::string GAIN_PER_NODE_1ST_PERCENTILE_TAG{"gain_per_node_1st_percentile"};
const std::string GAIN_PER_NODE_50TH_PERCENTILE_TAG{"gain_per_node_50th_percentile"};
const std::string GAIN_PER_NODE_90TH_PERCENTILE_TAG{"gain_per_node_90th_percentile"};
const std::string INITIALIZATION_CHECKPOINT_TAG{"initialization_checkpoint"};
const std::string LOG_DEPTH_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG{"log_depth_penalty_multiplier_search_interval"};
const std::string LOG_DOWNSAMPLE_FACTOR_SEARCH_INTERVAL_TAG{"log_downsample_factor_search_interval"};
const std::string LOG_ETA_SEARCH_INTERVAL_TAG{"log_eta_search_interval"};
const std::string LOG_FEATURE_BAG_FRACTION_INTERVAL_TAG{"log_feature_bag_fraction_interval"};
const std::string LOG_LEAF_WEIGHT_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG{"log_leaf_weight_penalty_multiplier_search_interval"};
const std::string LOG_TREE_SIZE_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG{"log_tree_size_penalty_multiplier_search_interval"};
const std::string LOG_TREE_TOPOLOGY_CHANGE_PENALTY_SEARCH_INTERVAL_TAG{"log_tree_topology_change_penalty_search_interval"};
const std::string SOFT_DEPTH_LIMIT_SEARCH_INTERVAL_TAG{"soft_depth_limit_search_interval"};
const std::string TOTAL_CURVATURE_PER_NODE_1ST_PERCENTILE_TAG{"total_curvature_per_node_1st_percentile"};
const std::string TOTAL_CURVATURE_PER_NODE_90TH_PERCENTILE_TAG{"total_curvature_per_node_90th_percentile"};
const std::string TREE_TAG{"tree"};
// clang-format on
}

void CBoostedTreeFactory::acceptPersistInserter(core::CStatePersistInserter& inserter_) const {
    inserter_.insertValue(INITIALIZATION_CHECKPOINT_TAG, "");
    inserter_.insertLevel(FACTORY_TAG, [this](core::CStatePersistInserter& inserter) {
        inserter.insertValue(VERSION_7_9_TAG, "");
        core::CPersistUtils::persist(GAIN_PER_NODE_1ST_PERCENTILE_TAG,
                                     m_GainPerNode1stPercentile, inserter);
        core::CPersistUtils::persist(GAIN_PER_NODE_50TH_PERCENTILE_TAG,
                                     m_GainPerNode50thPercentile, inserter);
        core::CPersistUtils::persist(GAIN_PER_NODE_90TH_PERCENTILE_TAG,
                                     m_GainPerNode90thPercentile, inserter);
        core::CPersistUtils::persist(LOG_DEPTH_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG,
                                     m_LogDepthPenaltyMultiplierSearchInterval, inserter);
        core::CPersistUtils::persist(LOG_DOWNSAMPLE_FACTOR_SEARCH_INTERVAL_TAG,
                                     m_LogDownsampleFactorSearchInterval, inserter);
        core::CPersistUtils::persist(LOG_ETA_SEARCH_INTERVAL_TAG,
                                     m_LogEtaSearchInterval, inserter);
        core::CPersistUtils::persist(LOG_FEATURE_BAG_FRACTION_INTERVAL_TAG,
                                     m_LogFeatureBagFractionInterval, inserter);
        core::CPersistUtils::persist(
            LOG_LEAF_WEIGHT_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG,
            m_LogLeafWeightPenaltyMultiplierSearchInterval, inserter);
        core::CPersistUtils::persist(LOG_TREE_SIZE_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG,
                                     m_LogTreeSizePenaltyMultiplierSearchInterval, inserter);
        core::CPersistUtils::persist(LOG_TREE_TOPOLOGY_CHANGE_PENALTY_SEARCH_INTERVAL_TAG,
                                     m_LogTreeTopologyChangePenaltySearchInterval, inserter);
        core::CPersistUtils::persist(SOFT_DEPTH_LIMIT_SEARCH_INTERVAL_TAG,
                                     m_SoftDepthLimitSearchInterval, inserter);
        core::CPersistUtils::persist(TOTAL_CURVATURE_PER_NODE_1ST_PERCENTILE_TAG,
                                     m_TotalCurvaturePerNode1stPercentile, inserter);
        core::CPersistUtils::persist(TOTAL_CURVATURE_PER_NODE_90TH_PERCENTILE_TAG,
                                     m_TotalCurvaturePerNode90thPercentile, inserter);
    });
    inserter_.insertLevel(TREE_TAG, [this](core::CStatePersistInserter& inserter) {
        m_TreeImpl->acceptPersistInserter(inserter);
    });
}

bool CBoostedTreeFactory::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser_) {
    if (traverser_.name() != INITIALIZATION_CHECKPOINT_TAG) {
        return m_TreeImpl->acceptRestoreTraverser(traverser_);
    }

    while (traverser_.next()) {
        const std::string& name_{traverser_.name()};
        if (name_ == FACTORY_TAG) {
            if (traverser_.traverseSubLevel([this](core::CStateRestoreTraverser& traverser) {
                    do {
                        const std::string& name{traverser.name()};
                        RESTORE(GAIN_PER_NODE_1ST_PERCENTILE_TAG,
                                core::CPersistUtils::restore(
                                    GAIN_PER_NODE_1ST_PERCENTILE_TAG,
                                    m_GainPerNode1stPercentile, traverser))
                        RESTORE(GAIN_PER_NODE_50TH_PERCENTILE_TAG,
                                core::CPersistUtils::restore(
                                    GAIN_PER_NODE_50TH_PERCENTILE_TAG,
                                    m_GainPerNode50thPercentile, traverser))
                        RESTORE(GAIN_PER_NODE_90TH_PERCENTILE_TAG,
                                core::CPersistUtils::restore(
                                    GAIN_PER_NODE_90TH_PERCENTILE_TAG,
                                    m_GainPerNode90thPercentile, traverser))
                        RESTORE(LOG_DEPTH_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG,
                                core::CPersistUtils::restore(
                                    LOG_DEPTH_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG,
                                    m_LogDepthPenaltyMultiplierSearchInterval, traverser))
                        RESTORE(LOG_DOWNSAMPLE_FACTOR_SEARCH_INTERVAL_TAG,
                                core::CPersistUtils::restore(
                                    LOG_DOWNSAMPLE_FACTOR_SEARCH_INTERVAL_TAG,
                                    m_LogDownsampleFactorSearchInterval, traverser))
                        RESTORE(LOG_ETA_SEARCH_INTERVAL_TAG,
                                core::CPersistUtils::restore(LOG_ETA_SEARCH_INTERVAL_TAG,
                                                             m_LogEtaSearchInterval, traverser))
                        RESTORE(LOG_FEATURE_BAG_FRACTION_INTERVAL_TAG,
                                core::CPersistUtils::restore(
                                    LOG_FEATURE_BAG_FRACTION_INTERVAL_TAG,
                                    m_LogFeatureBagFractionInterval, traverser))
                        RESTORE(LOG_LEAF_WEIGHT_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG,
                                core::CPersistUtils::restore(
                                    LOG_LEAF_WEIGHT_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG,
                                    m_LogLeafWeightPenaltyMultiplierSearchInterval, traverser))
                        RESTORE(LOG_TREE_SIZE_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG,
                                core::CPersistUtils::restore(
                                    LOG_TREE_SIZE_PENALTY_MULTIPLIER_SEARCH_INTERVAL_TAG,
                                    m_LogTreeSizePenaltyMultiplierSearchInterval, traverser))
                        RESTORE(LOG_TREE_TOPOLOGY_CHANGE_PENALTY_SEARCH_INTERVAL_TAG,
                                core::CPersistUtils::restore(
                                    LOG_TREE_TOPOLOGY_CHANGE_PENALTY_SEARCH_INTERVAL_TAG,
                                    m_LogTreeTopologyChangePenaltySearchInterval, traverser))
                        RESTORE(SOFT_DEPTH_LIMIT_SEARCH_INTERVAL_TAG,
                                core::CPersistUtils::restore(
                                    SOFT_DEPTH_LIMIT_SEARCH_INTERVAL_TAG,
                                    m_SoftDepthLimitSearchInterval, traverser))
                        RESTORE(TOTAL_CURVATURE_PER_NODE_1ST_PERCENTILE_TAG,
                                core::CPersistUtils::restore(
                                    TOTAL_CURVATURE_PER_NODE_1ST_PERCENTILE_TAG,
                                    m_TotalCurvaturePerNode1stPercentile, traverser))
                        RESTORE(TOTAL_CURVATURE_PER_NODE_90TH_PERCENTILE_TAG,
                                core::CPersistUtils::restore(
                                    TOTAL_CURVATURE_PER_NODE_90TH_PERCENTILE_TAG,
                                    m_TotalCurvaturePerNode90thPercentile, traverser))
                    } while (traverser.next());
                    return true;
                }) == false) {
                LOG_ERROR(<< "Failed to restore " << FACTORY_TAG);
                return false;
            }
            continue;
        }
        if (name_ == TREE_TAG) {
            if (traverser_.traverseSubLevel([this](core::CStateRestoreTraverser& traverser) {
                    return m_TreeImpl->acceptRestoreTraverser(traverser);
                }) == false) {
                LOG_ERROR(<< "Failed to restore " << TREE_TAG);
                return false;
            }
            continue;
        }
    }
    return true;
}

const std::string CBoostedTreeFactory::FEATURE_SELECTION{"feature_selection"};
const std::string CBoostedTreeFactory::COARSE_PARAMETER_SEARCH{"coarse_parameter_search"};
const std::string CBoostedTreeFactory::FINE_TUNING_PARAMETERS{"fine_tuning_parameters"};
const std::string CBoostedTreeFactory::FINAL_TRAINING{"final_training"};
const std::string CBoostedTreeFactory::INCREMENTAL_TRAIN{"inncremental_train"};
}
}
