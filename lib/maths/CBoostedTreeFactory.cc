/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <maths/CBoostedTreeFactory.h>

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CIEEE754.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBayesianOptimisation.h>
#include <maths/CBoostedTreeHyperparameters.h>
#include <maths/CBoostedTreeImpl.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CLowess.h>
#include <maths/CLowessDetail.h>
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
const double SMALL_RELATIVE_TEST_LOSS_INCREASE{0.01};
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
// This isn't a hard limit but we increase the number of default training folds
// if the initial downsample fraction would be larger than this.
const double MAX_DESIRED_INITIAL_DOWNSAMPLE_FRACTION{0.5};
const double MAX_NUMBER_FOLDS{5.0};
const std::size_t MAX_NUMBER_TREES{static_cast<std::size_t>(2.0 / MIN_ETA + 0.5)};

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
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized, [&] {
        if (frame.numberRows() > m_TreeImpl->m_NewTrainingRowMask.size()) {
            m_TreeImpl->m_NewTrainingRowMask.extend(
                false, frame.numberRows() - m_TreeImpl->m_NewTrainingRowMask.size());
        }
    });

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

    this->initializeSplitsCache(frame);

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
    m_TreeImpl->m_Hyperparameters.incrementalTraining(true);

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
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->determineFeatureDataTypes(frame); });
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { m_TreeImpl->selectTreesToRetrain(frame); });

    this->initializeSplitsCache(frame);

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
CBoostedTreeFactory::buildForPredict(core::CDataFrame& frame, std::size_t dependentVariable) {

    m_TreeImpl->m_DependentVariable = dependentVariable;

    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->initializeMissingFeatureMasks(frame); });
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized, [&] {
        if (frame.numberRows() > m_TreeImpl->m_NewTrainingRowMask.size()) {
            // We assume any additional rows are new examples.
            m_TreeImpl->m_NewTrainingRowMask.extend(
                true, frame.numberRows() - m_TreeImpl->m_NewTrainingRowMask.size());
        }
    });

    this->prepareDataFrameForTrain(frame);

    skipIfAfter(CBoostedTreeImpl::E_NotInitialized,
                [&] { this->determineFeatureDataTypes(frame); });
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized, [&] {
        m_TreeImpl->predict(frame);
        m_TreeImpl->computeClassificationWeights(frame);
    });

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

    if (m_TreeImpl->m_Hyperparameters.incrementalTraining() == false) {
        this->prepareDataFrameForTrain(frame);
    } else {
        this->prepareDataFrameForIncrementalTrain(frame);
    }
    this->initializeSplitsCache(frame);

    m_TreeImpl->m_Instrumentation->updateMemoryUsage(core::CMemory::dynamicSize(m_TreeImpl));
    m_TreeImpl->m_Instrumentation->lossType(m_TreeImpl->m_Loss->name());
    m_TreeImpl->m_Instrumentation->flush();

    if (m_TreeImpl->m_Hyperparameters.incrementalTraining() == false) {
        this->skipProgressMonitoringFeatureSelection();
        this->skipProgressMonitoringInitializeHyperparameters();
    }

    return TBoostedTreeUPtr{
        new CBoostedTree{frame, m_RecordTrainingState, std::move(m_TreeImpl)}};
}

void CBoostedTreeFactory::initializeHyperparameterOptimisation() const {

    // We need sensible bounds for the region we'll search for optimal values.
    // For all parameters where we have initial estimates we use bounds of the
    // form a * initial and b * initial for a < 1 < b. For other parameters we
    // use a fixed range.
    //
    // We also parameterise so the probability any subinterval contains a good
    // value is proportional to its length. For parameters whose difference is
    // naturally measured as a ratio, i.e. diff(p_1, p_0) = p_1 / p_0 for p_0
    // less than p_1, This translates to using log parameter values.

    m_TreeImpl->m_Hyperparameters.initializeSearch([this](EHyperparameter parameter,
                                                          TDoubleDoublePrVec& boundingBox) {
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
            double rate{m_TreeImpl->m_Hyperparameters.etaGrowthRatePerTree().value() - 1.0};
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
        case E_RetrainedTreeEta:
            boundingBox.emplace_back(
                m_LogRetrainedTreeEtaSearchInterval(MIN_PARAMETER_INDEX),
                m_LogRetrainedTreeEtaSearchInterval(MAX_PARAMETER_INDEX));
            break;
        case E_TreeTopologyChangePenalty:
            boundingBox.emplace_back(
                m_LogTreeTopologyChangePenaltySearchInterval(MIN_PARAMETER_INDEX),
                m_LogTreeTopologyChangePenaltySearchInterval(MAX_PARAMETER_INDEX));
            break;
        case E_MaximumNumberTrees:
            // Maximum number trees is not tuned directly.
            break;
        }
    });
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

    auto result = frame.readRows(
        m_NumberThreads,
        core::bindRetrievableState(
            [this](std::size_t& numberTrainingRows, const TRowItr& beginRows, const TRowItr& endRows) {
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
    //
    // In addition, we want to constrain the maximum amount of training data we'll
    // use during hyperparameter search to avoid very long run times. To do this
    // we use less than the implied 1 - 1/k : 1/k for the train : test split when
    // it results in more train rows than the defined maximum.

    double initialDownsampleFraction{(m_InitialDownsampleRowsPerFeature *
                                      static_cast<double>(frame.numberColumns() - 1)) /
                                     static_cast<double>(totalNumberTrainingRows)};
    LOG_TRACE(<< "initial downsample fraction = " << initialDownsampleFraction);
    m_TreeImpl->m_NumberFolds.set(static_cast<std::size_t>(
        std::ceil(1.0 / std::max(1.0 - initialDownsampleFraction / MAX_DESIRED_INITIAL_DOWNSAMPLE_FRACTION,
                                 1.0 / MAX_NUMBER_FOLDS))));
    m_TreeImpl->m_TrainFractionPerFold.set(
        std::min(1.0 - 1.0 / static_cast<double>(m_TreeImpl->m_NumberFolds.value()),
                 static_cast<double>(m_MaximumNumberOfTrainRows) /
                     static_cast<double>(totalNumberTrainingRows)));
    LOG_TRACE(<< "# folds = " << m_TreeImpl->m_NumberFolds.value() << ", train fraction per fold = "
              << m_TreeImpl->m_TrainFractionPerFold.value());
}

void CBoostedTreeFactory::prepareDataFrameForTrain(core::CDataFrame& frame) const {

    // Extend the frame with the bookkeeping columns used in train.
    std::size_t oldFrameMemory{core::CMemory::dynamicSize(frame)};
    TSizeVec extraColumns;
    std::size_t paddedExtraColumns;
    std::size_t numberLossParameters{m_TreeImpl->m_Loss->numberParameters()};
    std::tie(extraColumns, paddedExtraColumns) = frame.resizeColumns(
        m_TreeImpl->m_NumberThreads, extraColumnsForTrain(numberLossParameters));
    auto extraColumnTags = extraColumnTagsForTrain();
    m_TreeImpl->m_ExtraColumns.resize(E_NumberExtraColumnTags);
    for (std::size_t i = 0; i < extraColumnTags.size(); ++i) {
        m_TreeImpl->m_ExtraColumns[extraColumnTags[i]] = extraColumns[i];
    }
    m_PaddedExtraColumns += paddedExtraColumns;
    std::size_t newFrameMemory{core::CMemory::dynamicSize(frame)};
    m_TreeImpl->m_Instrumentation->updateMemoryUsage(newFrameMemory - oldFrameMemory);
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
    std::size_t oldFrameMemory{core::CMemory::dynamicSize(frame)};
    TSizeVec extraColumns;
    std::size_t paddedExtraColumns;
    std::size_t numberLossParameters{m_TreeImpl->m_Loss->numberParameters()};
    std::tie(extraColumns, paddedExtraColumns) = frame.resizeColumns(
        m_TreeImpl->m_NumberThreads, extraColumnsForIncrementalTrain(numberLossParameters));
    auto extraColumnTags = extraColumnTagsForIncrementalTrain();
    for (std::size_t i = 0; i < extraColumnTags.size(); ++i) {
        m_TreeImpl->m_ExtraColumns[extraColumnTags[i]] = extraColumns[i];
    }
    m_PaddedExtraColumns += paddedExtraColumns;
    std::size_t newFrameMemory{core::CMemory::dynamicSize(frame)};
    m_TreeImpl->m_Instrumentation->updateMemoryUsage(newFrameMemory - oldFrameMemory);
    m_TreeImpl->m_Instrumentation->flush();

    // Compute predictions from the old model.
    m_TreeImpl->predict(frame);

    // Copy all predictions to previous prediction column(s) in frame.
    frame.writeColumns(m_NumberThreads, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row_ = beginRows; row_ != endRows; ++row_) {
            auto row = *row_;
            writePreviousPrediction(
                row, m_TreeImpl->m_ExtraColumns, numberLossParameters,
                readPrediction(row, m_TreeImpl->m_ExtraColumns, numberLossParameters));
        }
    });
}

void CBoostedTreeFactory::initializeCrossValidation(core::CDataFrame& frame) const {

    core::CPackedBitVector allTrainingRowsMask{m_TreeImpl->allTrainingRowsMask()};
    std::size_t dependentVariable{m_TreeImpl->m_DependentVariable};

    std::size_t numberThreads{m_TreeImpl->m_NumberThreads};
    std::size_t numberFolds{m_TreeImpl->m_NumberFolds.value()};
    std::size_t numberBuckets(m_StratifyRegressionCrossValidation ? 10 : 1);
    double trainFractionPerFold{m_TreeImpl->m_TrainFractionPerFold.value()};
    auto& rng = m_TreeImpl->m_Rng;

    if (m_TreeImpl->m_Hyperparameters.incrementalTraining() == false) {
        std::tie(m_TreeImpl->m_TrainingRowMasks, m_TreeImpl->m_TestingRowMasks, std::ignore) =
            CDataFrameUtils::stratifiedCrossValidationRowMasks(
                numberThreads, frame, dependentVariable, rng, numberFolds,
                trainFractionPerFold, numberBuckets, allTrainingRowsMask);
    } else {

        // Use separate stratified samples on old and new training data to ensure
        // we have even splits of old and new data across all folds.

        const auto& newTrainingRowMask = m_TreeImpl->m_NewTrainingRowMask;

        std::tie(m_TreeImpl->m_TrainingRowMasks, m_TreeImpl->m_TestingRowMasks, std::ignore) =
            CDataFrameUtils::stratifiedCrossValidationRowMasks(
                numberThreads, frame, dependentVariable, rng, numberFolds, trainFractionPerFold,
                numberBuckets, allTrainingRowsMask & ~newTrainingRowMask);

        if (m_TreeImpl->m_NewTrainingRowMask.manhattan() > 0.0) {
            TPackedBitVectorVec newTrainingRowMasks;
            TPackedBitVectorVec newTestingRowMasks;
            std::tie(newTrainingRowMasks, newTestingRowMasks, std::ignore) =
                CDataFrameUtils::stratifiedCrossValidationRowMasks(
                    numberThreads, frame, dependentVariable, rng, numberFolds, trainFractionPerFold,
                    numberBuckets, allTrainingRowsMask & newTrainingRowMask);
            for (std::size_t i = 0; i < numberFolds; ++i) {
                m_TreeImpl->m_TrainingRowMasks[i] |= newTrainingRowMasks[i];
                m_TreeImpl->m_TestingRowMasks[i] |= newTestingRowMasks[i];
            }
        }
    }
}

void CBoostedTreeFactory::selectFeaturesAndEncodeCategories(core::CDataFrame& frame) const {

    // TODO we should do feature selection per fold.

    TSizeVec regressors(frame.numberColumns() - m_PaddedExtraColumns);
    std::iota(regressors.begin(), regressors.end(), 0);
    regressors.erase(regressors.begin() + m_TreeImpl->m_DependentVariable);
    std::size_t numberTrainingRows{
        static_cast<std::size_t>(m_TreeImpl->allTrainingRowsMask().manhattan())};
    LOG_TRACE(<< "candidate regressors = " << core::CContainerPrinter::print(regressors));

    m_TreeImpl->m_Encoder = std::make_unique<CDataFrameCategoryEncoder>(
        CMakeDataFrameCategoryEncoder{m_TreeImpl->m_NumberThreads, frame,
                                      m_TreeImpl->m_DependentVariable}
            .minimumRowsPerFeature(m_TreeImpl->rowsPerFeature(numberTrainingRows))
            .minimumFrequencyToOneHotEncode(m_MinimumFrequencyToOneHotEncode)
            .rowMask(m_TreeImpl->allTrainingRowsMask())
            .columnMask(std::move(regressors))
            .progressCallback(m_TreeImpl->m_Instrumentation->progressCallback()));
}

void CBoostedTreeFactory::initializeSplitsCache(core::CDataFrame& frame) const {
    std::size_t oldFrameMemory{core::CMemory::dynamicSize(frame)};
    std::size_t beginSplits{frame.numberColumns()};
    frame.resizeColumns(m_TreeImpl->m_NumberThreads,
                        beginSplits + (m_TreeImpl->numberFeatures() + 3) / 4);
    m_PaddedExtraColumns += frame.numberColumns() - beginSplits;
    m_TreeImpl->m_ExtraColumns[E_BeginSplits] = beginSplits;
    std::size_t newFrameMemory{core::CMemory::dynamicSize(frame)};
    m_TreeImpl->m_Instrumentation->updateMemoryUsage(newFrameMemory - oldFrameMemory);
    m_TreeImpl->m_Instrumentation->flush();
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

    if (m_TreeImpl->m_Hyperparameters.incrementalTraining() == false) {
        this->initializeUnsetRegularizationHyperparameters(frame);
        this->initializeUnsetDownsampleFactor(frame);
        this->initializeUnsetFeatureBagFraction(frame);
        this->initializeUnsetEta(frame);
    } else {
        skipIfAfter(CBoostedTreeImpl::E_NotInitialized, [&] {
            this->initializeUnsetTreeTopologyPenalty(frame);
            this->initializeUnsetRetrainedTreeEta();
        });
    }
}

void CBoostedTreeFactory::initializeHyperparametersSetup(core::CDataFrame& frame) {
    auto& hyperparameters = m_TreeImpl->m_Hyperparameters;

    double numberFeatures{static_cast<double>(m_TreeImpl->m_Encoder->numberEncodedColumns())};
    double featureBagFraction{std::min(hyperparameters.featureBagFraction().value(),
                                       m_TreeImpl->m_TrainingRowMasks[0].manhattan() /
                                           MIN_ROWS_PER_FEATURE / numberFeatures)};
    double downsampleFactor{m_InitialDownsampleRowsPerFeature * numberFeatures /
                            m_TreeImpl->m_TrainingRowMasks[0].manhattan()};

    // Note that values are only set if the parameters are not user overridden.
    hyperparameters.depthPenaltyMultiplier().set(0.0);
    hyperparameters.treeSizePenaltyMultiplier().set(0.0);
    hyperparameters.leafWeightPenaltyMultiplier().set(0.0);
    hyperparameters.softTreeDepthLimit().set(0.0);
    hyperparameters.softTreeDepthTolerance().set(0.0);
    hyperparameters.featureBagFraction().set(featureBagFraction);
    hyperparameters.downsampleFactor().set(CTools::truncate(
        downsampleFactor, MIN_INITIAL_DOWNSAMPLE_FACTOR, MAX_INITIAL_DOWNSAMPLE_FACTOR));
    hyperparameters.eta().set(computeEta(frame.numberColumns() - m_PaddedExtraColumns));
    hyperparameters.etaGrowthRatePerTree().set(1.0 + hyperparameters.eta().value() / 2.0);
    // This needs to be tied to the learn rate to avoid bias.
    hyperparameters.maximumNumberTrees().set(
        computeMaximumNumberTrees(hyperparameters.eta().value()));
    hyperparameters.treeTopologyChangePenalty().set(0.0);
    hyperparameters.predictionChangeCost().set(0.5);
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

    auto& hyperparameters = m_TreeImpl->m_Hyperparameters;
    double log2MaxTreeSize{std::log2(static_cast<double>(m_TreeImpl->maximumTreeSize(
                               m_TreeImpl->m_TrainingRowMasks[0]))) +
                           1.0};
    skipIfAfter(CBoostedTreeImpl::E_NotInitialized, [&] {
        hyperparameters.softTreeDepthLimit().set(log2MaxTreeSize);
        hyperparameters.softTreeDepthTolerance().set(
            0.5 * (MIN_SOFT_DEPTH_LIMIT_TOLERANCE + MAX_SOFT_DEPTH_LIMIT_TOLERANCE));

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

        LOG_TRACE(<< "max depth = " << hyperparameters.softTreeDepthLimit().print() << ", tolerance = "
                  << hyperparameters.softTreeDepthTolerance().print() << ", gains and total curvatures per node = "
                  << core::CContainerPrinter::print(gainAndTotalCurvaturePerNode));
    });

    // Search for depth limit at which the tree starts to overfit.
    if (hyperparameters.softTreeDepthLimit().fixed() == false) {
        if (this->skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_SoftTreeDepthLimitInitialized, [&] {
                if (m_GainPerNode90thPercentile > 0.0) {
                    hyperparameters.depthPenaltyMultiplier().set(m_GainPerNode50thPercentile);
                    double minSoftDepthLimit{MIN_SOFT_DEPTH_LIMIT};
                    double maxSoftDepthLimit{MIN_SOFT_DEPTH_LIMIT + log2MaxTreeSize};
                    double meanSoftDepthLimit{(minSoftDepthLimit + maxSoftDepthLimit) / 2.0};
                    LOG_TRACE(<< "mean soft depth limit = " << meanSoftDepthLimit);

                    auto applySoftDepthLimit = [](CBoostedTreeImpl& tree, double softDepthLimit) {
                        tree.m_Hyperparameters.softTreeDepthLimit().set(softDepthLimit);
                        return true;
                    };

                    TVector fallback{{minSoftDepthLimit, meanSoftDepthLimit, maxSoftDepthLimit}};
                    m_SoftDepthLimitSearchInterval =
                        this->testLossLineSearch(frame, applySoftDepthLimit,
                                                 minSoftDepthLimit, maxSoftDepthLimit)
                            .value_or(fallback);
                    m_SoftDepthLimitSearchInterval = max(
                        m_SoftDepthLimitSearchInterval, TVector{MIN_SOFT_DEPTH_LIMIT});
                    LOG_TRACE(<< "soft depth limit search interval = ["
                              << m_SoftDepthLimitSearchInterval.toDelimited() << "]");
                    hyperparameters.softTreeDepthLimit().set(
                        m_SoftDepthLimitSearchInterval(BEST_PARAMETER_INDEX));
                }
                if (m_GainPerNode90thPercentile <= 0.0 ||
                    intervalIsEmpty(m_SoftDepthLimitSearchInterval)) {
                    hyperparameters.softTreeDepthLimit().fix();
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }

    // Set the depth limit to its smallest value and search for the value of the
    // penalty multiplier at which the tree starts to overfit.
    if (hyperparameters.depthPenaltyMultiplier().fixed() == false) {
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
                    LOG_TRACE(<< "mean log depth penalty multiplier = "
                              << meanLogDepthPenaltyMultiplier);

                    auto applyDepthPenaltyMultiplier = [](CBoostedTreeImpl& tree,
                                                          double logDepthPenalty) {
                        tree.m_Hyperparameters.depthPenaltyMultiplier().set(
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
                                                 logMaxDepthPenaltyMultiplier)
                            .value_or(fallback);
                    LOG_TRACE(<< "log depth penalty multiplier search interval = ["
                              << m_LogDepthPenaltyMultiplierSearchInterval.toDelimited()
                              << "]");

                    hyperparameters.depthPenaltyMultiplier().set(CTools::stableExp(
                        m_LogDepthPenaltyMultiplierSearchInterval(BEST_PARAMETER_INDEX)));
                }
                if (m_GainPerNode90thPercentile <= 0.0 ||
                    intervalIsEmpty(m_LogDepthPenaltyMultiplierSearchInterval)) {
                    hyperparameters.depthPenaltyMultiplier().fix();
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }

    // Search for the value of the tree size penalty multiplier at which the tree
    // starts to overfit.
    if (hyperparameters.treeSizePenaltyMultiplier().fixed() == false) {
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
                    LOG_TRACE(<< "mean log tree size penalty multiplier = "
                              << meanLogTreeSizePenaltyMultiplier);

                    auto applyTreeSizePenaltyMultiplier =
                        [](CBoostedTreeImpl& tree, double logTreeSizePenalty) {
                            tree.m_Hyperparameters.treeSizePenaltyMultiplier().set(
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
                                                 logMaxTreeSizePenaltyMultiplier)
                            .value_or(fallback);
                    LOG_TRACE(<< "log tree size penalty multiplier search interval = ["
                              << m_LogTreeSizePenaltyMultiplierSearchInterval.toDelimited()
                              << "]");

                    hyperparameters.treeSizePenaltyMultiplier().set(CTools::stableExp(
                        m_LogTreeSizePenaltyMultiplierSearchInterval(BEST_PARAMETER_INDEX)));
                }
                if (m_GainPerNode90thPercentile <= 0.0 ||
                    intervalIsEmpty(m_LogTreeSizePenaltyMultiplierSearchInterval)) {
                    hyperparameters.treeSizePenaltyMultiplier().fix();
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }

    // Search for the value of the leaf weight penalty multiplier at which the
    // tree starts to overfit.
    if (hyperparameters.leafWeightPenaltyMultiplier().fixed() == false) {
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
                    LOG_TRACE(<< "mean log leaf weight penalty multiplier = "
                              << meanLogLeafWeightPenaltyMultiplier);

                    auto applyLeafWeightPenaltyMultiplier =
                        [](CBoostedTreeImpl& tree, double logLeafWeightPenalty) {
                            tree.m_Hyperparameters.leafWeightPenaltyMultiplier().set(
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
                                                 logMaxLeafWeightPenaltyMultiplier)
                            .value_or(fallback);
                    LOG_TRACE(<< "log leaf weight penalty multiplier search interval = ["
                              << m_LogLeafWeightPenaltyMultiplierSearchInterval.toDelimited()
                              << "]");
                    hyperparameters.leafWeightPenaltyMultiplier().set(CTools::stableExp(
                        m_LogLeafWeightPenaltyMultiplierSearchInterval(BEST_PARAMETER_INDEX)));
                }
                if (m_TotalCurvaturePerNode90thPercentile <= 0.0 ||
                    intervalIsEmpty(m_LogLeafWeightPenaltyMultiplierSearchInterval)) {
                    hyperparameters.leafWeightPenaltyMultiplier().fix();
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }

    if (hyperparameters.depthPenaltyMultiplier().fixed() &&
        hyperparameters.depthPenaltyMultiplier().value() == 0.0) {
        hyperparameters.softTreeDepthLimit().fix(MIN_SOFT_DEPTH_LIMIT);
        hyperparameters.softTreeDepthTolerance().fix(MIN_SOFT_DEPTH_LIMIT_TOLERANCE);
    }
}

void CBoostedTreeFactory::initializeUnsetDownsampleFactor(core::CDataFrame& frame) {

    auto& hyperparameters = m_TreeImpl->m_Hyperparameters;

    if (hyperparameters.downsampleFactor().fixed() == false) {
        if (this->skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_DownsampleFactorInitialized, [&] {
                double searchIntervalSize{CTools::truncate(
                    m_TreeImpl->m_TrainingRowMasks[0].manhattan() / 100.0,
                    MIN_DOWNSAMPLE_LINE_SEARCH_RANGE, MAX_DOWNSAMPLE_LINE_SEARCH_RANGE)};
                double logMaxDownsampleFactor{CTools::stableLog(
                    std::min(std::sqrt(searchIntervalSize) *
                                 hyperparameters.downsampleFactor().value(),
                             1.0))};
                double logMinDownsampleFactor{logMaxDownsampleFactor -
                                              CTools::stableLog(searchIntervalSize)};
                double meanLogDownSampleFactor{
                    (logMinDownsampleFactor + logMaxDownsampleFactor) / 2.0};
                LOG_TRACE(<< "mean log downsample factor = " << meanLogDownSampleFactor);

                double initialDownsampleFactor{
                    hyperparameters.downsampleFactor().value()};
                double initialDepthPenaltyMultiplier{
                    hyperparameters.depthPenaltyMultiplier().value()};
                double initialTreeSizePenaltyMultiplier{
                    hyperparameters.treeSizePenaltyMultiplier().value()};
                double initialLeafWeightPenaltyMultiplier{
                    hyperparameters.leafWeightPenaltyMultiplier().value()};

                // We need to scale the regularisation terms to account for the difference
                // in the downsample factor compared to the value used in the line search.
                auto scaleRegularizers = [&](CBoostedTreeImpl& tree, double downsampleFactor) {
                    double scale{initialDownsampleFactor / downsampleFactor};
                    tree.m_Hyperparameters.depthPenaltyMultiplier().set(initialDepthPenaltyMultiplier);
                    tree.m_Hyperparameters.treeSizePenaltyMultiplier().set(
                        initialTreeSizePenaltyMultiplier);
                    tree.m_Hyperparameters.leafWeightPenaltyMultiplier().set(
                        initialLeafWeightPenaltyMultiplier);
                    tree.m_Hyperparameters.scaleRegularizerMultipliers(scale);
                    return scale;
                };

                double numberTrainingRows{m_TreeImpl->m_TrainingRowMasks[0].manhattan()};

                auto applyDownsampleFactor = [&](CBoostedTreeImpl& tree,
                                                 double logDownsampleFactor) {
                    double downsampleFactor{CTools::stableExp(logDownsampleFactor)};
                    tree.m_Hyperparameters.downsampleFactor().set(downsampleFactor);
                    scaleRegularizers(tree, downsampleFactor);
                    return tree.m_Hyperparameters.downsampleFactor().value() * numberTrainingRows >
                           10.0;
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
                                          0.0, SMALL_RELATIVE_TEST_LOSS_INCREASE * minTestLoss,
                                          logDownsampleFactor);
                };

                TVector fallback;
                fallback(MIN_PARAMETER_INDEX) = logMinDownsampleFactor;
                fallback(BEST_PARAMETER_INDEX) = meanLogDownSampleFactor;
                fallback(MAX_PARAMETER_INDEX) = logMaxDownsampleFactor;

                m_LogDownsampleFactorSearchInterval =
                    this->testLossLineSearch(frame, applyDownsampleFactor, logMinDownsampleFactor,
                                             logMaxDownsampleFactor, adjustTestLoss)
                        .value_or(fallback);

                // Truncate the log(factor) to be less than or equal to log(1.0) and the
                // downsampled set contains at least ten examples on average.
                m_LogDownsampleFactorSearchInterval =
                    min(max(m_LogDownsampleFactorSearchInterval,
                            TVector{CTools::stableLog(10.0 / numberTrainingRows)}),
                        TVector{0.0});
                LOG_TRACE(<< "log downsample factor search interval = ["
                          << m_LogDownsampleFactorSearchInterval.toDelimited() << "]");

                hyperparameters.downsampleFactor().set(CTools::stableExp(
                    m_LogDownsampleFactorSearchInterval(BEST_PARAMETER_INDEX)));

                TVector logScale{CTools::stableLog(scaleRegularizers(
                    *m_TreeImpl, hyperparameters.downsampleFactor().value()))};
                m_LogTreeSizePenaltyMultiplierSearchInterval += logScale;
                m_LogLeafWeightPenaltyMultiplierSearchInterval += logScale;

                if (intervalIsEmpty(m_LogDownsampleFactorSearchInterval)) {
                    hyperparameters.downsampleFactor().fix();
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }
}

void CBoostedTreeFactory::initializeUnsetFeatureBagFraction(core::CDataFrame& frame) {

    auto& hyperparameters = m_TreeImpl->m_Hyperparameters;

    if (hyperparameters.featureBagFraction().fixed() == false) {
        if (this->skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_FeatureBagFractionInitialized, [&] {
                double searchIntervalSize{FEATURE_BAG_FRACTION_LINE_SEARCH_RANGE};
                double logMaxFeatureBagFraction{CTools::stableLog(
                    std::min(2.0 * hyperparameters.featureBagFraction().value(),
                             MAX_FEATURE_BAG_FRACTION))};
                double logMinFeatureBagFraction{logMaxFeatureBagFraction -
                                                CTools::stableLog(searchIntervalSize)};

                auto applyFeatureBagFraction = [&](CBoostedTreeImpl& tree,
                                                   double logFeatureBagFraction) {
                    tree.m_Hyperparameters.featureBagFraction().set(
                        CTools::stableExp(logFeatureBagFraction));
                    return tree.featureBagSize(
                               tree.m_Hyperparameters.featureBagFraction().value()) > 1;
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
                                          0.0, SMALL_RELATIVE_TEST_LOSS_INCREASE * minTestLoss,
                                          logFeatureBagFraction);
                };

                TVector fallback;
                fallback(MIN_PARAMETER_INDEX) = logMinFeatureBagFraction;
                fallback(BEST_PARAMETER_INDEX) = logMaxFeatureBagFraction;
                fallback(MAX_PARAMETER_INDEX) = logMaxFeatureBagFraction;
                m_LogFeatureBagFractionInterval =
                    this->testLossLineSearch(frame, applyFeatureBagFraction,
                                             logMinFeatureBagFraction,
                                             logMaxFeatureBagFraction, adjustTestLoss)
                        .value_or(fallback);

                // Truncate the log(fraction) to be less than or equal to log(MAX_FEATURE_BAG_FRACTION).
                m_LogFeatureBagFractionInterval =
                    min(m_LogFeatureBagFractionInterval,
                        TVector{CTools::stableLog(MAX_FEATURE_BAG_FRACTION)});
                LOG_TRACE(<< "log feature bag fraction search interval = ["
                          << m_LogFeatureBagFractionInterval.toDelimited() << "]");

                hyperparameters.featureBagFraction().set(CTools::stableExp(
                    m_LogFeatureBagFractionInterval(BEST_PARAMETER_INDEX)));

                if (intervalIsEmpty(m_LogFeatureBagFractionInterval)) {
                    hyperparameters.featureBagFraction().fix();
                }
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame));
        }
    }
}

void CBoostedTreeFactory::initializeUnsetEta(core::CDataFrame& frame) {

    auto& hyperparameters = m_TreeImpl->m_Hyperparameters;

    if (hyperparameters.eta().fixed() == false) {
        if (skipCheckpointIfAtOrAfter(CBoostedTreeImpl::E_EtaInitialized, [&] {
                double searchIntervalSize{5.0 * MAX_ETA_SCALE / MIN_ETA_SCALE};
                double logMaxEta{CTools::stableLog(std::sqrt(searchIntervalSize) *
                                                   hyperparameters.eta().value())};
                double logMinEta{logMaxEta - CTools::stableLog(searchIntervalSize)};
                double meanLogEta{(logMaxEta + logMinEta) / 2.0};
                LOG_TRACE(<< "mean log eta = " << meanLogEta);

                auto applyEta = [](CBoostedTreeImpl& tree, double eta) {
                    eta = CTools::stableExp(eta);
                    tree.m_Hyperparameters.eta().set(eta);
                    tree.m_Hyperparameters.etaGrowthRatePerTree().set(1.0 + eta / 2.0);
                    tree.m_Hyperparameters.maximumNumberTrees().set(
                        computeMaximumNumberTrees(eta));
                    return true;
                };

                TVector fallback;
                fallback(MIN_PARAMETER_INDEX) = logMinEta;
                fallback(BEST_PARAMETER_INDEX) = meanLogEta;
                fallback(MAX_PARAMETER_INDEX) = logMaxEta;

                m_LogEtaSearchInterval =
                    this->testLossLineSearch(frame, applyEta, logMinEta, logMaxEta)
                        .value_or(fallback);
                m_LogEtaSearchInterval = min(m_LogEtaSearchInterval, TVector{0.0});
                LOG_TRACE(<< "log eta search interval = ["
                          << m_LogEtaSearchInterval.toDelimited() << "]");
                applyEta(*m_TreeImpl, m_LogEtaSearchInterval(BEST_PARAMETER_INDEX));

                if (intervalIsEmpty(m_LogEtaSearchInterval)) {
                    hyperparameters.eta().fix();
                }
                hyperparameters.maximumNumberTrees().set(computeMaximumNumberTrees(
                    MIN_ETA_SCALE * hyperparameters.eta().value()));
            })) {
            m_TreeImpl->m_TrainingProgress.increment(
                this->lineSearchMaximumNumberIterations(frame, 0.5));
        }
    }
}

void CBoostedTreeFactory::initializeUnsetRetrainedTreeEta() {
    if (m_TreeImpl->m_Hyperparameters.retrainedTreeEta().fixed() == false) {
        // The incremental loss function keeps the leaf weights around the
        // magnitude of the old tree leaf weights so we search larger values
        // of eta for trees we retrain.
        m_LogRetrainedTreeEtaSearchInterval(MIN_PARAMETER_INDEX) =
            CTools::stableLog(m_TreeImpl->m_Hyperparameters.eta().value());
        m_LogRetrainedTreeEtaSearchInterval(BEST_PARAMETER_INDEX) = 0.0;
        m_LogRetrainedTreeEtaSearchInterval(MAX_PARAMETER_INDEX) = 0.0;
    }
}

void CBoostedTreeFactory::initializeUnsetTreeTopologyPenalty(core::CDataFrame& frame) {

    if (m_TreeImpl->m_Hyperparameters.treeTopologyChangePenalty().fixed() == false) {

        auto forest = m_TreeImpl
                          ->updateForest(frame, m_TreeImpl->m_TrainingRowMasks[0],
                                         m_TreeImpl->m_TestingRowMasks[0],
                                         m_TreeImpl->m_TrainingProgress)
                          .s_Forest;
        CFastQuantileSketch quantiles{CQuantileSketch::E_Linear, 50};
        for (const auto& tree : forest) {
            for (const auto& node : tree) {
                if (node.isLeaf() == false) {
                    quantiles.add(node.gainVariance());
                }
            }
        }

        if (quantiles.count() > 0) {
            // We use the best forest internal gain percentiles to bound the range to search
            // for the penalty. This ensures we search a range which encompasses the penalty
            // having little impact on split selected to strongly resisting changing the tree.

            double gainVariancePercentiles[3];
            quantiles.quantile(1.0, gainVariancePercentiles[0]);
            quantiles.quantile(50.0, gainVariancePercentiles[1]);
            quantiles.quantile(90.0, gainVariancePercentiles[2]);
            LOG_TRACE(<< "gain variances = "
                      << core::CContainerPrinter::print(gainVariancePercentiles));

            auto postiveGain =
                std::find_if(gainVariancePercentiles, gainVariancePercentiles + 3,
                             [](auto gainVariance) { return gainVariance > 0.0; });

            if (postiveGain != gainVariancePercentiles + 3) {
                double minimumGain{0.1 * *postiveGain};
                gainVariancePercentiles[0] =
                    CTools::stableLog(0.5) +
                    0.5 * CTools::stableLog(std::max(gainVariancePercentiles[0], minimumGain));
                gainVariancePercentiles[1] =
                    0.5 * CTools::stableLog(std::max(gainVariancePercentiles[1], minimumGain));
                gainVariancePercentiles[2] =
                    CTools::stableLog(3.0) +
                    0.5 * CTools::stableLog(std::max(gainVariancePercentiles[2], minimumGain));
                m_TreeImpl->m_Hyperparameters.treeTopologyChangePenalty().set(
                    CTools::stableExp(gainVariancePercentiles[1]));
                m_LogTreeTopologyChangePenaltySearchInterval = TVector{gainVariancePercentiles};
            }
        }

        if (intervalIsEmpty(m_LogTreeTopologyChangePenaltySearchInterval)) {
            m_TreeImpl->m_Hyperparameters.treeTopologyChangePenalty().fix(0.0);
        }
    }
}

CBoostedTreeFactory::TDoubleDoublePrVec
CBoostedTreeFactory::estimateTreeGainAndCurvature(core::CDataFrame& frame,
                                                  const TDoubleVec& percentiles) const {

    CBoostedTreeHyperparameters::CScopeForceSetMaximumNumberTrees to{
        1, m_TreeImpl->m_Hyperparameters};
    CBoostedTreeImpl::TNodeVecVec forest{
        m_TreeImpl
            ->trainForest(frame, m_TreeImpl->m_TrainingRowMasks[0],
                          m_TreeImpl->m_TestingRowMasks[0], m_TreeImpl->m_TrainingProgress)
            .s_Forest};

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

    using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;

    TMinAccumulator minTestLoss;
    TDoubleDoublePrVec testLosses;
    testLosses.reserve(MAX_LINE_SEARCH_ITERATIONS);
    // Ensure we choose one value based on expected improvement.
    std::size_t minNumberTestLosses{6};

    for (auto parameter :
         {intervalLeftEnd, (2.0 * intervalLeftEnd + intervalRightEnd) / 3.0,
          (intervalLeftEnd + 2.0 * intervalRightEnd) / 3.0, intervalRightEnd}) {
        if (applyParameter(*m_TreeImpl, parameter) == false) {
            m_TreeImpl->m_TrainingProgress.increment(
                (MAX_LINE_SEARCH_ITERATIONS - testLosses.size()) *
                m_TreeImpl->m_Hyperparameters.maximumNumberTrees().value());
            break;
        }

        CBoostedTreeImpl::TNodeVecVec forest;
        double testLoss;
        std::tie(forest, testLoss, std::ignore, std::ignore) =
            m_TreeImpl
                ->trainForest(frame, m_TreeImpl->m_TrainingRowMasks[0],
                              m_TreeImpl->m_TestingRowMasks[0], m_TreeImpl->m_TrainingProgress)
                .asTuple();
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
                (MAX_LINE_SEARCH_ITERATIONS - testLosses.size()) *
                m_TreeImpl->m_Hyperparameters.maximumNumberTrees().value());
            break;
        }

        CBoostedTreeImpl::TNodeVecVec forest;
        double testLoss;
        std::tie(forest, testLoss, std::ignore, std::ignore) =
            m_TreeImpl
                ->trainForest(frame, m_TreeImpl->m_TrainingRowMasks[0],
                              m_TreeImpl->m_TestingRowMasks[0], m_TreeImpl->m_TrainingProgress)
                .asTuple();

        minTestLoss.add(testLoss);

        double adjustedTestLoss{adjustTestLoss(parameter(0), testLoss)};
        bopt.add(parameter, adjustedTestLoss, 0.0);
        testLosses.emplace_back(parameter(0), adjustedTestLoss);
    }

    std::sort(testLosses.begin(), testLosses.end());
    LOG_TRACE(<< "test losses = " << core::CContainerPrinter::print(testLosses));

    CLowess<2> lowess;
    lowess.fit(std::move(testLosses), testLosses.size());

    double bestParameter;
    double bestParameterTestLoss;
    std::tie(bestParameter, bestParameterTestLoss) = lowess.minimum();
    LOG_TRACE(<< "best parameter = " << bestParameter << ", test loss = " << bestParameterTestLoss);

    double width{(intervalRightEnd - intervalLeftEnd) /
                 static_cast<double>(MAX_LINE_SEARCH_ITERATIONS)};
    intervalLeftEnd = bestParameter - width;
    intervalRightEnd = bestParameter + width;
    LOG_TRACE(<< "interval = [" << intervalLeftEnd << "," << intervalRightEnd << "]");

    return TVector{{intervalLeftEnd, bestParameter, intervalRightEnd}};
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
    auto& hyperparameters = result.m_TreeImpl->m_Hyperparameters;
    result.m_TreeImpl = std::move(model->m_Impl);
    result.m_TreeImpl->m_Rng.seed(result.m_TreeImpl->m_Seed);
    hyperparameters.depthPenaltyMultiplier().fix();
    hyperparameters.treeSizePenaltyMultiplier().fix();
    hyperparameters.leafWeightPenaltyMultiplier().fix();
    hyperparameters.softTreeDepthLimit().fix();
    hyperparameters.softTreeDepthTolerance().fix();
    hyperparameters.downsampleFactor().fix();
    hyperparameters.eta().fix();
    hyperparameters.etaGrowthRatePerTree().fix();
    hyperparameters.featureBagFraction().fix();
    hyperparameters.resetSearch();
    result.m_TreeImpl->m_PreviousTrainNumberRows = static_cast<std::size_t>(
        result.m_TreeImpl->allTrainingRowsMask().manhattan());
    result.m_TreeImpl->m_PreviousTrainLossGap = hyperparameters.bestForestLossGap();
    result.m_TreeImpl->m_FoldRoundTestLosses.clear();
    result.m_TreeImpl->m_InitializationStage = CBoostedTreeImpl::E_NotInitialized;
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

CBoostedTreeFactory& CBoostedTreeFactory::seed(std::uint64_t seed) {
    m_TreeImpl->m_Seed = seed;
    m_TreeImpl->m_Rng.seed(seed);
    return *this;
}

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
    m_TreeImpl->m_NumberFolds.fix(numberFolds);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::trainFractionPerFold(double fraction) {
    if (fraction <= 0.0 || fraction >= 1.0) {
        LOG_WARN(<< "Training data fraction " << fraction << " per fold out of range");
    } else {
        m_TreeImpl->m_TrainFractionPerFold.fix(fraction);
    }
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::maximumNumberTrainRows(std::size_t rows) {
    m_MaximumNumberOfTrainRows = rows;
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
    m_TreeImpl->m_Hyperparameters.downsampleFactor().fix(factor);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::depthPenaltyMultiplier(double depthPenaltyMultiplier) {
    if (depthPenaltyMultiplier < 0.0) {
        LOG_WARN(<< "Depth penalty multiplier must be non-negative");
        depthPenaltyMultiplier = 0.0;
    }
    m_TreeImpl->m_Hyperparameters.depthPenaltyMultiplier().fix(depthPenaltyMultiplier);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::treeSizePenaltyMultiplier(double treeSizePenaltyMultiplier) {
    if (treeSizePenaltyMultiplier < 0.0) {
        LOG_WARN(<< "Tree size penalty multiplier must be non-negative");
        treeSizePenaltyMultiplier = 0.0;
    }
    m_TreeImpl->m_Hyperparameters.treeSizePenaltyMultiplier().fix(treeSizePenaltyMultiplier);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::leafWeightPenaltyMultiplier(double leafWeightPenaltyMultiplier) {
    if (leafWeightPenaltyMultiplier < 0.0) {
        LOG_WARN(<< "Leaf weight penalty multiplier must be non-negative");
        leafWeightPenaltyMultiplier = 0.0;
    }
    m_TreeImpl->m_Hyperparameters.leafWeightPenaltyMultiplier().fix(leafWeightPenaltyMultiplier);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::treeTopologyChangePenalty(double treeTopologyChangePenalty) {
    if (treeTopologyChangePenalty < 0.0) {
        LOG_WARN(<< "tree topology change penalty must be non-negative");
        treeTopologyChangePenalty = 0.0;
    }
    m_TreeImpl->m_Hyperparameters.treeTopologyChangePenalty().fix(treeTopologyChangePenalty);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::softTreeDepthLimit(double softTreeDepthLimit) {
    if (softTreeDepthLimit < MIN_SOFT_DEPTH_LIMIT) {
        LOG_WARN(<< "Minimum tree depth must be at least two");
        softTreeDepthLimit = MIN_SOFT_DEPTH_LIMIT;
    }
    m_TreeImpl->m_Hyperparameters.softTreeDepthLimit().fix(softTreeDepthLimit);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::softTreeDepthTolerance(double softTreeDepthTolerance) {
    if (softTreeDepthTolerance < 0.01) {
        LOG_WARN(<< "Minimum tree depth tolerance must be at least 0.01");
        softTreeDepthTolerance = 0.01;
    }
    m_TreeImpl->m_Hyperparameters.softTreeDepthTolerance().fix(softTreeDepthTolerance);
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
    m_TreeImpl->m_Hyperparameters.eta().fix(eta);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::retrainedTreeEta(double eta) {
    if (eta < MIN_ETA) {
        LOG_WARN(<< "Truncating supplied learning rate " << eta
                 << " which must be no smaller than " << MIN_ETA);
        eta = std::max(eta, MIN_ETA);
    }
    if (eta > 1.0) {
        LOG_WARN(<< "Using a learning rate greater than one doesn't make sense");
        eta = 1.0;
    }
    m_TreeImpl->m_Hyperparameters.retrainedTreeEta().fix(eta);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::etaGrowthRatePerTree(double etaGrowthRatePerTree) {
    if (etaGrowthRatePerTree < MIN_ETA) {
        LOG_WARN(<< "Truncating supplied learning rate growth rate " << etaGrowthRatePerTree
                 << " which must be no smaller than " << MIN_ETA);
        etaGrowthRatePerTree = std::max(etaGrowthRatePerTree, MIN_ETA);
    }
    m_TreeImpl->m_Hyperparameters.etaGrowthRatePerTree().fix(etaGrowthRatePerTree);
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
    m_TreeImpl->m_Hyperparameters.maximumNumberTrees().fix(maximumNumberTrees);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::featureBagFraction(double featureBagFraction) {
    if (featureBagFraction < 0.0 || featureBagFraction > 1.0) {
        LOG_WARN(<< "Truncating supplied feature bag fraction " << featureBagFraction
                 << " which must be positive and not more than one");
        featureBagFraction = CTools::truncate(featureBagFraction, 0.0, 1.0);
    }
    m_TreeImpl->m_Hyperparameters.featureBagFraction().fix(featureBagFraction);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::predictionChangeCost(double predictionChangeCost) {
    if (predictionChangeCost < 0.0) {
        LOG_WARN(<< "tree topology change penalty must be non-negative");
        predictionChangeCost = 0.0;
    }
    m_TreeImpl->m_Hyperparameters.predictionChangeCost().fix(predictionChangeCost);
    return *this;
}

CBoostedTreeFactory&
CBoostedTreeFactory::maximumOptimisationRoundsPerHyperparameter(std::size_t rounds) {
    m_TreeImpl->m_Hyperparameters.maximumOptimisationRoundsPerHyperparameter(rounds);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::bayesianOptimisationRestarts(std::size_t restarts) {
    m_TreeImpl->m_Hyperparameters.bayesianOptimisationRestarts(
        std::max(restarts, std::size_t{1}));
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
    m_TreeImpl->m_Hyperparameters.stopHyperparameterOptimizationEarly(enable);
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::dataSummarizationFraction(double fraction) {
    m_TreeImpl->m_DataSummarizationFraction = fraction;
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

CBoostedTreeFactory& CBoostedTreeFactory::previousTrainLossGap(double gap) {
    m_TreeImpl->m_PreviousTrainLossGap = gap;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::previousTrainNumberRows(std::size_t numberRows) {
    m_TreeImpl->m_PreviousTrainNumberRows = numberRows;
    return *this;
}

CBoostedTreeFactory& CBoostedTreeFactory::forceAcceptIncrementalTraining(bool force) {
    m_TreeImpl->m_ForceAcceptIncrementalTraining = force;
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
        m_TreeImpl->m_Hyperparameters.eta().fixed()
            ? m_TreeImpl->m_Hyperparameters.eta().value()
            : computeEta(numberColumns))};
    CBoostedTreeHyperparameters::CScopeForceSetMaximumNumberTrees to{
        maximumNumberTrees, m_TreeImpl->m_Hyperparameters};
    std::size_t result{m_TreeImpl->estimateMemoryUsageTrain(numberRows, numberColumns)};
    return result;
}

std::size_t
CBoostedTreeFactory::estimateMemoryUsageTrainIncremental(std::size_t numberRows,
                                                         std::size_t numberColumns) const {
    return m_TreeImpl->estimateMemoryUsageTrainIncremental(numberRows, numberColumns);
}

std::size_t CBoostedTreeFactory::estimatedExtraColumnsForTrain(std::size_t numberColumns,
                                                               std::size_t numberLossParameters) {
    // We store as follows:
    //   1. The predicted values for the dependent variable
    //   2. The gradient of the loss function
    //   3. The upper triangle of the hessian of the loss function
    //   4. The example's weight
    //   5. The example's splits packed into uint8_t
    return numberLossParameters * (numberLossParameters + 5) / 2 + 1 +
           (numberColumns + 2) / 4;
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
    auto& hyperparameters = m_TreeImpl->m_Hyperparameters;

    std::size_t totalNumberSteps{0};
    if (hyperparameters.softTreeDepthLimit().fixed() == false) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (hyperparameters.depthPenaltyMultiplier().fixed() == false) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (hyperparameters.treeSizePenaltyMultiplier().fixed() == false) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (hyperparameters.leafWeightPenaltyMultiplier().fixed() == false) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (hyperparameters.featureBagFraction().fixed() == false) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (hyperparameters.downsampleFactor().fixed() == false) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame);
    }
    if (hyperparameters.eta().fixed() == false) {
        totalNumberSteps += this->lineSearchMaximumNumberIterations(frame, 0.5);
    }
    if (hyperparameters.treeTopologyChangePenalty().fixed() == false) {
        totalNumberSteps += m_TreeImpl->m_TreesToRetrain.size();
    }

    LOG_TRACE(<< "initial search total number steps = " << totalNumberSteps);
    m_TreeImpl->m_TrainingProgress = core::CLoopProgress{
        totalNumberSteps, m_TreeImpl->m_Instrumentation->progressCallback(), 1.0, 1024};
}

std::size_t
CBoostedTreeFactory::lineSearchMaximumNumberIterations(const core::CDataFrame& frame,
                                                       double etaScale) const {
    double eta{m_TreeImpl->m_Hyperparameters.eta().fixed()
                   ? m_TreeImpl->m_Hyperparameters.eta().value()
                   : computeEta(frame.numberColumns() - m_PaddedExtraColumns)};
    return MAX_LINE_SEARCH_ITERATIONS * computeMaximumNumberTrees(etaScale * eta);
}

std::size_t CBoostedTreeFactory::mainLoopMaximumNumberTrees(double eta) const {
    return m_TreeImpl->m_Hyperparameters.maximumNumberTrees().fixed()
               ? m_TreeImpl->m_Hyperparameters.maximumNumberTrees().value()
               : computeMaximumNumberTrees(MIN_ETA_SCALE * eta);
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
const std::string CBoostedTreeFactory::INCREMENTAL_TRAIN{"incremental_train"};
}
}
