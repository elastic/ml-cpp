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

#include <maths/CBoostedTreeHyperparameters.h>

#include <core/CContainerPrinter.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CBayesianOptimisation.h>
#include <maths/CDataFrameAnalysisInstrumentationInterface.h>
#include <maths/CLinearAlgebra.h>

#include <cmath>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;

CBoostedTreeHyperparameters::CBoostedTreeHyperparameters() {
    this->saveCurrent();
    this->initializeTunableHyperparameters();
}

double CBoostedTreeHyperparameters::penaltyForDepth(std::size_t depth) const {
    return std::exp((static_cast<double>(depth) / m_SoftTreeDepthLimit.value() - 1.0) /
                    m_SoftTreeDepthTolerance.value());
}

void CBoostedTreeHyperparameters::scaleRegularizationMultipliers(
    double scale,
    CScopeBoostedTreeParameterOverrides<double>& overrides,
    bool undo) {
    overrides.apply(m_DepthPenaltyMultiplier,
                    scale * m_DepthPenaltyMultiplier.value(), undo);
    overrides.apply(m_TreeSizePenaltyMultiplier,
                    scale * m_TreeSizePenaltyMultiplier.value(), undo);
    overrides.apply(m_LeafWeightPenaltyMultiplier,
                    scale * m_LeafWeightPenaltyMultiplier.value(), undo);
}

void CBoostedTreeHyperparameters::maximumOptimisationRoundsPerHyperparameter(std::size_t rounds) {
    m_MaximumOptimisationRoundsPerHyperparameter = rounds;
}

void CBoostedTreeHyperparameters::stopHyperparameterOptimizationEarly(bool enable) {
    m_StopHyperparameterOptimizationEarly = enable;
}

void CBoostedTreeHyperparameters::bayesianOptimisationRestarts(std::size_t restarts) {
    m_BayesianOptimisationRestarts = restarts;
}

std::size_t CBoostedTreeHyperparameters::numberToTune() const {
    std::size_t result((m_DepthPenaltyMultiplier.fixed() ? 0 : 1) +
                       (m_TreeSizePenaltyMultiplier.fixed() ? 0 : 1) +
                       (m_LeafWeightPenaltyMultiplier.fixed() ? 0 : 1) +
                       (m_SoftTreeDepthLimit.fixed() ? 0 : 1) +
                       (m_SoftTreeDepthTolerance.fixed() ? 0 : 1) +
                       (m_DownsampleFactor.fixed() ? 0 : 1) +
                       (m_FeatureBagFraction.fixed() ? 0 : 1) + (m_Eta.fixed() ? 0 : 1) +
                       (m_EtaGrowthRatePerTree.fixed() ? 0 : 1));
    if (m_IncrementalTraining) {
        result += (m_TreeTopologyChangePenalty.fixed() ? 0 : 1) +
                  (m_PredictionChangeCost.fixed() ? 0 : 1) +
                  (m_RetrainedTreeEta.fixed() ? 0 : 1);
    }
    return result;
}

void CBoostedTreeHyperparameters::resetSearch() {
    m_CurrentRound = 0;
    m_BestForestTestLoss = boosted_tree_detail::INF;
    m_MeanForestSizeAccumulator = TMeanAccumulator{};
    m_MeanTestLossAccumulator = TMeanAccumulator{};
    this->initializeTunableHyperparameters();
}

void CBoostedTreeHyperparameters::initializeSearch(const TAddInitialRangeFunc& addInitialRange) {

    this->initializeTunableHyperparameters();

    TDoubleDoublePrVec boundingBox;
    boundingBox.reserve(m_TunableHyperparameters.size());
    for (const auto& parameter : m_TunableHyperparameters) {
        addInitialRange(parameter, boundingBox);
    }
    LOG_TRACE(<< "hyperparameter search bounding box = "
              << core::CContainerPrinter::print(boundingBox));

    m_BayesianOptimization = std::make_unique<CBayesianOptimisation>(
        std::move(boundingBox),
        m_BayesianOptimisationRestarts.value_or(CBayesianOptimisation::RESTARTS));

    m_CurrentRound = 0;
    m_NumberRounds = m_MaximumOptimisationRoundsPerHyperparameter *
                     m_TunableHyperparameters.size();
    this->saveCurrent();
}

void CBoostedTreeHyperparameters::startSearch() {
    std::size_t dimension{m_TunableHyperparameters.size()};
    std::size_t n{m_NumberRounds / 3 + 1};
    CSampling::sobolSequenceSample(dimension, n, m_HyperparameterSamples);
}

void CBoostedTreeHyperparameters::addRoundStats(const TMeanAccumulator& meanForestSizeAccumulator,
                                                double meanTestLoss) {
    m_MeanForestSizeAccumulator += meanForestSizeAccumulator;
    m_MeanTestLossAccumulator.add(meanTestLoss);
}

bool CBoostedTreeHyperparameters::selectNext(const TMeanVarAccumulator& testLossMoments,
                                             double explainedVariance) {

    using TVector = CDenseVector<double>;

    TVector parameters{m_TunableHyperparameters.size()};

    TVector minBoundary;
    TVector maxBoundary;
    std::tie(minBoundary, maxBoundary) = m_BayesianOptimization->boundingBox();

    // Downsampling directly affects the loss terms: it multiplies the sums over
    // gradients and Hessians in expectation by the downsample factor. To preserve
    // the same effect for regularisers we need to scale these terms by the same
    // multiplier.
    double scale{1.0};
    if (m_DownsampleFactor.fixed() == false) {
        auto i = std::distance(m_TunableHyperparameters.begin(),
                               std::find(m_TunableHyperparameters.begin(),
                                         m_TunableHyperparameters.end(), E_DownsampleFactor));
        if (static_cast<std::size_t>(i) < m_TunableHyperparameters.size()) {
            scale = std::min(1.0, 2.0 * m_DownsampleFactor.value() /
                                      (CTools::stableExp(minBoundary(i)) +
                                       CTools::stableExp(maxBoundary(i))));
        }
    }

    // Read parameters for last round.
    for (std::size_t i = 0; i < m_TunableHyperparameters.size(); ++i) {
        switch (m_TunableHyperparameters[i]) {
        case E_Alpha:
            parameters(i) = CTools::stableLog(m_DepthPenaltyMultiplier.value() / scale);
            break;
        case E_DownsampleFactor:
            parameters(i) = CTools::stableLog(m_DownsampleFactor.value());
            break;
        case E_Eta:
            parameters(i) = CTools::stableLog(m_Eta.value());
            break;
        case E_EtaGrowthRatePerTree:
            parameters(i) = m_EtaGrowthRatePerTree.value();
            break;
        case E_FeatureBagFraction:
            parameters(i) = m_FeatureBagFraction.value();
            break;
        case E_MaximumNumberTrees:
            parameters(i) = static_cast<double>(m_MaximumNumberTrees.value());
            break;
        case E_Gamma:
            parameters(i) = CTools::stableLog(m_TreeSizePenaltyMultiplier.value() / scale);
            break;
        case E_Lambda:
            parameters(i) = CTools::stableLog(m_LeafWeightPenaltyMultiplier.value() / scale);
            break;
        case E_SoftTreeDepthLimit:
            parameters(i) = m_SoftTreeDepthLimit.value();
            break;
        case E_SoftTreeDepthTolerance:
            parameters(i) = m_SoftTreeDepthTolerance.value();
            break;
        case E_PredictionChangeCost:
            parameters(i) = CTools::stableLog(m_PredictionChangeCost.value());
            break;
        case E_RetrainedTreeEta:
            parameters(i) = CTools::stableLog(m_RetrainedTreeEta.value());
            break;
        case E_TreeTopologyChangePenalty:
            parameters(i) = CTools::stableLog(m_TreeTopologyChangePenalty.value());
            break;
        }
    }

    double meanTestLoss{CBasicStatistics::mean(testLossMoments)};
    double testLossVariance{CBasicStatistics::variance(testLossMoments)};

    LOG_TRACE(<< "round = " << m_CurrentRound << ", loss = " << meanTestLoss
              << ", total variance = " << testLossVariance
              << ", explained variance = " << explainedVariance);
    LOG_TRACE(<< "parameters = " << this->print());

    m_BayesianOptimization->add(parameters, meanTestLoss, testLossVariance);

    // One fold might have examples which are harder to predict on average than
    // another fold, particularly if the sample size is small. What we really care
    // about is the variation between fold loss values after accounting for any
    // systematic effect due to sampling. Running for multiple rounds allows us
    // to estimate this effect and we remove it when characterising the uncertainty
    // in the loss values in the Gaussian Process.
    m_BayesianOptimization->explainedErrorVariance(explainedVariance);

    if (m_CurrentRound < m_HyperparameterSamples.size()) {
        std::copy(m_HyperparameterSamples[m_CurrentRound].begin(),
                  m_HyperparameterSamples[m_CurrentRound].end(), parameters.data());
        parameters = minBoundary + parameters.cwiseProduct(maxBoundary - minBoundary);
    } else if (m_StopHyperparameterOptimizationEarly &&
               m_BayesianOptimization->anovaTotalVariance() < 1e-9) {
        return false;
    } else {
        std::tie(parameters, std::ignore) =
            m_BayesianOptimization->maximumExpectedImprovement();
    }

    // Write parameters for next round.
    if (m_DownsampleFactor.fixed() == false) {
        auto i = std::distance(m_TunableHyperparameters.begin(),
                               std::find(m_TunableHyperparameters.begin(),
                                         m_TunableHyperparameters.end(), E_DownsampleFactor));
        if (static_cast<std::size_t>(i) < m_TunableHyperparameters.size()) {
            scale = std::min(1.0, 2.0 * CTools::stableExp(parameters(i)) /
                                      (CTools::stableExp(minBoundary(i)) +
                                       CTools::stableExp(maxBoundary(i))));
        }
    }
    for (std::size_t i = 0; i < m_TunableHyperparameters.size(); ++i) {
        switch (m_TunableHyperparameters[i]) {
        case E_Alpha:
            m_DepthPenaltyMultiplier.set(scale * CTools::stableExp(parameters(i)));
            break;
        case E_DownsampleFactor:
            m_DownsampleFactor.set(CTools::stableExp(parameters(i)));
            break;
        case E_Eta:
            m_Eta.set(CTools::stableExp(parameters(i)));
            break;
        case E_EtaGrowthRatePerTree:
            m_EtaGrowthRatePerTree.set(parameters(i));
            break;
        case E_FeatureBagFraction:
            m_FeatureBagFraction.set(parameters(i));
            break;
        case E_MaximumNumberTrees:
            m_MaximumNumberTrees.set(static_cast<std::size_t>(std::ceil(parameters(i))));
            break;
        case E_Gamma:
            m_TreeSizePenaltyMultiplier.set(scale * CTools::stableExp(parameters(i)));
            break;
        case E_Lambda:
            m_LeafWeightPenaltyMultiplier.set(scale * CTools::stableExp(parameters(i)));
            break;
        case E_SoftTreeDepthLimit:
            m_SoftTreeDepthLimit.set(std::max(parameters(i), 2.0));
            break;
        case E_SoftTreeDepthTolerance:
            m_SoftTreeDepthTolerance.set(parameters(i));
            break;
        case E_PredictionChangeCost:
            m_PredictionChangeCost.set(CTools::stableExp(parameters(i)));
            break;
        case E_RetrainedTreeEta:
            m_RetrainedTreeEta.set(CTools::stableExp(parameters(i)));
            break;
        case E_TreeTopologyChangePenalty:
            m_TreeTopologyChangePenalty.set(CTools::stableExp(parameters(i)));
            break;
        }
    }

    return true;
}

void CBoostedTreeHyperparameters::captureBest(const TMeanVarAccumulator& testLossMoments,
                                              double meanLossGap,
                                              double numberKeptNodes,
                                              double numberNewNodes,
                                              std::size_t numberTrees) {

    // We capture the parameters with the lowest error at one standard
    // deviation above the mean. If the mean error improvement is marginal
    // we prefer the solution with the least variation across the folds.
    double testLoss{lossAtNSigma(1.0, testLossMoments) +
                    this->modelSizePenalty(numberKeptNodes, numberNewNodes)};

    if (testLoss < m_BestForestTestLoss) {
        m_BestForestTestLoss = testLoss;
        m_BestForestLossGap = meanLossGap;

        // During hyperparameter search we have a fixed upper bound on
        // the number of trees which we use for every round and we stop
        // adding trees early based on cross-validation loss. The stored
        // number of trees is used for final train when we train on all
        // the data and so can't measure when to stop.
        std::size_t numberTreesToRestore{m_MaximumNumberTrees.value()};
        m_MaximumNumberTrees.set(numberTrees);
        this->saveCurrent();
        m_MaximumNumberTrees.set(numberTreesToRestore);
    }
}

double CBoostedTreeHyperparameters::modelSizePenalty(double numberKeptNodes,
                                                     double numberNewNodes) const {
    // eps * "forest number nodes" * E[GP] / "average forest number nodes" to meanLoss.
    return RELATIVE_SIZE_PENALTY * (numberKeptNodes + numberNewNodes) /
           (numberKeptNodes + CBasicStatistics::mean(m_MeanForestSizeAccumulator)) *
           CBasicStatistics::mean(m_MeanTestLossAccumulator);
}

CBoostedTreeHyperparameters::THyperparameterImportanceVec
CBoostedTreeHyperparameters::importances() const {
    THyperparameterImportanceVec importances;
    importances.reserve(m_TunableHyperparameters.size());
    CBayesianOptimisation::TDoubleDoublePrVec anovaMainEffects{
        m_BayesianOptimization->anovaMainEffects()};
    for (std::size_t i = 0; i < static_cast<std::size_t>(NUMBER_HYPERPARAMETERS); ++i) {
        double absoluteImportance{0.0};
        double relativeImportance{0.0};
        double hyperparameterValue;
        SHyperparameterImportance::EType hyperparameterType{
            boosted_tree_detail::SHyperparameterImportance::E_Double};
        switch (static_cast<EHyperparameter>(i)) {
        case E_Alpha:
            hyperparameterValue = m_DepthPenaltyMultiplier.value();
            break;
        case E_DownsampleFactor:
            hyperparameterValue = m_DownsampleFactor.value();
            break;
        case E_Eta:
            hyperparameterValue = m_Eta.value();
            break;
        case E_EtaGrowthRatePerTree:
            hyperparameterValue = m_EtaGrowthRatePerTree.value();
            break;
        case E_FeatureBagFraction:
            hyperparameterValue = m_FeatureBagFraction.value();
            break;
        case E_MaximumNumberTrees:
            hyperparameterValue = static_cast<double>(m_MaximumNumberTrees.value());
            hyperparameterType = boosted_tree_detail::SHyperparameterImportance::E_Uint64;
            break;
        case E_Gamma:
            hyperparameterValue = m_TreeSizePenaltyMultiplier.value();
            break;
        case E_Lambda:
            hyperparameterValue = m_LeafWeightPenaltyMultiplier.value();
            break;
        case E_SoftTreeDepthLimit:
            hyperparameterValue = m_SoftTreeDepthLimit.value();
            break;
        case E_SoftTreeDepthTolerance:
            hyperparameterValue = m_SoftTreeDepthTolerance.value();
            break;
        case E_PredictionChangeCost:
            hyperparameterValue = m_PredictionChangeCost.value();
            break;
        case E_RetrainedTreeEta:
            hyperparameterValue = m_RetrainedTreeEta.value();
            break;
        case E_TreeTopologyChangePenalty:
            hyperparameterValue = m_TreeTopologyChangePenalty.value();
            break;
        }
        bool supplied{true};
        auto tunableIndex = std::distance(m_TunableHyperparameters.begin(),
                                          std::find(m_TunableHyperparameters.begin(),
                                                    m_TunableHyperparameters.end(), i));
        if (static_cast<std::size_t>(tunableIndex) < m_TunableHyperparameters.size()) {
            supplied = false;
            std::tie(absoluteImportance, relativeImportance) = anovaMainEffects[tunableIndex];
        }
        importances.push_back({static_cast<EHyperparameter>(i), hyperparameterValue, absoluteImportance,
                               relativeImportance, supplied, hyperparameterType});
    }
    return importances;
}

void CBoostedTreeHyperparameters::recordHyperparameters(
    CDataFrameTrainBoostedTreeInstrumentationInterface& instrumentation) const {
    auto& hyperparameters = instrumentation.hyperparameters();
    hyperparameters.s_Eta = m_Eta.value();
    hyperparameters.s_RetrainedTreeEta = m_RetrainedTreeEta.value();
    hyperparameters.s_DepthPenaltyMultiplier = m_DepthPenaltyMultiplier.value();
    hyperparameters.s_SoftTreeDepthLimit = m_SoftTreeDepthLimit.value();
    hyperparameters.s_SoftTreeDepthTolerance = m_SoftTreeDepthTolerance.value();
    hyperparameters.s_TreeSizePenaltyMultiplier = m_TreeSizePenaltyMultiplier.value();
    hyperparameters.s_LeafWeightPenaltyMultiplier =
        m_LeafWeightPenaltyMultiplier.value();
    hyperparameters.s_TreeTopologyChangePenalty = m_TreeTopologyChangePenalty.value();
    hyperparameters.s_DownsampleFactor = m_DownsampleFactor.value();
    hyperparameters.s_MaxTrees = m_MaximumNumberTrees.value();
    hyperparameters.s_FeatureBagFraction = m_FeatureBagFraction.value();
    hyperparameters.s_PredictionChangeCost = m_PredictionChangeCost.value();
    hyperparameters.s_EtaGrowthRatePerTree = m_EtaGrowthRatePerTree.value();
    hyperparameters.s_MaxOptimizationRoundsPerHyperparameter = m_MaximumOptimisationRoundsPerHyperparameter;
}

void CBoostedTreeHyperparameters::checkSearchInvariants() const {
    if (m_BayesianOptimization == nullptr) {
        HANDLE_FATAL(<< "Internal error: must supply an optimizer. Please report this problem.");
    }
}

void CBoostedTreeHyperparameters::checkRestoredInvariants(bool expectOptimizerInitialized) const {
    if (expectOptimizerInitialized) {
        VIOLATES_INVARIANT_NO_EVALUATION(m_BayesianOptimization, ==, nullptr);
    }
    VIOLATES_INVARIANT(m_CurrentRound, >, m_NumberRounds);
    for (const auto& samples : m_HyperparameterSamples) {
        VIOLATES_INVARIANT(m_TunableHyperparameters.size(), !=, samples.size());
    }
}

std::size_t CBoostedTreeHyperparameters::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_TunableHyperparameters)};
    mem += core::CMemory::dynamicSize(m_HyperparameterSamples);
    mem += core::CMemory::dynamicSize(m_BayesianOptimization);
    return mem;
}

void CBoostedTreeHyperparameters::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persistIfNotNull(BAYESIAN_OPTIMIZATION_TAG,
                                          m_BayesianOptimization, inserter);
    core::CPersistUtils::persist(BEST_FOREST_LOSS_GAP_TAG, m_BestForestLossGap, inserter);
    core::CPersistUtils::persist(BEST_FOREST_TEST_LOSS_TAG, m_BestForestTestLoss, inserter);
    core::CPersistUtils::persist(CURRENT_ROUND_TAG, m_CurrentRound, inserter);
    core::CPersistUtils::persist(DEPTH_PENALTY_MULTIPLIER_TAG,
                                 m_DepthPenaltyMultiplier, inserter);
    core::CPersistUtils::persist(DOWNSAMPLE_FACTOR_TAG, m_DownsampleFactor, inserter);
    core::CPersistUtils::persist(ETA_GROWTH_RATE_PER_TREE_TAG,
                                 m_EtaGrowthRatePerTree, inserter);
    core::CPersistUtils::persist(ETA_TAG, m_Eta, inserter);
    core::CPersistUtils::persist(FEATURE_BAG_FRACTION_TAG, m_FeatureBagFraction, inserter);
    core::CPersistUtils::persist(LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
                                 m_LeafWeightPenaltyMultiplier, inserter);
    core::CPersistUtils::persist(MAXIMUM_NUMBER_TREES_TAG, m_MaximumNumberTrees, inserter);
    core::CPersistUtils::persist(MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                                 m_MaximumOptimisationRoundsPerHyperparameter, inserter);
    core::CPersistUtils::persist(MEAN_FOREST_SIZE_ACCUMULATOR_TAG,
                                 m_MeanForestSizeAccumulator, inserter);
    core::CPersistUtils::persist(MEAN_TEST_LOSS_ACCUMULATOR_TAG,
                                 m_MeanTestLossAccumulator, inserter);
    core::CPersistUtils::persist(NUMBER_ROUNDS_TAG, m_NumberRounds, inserter);
    core::CPersistUtils::persist(PREDICTION_CHANGE_COST_TAG, m_PredictionChangeCost, inserter);
    core::CPersistUtils::persist(RETRAINED_TREE_ETA_TAG, m_RetrainedTreeEta, inserter);
    core::CPersistUtils::persist(SOFT_TREE_DEPTH_LIMIT_TAG, m_SoftTreeDepthLimit, inserter);
    core::CPersistUtils::persist(SOFT_TREE_DEPTH_TOLERANCE_TAG,
                                 m_SoftTreeDepthTolerance, inserter);
    core::CPersistUtils::persist(STOP_HYPERPARAMETER_OPTIMIZATION_EARLY_TAG,
                                 m_StopHyperparameterOptimizationEarly, inserter);
    core::CPersistUtils::persist(TREE_SIZE_PENALTY_MULTIPLIER_TAG,
                                 m_TreeSizePenaltyMultiplier, inserter);
    core::CPersistUtils::persist(TREE_TOPOLOGY_CHANGE_PENALTY_TAG,
                                 m_TreeTopologyChangePenalty, inserter);
    // m_TunableHyperparameters is not persisted explicitly, it is re-generated
    // from overriden hyperparameters.
    // m_HyperparameterSamples is not persisted explicitly, it is re-generated.
}

bool CBoostedTreeHyperparameters::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_NO_ERROR(BAYESIAN_OPTIMIZATION_TAG,
                         m_BayesianOptimization =
                             std::make_unique<CBayesianOptimisation>(traverser))
        RESTORE(BEST_FOREST_LOSS_GAP_TAG,
                core::CPersistUtils::restore(BEST_FOREST_LOSS_GAP_TAG,
                                             m_BestForestLossGap, traverser))
        RESTORE(BEST_FOREST_TEST_LOSS_TAG,
                core::CPersistUtils::restore(BEST_FOREST_TEST_LOSS_TAG,
                                             m_BestForestTestLoss, traverser))
        RESTORE(CURRENT_ROUND_TAG,
                core::CPersistUtils::restore(CURRENT_ROUND_TAG, m_CurrentRound, traverser))
        RESTORE(DEPTH_PENALTY_MULTIPLIER_TAG,
                core::CPersistUtils::restore(DEPTH_PENALTY_MULTIPLIER_TAG,
                                             m_DepthPenaltyMultiplier, traverser))
        RESTORE(DOWNSAMPLE_FACTOR_TAG,
                core::CPersistUtils::restore(DOWNSAMPLE_FACTOR_TAG, m_DownsampleFactor, traverser))
        RESTORE(ETA_TAG, core::CPersistUtils::restore(ETA_TAG, m_Eta, traverser))
        RESTORE(ETA_GROWTH_RATE_PER_TREE_TAG,
                core::CPersistUtils::restore(ETA_GROWTH_RATE_PER_TREE_TAG,
                                             m_EtaGrowthRatePerTree, traverser))
        RESTORE(FEATURE_BAG_FRACTION_TAG,
                core::CPersistUtils::restore(FEATURE_BAG_FRACTION_TAG,
                                             m_FeatureBagFraction, traverser))
        RESTORE(LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
                core::CPersistUtils::restore(LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
                                             m_LeafWeightPenaltyMultiplier, traverser))
        RESTORE(MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                core::CPersistUtils::restore(
                    MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                    m_MaximumOptimisationRoundsPerHyperparameter, traverser))
        RESTORE(MAXIMUM_NUMBER_TREES_TAG,
                core::CPersistUtils::restore(MAXIMUM_NUMBER_TREES_TAG,
                                             m_MaximumNumberTrees, traverser))
        RESTORE(MEAN_FOREST_SIZE_ACCUMULATOR_TAG,
                core::CPersistUtils::restore(MEAN_FOREST_SIZE_ACCUMULATOR_TAG,
                                             m_MeanForestSizeAccumulator, traverser))
        RESTORE(MEAN_TEST_LOSS_ACCUMULATOR_TAG,
                core::CPersistUtils::restore(MEAN_TEST_LOSS_ACCUMULATOR_TAG,
                                             m_MeanTestLossAccumulator, traverser))
        RESTORE(NUMBER_ROUNDS_TAG,
                core::CPersistUtils::restore(NUMBER_ROUNDS_TAG, m_NumberRounds, traverser))
        RESTORE(PREDICTION_CHANGE_COST_TAG,
                core::CPersistUtils::restore(PREDICTION_CHANGE_COST_TAG,
                                             m_PredictionChangeCost, traverser))
        RESTORE(RETRAINED_TREE_ETA_TAG,
                core::CPersistUtils::restore(RETRAINED_TREE_ETA_TAG, m_RetrainedTreeEta, traverser))
        RESTORE(SOFT_TREE_DEPTH_LIMIT_TAG,
                core::CPersistUtils::restore(SOFT_TREE_DEPTH_LIMIT_TAG,
                                             m_SoftTreeDepthLimit, traverser))
        RESTORE(SOFT_TREE_DEPTH_TOLERANCE_TAG,
                core::CPersistUtils::restore(SOFT_TREE_DEPTH_TOLERANCE_TAG,
                                             m_SoftTreeDepthTolerance, traverser))
        RESTORE(STOP_HYPERPARAMETER_OPTIMIZATION_EARLY_TAG,
                core::CPersistUtils::restore(STOP_HYPERPARAMETER_OPTIMIZATION_EARLY_TAG,
                                             m_StopHyperparameterOptimizationEarly, traverser))
        RESTORE(TREE_SIZE_PENALTY_MULTIPLIER_TAG,
                core::CPersistUtils::restore(TREE_SIZE_PENALTY_MULTIPLIER_TAG,
                                             m_TreeSizePenaltyMultiplier, traverser))
        RESTORE(TREE_TOPOLOGY_CHANGE_PENALTY_TAG,
                core::CPersistUtils::restore(TREE_TOPOLOGY_CHANGE_PENALTY_TAG,
                                             m_TreeTopologyChangePenalty, traverser))
    } while (traverser.next());

    this->initializeTunableHyperparameters();

    return true;
}

std::uint64_t CBoostedTreeHyperparameters::checksum(std::uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_BayesianOptimization);
    seed = CChecksum::calculate(seed, m_BestForestLossGap);
    seed = CChecksum::calculate(seed, m_BestForestTestLoss);
    seed = CChecksum::calculate(seed, m_CurrentRound);
    seed = CChecksum::calculate(seed, m_DepthPenaltyMultiplier);
    seed = CChecksum::calculate(seed, m_DownsampleFactor);
    seed = CChecksum::calculate(seed, m_Eta);
    seed = CChecksum::calculate(seed, m_EtaGrowthRatePerTree);
    seed = CChecksum::calculate(seed, m_FeatureBagFraction);
    seed = CChecksum::calculate(seed, m_HyperparameterSamples);
    seed = CChecksum::calculate(seed, m_LeafWeightPenaltyMultiplier);
    seed = CChecksum::calculate(seed, m_MaximumNumberTrees);
    seed = CChecksum::calculate(seed, m_MaximumOptimisationRoundsPerHyperparameter);
    seed = CChecksum::calculate(seed, m_MeanForestSizeAccumulator);
    seed = CChecksum::calculate(seed, m_MeanTestLossAccumulator);
    seed = CChecksum::calculate(seed, m_NumberRounds);
    seed = CChecksum::calculate(seed, m_PredictionChangeCost);
    seed = CChecksum::calculate(seed, m_RetrainedTreeEta);
    seed = CChecksum::calculate(seed, m_SoftTreeDepthLimit);
    seed = CChecksum::calculate(seed, m_SoftTreeDepthTolerance);
    seed = CChecksum::calculate(seed, m_StopHyperparameterOptimizationEarly);
    seed = CChecksum::calculate(seed, m_TreeSizePenaltyMultiplier);
    seed = CChecksum::calculate(seed, m_TreeTopologyChangePenalty);
    return CChecksum::calculate(seed, m_TunableHyperparameters);
}

std::string CBoostedTreeHyperparameters::print() const {
    return "(\ndepth penalty multiplier = " + m_DepthPenaltyMultiplier.print() +
           "\nsoft depth limit = " + m_SoftTreeDepthLimit.print() +
           "\nsoft depth tolerance = " + m_SoftTreeDepthTolerance.print() +
           "\ntree size penalty multiplier = " + m_TreeSizePenaltyMultiplier.print() +
           "\nleaf weight penalty multiplier = " + m_LeafWeightPenaltyMultiplier.print() +
           "\ntree topology change penalty = " + m_TreeTopologyChangePenalty.print() +
           "\ndownsample factor = " + m_DownsampleFactor.print() +
           "\nfeature bag fraction = " + m_FeatureBagFraction.print() +
           "\neta = " + m_Eta.print() +
           "\neta growth rate per tree = " + m_EtaGrowthRatePerTree.print() +
           "\nretrained tree eta = " + m_RetrainedTreeEta.print() +
           "\nprediction change cost = " + m_PredictionChangeCost.print() +
           "\nmaximum number trees = " + m_MaximumNumberTrees.print() + "\n)";
}

CBoostedTreeHyperparameters::TStrVec CBoostedTreeHyperparameters::names() {
    return {DOWNSAMPLE_FACTOR_TAG,
            ETA_TAG,
            ETA_GROWTH_RATE_PER_TREE_TAG,
            RETRAINED_TREE_ETA_TAG,
            FEATURE_BAG_FRACTION_TAG,
            PREDICTION_CHANGE_COST_TAG,
            DEPTH_PENALTY_MULTIPLIER_TAG,
            TREE_SIZE_PENALTY_MULTIPLIER_TAG,
            LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
            SOFT_TREE_DEPTH_LIMIT_TAG,
            SOFT_TREE_DEPTH_TOLERANCE_TAG,
            TREE_TOPOLOGY_CHANGE_PENALTY_TAG};
}

void CBoostedTreeHyperparameters::initializeTunableHyperparameters() {
    m_TunableHyperparameters.clear();
    m_TunableHyperparameters.reserve(NUMBER_HYPERPARAMETERS);
    for (int i = 0; i < static_cast<int>(NUMBER_HYPERPARAMETERS); ++i) {
        switch (static_cast<EHyperparameter>(i)) {
        // Train hyperparameters.
        case E_DownsampleFactor:
            if ((m_IncrementalTraining || m_DownsampleFactor.fixed()) == false) {
                m_TunableHyperparameters.push_back(E_DownsampleFactor);
            }
            break;
        case E_Alpha:
            if ((m_IncrementalTraining || m_DepthPenaltyMultiplier.fixed()) == false) {
                m_TunableHyperparameters.push_back(E_Alpha);
            }
            break;
        case E_Lambda:
            if ((m_IncrementalTraining || m_LeafWeightPenaltyMultiplier.fixed()) == false) {
                m_TunableHyperparameters.push_back(E_Lambda);
            }
            break;
        case E_Gamma:
            if ((m_IncrementalTraining || m_TreeSizePenaltyMultiplier.fixed()) == false) {
                m_TunableHyperparameters.push_back(E_Gamma);
            }
            break;
        case E_SoftTreeDepthLimit:
            if ((m_IncrementalTraining || m_SoftTreeDepthLimit.fixed()) == false) {
                m_TunableHyperparameters.push_back(E_SoftTreeDepthLimit);
            }
            break;
        case E_SoftTreeDepthTolerance:
            if ((m_IncrementalTraining || m_SoftTreeDepthTolerance.fixed()) == false) {
                m_TunableHyperparameters.push_back(E_SoftTreeDepthTolerance);
            }
            break;
        case E_Eta:
            if ((m_IncrementalTraining || m_Eta.fixed()) == false) {
                m_TunableHyperparameters.push_back(E_Eta);
            }
            break;
        case E_EtaGrowthRatePerTree:
            if ((m_IncrementalTraining || m_Eta.fixed() ||
                 m_EtaGrowthRatePerTree.fixed()) == false) {
                m_TunableHyperparameters.push_back(E_EtaGrowthRatePerTree);
            }
            break;
        case E_FeatureBagFraction:
            if ((m_IncrementalTraining || m_FeatureBagFraction.fixed()) == false) {
                m_TunableHyperparameters.push_back(E_FeatureBagFraction);
            }
            break;
        // Incremental train hyperparameters.
        case E_PredictionChangeCost:
            if (m_IncrementalTraining && (m_PredictionChangeCost.fixed() == false)) {
                m_TunableHyperparameters.push_back(E_PredictionChangeCost);
            }
            break;
        case E_RetrainedTreeEta:
            if (m_IncrementalTraining && (m_RetrainedTreeEta.fixed() == false)) {
                m_TunableHyperparameters.push_back(E_RetrainedTreeEta);
            }
            break;
        case E_TreeTopologyChangePenalty:
            if (m_IncrementalTraining && (m_TreeTopologyChangePenalty.fixed() == false)) {
                m_TunableHyperparameters.push_back(E_TreeTopologyChangePenalty);
            }
            break;
        // Not tuned directly.
        case E_MaximumNumberTrees:
            break;
        }
    }
}

void CBoostedTreeHyperparameters::saveCurrent() {
    m_DepthPenaltyMultiplier.save();
    m_TreeSizePenaltyMultiplier.save();
    m_LeafWeightPenaltyMultiplier.save();
    m_SoftTreeDepthLimit.save();
    m_SoftTreeDepthTolerance.save();
    m_TreeTopologyChangePenalty.save();
    m_DownsampleFactor.save();
    m_FeatureBagFraction.save();
    m_Eta.save();
    m_EtaGrowthRatePerTree.save();
    m_RetrainedTreeEta.save();
    m_PredictionChangeCost.save();
    m_MaximumNumberTrees.save();
}

void CBoostedTreeHyperparameters::restoreSaved() {
    m_DepthPenaltyMultiplier.load();
    m_TreeSizePenaltyMultiplier.load();
    m_LeafWeightPenaltyMultiplier.load();
    m_SoftTreeDepthLimit.load();
    m_SoftTreeDepthTolerance.load();
    m_TreeTopologyChangePenalty.load();
    m_DownsampleFactor.load();
    m_FeatureBagFraction.load();
    m_Eta.load();
    m_EtaGrowthRatePerTree.load();
    m_RetrainedTreeEta.load();
    m_PredictionChangeCost.load();
    m_MaximumNumberTrees.load();
    LOG_TRACE(<< "loss* = " << m_BestForestTestLoss);
    LOG_TRACE(<< "parameters*= " << this->print());
}

// clang-format off
const std::string CBoostedTreeHyperparameters::BAYESIAN_OPTIMIZATION_TAG{"bayesian_optimization"};
const std::string CBoostedTreeHyperparameters::BEST_FOREST_LOSS_GAP_TAG{"best_forest_loss_gap"};
const std::string CBoostedTreeHyperparameters::BEST_FOREST_TEST_LOSS_TAG{"best_forest_test_loss"};
const std::string CBoostedTreeHyperparameters::CURRENT_ROUND_TAG{"current_round"};
const std::string CBoostedTreeHyperparameters::DEPTH_PENALTY_MULTIPLIER_TAG{"depth_penalty_multiplier"};
const std::string CBoostedTreeHyperparameters::DOWNSAMPLE_FACTOR_TAG{"downsample_factor"};
const std::string CBoostedTreeHyperparameters::ETA_GROWTH_RATE_PER_TREE_TAG{"eta_growth_rate_per_tree"};
const std::string CBoostedTreeHyperparameters::ETA_TAG{"eta"};
const std::string CBoostedTreeHyperparameters::FEATURE_BAG_FRACTION_TAG{"feature_bag_fraction"};
const std::string CBoostedTreeHyperparameters::LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG{"leaf_weight_penalty_multiplier"};
const std::string CBoostedTreeHyperparameters::MAXIMUM_NUMBER_TREES_TAG{"maximum_number_trees"};
const std::string CBoostedTreeHyperparameters::MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG{"maximum_optimisation_rounds_per_hyperparameter"};
const std::string CBoostedTreeHyperparameters::MEAN_FOREST_SIZE_ACCUMULATOR_TAG{"mean_forest_size_accumulator"};
const std::string CBoostedTreeHyperparameters::MEAN_TEST_LOSS_ACCUMULATOR_TAG{"mean_test_loss_accumulator"};
const std::string CBoostedTreeHyperparameters::NUMBER_ROUNDS_TAG{"number_rounds"};
const std::string CBoostedTreeHyperparameters::PREDICTION_CHANGE_COST_TAG{"prediction_change_cost"};
const std::string CBoostedTreeHyperparameters::RETRAINED_TREE_ETA_TAG{"retrained_tree_eta"};
const std::string CBoostedTreeHyperparameters::SOFT_TREE_DEPTH_LIMIT_TAG{"soft_tree_depth_limit"};
const std::string CBoostedTreeHyperparameters::SOFT_TREE_DEPTH_TOLERANCE_TAG{"soft_tree_depth_tolerance"};
const std::string CBoostedTreeHyperparameters::STOP_HYPERPARAMETER_OPTIMIZATION_EARLY_TAG{"stop_hyperparameter_optimization_early"};
const std::string CBoostedTreeHyperparameters::TREE_SIZE_PENALTY_MULTIPLIER_TAG{"tree_size_penalty_multiplier"};
const std::string CBoostedTreeHyperparameters::TREE_TOPOLOGY_CHANGE_PENALTY_TAG{"tree_topology_change_penalty"};
// clang-format on
}
}
