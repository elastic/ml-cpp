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

#include <maths/analytics/CBoostedTreeHyperparameters.h>

#include <core/CContainerPrinter.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/analytics/CBoostedTreeImpl.h>
#include <maths/analytics/CDataFrameAnalysisInstrumentationInterface.h>

#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CBayesianOptimisation.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CLowess.h>
#include <maths/common/CLowessDetail.h>

#include <boost/optional/optional_io.hpp>

#include <cmath>

namespace ml {
namespace maths {
namespace analytics {
using namespace boosted_tree_detail;
namespace {
using TVector3x1 = CBoostedTreeHyperparameters::TVector3x1;

const std::size_t MIN_VALUE_INDEX{0};
const std::size_t MID_VALUE_INDEX{1};
const std::size_t MAX_VALUE_INDEX{2};
const double LINE_SEARCH_MINIMUM_RELATIVE_EI_TO_CONTINUE{0.01};
}

CBoostedTreeHyperparameters::CBoostedTreeHyperparameters() {
    this->saveCurrent();
    this->initializeTunableHyperparameters();
}

double CBoostedTreeHyperparameters::penaltyForDepth(std::size_t depth) const {
    return std::exp((static_cast<double>(depth) / m_SoftTreeDepthLimit.value() - 1.0) /
                    m_SoftTreeDepthTolerance.value());
}

double CBoostedTreeHyperparameters::etaForTreeAtPosition(std::size_t index) const {
    return std::min(m_Eta.value() * common::CTools::stable(
                                        std::pow(m_EtaGrowthRatePerTree.value(),
                                                 static_cast<double>(index))),
                    1.0);
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

void CBoostedTreeHyperparameters::initializeFineTuneSearchInterval(
    const CInitializeFineTuneArguments& args,
    TDoubleParameter& parameter) const {

    if (parameter.valid(args.maxValue())) {

        double maxValue{parameter.toSearchValue(args.maxValue())};
        double minValue{maxValue - parameter.toSearchValue(args.searchInterval())};
        double meanValue{(minValue + maxValue) / 2.0};
        LOG_TRACE(<< "mean value = " << meanValue);

        TVector3x1 fallback;
        fallback(MIN_VALUE_INDEX) = minValue;
        fallback(MID_VALUE_INDEX) = meanValue;
        fallback(MAX_VALUE_INDEX) = maxValue;
        auto interval = this->testLossLineSearch(args, minValue, maxValue).value_or(fallback);
        LOG_TRACE(<< "search interval = [" << interval.toDelimited() << "]");

        parameter.fixToRange(parameter.fromSearchValue(interval(MIN_VALUE_INDEX)),
                             parameter.fromSearchValue(interval(MAX_VALUE_INDEX)));
        parameter.set(parameter.fromSearchValue(interval(MID_VALUE_INDEX)));

    } else {
        parameter.fix();
    }
}

CBoostedTreeHyperparameters::TOptionalVector3x1
CBoostedTreeHyperparameters::testLossLineSearch(const CInitializeFineTuneArguments& args,
                                                double intervalLeftEnd,
                                                double intervalRightEnd) const {

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

    using TMinAccumulator = common::CBasicStatistics::SMin<double>::TAccumulator;

    TMinAccumulator minTestLoss;
    TDoubleDoublePrVec testLosses;
    testLosses.reserve(this->maxLineSearchIterations());
    // Ensure we choose one value based on expected improvement.
    std::size_t minNumberTestLosses{6};

    for (auto parameter :
         {intervalLeftEnd, (2.0 * intervalLeftEnd + intervalRightEnd) / 3.0,
          (intervalLeftEnd + 2.0 * intervalRightEnd) / 3.0, intervalRightEnd}) {

        if (args.updateParameter()(args.tree(), parameter) == false) {
            args.tree().m_TrainingProgress.increment(
                (this->maxLineSearchIterations() - testLosses.size()) *
                args.tree().m_Hyperparameters.maximumNumberTrees().value());
            break;
        }

        CBoostedTreeImpl::TNodeVecVec forest;
        double testLoss;
        std::tie(forest, testLoss, std::ignore, std::ignore) =
            args.tree()
                .trainForest(args.frame(), args.tree().m_TrainingRowMasks[0],
                             args.tree().m_TestingRowMasks[0], args.tree().m_TrainingProgress)
                .asTuple();
        minTestLoss.add(testLoss);
        testLosses.emplace_back(parameter, testLoss);
    }

    if (testLosses.empty()) {
        return {};
    }

    auto boptVector = [](double parameter) {
        return common::SConstant<common::CBayesianOptimisation::TVector>::get(1, parameter);
    };
    auto adjustTestLoss = [&](double parameter, double testLoss) {
        auto min = std::min_element(testLosses.begin(), testLosses.end(),
                                    common::COrderings::SSecondLess{});
        return args.adjustLoss()(parameter, min->second, testLoss);
    };

    common::CBayesianOptimisation bopt{{{intervalLeftEnd, intervalRightEnd}}};
    for (auto& parameterAndTestLoss : testLosses) {
        double parameter;
        double testLoss;
        std::tie(parameter, testLoss) = parameterAndTestLoss;
        double adjustedTestLoss{adjustTestLoss(parameter, testLoss)};
        bopt.add(boptVector(parameter), adjustedTestLoss, 0.0);
        parameterAndTestLoss.second = adjustedTestLoss;
    }

    while (testLosses.size() > 0 && testLosses.size() < this->maxLineSearchIterations()) {
        common::CBayesianOptimisation::TVector parameter;
        common::CBayesianOptimisation::TOptionalDouble EI;
        std::tie(parameter, EI) = bopt.maximumExpectedImprovement();
        double threshold{LINE_SEARCH_MINIMUM_RELATIVE_EI_TO_CONTINUE * minTestLoss[0]};
        LOG_TRACE(<< "EI = " << EI << " threshold to continue = " << threshold);
        if ((testLosses.size() >= minNumberTestLosses && EI != boost::none && *EI < threshold) ||
            args.updateParameter()(args.tree(), parameter(0)) == false) {
            args.tree().m_TrainingProgress.increment(
                (this->maxLineSearchIterations() - testLosses.size()) *
                args.tree().m_Hyperparameters.maximumNumberTrees().value());
            break;
        }

        CBoostedTreeImpl::TNodeVecVec forest;
        double testLoss;
        std::tie(forest, testLoss, std::ignore, std::ignore) =
            args.tree()
                .trainForest(args.frame(), args.tree().m_TrainingRowMasks[0],
                             args.tree().m_TestingRowMasks[0], args.tree().m_TrainingProgress)
                .asTuple();

        minTestLoss.add(testLoss);

        double adjustedTestLoss{adjustTestLoss(parameter(0), testLoss)};
        bopt.add(parameter, adjustedTestLoss, 0.0);
        testLosses.emplace_back(parameter(0), adjustedTestLoss);
    }

    std::sort(testLosses.begin(), testLosses.end());
    LOG_TRACE(<< "test losses = " << core::CContainerPrinter::print(testLosses));

    common::CLowess<2> lowess;
    lowess.fit(std::move(testLosses), testLosses.size());

    double bestParameter;
    double bestParameterTestLoss;
    std::tie(bestParameter, bestParameterTestLoss) = lowess.minimum();
    LOG_TRACE(<< "best parameter = " << bestParameter << ", test loss = " << bestParameterTestLoss);

    double width{(intervalRightEnd - intervalLeftEnd) /
                 static_cast<double>(this->maxLineSearchIterations())};
    intervalLeftEnd = bestParameter - width;
    intervalRightEnd = bestParameter + width;
    LOG_TRACE(<< "interval = [" << intervalLeftEnd << "," << intervalRightEnd << "]");

    return TVector3x1{{intervalLeftEnd, bestParameter, intervalRightEnd}};
}

void CBoostedTreeHyperparameters::initializeSearch() {

    // We need sensible bounds for the region we'll search for optimal values.
    // For all parameters where we have initial estimates we use bounds of the
    // form a * initial and b * initial for a < 1 < b. For other parameters we
    // use a fixed range.
    //
    // We also parameterise so the probability any subinterval contains a good
    // value is proportional to its length. For parameters whose difference is
    // naturally measured as a ratio, i.e. diff(p_1, p_0) = p_1 / p_0 for p_0
    // less than p_1, This translates to using log parameter values.

    this->initializeTunableHyperparameters();

    TDoubleDoublePrVec boundingBox;
    boundingBox.reserve(m_TunableHyperparameters.size());
    for (const auto& parameter : m_TunableHyperparameters) {
        switch (parameter) {
        case E_Alpha:
            boundingBox.push_back(m_TreeSizePenaltyMultiplier.range());
            break;
        case E_DownsampleFactor:
            boundingBox.push_back(m_DownsampleFactor.range());
            break;
        case E_Eta:
            boundingBox.push_back(m_Eta.range());
            break;
        case E_EtaGrowthRatePerTree: {
            boundingBox.push_back(m_EtaGrowthRatePerTree.range());
            break;
        }
        case E_FeatureBagFraction:
            boundingBox.push_back(m_FeatureBagFraction.range());
            break;
        case E_Gamma:
            boundingBox.push_back(m_TreeSizePenaltyMultiplier.range());
            break;
        case E_Lambda:
            boundingBox.push_back(m_LeafWeightPenaltyMultiplier.range());
            break;
        case E_MaximumNumberTrees:
            // Maximum number trees is not tuned directly.
            break;
        case E_SoftTreeDepthLimit:
            boundingBox.push_back(m_SoftTreeDepthLimit.range());
            break;
        case E_SoftTreeDepthTolerance:
            boundingBox.push_back(m_SoftTreeDepthTolerance.range());
            break;
        case E_PredictionChangeCost:
            boundingBox.push_back(m_PredictionChangeCost.range());
            break;
        case E_RetrainedTreeEta:
            boundingBox.push_back(m_RetrainedTreeEta.range());
            break;
        case E_TreeTopologyChangePenalty:
            boundingBox.push_back(m_TreeTopologyChangePenalty.range());
            break;
        }
    }
    LOG_TRACE(<< "hyperparameter search bounding box = "
              << core::CContainerPrinter::print(boundingBox));

    m_BayesianOptimization = std::make_unique<common::CBayesianOptimisation>(
        std::move(boundingBox),
        m_BayesianOptimisationRestarts.value_or(common::CBayesianOptimisation::RESTARTS));

    m_CurrentRound = 0;
    m_NumberRounds = m_MaximumOptimisationRoundsPerHyperparameter *
                     m_TunableHyperparameters.size();
    this->saveCurrent();
}

void CBoostedTreeHyperparameters::startSearch() {
    std::size_t dimension{m_TunableHyperparameters.size()};
    std::size_t n{m_NumberRounds / 3 + 1};
    common::CSampling::sobolSequenceSample(dimension, n, m_HyperparameterSamples);
}

void CBoostedTreeHyperparameters::addRoundStats(const TMeanAccumulator& meanForestSizeAccumulator,
                                                double meanTestLoss) {
    m_MeanForestSizeAccumulator += meanForestSizeAccumulator;
    m_MeanTestLossAccumulator.add(meanTestLoss);
}

bool CBoostedTreeHyperparameters::selectNext(const TMeanVarAccumulator& testLossMoments,
                                             double explainedVariance) {

    using TVector = common::CDenseVector<double>;

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
                                      (common::CTools::stableExp(minBoundary(i)) +
                                       common::CTools::stableExp(maxBoundary(i))));
        }
    }

    // Read parameters for last round.
    for (std::size_t i = 0; i < m_TunableHyperparameters.size(); ++i) {
        switch (m_TunableHyperparameters[i]) {
        case E_Alpha:
            parameters(i) = m_DepthPenaltyMultiplier.toSearchValue(
                m_DepthPenaltyMultiplier.value() / scale);
            break;
        case E_DownsampleFactor:
            parameters(i) = m_DownsampleFactor.toSearchValue(m_DownsampleFactor.value());
            break;
        case E_Eta:
            parameters(i) = m_Eta.toSearchValue(m_Eta.value());
            break;
        case E_EtaGrowthRatePerTree:
            parameters(i) =
                m_EtaGrowthRatePerTree.toSearchValue(m_EtaGrowthRatePerTree.value());
            break;
        case E_FeatureBagFraction:
            parameters(i) =
                m_FeatureBagFraction.toSearchValue(m_FeatureBagFraction.value());
            break;
        case E_MaximumNumberTrees:
            parameters(i) =
                m_MaximumNumberTrees.toSearchValue(m_MaximumNumberTrees.value());
            break;
        case E_Gamma:
            parameters(i) = m_TreeSizePenaltyMultiplier.toSearchValue(
                m_TreeSizePenaltyMultiplier.value() / scale);
            break;
        case E_Lambda:
            parameters(i) = m_LeafWeightPenaltyMultiplier.toSearchValue(
                m_LeafWeightPenaltyMultiplier.value() / scale);
            break;
        case E_SoftTreeDepthLimit:
            parameters(i) =
                m_SoftTreeDepthLimit.toSearchValue(m_SoftTreeDepthLimit.value());
            break;
        case E_SoftTreeDepthTolerance:
            parameters(i) = m_SoftTreeDepthTolerance.toSearchValue(
                m_SoftTreeDepthTolerance.value());
            break;
        case E_PredictionChangeCost:
            parameters(i) =
                m_PredictionChangeCost.toSearchValue(m_PredictionChangeCost.value());
            break;
        case E_RetrainedTreeEta:
            parameters(i) = m_RetrainedTreeEta.toSearchValue(m_RetrainedTreeEta.value());
            break;
        case E_TreeTopologyChangePenalty:
            parameters(i) = m_TreeTopologyChangePenalty.toSearchValue(
                m_TreeTopologyChangePenalty.value());
            break;
        }
    }

    double meanTestLoss{common::CBasicStatistics::mean(testLossMoments)};
    double testLossVariance{common::CBasicStatistics::variance(testLossMoments)};

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
               m_BayesianOptimization->anovaTotalCoefficientOfVariation() < 1e-3) {
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
            scale = std::min(1.0, 2.0 * common::CTools::stableExp(parameters(i)) /
                                      (common::CTools::stableExp(minBoundary(i)) +
                                       common::CTools::stableExp(maxBoundary(i))));
        }
    }
    for (std::size_t i = 0; i < m_TunableHyperparameters.size(); ++i) {
        switch (m_TunableHyperparameters[i]) {
        case E_Alpha:
            m_DepthPenaltyMultiplier.set(
                scale * m_DepthPenaltyMultiplier.fromSearchValue(parameters(i)));
            break;
        case E_DownsampleFactor:
            m_DownsampleFactor.set(m_DownsampleFactor.fromSearchValue(parameters(i)));
            break;
        case E_Eta:
            m_Eta.set(m_Eta.fromSearchValue(parameters(i)));
            break;
        case E_EtaGrowthRatePerTree:
            m_EtaGrowthRatePerTree.set(
                m_EtaGrowthRatePerTree.fromSearchValue(parameters(i)));
            break;
        case E_FeatureBagFraction:
            m_FeatureBagFraction.set(m_FeatureBagFraction.fromSearchValue(parameters(i)));
            break;
        case E_MaximumNumberTrees:
            m_MaximumNumberTrees.set(
                m_MaximumNumberTrees.fromSearchValue(std::ceil(parameters(i))));
            break;
        case E_Gamma:
            m_TreeSizePenaltyMultiplier.set(
                scale * m_TreeSizePenaltyMultiplier.fromSearchValue(parameters(i)));
            break;
        case E_Lambda:
            m_LeafWeightPenaltyMultiplier.set(
                scale * m_LeafWeightPenaltyMultiplier.fromSearchValue(parameters(i)));
            break;
        case E_SoftTreeDepthLimit:
            m_SoftTreeDepthLimit.set(m_SoftTreeDepthLimit.fromSearchValue(parameters(i)));
            break;
        case E_SoftTreeDepthTolerance:
            m_SoftTreeDepthTolerance.set(
                m_SoftTreeDepthTolerance.fromSearchValue(parameters(i)));
            break;
        case E_PredictionChangeCost:
            m_PredictionChangeCost.set(
                m_PredictionChangeCost.fromSearchValue(parameters(i)));
            break;
        case E_RetrainedTreeEta:
            m_RetrainedTreeEta.set(m_RetrainedTreeEta.fromSearchValue(parameters(i)));
            break;
        case E_TreeTopologyChangePenalty:
            m_TreeTopologyChangePenalty.set(
                m_TreeTopologyChangePenalty.fromSearchValue(parameters(i)));
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
           (numberKeptNodes + common::CBasicStatistics::mean(m_MeanForestSizeAccumulator)) *
           common::CBasicStatistics::mean(m_MeanTestLossAccumulator);
}

CBoostedTreeHyperparameters::THyperparameterImportanceVec
CBoostedTreeHyperparameters::importances() const {
    THyperparameterImportanceVec importances;
    importances.reserve(m_TunableHyperparameters.size());
    common::CBayesianOptimisation::TDoubleDoublePrVec anovaMainEffects{
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

std::size_t CBoostedTreeHyperparameters::estimateMemoryUsage() const {
    std::size_t numberToTune{this->numberToTune()};
    return sizeof(*this) + numberToTune * sizeof(int) +
           (m_NumberRounds / 3 + 1) * numberToTune * sizeof(double) +
           common::CBayesianOptimisation::estimateMemoryUsage(numberToTune, m_NumberRounds);
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
        const std::string& name{traverser.name()};
        RESTORE_NO_ERROR(BAYESIAN_OPTIMIZATION_TAG,
                         m_BayesianOptimization =
                             std::make_unique<common::CBayesianOptimisation>(traverser))
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
    seed = common::CChecksum::calculate(seed, m_BayesianOptimization);
    seed = common::CChecksum::calculate(seed, m_BestForestLossGap);
    seed = common::CChecksum::calculate(seed, m_BestForestTestLoss);
    seed = common::CChecksum::calculate(seed, m_CurrentRound);
    seed = common::CChecksum::calculate(seed, m_DepthPenaltyMultiplier);
    seed = common::CChecksum::calculate(seed, m_DownsampleFactor);
    seed = common::CChecksum::calculate(seed, m_Eta);
    seed = common::CChecksum::calculate(seed, m_EtaGrowthRatePerTree);
    seed = common::CChecksum::calculate(seed, m_FeatureBagFraction);
    seed = common::CChecksum::calculate(seed, m_HyperparameterSamples);
    seed = common::CChecksum::calculate(seed, m_LeafWeightPenaltyMultiplier);
    seed = common::CChecksum::calculate(seed, m_MaximumNumberTrees);
    seed = common::CChecksum::calculate(seed, m_MaximumOptimisationRoundsPerHyperparameter);
    seed = common::CChecksum::calculate(seed, m_MeanForestSizeAccumulator);
    seed = common::CChecksum::calculate(seed, m_MeanTestLossAccumulator);
    seed = common::CChecksum::calculate(seed, m_NumberRounds);
    seed = common::CChecksum::calculate(seed, m_PredictionChangeCost);
    seed = common::CChecksum::calculate(seed, m_RetrainedTreeEta);
    seed = common::CChecksum::calculate(seed, m_SoftTreeDepthLimit);
    seed = common::CChecksum::calculate(seed, m_SoftTreeDepthTolerance);
    seed = common::CChecksum::calculate(seed, m_StopHyperparameterOptimizationEarly);
    seed = common::CChecksum::calculate(seed, m_TreeSizePenaltyMultiplier);
    seed = common::CChecksum::calculate(seed, m_TreeTopologyChangePenalty);
    return common::CChecksum::calculate(seed, m_TunableHyperparameters);
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

void CBoostedTreeHyperparameters::restoreBest() {
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
}
