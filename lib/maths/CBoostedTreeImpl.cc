/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeImpl.h>

#include <core/CImmutableRadixSet.h>
#include <core/CLoopProgress.h>
#include <core/CPersistUtils.h>
#include <core/CProgramCounters.h>
#include <core/CStopWatch.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CBayesianOptimisation.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CQuantileSketch.h>
#include <maths/CSampling.h>
#include <maths/CSetTools.h>
#include <maths/CTreeShapFeatureImportance.h>

#include <limits>

namespace ml {
namespace maths {
using namespace boosted_tree;
using namespace boosted_tree_detail;

namespace {
using TStrVec = CBoostedTreeImpl::TStrVec;
using TMeanVarAccumulator = CBoostedTreeImpl::TMeanVarAccumulator;
using TRowRef = core::CDataFrame::TRowRef;

class CScopeRecordMemoryUsage {
public:
    using TMemoryUsageCallback = CBoostedTreeImpl::TMemoryUsageCallback;

public:
    template<typename T>
    CScopeRecordMemoryUsage(const T& object, const TMemoryUsageCallback& recordMemoryUsage)
        : m_RecordMemoryUsage{recordMemoryUsage},
          m_MemoryUsage(core::CMemory::dynamicSize(object)) {
        m_RecordMemoryUsage(m_MemoryUsage);
    }

    ~CScopeRecordMemoryUsage() { m_RecordMemoryUsage(-m_MemoryUsage); }

    CScopeRecordMemoryUsage(const CScopeRecordMemoryUsage&) = delete;

    CScopeRecordMemoryUsage& operator=(const CScopeRecordMemoryUsage&) = delete;

    template<typename T>
    void add(const T& object) {
        std::int64_t memoryUsage(core::CMemory::dynamicSize(object));
        m_MemoryUsage += memoryUsage;
        m_RecordMemoryUsage(memoryUsage);
    }

    template<typename T>
    void remove(const T& object) {
        std::int64_t memoryUsage(core::CMemory::dynamicSize(object));
        m_MemoryUsage -= memoryUsage;
        m_RecordMemoryUsage(-memoryUsage);
    }

private:
    const TMemoryUsageCallback& m_RecordMemoryUsage;
    std::int64_t m_MemoryUsage;
};

//! \brief Manages exiting from the loop adding trees to the forest.
//!
//! DESCRIPTION:\n
//! Typically, the test error will decrease exponentially to some minimum then
//! slightly increase thereafter as more trees are added. The logic for exiting
//! training a forest is simple: continue to add trees for some fraction of the
//! maximum forest size after we see the smallest test loss. This amounts to a
//! fixed relative runtime penalty for ensuring we don't stop too early since
//! we record the forest size corresponding to the minimum test loss and simply
//! discard the extra trees at the end of training.
class CTrainForestStoppingCondition {
public:
    CTrainForestStoppingCondition(std::size_t maximumNumberTrees)
        : m_MaximumNumberTrees{maximumNumberTrees},
          m_MaximumNumberTreesWithoutImprovement{std::max(
              static_cast<std::size_t>(0.075 * static_cast<double>(maximumNumberTrees) + 0.5),
              std::size_t{1})} {}

    std::size_t bestSize() const { return m_BestTestLoss[0].second; }

    double bestLoss() const { return m_BestTestLoss[0].first; }

    template<typename FUNC>
    bool shouldStop(std::size_t numberTrees, FUNC computeLoss) {
        double loss{computeLoss()};
        m_BestTestLoss.add({loss, numberTrees});
        LOG_TRACE(<< "test loss = " << loss);
        if (numberTrees - m_BestTestLoss[0].second > m_MaximumNumberTreesWithoutImprovement) {
            return true;
        }
        return numberTrees > m_MaximumNumberTrees;
    }

private:
    using TDoubleSizePrMinAccumulator =
        CBasicStatistics::SMin<std::pair<double, std::size_t>>::TAccumulator;

private:
    std::size_t m_MaximumNumberTrees;
    std::size_t m_MaximumNumberTreesWithoutImprovement;
    TDoubleSizePrMinAccumulator m_BestTestLoss;
};

double readPrediction(const TRowRef& row) {
    return row[predictionColumn(row.numberColumns())];
}

double readLossGradient(const TRowRef& row) {
    return row[lossGradientColumn(row.numberColumns())];
}

double readLossCurvature(const TRowRef& row) {
    return row[lossCurvatureColumn(row.numberColumns())];
}

double readExampleWeight(const TRowRef& row) {
    return row[exampleWeightColumn(row.numberColumns())];
}

double readActual(const TRowRef& row, std::size_t dependentVariable) {
    return row[dependentVariable];
}

double lossAtNSigma(double n, const TMeanVarAccumulator& lossMoments) {
    return CBasicStatistics::mean(lossMoments) +
           n * std::sqrt(CBasicStatistics::variance(lossMoments));
}

const std::size_t ASSIGN_MISSING_TO_LEFT{0};
const std::size_t ASSIGN_MISSING_TO_RIGHT{1};
const double SMALLEST_RELATIVE_CURVATURE{1e-20};
// It isn't critical to recompute splits every tree we add because random
// downsampling means they're only approximate estimates of the full data
// quantiles anyway. So we amortise their compute cost w.r.t. training trees
// by only refreshing once every MINIMUM_SPLIT_REFRESH_INTERVAL trees we add.
const double MINIMUM_SPLIT_REFRESH_INTERVAL{3.0};
}

CBoostedTreeImpl::CLeafNodeStatistics::CLeafNodeStatistics(
    std::size_t id,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    const TRegularization& regularization,
    const TImmutableRadixSetVec& candidateSplits,
    const TSizeVec& featureBag,
    std::size_t depth,
    const core::CPackedBitVector& rowMask)
    : m_Id{id}, m_CandidateSplits{candidateSplits}, m_Depth{depth}, m_RowMask{rowMask} {

    LOG_TRACE(<< "row mask = " << m_RowMask);
    this->computeAggregateLossDerivatives(numberThreads, frame, encoder);
    m_BestSplit = this->computeBestSplitStatistics(regularization, featureBag);
}

CBoostedTreeImpl::CLeafNodeStatistics::CLeafNodeStatistics(
    std::size_t id,
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    const TRegularization& regularization,
    const TImmutableRadixSetVec& candidateSplits,
    const TSizeVec& featureBag,
    bool isLeftChild,
    std::size_t depth,
    const CBoostedTreeNode& split,
    const core::CPackedBitVector& parentRowMask)
    : m_Id{id}, m_CandidateSplits{candidateSplits}, m_Depth{depth} {

    this->computeRowMaskAndAggregateLossDerivatives(
        numberThreads, frame, encoder, isLeftChild, split, parentRowMask);
    m_BestSplit = this->computeBestSplitStatistics(regularization, featureBag);
}

CBoostedTreeImpl::CLeafNodeStatistics::CLeafNodeStatistics(std::size_t id,
                                                           const CLeafNodeStatistics& parent,
                                                           const CLeafNodeStatistics& sibling,
                                                           const TRegularization& regularization,
                                                           const TSizeVec& featureBag,
                                                           core::CPackedBitVector rowMask)
    : m_Id{id}, m_CandidateSplits{sibling.m_CandidateSplits},
      m_Depth{sibling.m_Depth}, m_RowMask{std::move(rowMask)} {

    LOG_TRACE(<< "row mask = " << m_RowMask);

    m_Derivatives.resize(m_CandidateSplits.size());
    m_MissingDerivatives.resize(m_CandidateSplits.size());

    for (std::size_t i = 0; i < m_CandidateSplits.size(); ++i) {
        std::size_t numberSplits{m_CandidateSplits[i].size() + 1};
        m_Derivatives[i].resize(numberSplits);
        for (std::size_t j = 0; j < numberSplits; ++j) {
            // Numeric errors mean that it's possible the sum curvature for a candidate
            // split is identically zero while the gradient is epsilon. This can cause
            // the node gain to appear infinite (when there is no weight regularisation)
            // which in turns causes problems initialising the region we search for optimal
            // hyperparameter values. We can safely force the gradient and curvature to
            // be zero if we detect that the count is zero. Also, none of our loss functions
            // have negative curvature therefore we shouldn't allow the cumulative curvature
            // to be negative either. In this case we force it to be a v.small multiple
            // of the magnitude of the gradient since this is the closest feasible estimate.
            std::size_t count{parent.m_Derivatives[i][j].s_Count -
                              sibling.m_Derivatives[i][j].s_Count};
            if (count > 0) {
                double gradient{parent.m_Derivatives[i][j].s_Gradient -
                                sibling.m_Derivatives[i][j].s_Gradient};
                double curvature{parent.m_Derivatives[i][j].s_Curvature -
                                 sibling.m_Derivatives[i][j].s_Curvature};
                curvature = std::max(curvature, SMALLEST_RELATIVE_CURVATURE *
                                                    std::fabs(gradient));
                m_Derivatives[i][j] = SAggregateDerivatives{count, gradient, curvature};
            }
        }
        std::size_t count{parent.m_MissingDerivatives[i].s_Count -
                          sibling.m_MissingDerivatives[i].s_Count};
        if (count > 0) {
            double gradient{parent.m_MissingDerivatives[i].s_Gradient -
                            sibling.m_MissingDerivatives[i].s_Gradient};
            double curvature{parent.m_MissingDerivatives[i].s_Curvature -
                             sibling.m_MissingDerivatives[i].s_Curvature};
            curvature = std::max(curvature, SMALLEST_RELATIVE_CURVATURE * std::fabs(gradient));
            m_MissingDerivatives[i] = SAggregateDerivatives{count, gradient, curvature};
        }
    }
    LOG_TRACE(<< "derivatives = " << core::CContainerPrinter::print(m_Derivatives));
    LOG_TRACE(<< "missing derivatives = "
              << core::CContainerPrinter::print(m_MissingDerivatives));

    m_BestSplit = this->computeBestSplitStatistics(regularization, featureBag);
}

auto CBoostedTreeImpl::CLeafNodeStatistics::split(std::size_t leftChildId,
                                                  std::size_t rightChildId,
                                                  std::size_t numberThreads,
                                                  const core::CDataFrame& frame,
                                                  const CDataFrameCategoryEncoder& encoder,
                                                  const TRegularization& regularization,
                                                  const TImmutableRadixSetVec& candidateSplits,
                                                  const TSizeVec& featureBag,
                                                  const CBoostedTreeNode& split,
                                                  bool leftChildHasFewerRows) {

    if (leftChildHasFewerRows) {
        auto leftChild = std::make_shared<CLeafNodeStatistics>(
            leftChildId, numberThreads, frame, encoder, regularization, candidateSplits,
            featureBag, true /*is left child*/, m_Depth + 1, split, m_RowMask);
        core::CPackedBitVector rightChildRowMask{m_RowMask};
        rightChildRowMask ^= leftChild->rowMask();
        auto rightChild = std::make_shared<CLeafNodeStatistics>(
            rightChildId, *this, *leftChild, regularization, featureBag,
            std::move(rightChildRowMask));

        return std::make_pair(leftChild, rightChild);
    }

    auto rightChild = std::make_shared<CLeafNodeStatistics>(
        rightChildId, numberThreads, frame, encoder, regularization, candidateSplits,
        featureBag, false /*is left child*/, m_Depth + 1, split, m_RowMask);
    core::CPackedBitVector leftChildRowMask{m_RowMask};
    leftChildRowMask ^= rightChild->rowMask();
    auto leftChild = std::make_shared<CLeafNodeStatistics>(
        leftChildId, *this, *rightChild, regularization, featureBag,
        std::move(leftChildRowMask));

    return std::make_pair(leftChild, rightChild);
}

void CBoostedTreeImpl::CLeafNodeStatistics::computeAggregateLossDerivatives(
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder) {

    auto result = frame.readRows(
        numberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](SSplitAggregateDerivatives& splitAggregateDerivatives,
                TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    this->addRowDerivatives(encoder.encode(*row), splitAggregateDerivatives);
                }
            },
            SSplitAggregateDerivatives{m_CandidateSplits}),
        &m_RowMask);
    auto& state = result.first;

    SSplitAggregateDerivatives derivatives{std::move(state[0].s_FunctionState)};
    for (std::size_t i = 1; i < state.size(); ++i) {
        derivatives.merge(state[i].s_FunctionState);
    }

    std::tie(m_Derivatives, m_MissingDerivatives) = derivatives.move();

    LOG_TRACE(<< "derivatives = " << core::CContainerPrinter::print(m_Derivatives));
    LOG_TRACE(<< "missing derivatives = "
              << core::CContainerPrinter::print(m_MissingDerivatives));
}

void CBoostedTreeImpl::CLeafNodeStatistics::computeRowMaskAndAggregateLossDerivatives(
    std::size_t numberThreads,
    const core::CDataFrame& frame,
    const CDataFrameCategoryEncoder& encoder,
    bool isLeftChild,
    const CBoostedTreeNode& split,
    const core::CPackedBitVector& parentRowMask) {

    auto result = frame.readRows(
        numberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](std::pair<core::CPackedBitVector, SSplitAggregateDerivatives>& state,
                TRowItr beginRows, TRowItr endRows) {
                auto& mask = state.first;
                auto& splitAggregateDerivatives = state.second;
                for (auto row = beginRows; row != endRows; ++row) {
                    auto encodedRow = encoder.encode(*row);
                    if (split.assignToLeft(encodedRow) == isLeftChild) {
                        std::size_t index{row->index()};
                        mask.extend(false, index - mask.size());
                        mask.extend(true);
                        this->addRowDerivatives(encodedRow, splitAggregateDerivatives);
                    }
                }
            },
            std::make_pair(core::CPackedBitVector{}, SSplitAggregateDerivatives{m_CandidateSplits})),
        &parentRowMask);
    auto& state = result.first;

    for (auto& mask_ : state) {
        auto& mask = mask_.s_FunctionState.first;
        mask.extend(false, parentRowMask.size() - mask.size());
    }

    m_RowMask = std::move(state[0].s_FunctionState.first);
    SSplitAggregateDerivatives derivatives{std::move(state[0].s_FunctionState.second)};
    for (std::size_t i = 1; i < state.size(); ++i) {
        m_RowMask |= state[i].s_FunctionState.first;
        derivatives.merge(state[i].s_FunctionState.second);
    }

    std::tie(m_Derivatives, m_MissingDerivatives) = derivatives.move();

    LOG_TRACE(<< "row mask = " << m_RowMask);
    LOG_TRACE(<< "derivatives = " << core::CContainerPrinter::print(m_Derivatives));
    LOG_TRACE(<< "missing derivatives = "
              << core::CContainerPrinter::print(m_MissingDerivatives));
}

void CBoostedTreeImpl::CLeafNodeStatistics::addRowDerivatives(
    const CEncodedDataFrameRowRef& row,
    SSplitAggregateDerivatives& splitAggregateDerivatives) const {

    const TRowRef& unencodedRow{row.unencodedRow()};
    double gradient{readLossGradient(unencodedRow)};
    double curvature{readLossCurvature(unencodedRow)};

    for (std::size_t i = 0; i < m_CandidateSplits.size(); ++i) {
        double featureValue{row[i]};
        if (CDataFrameUtils::isMissing(featureValue)) {
            splitAggregateDerivatives.s_MissingDerivatives[i].add(1, gradient, curvature);
        } else {
            std::ptrdiff_t j{m_CandidateSplits[i].upperBound(featureValue)};
            splitAggregateDerivatives.s_Derivatives[i][j].add(1, gradient, curvature);
        }
    }
}

CBoostedTreeImpl::CLeafNodeStatistics::SSplitStatistics
CBoostedTreeImpl::CLeafNodeStatistics::computeBestSplitStatistics(const TRegularization& regularization,
                                                                  const TSizeVec& featureBag) const {

    // We have three possible regularization terms we'll use:
    //   1. Tree size: gamma * "node count"
    //   2. Sum square weights: lambda * sum{"leaf weight" ^ 2)}
    //   3. Tree depth: alpha * sum{exp(("depth" / "target depth" - 1.0) / "tolerance")}

    SSplitStatistics result;

    for (auto i : featureBag) {
        std::size_t c{m_MissingDerivatives[i].s_Count};
        double g{m_MissingDerivatives[i].s_Gradient};
        double h{m_MissingDerivatives[i].s_Curvature};
        for (const auto& derivatives : m_Derivatives[i]) {
            c += derivatives.s_Count;
            g += derivatives.s_Gradient;
            h += derivatives.s_Curvature;
        }
        std::size_t cl[]{m_MissingDerivatives[i].s_Count, 0};
        double gl[]{m_MissingDerivatives[i].s_Gradient, 0.0};
        double hl[]{m_MissingDerivatives[i].s_Curvature, 0.0};

        double maximumGain{-INF};
        double splitAt{-INF};
        bool leftChildHasFewerRows{true};
        bool assignMissingToLeft{true};

        for (std::size_t j = 0; j + 1 < m_Derivatives[i].size(); ++j) {
            cl[ASSIGN_MISSING_TO_LEFT] += m_Derivatives[i][j].s_Count;
            gl[ASSIGN_MISSING_TO_LEFT] += m_Derivatives[i][j].s_Gradient;
            hl[ASSIGN_MISSING_TO_LEFT] += m_Derivatives[i][j].s_Curvature;
            cl[ASSIGN_MISSING_TO_RIGHT] += m_Derivatives[i][j].s_Count;
            gl[ASSIGN_MISSING_TO_RIGHT] += m_Derivatives[i][j].s_Gradient;
            hl[ASSIGN_MISSING_TO_RIGHT] += m_Derivatives[i][j].s_Curvature;

            double gain[]{CTools::pow2(gl[ASSIGN_MISSING_TO_LEFT]) /
                                  (hl[ASSIGN_MISSING_TO_LEFT] +
                                   regularization.leafWeightPenaltyMultiplier()) +
                              CTools::pow2(g - gl[ASSIGN_MISSING_TO_LEFT]) /
                                  (h - hl[ASSIGN_MISSING_TO_LEFT] +
                                   regularization.leafWeightPenaltyMultiplier()),
                          CTools::pow2(gl[ASSIGN_MISSING_TO_RIGHT]) /
                                  (hl[ASSIGN_MISSING_TO_RIGHT] +
                                   regularization.leafWeightPenaltyMultiplier()) +
                              CTools::pow2(g - gl[ASSIGN_MISSING_TO_RIGHT]) /
                                  (h - hl[ASSIGN_MISSING_TO_RIGHT] +
                                   regularization.leafWeightPenaltyMultiplier())};

            if (gain[ASSIGN_MISSING_TO_LEFT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_LEFT];
                splitAt = m_CandidateSplits[i][j];
                leftChildHasFewerRows = (2 * cl[ASSIGN_MISSING_TO_LEFT] < c);
                assignMissingToLeft = true;
            }
            if (gain[ASSIGN_MISSING_TO_RIGHT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_RIGHT];
                splitAt = m_CandidateSplits[i][j];
                leftChildHasFewerRows = (2 * cl[ASSIGN_MISSING_TO_RIGHT] < c);
                assignMissingToLeft = false;
            }
        }

        double penaltyForDepth{regularization.penaltyForDepth(m_Depth)};
        double penaltyForDepthPlusOne{regularization.penaltyForDepth(m_Depth + 1)};
        double gain{0.5 * (maximumGain - CTools::pow2(g) / (h + regularization.leafWeightPenaltyMultiplier())) -
                    regularization.treeSizePenaltyMultiplier() -
                    regularization.depthPenaltyMultiplier() *
                        (2.0 * penaltyForDepthPlusOne - penaltyForDepth)};

        SSplitStatistics candidate{
            gain, h, i, splitAt, leftChildHasFewerRows, assignMissingToLeft};
        LOG_TRACE(<< "candidate split: " << candidate.print());

        if (candidate > result) {
            result = candidate;
        }
    }

    LOG_TRACE(<< "best split: " << result.print());

    return result;
}

CBoostedTreeImpl::CBoostedTreeImpl(std::size_t numberThreads, CBoostedTree::TLossFunctionUPtr loss)
    : m_NumberThreads{numberThreads}, m_Loss{std::move(loss)},
      m_BestHyperparameters{
          m_Regularization,       m_DownsampleFactor,   m_Eta,
          m_EtaGrowthRatePerTree, m_MaximumNumberTrees, m_FeatureBagFraction} {
}

CBoostedTreeImpl::CBoostedTreeImpl() = default;

CBoostedTreeImpl::~CBoostedTreeImpl() = default;

CBoostedTreeImpl& CBoostedTreeImpl::operator=(CBoostedTreeImpl&&) = default;

void CBoostedTreeImpl::train(core::CDataFrame& frame,
                             const TProgressCallback& recordProgress,
                             const TMemoryUsageCallback& recordMemoryUsage,
                             const TTrainingStateCallback& recordTrainStateCallback) {

    if (m_DependentVariable >= frame.numberColumns()) {
        HANDLE_FATAL(<< "Internal error: dependent variable '" << m_DependentVariable
                     << "' was incorrectly initialized. Please report this problem.");
        return;
    }

    LOG_TRACE(<< "Main training loop...");

    m_TrainingProgress.progressCallback(recordProgress);

    std::int64_t lastMemoryUsage(this->memoryUsage());
    recordMemoryUsage(lastMemoryUsage);

    core::CPackedBitVector allTrainingRowsMask{this->allTrainingRowsMask()};
    core::CPackedBitVector noRowsMask{allTrainingRowsMask.size(), false};

    if (this->canTrain() == false) {

        // Fallback to using the constant predictor which minimises the loss.

        m_BestForest.assign(1, this->initializePredictionsAndLossDerivatives(
                                   frame, allTrainingRowsMask, noRowsMask));
        m_BestForestTestLoss = this->meanLoss(frame, allTrainingRowsMask);
        LOG_TRACE(<< "Test loss = " << m_BestForestTestLoss);

    } else if (m_CurrentRound < m_NumberRounds || m_BestForest.empty()) {
        TMeanVarAccumulator timeAccumulator;
        core::CStopWatch stopWatch;
        stopWatch.start();
        std::uint64_t lastLap{stopWatch.lap()};

        // Hyperparameter optimisation loop.

        this->initializePerFoldTestLosses();

        while (m_CurrentRound < m_NumberRounds) {

            LOG_TRACE(<< "Optimisation round = " << m_CurrentRound + 1);

            TMeanVarAccumulator lossMoments;
            std::size_t maximumNumberTrees;
            std::tie(lossMoments, maximumNumberTrees) =
                this->crossValidateForest(frame, recordMemoryUsage);

            this->captureBestHyperparameters(lossMoments, maximumNumberTrees);

            if (this->selectNextHyperparameters(lossMoments, *m_BayesianOptimization) == false) {
                LOG_WARN(<< "Hyperparameter selection failed: exiting loop early");
                break;
            }

            std::int64_t memoryUsage(this->memoryUsage());
            recordMemoryUsage(memoryUsage - lastMemoryUsage);
            lastMemoryUsage = memoryUsage;

            // Store the training state after each hyperparameter search step.
            m_CurrentRound += 1;
            LOG_TRACE(<< "Round " << m_CurrentRound << " state recording started");
            this->recordState(recordTrainStateCallback);
            LOG_TRACE(<< "Round " << m_CurrentRound << " state recording finished");

            std::uint64_t currentLap{stopWatch.lap()};
            timeAccumulator.add(static_cast<double>(currentLap - lastLap));
            lastLap = currentLap;
        }

        LOG_TRACE(<< "Test loss = " << m_BestForestTestLoss);

        this->restoreBestHyperparameters();
        std::tie(m_BestForest, std::ignore) =
            this->trainForest(frame, allTrainingRowsMask, allTrainingRowsMask,
                              m_TrainingProgress, recordMemoryUsage);

        this->recordState(recordTrainStateCallback);

        timeAccumulator.add(static_cast<double>(stopWatch.stop()));

        LOG_INFO(<< "Training finished after " << m_CurrentRound << " iterations. "
                 << "Time per iteration in ms mean: "
                 << CBasicStatistics::mean(timeAccumulator) << " std. dev:  "
                 << std::sqrt(CBasicStatistics::variance(timeAccumulator)));

        core::CProgramCounters::counter(counter_t::E_DFTPMTrainedForestNumberTrees) =
            m_BestForest.size();
    }

    // Force to at least one here because we can have early exit from loop or take
    // a different path.
    recordProgress(1.0);

    std::int64_t memoryUsage(this->memoryUsage());
    recordMemoryUsage(memoryUsage - lastMemoryUsage);
}

void CBoostedTreeImpl::recordState(const TTrainingStateCallback& recordTrainState) const {
    recordTrainState([this](core::CStatePersistInserter& inserter) {
        this->acceptPersistInserter(inserter);
    });
}

void CBoostedTreeImpl::predict(core::CDataFrame& frame,
                               const TProgressCallback& /*recordProgress*/) const {
    if (m_BestForestTestLoss == INF) {
        HANDLE_FATAL(<< "Internal error: no model available for prediction. "
                     << "Please report this problem.");
        return;
    }
    bool successful;
    std::tie(std::ignore, successful) = frame.writeColumns(
        m_NumberThreads, 0, frame.numberRows(), [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                row->writeColumn(predictionColumn(row->numberColumns()),
                                 predictRow(m_Encoder->encode(*row), m_BestForest));
            }
        });
    if (successful == false) {
        HANDLE_FATAL(<< "Internal error: failed model inference. "
                     << "Please report this problem.");
    }
}

const CBoostedTreeImpl::TDoubleVec& CBoostedTreeImpl::featureWeights() const {
    return m_FeatureSampleProbabilities;
}

const CBoostedTreeImpl::TNodeVecVec& CBoostedTreeImpl::trainedModel() const {
    return m_BestForest;
}

std::size_t CBoostedTreeImpl::columnHoldingDependentVariable() const {
    return m_DependentVariable;
}

std::size_t CBoostedTreeImpl::estimateMemoryUsage(std::size_t numberRows,
                                                  std::size_t numberColumns) const {
    // The maximum tree size is defined is the maximum number of leaves minus one.
    // A binary tree with n + 1 leaves has 2n + 1 nodes in total.
    std::size_t maximumNumberLeaves{this->maximumTreeSize(numberRows) + 1};
    std::size_t maximumNumberNodes{2 * maximumNumberLeaves - 1};
    std::size_t forestMemoryUsage{
        m_MaximumNumberTrees *
        (sizeof(TNodeVec) + maximumNumberNodes * sizeof(CBoostedTreeNode))};
    std::size_t extraColumnsMemoryUsage{this->numberExtraColumnsForTrain() *
                                        numberRows * sizeof(CFloatStorage)};
    std::size_t hyperparametersMemoryUsage{numberColumns * sizeof(double)};
    std::size_t leafNodeStatisticsMemoryUsage{
        maximumNumberLeaves * CLeafNodeStatistics::estimateMemoryUsage(
                                  numberRows, numberColumns, m_NumberSplitsPerFeature)};
    std::size_t dataTypeMemoryUsage{numberColumns * sizeof(CDataFrameUtils::SDataType)};
    std::size_t featureSampleProbabilities{numberColumns * sizeof(double)};
    std::size_t missingFeatureMaskMemoryUsage{
        numberColumns * numberRows / PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE};
    std::size_t trainTestMaskMemoryUsage{2 * m_NumberFolds * numberRows /
                                         PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE};
    std::size_t bayesianOptimisationMemoryUsage{CBayesianOptimisation::estimateMemoryUsage(
        this->numberHyperparametersToTune(), m_NumberRounds)};
    return sizeof(*this) + forestMemoryUsage + extraColumnsMemoryUsage +
           hyperparametersMemoryUsage + leafNodeStatisticsMemoryUsage +
           dataTypeMemoryUsage + featureSampleProbabilities + missingFeatureMaskMemoryUsage +
           trainTestMaskMemoryUsage + bayesianOptimisationMemoryUsage;
}

bool CBoostedTreeImpl::canTrain() const {
    return std::accumulate(m_FeatureSampleProbabilities.begin(),
                           m_FeatureSampleProbabilities.end(), 0.0) > 0.0;
}

core::CPackedBitVector CBoostedTreeImpl::allTrainingRowsMask() const {
    return ~m_MissingFeatureRowMasks[m_DependentVariable];
}

CBoostedTreeImpl::TDoubleDoublePr
CBoostedTreeImpl::gainAndCurvatureAtPercentile(double percentile,
                                               const TNodeVecVec& forest) const {

    TDoubleVec gains;
    TDoubleVec curvatures;

    for (const auto& tree : forest) {
        for (const auto& node : tree) {
            if (node.isLeaf() == false) {
                gains.push_back(node.gain());
                curvatures.push_back(node.curvature());
            }
        }
    }

    if (gains.size() == 0) {
        return {0.0, 0.0};
    }

    std::size_t index{std::min(
        static_cast<std::size_t>(percentile * static_cast<double>(gains.size()) / 100.0 + 0.5),
        gains.size() - 1)};
    std::nth_element(gains.begin(), gains.begin() + index, gains.end());
    std::nth_element(curvatures.begin(), curvatures.begin() + index, curvatures.end());

    return {gains[index], curvatures[index]};
}

void CBoostedTreeImpl::initializePerFoldTestLosses() {
    m_FoldRoundTestLosses.resize(m_NumberFolds);
    for (auto& losses : m_FoldRoundTestLosses) {
        losses.resize(m_NumberRounds);
    }
}

CBoostedTreeImpl::TMeanVarAccumulatorSizePr
CBoostedTreeImpl::crossValidateForest(core::CDataFrame& frame,
                                      const TMemoryUsageCallback& recordMemoryUsage) {

    // We want to ensure we evaluate on equal proportions for each fold.
    TSizeVec folds(m_NumberFolds);
    std::iota(folds.begin(), folds.end(), 0);
    CSampling::random_shuffle(m_Rng, folds.begin(), folds.end());

    auto stopCrossValidationEarly = [&](TMeanVarAccumulator testLossMoments) {
        // Always train on at least one fold and every fold for the first
        // "number folds" rounds. Exit cross-validation early if it's clear
        // that the test error is not close to the minimum test error. We use
        // the estimated test error for each remaining fold at two standard
        // deviations below the mean for this.
        if (m_StopCrossValidationEarly && m_CurrentRound >= m_NumberFolds &&
            folds.size() < m_NumberFolds) {
            for (const auto& testLoss : this->estimateMissingTestLosses(folds)) {
                testLossMoments.add(
                    CBasicStatistics::mean(testLoss) -
                    2.0 * std::sqrt(CBasicStatistics::maximumLikelihoodVariance(testLoss)));
            }
            return CBasicStatistics::mean(testLossMoments) > this->minimumTestLoss();
        }
        return false;
    };

    TMeanVarAccumulator lossMoments;
    TDoubleVec numberTrees;
    numberTrees.reserve(m_NumberFolds);

    while (folds.size() > 0 && stopCrossValidationEarly(lossMoments) == false) {
        std::size_t fold{folds.back()};
        folds.pop_back();
        TNodeVecVec forest;
        double loss;
        std::tie(forest, loss) = this->trainForest(
            frame, m_TrainingRowMasks[fold], m_TestingRowMasks[fold],
            m_TrainingProgress, recordMemoryUsage);
        LOG_TRACE(<< "fold = " << fold << " forest size = " << forest.size()
                  << " test set loss = " << loss);
        lossMoments.add(loss);
        m_FoldRoundTestLosses[fold][m_CurrentRound] = loss;
        numberTrees.push_back(static_cast<double>(forest.size()));
    }
    m_TrainingProgress.increment(m_MaximumNumberTrees * folds.size());
    LOG_TRACE(<< "skipped " << folds.size() << " folds");

    std::sort(numberTrees.begin(), numberTrees.end());
    std::size_t medianNumberTrees{
        static_cast<std::size_t>(CBasicStatistics::median(numberTrees))};
    lossMoments = this->correctTestLossMoments(std::move(folds), lossMoments);
    LOG_TRACE(<< "test mean loss = " << CBasicStatistics::mean(lossMoments)
              << ", sigma = " << std::sqrt(CBasicStatistics::mean(lossMoments)));

    return {lossMoments, medianNumberTrees};
}

CBoostedTreeImpl::TNodeVec CBoostedTreeImpl::initializePredictionsAndLossDerivatives(
    core::CDataFrame& frame,
    const core::CPackedBitVector& trainingRowMask,
    const core::CPackedBitVector& testingRowMask) const {

    core::CPackedBitVector updateRowMask{trainingRowMask | testingRowMask};
    frame.writeColumns(m_NumberThreads, 0, frame.numberRows(),
                       [](TRowItr beginRows, TRowItr endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               std::size_t numberColumns{row->numberColumns()};
                               row->writeColumn(predictionColumn(numberColumns), 0.0);
                               row->writeColumn(lossGradientColumn(numberColumns), 0.0);
                               row->writeColumn(lossCurvatureColumn(numberColumns), 0.0);
                           }
                       },
                       &updateRowMask);

    // At the start we will centre the data w.r.t. the given loss function.
    TNodeVec tree(1);
    this->refreshPredictionsAndLossDerivatives(frame, trainingRowMask,
                                               testingRowMask, 1.0, tree);

    return tree;
}

CBoostedTreeImpl::TNodeVecVecDoublePr
CBoostedTreeImpl::trainForest(core::CDataFrame& frame,
                              const core::CPackedBitVector& trainingRowMask,
                              const core::CPackedBitVector& testingRowMask,
                              core::CLoopProgress& trainingProgress,
                              const TMemoryUsageCallback& recordMemoryUsage) const {

    LOG_TRACE(<< "Training one forest...");

    std::size_t maximumTreeSize{this->maximumTreeSize(trainingRowMask)};

    TNodeVecVec forest{this->initializePredictionsAndLossDerivatives(
        frame, trainingRowMask, testingRowMask)};
    forest.reserve(m_MaximumNumberTrees);

    CScopeRecordMemoryUsage scopeMemoryUsage{forest, recordMemoryUsage};

    // For each iteration:
    //  1. Compute weighted quantiles for features F
    //  2. Periodically compute candidate split set S from quantiles of F
    //  3. Build one tree on (F, S)
    //  4. Update predictions and loss derivatives

    double eta{m_Eta};

    // Computing feature quantiles is surprisingly runtime expensive and there may
    // be mileage in seeing if we can make the sketch more efficient. However, we
    // should easily be able to build multiple trees on the same set of candidate
    // splits and decrease the loss function since the space of candidate trees for
    // a fixed set of candidate splits is very large. Furthermore, we use a greedy
    // heuristic to search this space and so don't expect to find the tree to add
    // which minimises total loss. As a result, we choose to amortise the cost of
    // computing feature quantiles by only refreshing candidate splits periodically.
    std::size_t nextTreeCountToRefreshSplits{
        forest.size() + static_cast<std::size_t>(std::max(0.5 / eta, 1.0))};

    auto downsampledRowMask = this->downsample(trainingRowMask);
    auto candidateSplits = this->candidateSplits(frame, downsampledRowMask);
    scopeMemoryUsage.add(candidateSplits);

    std::size_t retries{0};
    CTrainForestStoppingCondition stoppingCondition{m_MaximumNumberTrees};
    do {
        auto tree = this->trainTree(frame, downsampledRowMask, candidateSplits,
                                    maximumTreeSize, recordMemoryUsage);

        retries = tree.size() == 1 ? retries + 1 : 0;

        if (retries == m_MaximumAttemptsToAddTree) {
            break;
        }

        if (tree.size() > 1) {
            scopeMemoryUsage.add(tree);
            this->refreshPredictionsAndLossDerivatives(frame, trainingRowMask,
                                                       testingRowMask, eta, tree);
            forest.push_back(std::move(tree));
            eta = std::min(1.0, m_EtaGrowthRatePerTree * eta);
            retries = 0;
            trainingProgress.increment();
        } else {
            // Refresh splits in case it allows us to find tree which can reduce loss.
            candidateSplits = this->candidateSplits(frame, downsampledRowMask);
            nextTreeCountToRefreshSplits += static_cast<std::size_t>(
                std::max(0.5 / eta, MINIMUM_SPLIT_REFRESH_INTERVAL));
        }

        downsampledRowMask = this->downsample(trainingRowMask);

        if (forest.size() == nextTreeCountToRefreshSplits) {
            candidateSplits = this->candidateSplits(frame, downsampledRowMask);
            nextTreeCountToRefreshSplits += static_cast<std::size_t>(
                std::max(0.5 / eta, MINIMUM_SPLIT_REFRESH_INTERVAL));
        }
    } while (stoppingCondition.shouldStop(forest.size(), [&]() {
        return this->meanLoss(frame, testingRowMask);
    }) == false);

    LOG_TRACE(<< "Stopped at " << forest.size() - 1 << "/" << m_MaximumNumberTrees);

    trainingProgress.increment(std::max(m_MaximumNumberTrees, forest.size()) -
                               forest.size());

    forest.resize(stoppingCondition.bestSize());

    LOG_TRACE(<< "Trained one forest");

    return {forest, stoppingCondition.bestLoss()};
}

core::CPackedBitVector
CBoostedTreeImpl::downsample(const core::CPackedBitVector& trainingRowMask) const {
    // We compute a stochastic version of the candidate splits, gradients and
    // curvatures for each tree we train. The sampling scheme should minimize
    // the correlation with previous trees for fixed sample size so randomly
    // sampling without replacement is appropriate.
    core::CPackedBitVector result;
    do {
        result = core::CPackedBitVector{};
        for (auto i = trainingRowMask.beginOneBits();
             i != trainingRowMask.endOneBits(); ++i) {
            if (CSampling::uniformSample(m_Rng, 0.0, 1.0) < m_DownsampleFactor) {
                result.extend(false, *i - result.size());
                result.extend(true);
            }
        }
    } while (result.manhattan() == 0.0);
    result.extend(false, trainingRowMask.size() - result.size());
    return result;
}

CBoostedTreeImpl::TImmutableRadixSetVec
CBoostedTreeImpl::candidateSplits(const core::CDataFrame& frame,
                                  const core::CPackedBitVector& trainingRowMask) const {

    TSizeVec features{this->candidateRegressorFeatures()};
    LOG_TRACE(<< "candidate features = " << core::CContainerPrinter::print(features));

    TSizeVec binaryFeatures(features);
    binaryFeatures.erase(std::remove_if(binaryFeatures.begin(), binaryFeatures.end(),
                                        [this](std::size_t index) {
                                            return m_Encoder->isBinary(index) == false;
                                        }),
                         binaryFeatures.end());
    CSetTools::inplace_set_difference(features, binaryFeatures.begin(),
                                      binaryFeatures.end());
    LOG_TRACE(<< "binary features = " << core::CContainerPrinter::print(binaryFeatures)
              << " other features = " << core::CContainerPrinter::print(features));

    auto featureQuantiles =
        CDataFrameUtils::columnQuantiles(
            m_NumberThreads, frame, trainingRowMask, features,
            CFastQuantileSketch{CFastQuantileSketch::E_Linear,
                                std::max(m_NumberSplitsPerFeature, std::size_t{50}), m_Rng},
            m_Encoder.get(), readLossCurvature)
            .first;

    TImmutableRadixSetVec candidateSplits(this->numberFeatures());

    for (auto i : binaryFeatures) {
        candidateSplits[i] = core::CImmutableRadixSet<double>{0.5};
        LOG_TRACE(<< "feature '" << i << "' splits = " << candidateSplits[i].print());
    }
    for (std::size_t i = 0; i < features.size(); ++i) {

        TDoubleVec featureSplits;
        featureSplits.reserve(m_NumberSplitsPerFeature - 1);

        for (std::size_t j = 1; j < m_NumberSplitsPerFeature; ++j) {
            double rank{100.0 * static_cast<double>(j) / static_cast<double>(m_NumberSplitsPerFeature) +
                        CSampling::uniformSample(m_Rng, -0.1, 0.1)};
            double q;
            if (featureQuantiles[i].quantile(rank, q)) {
                featureSplits.push_back(q);
            } else {
                LOG_WARN(<< "Failed to compute quantile " << rank << ": ignoring split");
            }
        }

        const auto& dataType = m_FeatureDataTypes[features[i]];

        if (dataType.s_IsInteger) {
            // The key point here is that we know that if two distinct splits fall
            // between two consecutive integers they must produce identical partitions
            // of the data and so always have the same loss. We only need to retain
            // one such split for training. We achieve this by snapping to the midpoint
            // and subsquently deduplicating.
            std::for_each(featureSplits.begin(), featureSplits.end(),
                          [](double& split) { split = std::floor(split) + 0.5; });
        }
        featureSplits.erase(std::unique(featureSplits.begin(), featureSplits.end()),
                            featureSplits.end());
        featureSplits.erase(std::remove_if(featureSplits.begin(), featureSplits.end(),
                                           [&dataType](double split) {
                                               return split < dataType.s_Min ||
                                                      split > dataType.s_Max;
                                           }),
                            featureSplits.end());
        candidateSplits[features[i]] =
            core::CImmutableRadixSet<double>{std::move(featureSplits)};

        LOG_TRACE(<< "feature '" << features[i]
                  << "' splits = " << candidateSplits[features[i]].print());
    }

    return candidateSplits;
}

CBoostedTreeImpl::TNodeVec
CBoostedTreeImpl::trainTree(core::CDataFrame& frame,
                            const core::CPackedBitVector& trainingRowMask,
                            const TImmutableRadixSetVec& candidateSplits,
                            const std::size_t maximumTreeSize,
                            const TMemoryUsageCallback& recordMemoryUsage) const {

    LOG_TRACE(<< "Training one tree...");

    using TLeafNodeStatisticsPtr = std::shared_ptr<CLeafNodeStatistics>;
    using TLeafNodeStatisticsPtrQueue =
        std::priority_queue<TLeafNodeStatisticsPtr, std::vector<TLeafNodeStatisticsPtr>, COrderings::SLess>;

    TNodeVec tree(1);
    tree.reserve(2 * maximumTreeSize + 1);

    TLeafNodeStatisticsPtrQueue leaves;
    leaves.push(std::make_shared<CLeafNodeStatistics>(
        0 /*root*/, m_NumberThreads, frame, *m_Encoder, m_Regularization,
        candidateSplits, this->featureBag(), 0 /*depth*/, trainingRowMask));

    // We update local variables because the callback can be expensive if it
    // requires accessing atomics.
    std::int64_t memory{0};
    std::int64_t maxMemory{0};
    TMemoryUsageCallback localRecordMemoryUsage{[&](std::int64_t delta) {
        memory += delta;
        maxMemory = std::max(maxMemory, memory);
    }};
    CScopeRecordMemoryUsage scopeMemoryUsage{leaves, localRecordMemoryUsage};

    // For each iteration we:
    //   1. Find the leaf with the greatest decrease in loss
    //   2. If no split (significantly) reduced the loss we terminate
    //   3. Otherwise we split that leaf

    double totalGain{0.0};

    for (std::size_t i = 0; i < maximumTreeSize; ++i) {

        auto leaf = leaves.top();
        leaves.pop();

        scopeMemoryUsage.remove(leaf);

        if (leaf->gain() < MINIMUM_RELATIVE_GAIN_PER_SPLIT * totalGain) {
            break;
        }

        totalGain += leaf->gain();
        LOG_TRACE(<< "splitting " << leaf->id() << " total gain = " << totalGain);

        std::size_t splitFeature;
        double splitValue;
        std::tie(splitFeature, splitValue) = leaf->bestSplit();

        bool leftChildHasFewerRows{leaf->leftChildHasFewerRows()};
        bool assignMissingToLeft{leaf->assignMissingToLeft()};

        std::size_t leftChildId, rightChildId;
        std::tie(leftChildId, rightChildId) =
            tree[leaf->id()].split(splitFeature, splitValue, assignMissingToLeft,
                                   leaf->gain(), leaf->curvature(), tree);

        TLeafNodeStatisticsPtr leftChild;
        TLeafNodeStatisticsPtr rightChild;
        std::tie(leftChild, rightChild) =
            leaf->split(leftChildId, rightChildId, m_NumberThreads, frame, *m_Encoder,
                        m_Regularization, candidateSplits, this->featureBag(),
                        tree[leaf->id()], leftChildHasFewerRows);

        scopeMemoryUsage.add(leftChild);
        scopeMemoryUsage.add(rightChild);

        leaves.push(std::move(leftChild));
        leaves.push(std::move(rightChild));
    }

    tree.shrink_to_fit();

    // Flush the maximum memory used by the leaf statistics to the callback.
    recordMemoryUsage(maxMemory);
    recordMemoryUsage(-maxMemory);

    LOG_TRACE(<< "Trained one tree. # nodes = " << tree.size());

    return tree;
}

double CBoostedTreeImpl::minimumTestLoss() const {
    using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;
    TMinAccumulator minimumTestLoss;
    for (std::size_t round = 0; round < m_CurrentRound - 1; ++round) {
        TMeanVarAccumulator roundLossMoments;
        for (std::size_t fold = 0; fold < m_NumberFolds; ++fold) {
            if (m_FoldRoundTestLosses[fold][round] != boost::none) {
                roundLossMoments.add(*m_FoldRoundTestLosses[fold][round]);
            }
        }
        if (static_cast<std::size_t>(CBasicStatistics::count(roundLossMoments)) == m_NumberFolds) {
            minimumTestLoss.add(CBasicStatistics::mean(roundLossMoments));
        }
    }
    return minimumTestLoss[0];
}

TMeanVarAccumulator
CBoostedTreeImpl::correctTestLossMoments(const TSizeVec& missing,
                                         TMeanVarAccumulator lossMoments) const {
    if (missing.empty()) {
        return lossMoments;
    }
    for (const auto& loss : this->estimateMissingTestLosses(missing)) {
        lossMoments += loss;
    }
    return lossMoments;
}

CBoostedTreeImpl::TMeanVarAccumulatorVec
CBoostedTreeImpl::estimateMissingTestLosses(const TSizeVec& missing) const {

    // We have a subset of folds for which we've computed test loss. We want to
    // estimate the test loss we'll see for the remaining folds to decide if it
    // is worthwhile to continue training with these parameters and to correct
    // the loss value supplied to Bayesian Optimisation to account for the folds
    // we haven't trained on. To this end, for each missing fold fitting an
    // OLS to the data (x_i, loss(m_i)) where i ranges over the previous rounds
    // and x_i is the i'th vector whose components comprise the losses for which
    // we have values in the current round and indicators for whether they were
    // missing in the i'th round. We only include a round if we've trained for at
    // least one of the same folds in the current round.

    TSizeVec present(m_NumberFolds);
    std::iota(present.begin(), present.end(), 0);
    TSizeVec ordered{missing};
    std::sort(ordered.begin(), ordered.end());
    CSetTools::inplace_set_difference(present, ordered.begin(), ordered.end());
    LOG_TRACE(<< "present = " << core::CContainerPrinter::print(present));

    // Get the current round feature vector.
    TVector x(2 * present.size());
    for (std::size_t col = 0; col < present.size(); ++col) {
        x(col) = *m_FoldRoundTestLosses[present[col]][m_CurrentRound];
        x(present.size() + col) = 0.0;
    }

    TMeanVarAccumulatorVec predictedTestLosses;
    predictedTestLosses.reserve(missing.size());

    for (std::size_t target : missing) {
        // Extract the training mask.
        TSizeVec trainingMask;
        trainingMask.reserve(m_CurrentRound);
        for (std::size_t round = 0; round < m_CurrentRound; ++round) {
            if (m_FoldRoundTestLosses[target][round] &&
                std::find_if(present.begin(), present.end(), [&](std::size_t fold) {
                    return m_FoldRoundTestLosses[fold][round];
                }) != present.end()) {
                trainingMask.push_back(round);
            }
        }

        // Fit the OLS regression.
        CDenseMatrix<double> A(trainingMask.size(), 2 * present.size());
        TVector b(trainingMask.size());
        for (std::size_t row = 0; row < trainingMask.size(); ++row) {
            for (std::size_t col = 0; col < present.size(); ++col) {
                if (m_FoldRoundTestLosses[present[col]][trainingMask[row]]) {
                    A(row, col) = *m_FoldRoundTestLosses[present[col]][trainingMask[row]];
                    A(row, present.size() + col) = 0.0;
                } else {
                    A(row, col) = 0.0;
                    A(row, present.size() + col) = 1.0;
                }
            }
            b(row) = *m_FoldRoundTestLosses[target][trainingMask[row]];
        }
        TVector params{A.colPivHouseholderQr().solve(b)};

        TMeanVarAccumulator residualMoments;
        for (int row = 0; row < A.rows(); ++row) {
            residualMoments.add(b(row) - A.row(row) * params);
        }

        double predictedTestLoss{params.transpose() * x};
        double predictedTestLossVariance{
            CBasicStatistics::maximumLikelihoodVariance(residualMoments)};
        predictedTestLosses.push_back(CBasicStatistics::momentsAccumulator(
            1.0, predictedTestLoss, predictedTestLossVariance));
        LOG_TRACE(<< "prediction(x = " << x.transpose() << ", fold = " << target
                  << ") = (mean = " << predictedTestLoss
                  << ", variance = " << predictedTestLossVariance << ")");
    }

    return predictedTestLosses;
}

std::size_t CBoostedTreeImpl::numberFeatures() const {
    return m_Encoder->numberEncodedColumns();
}

std::size_t CBoostedTreeImpl::featureBagSize() const {
    return static_cast<std::size_t>(std::max(
        std::ceil(m_FeatureBagFraction * static_cast<double>(this->numberFeatures())), 1.0));
}

CBoostedTreeImpl::TSizeVec CBoostedTreeImpl::featureBag() const {

    std::size_t size{this->featureBagSize()};

    TSizeVec features{this->candidateRegressorFeatures()};
    if (size >= features.size()) {
        return features;
    }

    TSizeVec sample;
    TDoubleVec probabilities(m_FeatureSampleProbabilities);
    CSampling::categoricalSampleWithoutReplacement(m_Rng, probabilities, size, sample);

    return sample;
}

void CBoostedTreeImpl::refreshPredictionsAndLossDerivatives(core::CDataFrame& frame,
                                                            const core::CPackedBitVector& trainingRowMask,
                                                            const core::CPackedBitVector& testingRowMask,
                                                            double eta,
                                                            TNodeVec& tree) const {

    using TArgMinLossVec = std::vector<CArgMinLoss>;

    TArgMinLossVec leafValues(
        tree.size(), m_Loss->minimizer(m_Regularization.leafWeightPenaltyMultiplier()));
    auto nextPass = [&] {
        bool done{true};
        for (const auto& value : leafValues) {
            done &= (value.nextPass() == false);
        }
        return done == false;
    };

    do {
        auto result = frame.readRows(
            m_NumberThreads, 0, frame.numberRows(),
            core::bindRetrievableState(
                [&](TArgMinLossVec& leafValues_, TRowItr beginRows, TRowItr endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        double prediction{readPrediction(*row)};
                        double actual{readActual(*row, m_DependentVariable)};
                        double weight{readExampleWeight(*row)};
                        leafValues_[root(tree).leafIndex(m_Encoder->encode(*row), tree)]
                            .add(prediction, actual, weight);
                    }
                },
                std::move(leafValues)),
            &trainingRowMask);

        leafValues = std::move(result.first[0].s_FunctionState);
        for (std::size_t i = 1; i < result.first.size(); ++i) {
            for (std::size_t j = 0; j < leafValues.size(); ++j) {
                leafValues[j].merge(result.first[i].s_FunctionState[j]);
            }
        }
    } while (nextPass());

    for (std::size_t i = 0; i < tree.size(); ++i) {
        tree[i].value(eta * leafValues[i].value());
    }

    LOG_TRACE(<< "tree =\n" << root(tree).print(tree));

    core::CPackedBitVector updateRowMask{trainingRowMask | testingRowMask};
    auto results = frame.writeColumns(
        m_NumberThreads, 0, frame.numberRows(),
        [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                std::size_t numberColumns{row->numberColumns()};
                double prediction{readPrediction(*row) +
                                  root(tree).value(m_Encoder->encode(*row), tree)};
                double actual{readActual(*row, m_DependentVariable)};
                double weight{readExampleWeight(*row)};

                row->writeColumn(predictionColumn(numberColumns), prediction);
                row->writeColumn(lossGradientColumn(numberColumns),
                                 m_Loss->gradient(prediction, actual, weight));
                row->writeColumn(lossCurvatureColumn(numberColumns),
                                 m_Loss->curvature(prediction, actual, weight));
            }
        },
        &updateRowMask);
}

double CBoostedTreeImpl::meanLoss(const core::CDataFrame& frame,
                                  const core::CPackedBitVector& rowMask) const {

    auto results = frame.readRows(
        m_NumberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](TMeanAccumulator& loss, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    double prediction{readPrediction(*row)};
                    double actual{readActual(*row, m_DependentVariable)};
                    loss.add(m_Loss->value(prediction, actual));
                }
            },
            TMeanAccumulator{}),
        &rowMask);

    TMeanAccumulator loss;
    for (const auto& result : results.first) {
        loss += result.s_FunctionState;
    }

    LOG_TRACE(<< "mean loss = " << CBasicStatistics::mean(loss));

    return CBasicStatistics::mean(loss);
}

CBoostedTreeImpl::TSizeVec CBoostedTreeImpl::candidateRegressorFeatures() const {
    TSizeVec result;
    result.reserve(m_FeatureSampleProbabilities.size());
    for (std::size_t i = 0; i < m_FeatureSampleProbabilities.size(); ++i) {
        if (m_FeatureSampleProbabilities[i] > 0.0) {
            result.push_back(i);
        }
    }
    return result;
}

const CBoostedTreeNode& CBoostedTreeImpl::root(const TNodeVec& tree) {
    return tree[0];
}

double CBoostedTreeImpl::predictRow(const CEncodedDataFrameRowRef& row,
                                    const TNodeVecVec& forest) {
    double result{0.0};
    for (const auto& tree : forest) {
        result += root(tree).value(row, tree);
    }
    return result;
}

bool CBoostedTreeImpl::selectNextHyperparameters(const TMeanVarAccumulator& lossMoments,
                                                 CBayesianOptimisation& bopt) {

    TVector parameters{this->numberHyperparametersToTune()};

    // Read parameters for last round.
    int i{0};
    if (m_DownsampleFactorOverride == boost::none) {
        parameters(i++) = std::log(m_DownsampleFactor);
    }
    if (m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        parameters(i++) = std::log(m_Regularization.depthPenaltyMultiplier());
    }
    if (m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        parameters(i++) = std::log(m_Regularization.leafWeightPenaltyMultiplier());
    }
    if (m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        parameters(i++) = std::log(m_Regularization.treeSizePenaltyMultiplier());
    }
    if (m_RegularizationOverride.softTreeDepthLimit() == boost::none) {
        parameters(i++) = m_Regularization.softTreeDepthLimit();
    }
    if (m_RegularizationOverride.softTreeDepthTolerance() == boost::none) {
        parameters(i++) = m_Regularization.softTreeDepthTolerance();
    }
    if (m_EtaOverride == boost::none) {
        parameters(i++) = std::log(m_Eta);
        parameters(i++) = m_EtaGrowthRatePerTree;
    }
    if (m_FeatureBagFractionOverride == boost::none) {
        parameters(i++) = m_FeatureBagFraction;
    }

    double meanLoss{CBasicStatistics::mean(lossMoments)};
    double lossVariance{CBasicStatistics::variance(lossMoments)};

    LOG_TRACE(<< "round = " << m_CurrentRound << " loss = " << meanLoss
              << ": regularization = " << m_Regularization.print()
              << ", downsample factor = " << m_DownsampleFactor << ", eta = " << m_Eta
              << ", eta growth rate per tree = " << m_EtaGrowthRatePerTree
              << ", feature bag fraction = " << m_FeatureBagFraction);

    bopt.add(parameters, meanLoss, lossVariance);
    if (3 * m_CurrentRound < m_NumberRounds) {
        std::generate_n(parameters.data(), parameters.size(), [&]() {
            return CSampling::uniformSample(m_Rng, 0.0, 1.0);
        });
        TVector minBoundary;
        TVector maxBoundary;
        std::tie(minBoundary, maxBoundary) = bopt.boundingBox();
        parameters = minBoundary + parameters.cwiseProduct(maxBoundary - minBoundary);
    } else {
        std::tie(parameters, std::ignore) = bopt.maximumExpectedImprovement();
    }

    // Downsampling acts as a regularisation and also increases the variance
    // of each of the base learners so we scale the other regularisation terms
    // and the weight shrinkage to compensate.
    double scale{1.0};

    // Write parameters for next round.
    i = 0;
    if (m_DownsampleFactorOverride == boost::none) {
        m_DownsampleFactor = std::exp(parameters(i++));
        TVector minBoundary;
        TVector maxBoundary;
        std::tie(minBoundary, maxBoundary) = bopt.boundingBox();
        scale = std::min(scale, 2.0 * m_DownsampleFactor /
                                    (std::exp(minBoundary(0)) + std::exp(maxBoundary(0))));
    }
    if (m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        m_Regularization.depthPenaltyMultiplier(std::exp(parameters(i++)));
    }
    if (m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        m_Regularization.leafWeightPenaltyMultiplier(scale * std::exp(parameters(i++)));
    }
    if (m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        m_Regularization.treeSizePenaltyMultiplier(scale * std::exp(parameters(i++)));
    }
    if (m_RegularizationOverride.softTreeDepthLimit() == boost::none) {
        m_Regularization.softTreeDepthLimit(parameters(i++));
    }
    if (m_RegularizationOverride.softTreeDepthTolerance() == boost::none) {
        m_Regularization.softTreeDepthTolerance(parameters(i++));
    }
    if (m_EtaOverride == boost::none) {
        m_Eta = std::exp(scale * parameters(i++));
        m_EtaGrowthRatePerTree = parameters(i++);
    }
    if (m_FeatureBagFractionOverride == boost::none) {
        m_FeatureBagFraction = parameters(i++);
    }

    return true;
}

void CBoostedTreeImpl::captureBestHyperparameters(const TMeanVarAccumulator& lossMoments,
                                                  std::size_t maximumNumberTrees) {
    // We capture the parameters with the lowest error at one standard
    // deviation above the mean. If the mean error improvement is marginal
    // we prefer the solution with the least variation across the folds.
    double loss{lossAtNSigma(1.0, lossMoments)};
    if (loss < m_BestForestTestLoss) {
        m_BestForestTestLoss = loss;
        m_BestHyperparameters = CBoostedTreeHyperparameters{
            m_Regularization,       m_DownsampleFactor, m_Eta,
            m_EtaGrowthRatePerTree, maximumNumberTrees, m_FeatureBagFraction};
    }
}

void CBoostedTreeImpl::restoreBestHyperparameters() {
    m_Regularization = m_BestHyperparameters.regularization();
    m_DownsampleFactor = m_BestHyperparameters.downsampleFactor();
    m_Eta = m_BestHyperparameters.eta();
    m_EtaGrowthRatePerTree = m_BestHyperparameters.etaGrowthRatePerTree();
    m_MaximumNumberTrees = m_BestHyperparameters.maximumNumberTrees();
    m_FeatureBagFraction = m_BestHyperparameters.featureBagFraction();
    LOG_INFO(<< "regularization* = " << m_Regularization.print()
             << ", downsample factor* = " << m_DownsampleFactor << ", eta* = " << m_Eta
             << ", eta growth rate per tree* = " << m_EtaGrowthRatePerTree
             << ", maximum number trees* = " << m_MaximumNumberTrees
             << ", feature bag fraction* = " << m_FeatureBagFraction);
}

std::size_t CBoostedTreeImpl::numberHyperparametersToTune() const {
    return m_RegularizationOverride.countNotSet() +
           (m_DownsampleFactorOverride != boost::none ? 0 : 1) +
           (m_EtaOverride != boost::none ? 0 : 2) +
           (m_FeatureBagFractionOverride != boost::none ? 0 : 1);
}

std::size_t CBoostedTreeImpl::maximumTreeSize(const core::CPackedBitVector& trainingRowMask) const {
    return this->maximumTreeSize(static_cast<std::size_t>(trainingRowMask.manhattan()));
}

std::size_t CBoostedTreeImpl::maximumTreeSize(std::size_t numberRows) const {
    return static_cast<std::size_t>(
        std::ceil(10.0 * std::sqrt(static_cast<double>(numberRows))));
}

const std::size_t CBoostedTreeImpl::PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE{256};

namespace {
const std::string VERSION_7_5_TAG{"7.5"};
const std::string VERSION_7_6_TAG{"7.6"};
const TStrVec SUPPORTED_VERSIONS{VERSION_7_5_TAG, VERSION_7_6_TAG};

const std::string BAYESIAN_OPTIMIZATION_TAG{"bayesian_optimization"};
const std::string BEST_FOREST_TAG{"best_forest"};
const std::string BEST_FOREST_TEST_LOSS_TAG{"best_forest_test_loss"};
const std::string BEST_HYPERPARAMETERS_TAG{"best_hyperparameters"};
const std::string CURRENT_ROUND_TAG{"current_round"};
const std::string DEPENDENT_VARIABLE_TAG{"dependent_variable"};
const std::string DOWNSAMPLE_FACTOR_TAG{"downsample_factor"};
const std::string ENCODER_TAG{"encoder"};
const std::string ETA_GROWTH_RATE_PER_TREE_TAG{"eta_growth_rate_per_tree"};
const std::string ETA_OVERRIDE_TAG{"eta_override"};
const std::string ETA_TAG{"eta"};
const std::string FEATURE_BAG_FRACTION_OVERRIDE_TAG{"feature_bag_fraction_override"};
const std::string FEATURE_BAG_FRACTION_TAG{"feature_bag_fraction"};
const std::string FEATURE_DATA_TYPES_TAG{"feature_data_types"};
const std::string FEATURE_SAMPLE_PROBABILITIES_TAG{"feature_sample_probabilities"};
const std::string FOLD_ROUND_TEST_LOSSES_TAG{"fold_round_test_losses"};
const std::string LOSS_TAG{"loss"};
const std::string MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG{"maximum_attempts_to_add_tree"};
const std::string MAXIMUM_NUMBER_TREES_OVERRIDE_TAG{"maximum_number_trees_override"};
const std::string MAXIMUM_NUMBER_TREES_TAG{"maximum_number_trees"};
const std::string MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG{
    "maximum_optimisation_rounds_per_hyperparameter"};
const std::string MISSING_FEATURE_ROW_MASKS_TAG{"missing_feature_row_masks"};
const std::string NUMBER_FOLDS_TAG{"number_folds"};
const std::string NUMBER_FOLDS_OVERRIDE_TAG{"number_folds_override"};
const std::string NUMBER_ROUNDS_TAG{"number_rounds"};
const std::string NUMBER_SPLITS_PER_FEATURE_TAG{"number_splits_per_feature"};
const std::string NUMBER_THREADS_TAG{"number_threads"};
const std::string RANDOM_NUMBER_GENERATOR_TAG{"random_number_generator"};
const std::string REGULARIZATION_TAG{"regularization"};
const std::string REGULARIZATION_OVERRIDE_TAG{"regularization_override"};
const std::string ROWS_PER_FEATURE_TAG{"rows_per_feature"};
const std::string STOP_CROSS_VALIDATION_EARLY_TAG{"stop_cross_validation_eraly"};
const std::string TESTING_ROW_MASKS_TAG{"testing_row_masks"};
const std::string TRAINING_ROW_MASKS_TAG{"training_row_masks"};
const std::string TRAINING_PROGRESS_TAG{"training_progress"};
const std::string TOP_SHAP_VALUES_TAG{"top_shap_values"};
const std::string FIRST_SHAP_COLUMN_INDEX{"first_shap_column_index"};
const std::string LAST_SHAP_COLUMN_INDEX{"last_shap_column_index"};
}

const std::string& CBoostedTreeImpl::bestHyperparametersName() {
    return BEST_HYPERPARAMETERS_TAG;
}

const std::string& CBoostedTreeImpl::bestRegularizationHyperparametersName() {
    return CBoostedTreeHyperparameters::HYPERPARAM_REGULARIZATION_TAG;
}

CBoostedTreeImpl::TStrVec CBoostedTreeImpl::bestHyperparameterNames() {
    return {CBoostedTreeHyperparameters::HYPERPARAM_DOWNSAMPLE_FACTOR_TAG,
            CBoostedTreeHyperparameters::HYPERPARAM_ETA_TAG,
            CBoostedTreeHyperparameters::HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
            CBoostedTreeHyperparameters::HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
            TRegularization::REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG,
            TRegularization::REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG,
            TRegularization::REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
            TRegularization::REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG,
            TRegularization::REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG};
}

void CBoostedTreeImpl::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(VERSION_7_6_TAG, "", inserter);
    core::CPersistUtils::persist(BAYESIAN_OPTIMIZATION_TAG, *m_BayesianOptimization, inserter);
    core::CPersistUtils::persist(BEST_FOREST_TEST_LOSS_TAG, m_BestForestTestLoss, inserter);
    core::CPersistUtils::persist(CURRENT_ROUND_TAG, m_CurrentRound, inserter);
    core::CPersistUtils::persist(DEPENDENT_VARIABLE_TAG, m_DependentVariable, inserter);
    core::CPersistUtils::persist(DOWNSAMPLE_FACTOR_TAG, m_DownsampleFactor, inserter);
    core::CPersistUtils::persist(ENCODER_TAG, *m_Encoder, inserter);
    core::CPersistUtils::persist(ETA_GROWTH_RATE_PER_TREE_TAG,
                                 m_EtaGrowthRatePerTree, inserter);
    core::CPersistUtils::persist(ETA_TAG, m_Eta, inserter);
    core::CPersistUtils::persist(FEATURE_BAG_FRACTION_TAG, m_FeatureBagFraction, inserter);
    core::CPersistUtils::persist(FEATURE_DATA_TYPES_TAG, m_FeatureDataTypes, inserter);
    core::CPersistUtils::persist(FEATURE_SAMPLE_PROBABILITIES_TAG,
                                 m_FeatureSampleProbabilities, inserter);
    core::CPersistUtils::persist(FOLD_ROUND_TEST_LOSSES_TAG, m_FoldRoundTestLosses, inserter);
    core::CPersistUtils::persist(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                                 m_MaximumAttemptsToAddTree, inserter);
    core::CPersistUtils::persist(MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                                 m_MaximumOptimisationRoundsPerHyperparameter, inserter);
    core::CPersistUtils::persist(MISSING_FEATURE_ROW_MASKS_TAG,
                                 m_MissingFeatureRowMasks, inserter);
    core::CPersistUtils::persist(NUMBER_FOLDS_TAG, m_NumberFolds, inserter);
    core::CPersistUtils::persist(NUMBER_FOLDS_OVERRIDE_TAG, m_NumberFoldsOverride, inserter);
    core::CPersistUtils::persist(NUMBER_ROUNDS_TAG, m_NumberRounds, inserter);
    core::CPersistUtils::persist(NUMBER_SPLITS_PER_FEATURE_TAG,
                                 m_NumberSplitsPerFeature, inserter);
    core::CPersistUtils::persist(NUMBER_THREADS_TAG, m_NumberThreads, inserter);
    inserter.insertValue(RANDOM_NUMBER_GENERATOR_TAG, m_Rng.toString());
    core::CPersistUtils::persist(REGULARIZATION_OVERRIDE_TAG,
                                 m_RegularizationOverride, inserter);
    core::CPersistUtils::persist(REGULARIZATION_TAG, m_Regularization, inserter);
    core::CPersistUtils::persist(ROWS_PER_FEATURE_TAG, m_RowsPerFeature, inserter);
    core::CPersistUtils::persist(STOP_CROSS_VALIDATION_EARLY_TAG,
                                 m_StopCrossValidationEarly, inserter);
    core::CPersistUtils::persist(TESTING_ROW_MASKS_TAG, m_TestingRowMasks, inserter);
    core::CPersistUtils::persist(MAXIMUM_NUMBER_TREES_TAG, m_MaximumNumberTrees, inserter);
    core::CPersistUtils::persist(TRAINING_ROW_MASKS_TAG, m_TrainingRowMasks, inserter);
    core::CPersistUtils::persist(TRAINING_PROGRESS_TAG, m_TrainingProgress, inserter);
    core::CPersistUtils::persist(BEST_FOREST_TAG, m_BestForest, inserter);
    core::CPersistUtils::persist(BEST_HYPERPARAMETERS_TAG, m_BestHyperparameters, inserter);
    core::CPersistUtils::persist(ETA_OVERRIDE_TAG, m_EtaOverride, inserter);
    core::CPersistUtils::persist(FEATURE_BAG_FRACTION_OVERRIDE_TAG,
                                 m_FeatureBagFractionOverride, inserter);
    core::CPersistUtils::persist(MAXIMUM_NUMBER_TREES_OVERRIDE_TAG,
                                 m_MaximumNumberTreesOverride, inserter);
    inserter.insertValue(LOSS_TAG, m_Loss->name());
    core::CPersistUtils::persist(TOP_SHAP_VALUES_TAG, m_TopShapValues, inserter);
    core::CPersistUtils::persist(FIRST_SHAP_COLUMN_INDEX, m_FirstShapColumnIndex, inserter);
    core::CPersistUtils::persist(LAST_SHAP_COLUMN_INDEX, m_LastShapColumnIndex, inserter);
}

bool CBoostedTreeImpl::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    if (traverser.name() == VERSION_7_5_TAG) {
        // Force downsample factor to 1.0.
        m_DownsampleFactorOverride = 1.0;
        m_DownsampleFactor = 1.0;
        m_BestHyperparameters.downsampleFactor(1.0);
        // We can't stop cross-validation early because we haven't gathered the
        // per fold test losses.
        m_StopCrossValidationEarly = false;
    } else if (traverser.name() != VERSION_7_6_TAG) {
        LOG_ERROR(<< "Input error: unsupported state serialization version. "
                  << "Currently supported versions: "
                  << core::CContainerPrinter::print(SUPPORTED_VERSIONS) << ".");
        return false;
    }

    do {
        const std::string& name = traverser.name();
        RESTORE_NO_ERROR(BAYESIAN_OPTIMIZATION_TAG,
                         m_BayesianOptimization =
                             std::make_unique<CBayesianOptimisation>(traverser))
        RESTORE(BEST_FOREST_TEST_LOSS_TAG,
                core::CPersistUtils::restore(BEST_FOREST_TEST_LOSS_TAG,
                                             m_BestForestTestLoss, traverser))
        RESTORE(CURRENT_ROUND_TAG,
                core::CPersistUtils::restore(CURRENT_ROUND_TAG, m_CurrentRound, traverser))
        RESTORE(DEPENDENT_VARIABLE_TAG,
                core::CPersistUtils::restore(DEPENDENT_VARIABLE_TAG,
                                             m_DependentVariable, traverser))
        RESTORE(DOWNSAMPLE_FACTOR_TAG,
                core::CPersistUtils::restore(DOWNSAMPLE_FACTOR_TAG, m_DownsampleFactor, traverser))
        RESTORE_NO_ERROR(ENCODER_TAG,
                         m_Encoder = std::make_unique<CDataFrameCategoryEncoder>(traverser))
        RESTORE(ETA_GROWTH_RATE_PER_TREE_TAG,
                core::CPersistUtils::restore(ETA_GROWTH_RATE_PER_TREE_TAG,
                                             m_EtaGrowthRatePerTree, traverser))
        RESTORE(ETA_TAG, core::CPersistUtils::restore(ETA_TAG, m_Eta, traverser))
        RESTORE(FEATURE_BAG_FRACTION_TAG,
                core::CPersistUtils::restore(FEATURE_BAG_FRACTION_TAG,
                                             m_FeatureBagFraction, traverser))
        RESTORE(FEATURE_DATA_TYPES_TAG,
                core::CPersistUtils::restore(FEATURE_DATA_TYPES_TAG,
                                             m_FeatureDataTypes, traverser));
        RESTORE(FEATURE_SAMPLE_PROBABILITIES_TAG,
                core::CPersistUtils::restore(FEATURE_SAMPLE_PROBABILITIES_TAG,
                                             m_FeatureSampleProbabilities, traverser))
        RESTORE(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                core::CPersistUtils::restore(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                                             m_MaximumAttemptsToAddTree, traverser))
        RESTORE(MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                core::CPersistUtils::restore(
                    MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                    m_MaximumOptimisationRoundsPerHyperparameter, traverser))
        RESTORE(MISSING_FEATURE_ROW_MASKS_TAG,
                core::CPersistUtils::restore(MISSING_FEATURE_ROW_MASKS_TAG,
                                             m_MissingFeatureRowMasks, traverser))
        RESTORE(NUMBER_FOLDS_TAG,
                core::CPersistUtils::restore(NUMBER_FOLDS_TAG, m_NumberFolds, traverser))
        RESTORE(NUMBER_FOLDS_OVERRIDE_TAG,
                core::CPersistUtils::restore(NUMBER_FOLDS_OVERRIDE_TAG,
                                             m_NumberFoldsOverride, traverser))
        RESTORE(NUMBER_ROUNDS_TAG,
                core::CPersistUtils::restore(NUMBER_ROUNDS_TAG, m_NumberRounds, traverser))
        RESTORE(NUMBER_SPLITS_PER_FEATURE_TAG,
                core::CPersistUtils::restore(NUMBER_SPLITS_PER_FEATURE_TAG,
                                             m_NumberSplitsPerFeature, traverser))
        RESTORE(NUMBER_THREADS_TAG,
                core::CPersistUtils::restore(NUMBER_THREADS_TAG, m_NumberThreads, traverser))
        RESTORE(RANDOM_NUMBER_GENERATOR_TAG, m_Rng.fromString(traverser.value()))
        RESTORE(REGULARIZATION_TAG,
                core::CPersistUtils::restore(REGULARIZATION_TAG, m_Regularization, traverser))
        RESTORE(REGULARIZATION_OVERRIDE_TAG,
                core::CPersistUtils::restore(REGULARIZATION_OVERRIDE_TAG,
                                             m_RegularizationOverride, traverser))
        RESTORE(ROWS_PER_FEATURE_TAG,
                core::CPersistUtils::restore(ROWS_PER_FEATURE_TAG, m_RowsPerFeature, traverser))
        RESTORE(STOP_CROSS_VALIDATION_EARLY_TAG,
                core::CPersistUtils::restore(STOP_CROSS_VALIDATION_EARLY_TAG,
                                             m_StopCrossValidationEarly, traverser))
        RESTORE(TESTING_ROW_MASKS_TAG,
                core::CPersistUtils::restore(TESTING_ROW_MASKS_TAG, m_TestingRowMasks, traverser))
        RESTORE(MAXIMUM_NUMBER_TREES_TAG,
                core::CPersistUtils::restore(MAXIMUM_NUMBER_TREES_TAG,
                                             m_MaximumNumberTrees, traverser))
        RESTORE(TRAINING_ROW_MASKS_TAG,
                core::CPersistUtils::restore(TRAINING_ROW_MASKS_TAG, m_TrainingRowMasks, traverser))
        RESTORE(TRAINING_PROGRESS_TAG,
                core::CPersistUtils::restore(TRAINING_PROGRESS_TAG, m_TrainingProgress, traverser))
        RESTORE(BEST_FOREST_TAG,
                core::CPersistUtils::restore(BEST_FOREST_TAG, m_BestForest, traverser))
        RESTORE(BEST_HYPERPARAMETERS_TAG,
                core::CPersistUtils::restore(BEST_HYPERPARAMETERS_TAG,
                                             m_BestHyperparameters, traverser))
        RESTORE(ETA_OVERRIDE_TAG,
                core::CPersistUtils::restore(ETA_OVERRIDE_TAG, m_EtaOverride, traverser))
        RESTORE(FEATURE_BAG_FRACTION_OVERRIDE_TAG,
                core::CPersistUtils::restore(FEATURE_BAG_FRACTION_OVERRIDE_TAG,
                                             m_FeatureBagFractionOverride, traverser))
        RESTORE(MAXIMUM_NUMBER_TREES_OVERRIDE_TAG,
                core::CPersistUtils::restore(MAXIMUM_NUMBER_TREES_OVERRIDE_TAG,
                                             m_MaximumNumberTreesOverride, traverser))
        RESTORE(LOSS_TAG, restoreLoss(m_Loss, traverser))
        RESTORE(TOP_SHAP_VALUES_TAG,
                core::CPersistUtils::restore(TOP_SHAP_VALUES_TAG, m_TopShapValues, traverser))
        RESTORE(FIRST_SHAP_COLUMN_INDEX,
                core::CPersistUtils::restore(FIRST_SHAP_COLUMN_INDEX,
                                             m_FirstShapColumnIndex, traverser))
        RESTORE(LAST_SHAP_COLUMN_INDEX,
                core::CPersistUtils::restore(LAST_SHAP_COLUMN_INDEX,
                                             m_LastShapColumnIndex, traverser))
    } while (traverser.next());

    return true;
}

bool CBoostedTreeImpl::restoreLoss(CBoostedTree::TLossFunctionUPtr& loss,
                                   core::CStateRestoreTraverser& traverser) {
    const std::string& lossFunctionName{traverser.value()};
    if (lossFunctionName == CMse::NAME) {
        loss = std::make_unique<CMse>();
        return true;
    }
    LOG_ERROR(<< "Error restoring loss function. Unknown loss function type '"
              << lossFunctionName << "'.");
    return false;
}

std::size_t CBoostedTreeImpl::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_Loss)};
    mem += core::CMemory::dynamicSize(m_Encoder);
    mem += core::CMemory::dynamicSize(m_FeatureSampleProbabilities);
    mem += core::CMemory::dynamicSize(m_MissingFeatureRowMasks);
    mem += core::CMemory::dynamicSize(m_TrainingRowMasks);
    mem += core::CMemory::dynamicSize(m_TestingRowMasks);
    mem += core::CMemory::dynamicSize(m_FoldRoundTestLosses);
    mem += core::CMemory::dynamicSize(m_BestForest);
    mem += core::CMemory::dynamicSize(m_BayesianOptimization);
    return mem;
}

void CBoostedTreeImpl::accept(CBoostedTree::CVisitor& visitor) {
    m_Encoder->accept(visitor);
    for (const auto& tree : m_BestForest) {
        visitor.addTree();
        for (std::size_t i = 0; i < tree.size(); ++i) {
            tree[i].accept(visitor);
        }
    }
}

const double CBoostedTreeImpl::MINIMUM_RELATIVE_GAIN_PER_SPLIT{1e-7};
const double CBoostedTreeImpl::INF{std::numeric_limits<double>::max()};

const CBoostedTreeHyperparameters& CBoostedTreeImpl::bestHyperparameters() const {
    return m_BestHyperparameters;
}

void CBoostedTreeImpl::computeShapValues(core::CDataFrame& frame, const TProgressCallback&) {
    if (m_TopShapValues > 0) {
        if (m_BestForestTestLoss == INF) {
            HANDLE_FATAL(<< "Internal error: no model available for prediction. "
                         << "Please report this problem.");
            return;
        }
        auto treeFeatureImportance = std::make_unique<CTreeShapFeatureImportance>(
            m_BestForest, m_NumberThreads);
        std::size_t numberInputFields = m_NumberInputColumns - 1;
        // resize data frame to write SHAP values
        std::size_t offset{frame.numberColumns()};
        frame.resizeColumns(m_NumberThreads, frame.numberColumns() + numberInputFields);
        m_FirstShapColumnIndex = offset;
        m_LastShapColumnIndex = frame.numberColumns() - 1;
        TStrVec columnNames(frame.columnNames());
        for (std::size_t i = 0; i < numberInputFields; ++i) {
            columnNames[offset + i] = CDataFrameRegressionModel::SHAP_PREFIX +
                                      frame.columnNames()[i];
        }
        frame.columnNames(columnNames);
        treeFeatureImportance->shap(frame, *m_Encoder, offset);
    }
}

CBoostedTreeImpl::TSizeRange CBoostedTreeImpl::columnsHoldingShapValues() const {
    return TSizeRange{m_FirstShapColumnIndex, m_LastShapColumnIndex + 1};
}

std::size_t CBoostedTreeImpl::topShapValues() const {
    return m_TopShapValues;
}

std::size_t CBoostedTreeImpl::numberInputColumns() const {
    return m_NumberInputColumns;
}
}
}
