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

#include <maths/analytics/CBoostedTreeImpl.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CLoopProgress.h>
#include <core/CMemory.h>
#include <core/CPackedBitVector.h>
#include <core/CPersistUtils.h>
#include <core/CProgramCounters.h>
#include <core/CStopWatch.h>
#include <core/Concurrency.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CBoostedTreeFactory.h>
#include <maths/analytics/CBoostedTreeLeafNodeStatistics.h>
#include <maths/analytics/CBoostedTreeLeafNodeStatisticsIncremental.h>
#include <maths/analytics/CBoostedTreeLeafNodeStatisticsScratch.h>
#include <maths/analytics/CBoostedTreeLoss.h>
#include <maths/analytics/CBoostedTreeUtils.h>
#include <maths/analytics/CDataFrameAnalysisInstrumentationInterface.h>
#include <maths/analytics/CDataFrameCategoryEncoder.h>
#include <maths/analytics/CDataFrameUtils.h>
#include <maths/analytics/CTreeShapFeatureImportance.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CBayesianOptimisation.h>
#include <maths/common/CQuantileSketch.h>
#include <maths/common/CSampling.h>
#include <maths/common/CSetTools.h>
#include <maths/common/CSpline.h>
#include <maths/common/MathsTypes.h>

#include <boost/circular_buffer.hpp>
#include <boost/unordered_set.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>

namespace ml {
namespace maths {
namespace analytics {
using namespace boosted_tree;
using namespace boosted_tree_detail;
using TStrVec = std::vector<std::string>;
using TRowItr = core::CDataFrame::TRowItr;
using TMeanAccumulator = common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMemoryUsageCallback = CDataFrameAnalysisInstrumentationInterface::TMemoryUsageCallback;

namespace {
// It isn't critical to recompute splits every tree we add because random
// downsampling means they're only approximate estimates of the full data
// quantiles anyway. So we amortise their compute cost w.r.t. training trees
// by only refreshing once every MINIMUM_SPLIT_REFRESH_INTERVAL trees we add.
const double MINIMUM_SPLIT_REFRESH_INTERVAL{3.0};
const std::string HYPERPARAMETER_OPTIMIZATION_ROUND{"hyperparameter_optimization_round_"};
const std::string TRAIN_FINAL_FOREST{"train_final_forest"};
const double BYTES_IN_MB{static_cast<double>(core::constants::BYTES_IN_MEGABYTES)};

//! \brief Record the memory used by a supplied object using the RAII idiom.
class CScopeRecordMemoryUsage {
public:
    template<typename T>
    CScopeRecordMemoryUsage(const T& object, TMemoryUsageCallback&& recordMemoryUsage)
        : m_RecordMemoryUsage{std::move(recordMemoryUsage)},
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
    TMemoryUsageCallback m_RecordMemoryUsage;
    std::int64_t m_MemoryUsage;
};

//! \brief Resets a random number generator then jumps 2^64 values using the RAII idiom.
class CResetAndJumpOnExit {
public:
    explicit CResetAndJumpOnExit(common::CPRNG::CXorOShiro128Plus& rng)
        : m_Rng{rng}, m_RngCopy{rng} {}

    ~CResetAndJumpOnExit() {
        m_Rng = m_RngCopy;
        m_Rng.jump();
    }

    CResetAndJumpOnExit(const CResetAndJumpOnExit&) = delete;
    CResetAndJumpOnExit& operator=(const CResetAndJumpOnExit&) = delete;

private:
    common::CPRNG::CXorOShiro128Plus& m_Rng;
    common::CPRNG::CXorOShiro128Plus m_RngCopy;
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
    explicit CTrainForestStoppingCondition(std::size_t maximumNumberTrees)
        : m_MaximumNumberTrees{maximumNumberTrees},
          m_MaximumNumberTreesWithoutImprovement{std::max(
              static_cast<std::size_t>(0.075 * static_cast<double>(maximumNumberTrees) + 0.5),
              std::size_t{1})} {}

    std::size_t bestSize() const { return std::get<SIZE>(m_BestTestLoss[0]); }

    double bestTestLoss() const {
        return std::get<TEST_LOSS>(m_BestTestLoss[0]);
    }

    double lossGap() const {
        return std::max(std::get<TEST_LOSS>(m_BestTestLoss[0]) -
                            std::get<TRAIN_LOSS>(m_BestTestLoss[0]),
                        0.0);
    }

    template<typename FUNC>
    bool shouldStop(std::size_t numberTrees, FUNC computeLoss) {
        double trainLoss;
        double testLoss;
        std::tie(trainLoss, testLoss) = computeLoss();
        m_BestTestLoss.add({testLoss, numberTrees, trainLoss});
        LOG_TRACE(<< "number trees = " << numberTrees << ", train loss = " << trainLoss
                  << ", test loss = " << testLoss);
        if (numberTrees - std::get<SIZE>(m_BestTestLoss[0]) > m_MaximumNumberTreesWithoutImprovement) {
            return true;
        }
        return numberTrees > m_MaximumNumberTrees;
    }

private:
    using TDoubleSizeDoubleTrMinAccumulator =
        common::CBasicStatistics::SMin<std::tuple<double, std::size_t, double>>::TAccumulator;

private:
    static constexpr std::size_t TEST_LOSS{0};
    static constexpr std::size_t SIZE{1};
    static constexpr std::size_t TRAIN_LOSS{2};

private:
    std::size_t m_MaximumNumberTrees;
    std::size_t m_MaximumNumberTreesWithoutImprovement;
    TDoubleSizeDoubleTrMinAccumulator m_BestTestLoss;
};

double trace(std::size_t columns, const TMemoryMappedFloatVector& upperTriangle) {
    // This assumes the upper triangle of the matrix is stored row major.
    double result{0.0};
    for (int i = 0, j = static_cast<int>(columns);
         i < upperTriangle.size() && j > 0; i += j, --j) {
        result += upperTriangle(i);
    }
    return result;
}

TSizeVec merge(const TSizeVec& x, TSizeVec y) {
    std::size_t split{y.size()};
    y.insert(y.end(), x.begin(), x.end());
    std::inplace_merge(y.begin(), y.begin() + split, y.end());
    y.erase(std::unique(y.begin(), y.end()), y.end());
    return y;
}

CDataFrameTrainBoostedTreeInstrumentationStub INSTRUMENTATION_STUB;

double numberForestNodes(const CBoostedTreeImpl::TNodeVecVec& forest) {
    double numberNodes{0.0};
    for (const auto& tree : forest) {
        numberNodes += static_cast<double>(tree.size());
    }
    return numberNodes;
}
}

CBoostedTreeImpl::CBoostedTreeImpl(std::size_t numberThreads,
                                   CBoostedTree::TLossFunctionUPtr loss,
                                   TAnalysisInstrumentationPtr instrumentation)
    : m_NumberThreads{numberThreads}, m_Loss{std::move(loss)},
      m_Instrumentation{instrumentation != nullptr ? instrumentation : &INSTRUMENTATION_STUB} {
}

CBoostedTreeImpl::CBoostedTreeImpl() = default;
CBoostedTreeImpl::~CBoostedTreeImpl() = default;
CBoostedTreeImpl::CBoostedTreeImpl(CBoostedTreeImpl&&) noexcept = default;
CBoostedTreeImpl& CBoostedTreeImpl::operator=(CBoostedTreeImpl&&) noexcept = default;

void CBoostedTreeImpl::train(core::CDataFrame& frame,
                             const TTrainingStateCallback& recordTrainStateCallback) {

    this->checkTrainInvariants(frame);

    m_Instrumentation->type(
        m_Loss->isRegression()
            ? CDataFrameTrainBoostedTreeInstrumentationInterface::E_Regression
            : CDataFrameTrainBoostedTreeInstrumentationInterface::E_Classification);

    LOG_TRACE(<< "Main training loop...");

    m_TrainingProgress.progressCallback(m_Instrumentation->progressCallback());

    std::int64_t lastMemoryUsage(this->memoryUsage());

    core::CPackedBitVector allTrainingRowsMask{this->allTrainingRowsMask()};
    core::CPackedBitVector noRowsMask{allTrainingRowsMask.size(), false};

    this->startProgressMonitoringFineTuneHyperparameters();

    if (this->canTrain() == false) {

        // Fallback to using the constant predictor which minimises the loss.

        this->startProgressMonitoringFinalTrain();
        m_BestForest.assign(1, this->initializePredictionsAndLossDerivatives(
                                   frame, allTrainingRowsMask, noRowsMask));
        TMeanVarAccumulator testLossMoments;
        testLossMoments.add(this->meanLoss(frame, allTrainingRowsMask));
        m_Hyperparameters.captureBest(
            testLossMoments, 0.0 /*no loss gap*/, 0.0 /*no kept nodes*/,
            1.0 /*single node used to centre the data*/, 1 /*single tree*/);
        LOG_TRACE(<< "Test loss = " << m_Hyperparameters.bestForestTestLoss());

    } else if (m_Hyperparameters.stopEarly() == false &&
               (m_Hyperparameters.searchNotFinished() || m_BestForest.empty())) {
        m_Hyperparameters.clearObservations();

        TMeanVarAccumulator timeAccumulator;
        core::CStopWatch stopWatch;
        stopWatch.start();
        std::uint64_t lastLap{stopWatch.lap()};

        // Hyperparameter optimisation loop.

        this->initializePerFoldTestLosses();

        for (m_Hyperparameters.startSearch(); m_Hyperparameters.searchNotFinished(); /**/) {

            LOG_TRACE(<< "Optimisation round = " << m_Hyperparameters.currentRound() + 1);
            m_Instrumentation->iteration(m_Hyperparameters.currentRound() + 1);

            this->recordHyperparameters();

            auto crossValidationResult = this->crossValidateForest(
                frame, m_Hyperparameters.maximumNumberTrees().value(),
                [this](core::CDataFrame& frame_, const core::CPackedBitVector& trainingRowMask,
                       const core::CPackedBitVector& testingRowMask,
                       core::CLoopProgress& trainingProgress) {
                    return this->trainForest(frame_, trainingRowMask,
                                             testingRowMask, trainingProgress);
                });

            // If we have one fold we're evaluating using a hold-out set and will
            // not retrain on the full data set at the end.
            if (m_Hyperparameters.captureBest(
                    crossValidationResult.s_TestLossMoments,
                    crossValidationResult.s_MeanLossGap, 0.0 /*no kept nodes*/,
                    crossValidationResult.s_NumberNodes, crossValidationResult.s_NumberTrees) &&
                m_NumberFolds.value() == 1) {
                m_BestForest = std::move(crossValidationResult.s_Forest);
            }

            if (m_Hyperparameters.selectNext(crossValidationResult.s_TestLossMoments,
                                             this->betweenFoldTestLossVariance()) == false) {
                LOG_INFO(<< "Exiting hyperparameter optimisation loop on round "
                         << m_Hyperparameters.currentRound() << " out of "
                         << m_Hyperparameters.numberRounds() << ".");
                break;
            }

            std::int64_t memoryUsage(this->memoryUsage());
            m_Instrumentation->updateMemoryUsage(memoryUsage - lastMemoryUsage);
            lastMemoryUsage = memoryUsage;

            // We need to update the current round before we persist so we don't
            // perform an extra round when we fail over.
            m_Hyperparameters.startNextSearchRound();

            // Store the training state after each hyperparameter search step.
            LOG_TRACE(<< "Round " << m_Hyperparameters.currentRound()
                      << " state recording started");
            this->recordState(recordTrainStateCallback);
            LOG_TRACE(<< "Round " << m_Hyperparameters.currentRound()
                      << " state recording finished");

            std::uint64_t currentLap{stopWatch.lap()};
            std::uint64_t delta{currentLap - lastLap};
            m_Instrumentation->iterationTime(delta);

            timeAccumulator.add(static_cast<double>(delta));
            lastLap = currentLap;
            m_Instrumentation->flush(HYPERPARAMETER_OPTIMIZATION_ROUND +
                                     std::to_string(m_Hyperparameters.currentRound()));
        }

        LOG_TRACE(<< "Test loss = " << m_Hyperparameters.bestForestTestLoss());

        if (m_BestForest.empty()) {
            m_Hyperparameters.restoreBest();
            m_Hyperparameters.recordHyperparameters(*m_Instrumentation);
            m_Hyperparameters.captureScale();
            this->startProgressMonitoringFinalTrain();
            this->scaleRegularizationMultipliers(this->allTrainingRowsMask().manhattan() /
                                                 this->meanNumberTrainingRowsPerFold());

            // Reinitialize random number generator for reproducible results.
            m_Rng.seed(m_Seed);

            m_BestForest = this->trainForest(frame, allTrainingRowsMask,
                                             allTrainingRowsMask, m_TrainingProgress)
                               .s_Forest;

            this->recordState(recordTrainStateCallback);
        } else {
            this->skipProgressMonitoringFinalTrain();
        }
        m_Instrumentation->iteration(m_Hyperparameters.currentRound());
        m_Instrumentation->flush(TRAIN_FINAL_FOREST);

        timeAccumulator.add(static_cast<double>(stopWatch.stop() - lastLap));

        LOG_TRACE(<< "Training finished after " << m_Hyperparameters.currentRound()
                  << " iterations. Time per iteration in ms mean: "
                  << common::CBasicStatistics::mean(timeAccumulator) << " std. dev:  "
                  << std::sqrt(common::CBasicStatistics::variance(timeAccumulator)));

        core::CProgramCounters::counter(counter_t::E_DFTPMTrainedForestNumberTrees) =
            m_BestForest.size();
    } else {
        this->skipProgressMonitoringFinalTrain();
    }

    this->computeClassificationWeights(frame);
    this->initializeTreeShap(frame);

    // Force progress to one and record the final memory usage.
    m_Instrumentation->updateProgress(1.0);
    m_Instrumentation->updateMemoryUsage(
        static_cast<std::int64_t>(this->memoryUsage()) - lastMemoryUsage);
}

void CBoostedTreeImpl::trainIncremental(core::CDataFrame& frame,
                                        const TTrainingStateCallback& recordTrainStateCallback) {

    this->checkIncrementalTrainInvariants(frame);

    if (m_BestForest.size() == 1 || m_NewTrainingRowMask.manhattan() == 0.0) {
        return;
    }

    LOG_DEBUG(<< "Main incremental training loop...");

    this->selectTreesToRetrain(frame);
    // Add dummy trees that can be replaced with the new trees in the forest.
    std::size_t oldBestForestSize{m_BestForest.size()};
    m_BestForest.resize(oldBestForestSize + m_MaximumNumberNewTrees);
    for (auto i = oldBestForestSize; i < m_BestForest.size(); ++i) {
        m_BestForest[i] = {CBoostedTreeNode(m_Loss->numberParameters())};
    }
    m_TreesToRetrain.resize(m_TreesToRetrain.size() + m_MaximumNumberNewTrees);
    std::iota(m_TreesToRetrain.end() - m_MaximumNumberNewTrees,
              m_TreesToRetrain.end(), oldBestForestSize);
    TNodeVecVec retrainedTrees;

    std::int64_t lastMemoryUsage(this->memoryUsage());

    this->startProgressMonitoringTrainIncremental();

    double retrainedNumberNodes{0.0};
    for (const auto& i : m_TreesToRetrain) {
        retrainedNumberNodes += static_cast<double>(m_BestForest[i].size());
    }
    double numberKeptNodes{numberForestNodes(m_BestForest) - retrainedNumberNodes};

    // Make sure that our predictions are correctly initialised before computing
    // the initial loss.
    auto allTrainingRowsMask = this->allTrainingRowsMask();
    auto noRowsMask = core::CPackedBitVector{allTrainingRowsMask.size(), false};
    this->initializePredictionsAndLossDerivatives(frame, allTrainingRowsMask, noRowsMask);

    // When we decide whether to accept the results of incremental training below
    // we compare the loss calculated for the best candidate forest with the loss
    // calculated with the original model. Since the data summary comprises a subset
    // of the training data we are in effect comparing training error on old data +
    // validation error on new training data with something closer to validation
    // error on all data. If we don't have much new data or the improvement we can
    // make on it is small this typically causes us to reject models which actually
    // perform better in test. We record gap between the train and validation loss
    // on the old training data in train and add it on to the threshold to accept
    // adjusting for the proportion of old training data we have.
    double numberNewTrainingRows{m_NewTrainingRowMask.manhattan()};
    double numberOldTrainingRows{allTrainingRowsMask.manhattan() - numberNewTrainingRows};
    double initialLoss{
        CBoostedTreeHyperparameters::lossAtNSigma(
            1.0,
            [&] {
                TMeanVarAccumulator lossMoments;
                for (const auto& mask : m_TestingRowMasks) {
                    lossMoments.add(this->meanChangePenalisedLoss(frame, mask));
                }
                return lossMoments;
            }()) +
        this->expectedLossGapAfterTrainIncremental(numberOldTrainingRows, numberNewTrainingRows)};

    // Hyperparameter optimisation loop.

    this->initializePerFoldTestLosses();

    std::size_t numberTreesToRetrain{this->numberTreesToRetrain()};
    TMeanVarAccumulator timeAccumulator;
    core::CStopWatch stopWatch;
    stopWatch.start();
    std::uint64_t lastLap{stopWatch.lap()};
    LOG_TRACE(<< "Number trees to retrain = " << numberTreesToRetrain << "/"
              << m_BestForest.size());

    for (m_Hyperparameters.startSearch(); m_Hyperparameters.searchNotFinished(); /**/) {

        LOG_TRACE(<< "Optimisation round = " << m_Hyperparameters.currentRound() + 1);
        m_Instrumentation->iteration(m_Hyperparameters.currentRound() + 1);

        this->recordHyperparameters();

        auto crossValidationResult = this->crossValidateForest(
            frame, numberTreesToRetrain,
            [this](core::CDataFrame& frame_, const core::CPackedBitVector& trainingRowMask,
                   const core::CPackedBitVector& testingRowMask,
                   core::CLoopProgress& trainingProgress) {
                return this->updateForest(frame_, trainingRowMask,
                                          testingRowMask, trainingProgress);
            });

        // If we have one fold we're evaluating using a hold-out set and will
        // not retrain on the full data set at the end.
        if (m_Hyperparameters.captureBest(crossValidationResult.s_TestLossMoments,
                                          crossValidationResult.s_MeanLossGap, numberKeptNodes,
                                          crossValidationResult.s_NumberNodes,
                                          crossValidationResult.s_NumberTrees) &&
            m_NumberFolds.value() == 1) {
            retrainedTrees = std::move(crossValidationResult.s_Forest);
        }

        if (m_Hyperparameters.selectNext(crossValidationResult.s_TestLossMoments,
                                         this->betweenFoldTestLossVariance()) == false) {
            LOG_INFO(<< "Exiting hyperparameter optimisation loop on round "
                     << m_Hyperparameters.currentRound() << " out of "
                     << m_Hyperparameters.numberRounds() << ".");
            break;
        }

        std::int64_t memoryUsage(this->memoryUsage());
        m_Instrumentation->updateMemoryUsage(memoryUsage - lastMemoryUsage);
        lastMemoryUsage = memoryUsage;

        // We need to update the current round before we persist so we don't
        // perform an extra round when we fail over.
        m_Hyperparameters.startNextSearchRound();

        LOG_TRACE(<< "Round " << m_Hyperparameters.currentRound() << " state recording started");
        this->recordState(recordTrainStateCallback);
        LOG_TRACE(<< "Round " << m_Hyperparameters.currentRound() << " state recording finished");

        std::uint64_t currentLap{stopWatch.lap()};
        std::uint64_t delta{currentLap - lastLap};
        m_Instrumentation->iterationTime(delta);

        timeAccumulator.add(static_cast<double>(delta));
        lastLap = currentLap;
        m_Instrumentation->flush(HYPERPARAMETER_OPTIMIZATION_ROUND +
                                 std::to_string(m_Hyperparameters.currentRound()));
    }

    initialLoss += m_Hyperparameters.modelSizePenalty(numberKeptNodes, retrainedNumberNodes);

    LOG_TRACE(<< "Incremental training finished after "
              << m_Hyperparameters.currentRound() << " iterations. "
              << "Time per iteration in ms mean: "
              << common::CBasicStatistics::mean(timeAccumulator) << " std. dev:  "
              << std::sqrt(common::CBasicStatistics::variance(timeAccumulator)));
    LOG_TRACE(<< "best forest loss = " << m_Hyperparameters.bestForestTestLoss()
              << ", initial loss = " << initialLoss);

    if (m_ForceAcceptIncrementalTraining || m_Hyperparameters.bestForestTestLoss() < initialLoss) {
        m_Hyperparameters.restoreBest();
        m_Hyperparameters.recordHyperparameters(*m_Instrumentation);
        m_Hyperparameters.captureScale();

        if (retrainedTrees.empty()) {
            this->scaleRegularizationMultipliers(this->allTrainingRowsMask().manhattan() /
                                                 this->meanNumberTrainingRowsPerFold());

            // Reinitialize random number generator for reproducible results.
            m_Rng.seed(m_Seed);

            retrainedTrees = this->updateForest(frame, allTrainingRowsMask,
                                                allTrainingRowsMask, m_TrainingProgress)
                                 .s_Forest;
        }

        for (std::size_t i = 0; i < retrainedTrees.size(); ++i) {
            m_BestForest[m_TreesToRetrain[i]] = std::move(retrainedTrees[i]);
        }
        // Resize the forest to eliminate the unused dummy trees.
        auto lastChangedTreeIndex = m_TreesToRetrain[retrainedTrees.size() - 1];
        auto bestForestSize = std::max(lastChangedTreeIndex + 1,
                                       m_BestForest.size() - m_MaximumNumberNewTrees);
        m_BestForest.resize(bestForestSize);
    }

    this->computeClassificationWeights(frame);
    this->initializeTreeShap(frame);

    // Force progress to one and record the final memory usage.
    m_Instrumentation->updateProgress(1.0);
    m_Instrumentation->updateMemoryUsage(
        static_cast<std::int64_t>(this->memoryUsage()) - lastMemoryUsage);
}

void CBoostedTreeImpl::recordState(const TTrainingStateCallback& recordTrainState) const {
    recordTrainState([this](core::CStatePersistInserter& inserter) {
        this->acceptPersistInserter(inserter);
    });
}

void CBoostedTreeImpl::predict(core::CDataFrame& frame) const {
    core::CPackedBitVector rowMask{frame.numberRows(), true};
    this->predict(rowMask, frame);
}

void CBoostedTreeImpl::predict(const core::CPackedBitVector& rowMask,
                               core::CDataFrame& frame) const {
    if (m_BestForest.empty()) {
        HANDLE_FATAL(<< "Internal error: no model available for prediction. "
                     << "Please report this problem.");
        return;
    }
    bool successful;
    std::tie(std::ignore, successful) = frame.writeColumns(
        m_NumberThreads, 0, frame.numberRows(),
        [&](const TRowItr& beginRows, const TRowItr& endRows) {
            std::size_t numberLossParameters{m_Loss->numberParameters()};
            for (auto row = beginRows; row != endRows; ++row) {
                auto prediction = readPrediction(*row, m_ExtraColumns, numberLossParameters);
                prediction = this->predictRow(m_Encoder->encode(*row));
            }
        },
        &rowMask);
    if (successful == false) {
        HANDLE_FATAL(<< "Internal error: failed model inference. "
                     << "Please report this problem.");
    }
}

std::size_t CBoostedTreeImpl::estimateMemoryUsageTrain(std::size_t numberRows,
                                                       std::size_t numberColumns) const {
    // The maximum tree size is defined is the maximum number of leaves minus one.
    // A binary tree with n + 1 leaves has 2n + 1 nodes in total.
    std::size_t maximumNumberLeaves{maximumTreeSize(numberRows) + 1};
    std::size_t maximumNumberNodes{2 * maximumNumberLeaves - 1};
    std::size_t maximumNumberFeatures{
        std::min(numberColumns - 1, numberRows / this->rowsPerFeature(numberRows))};
    std::size_t hyperparametersMemoryUsage{m_Hyperparameters.estimateMemoryUsage()};
    std::size_t forestMemoryUsage{
        m_Hyperparameters.maximumNumberTrees().value() *
        (sizeof(TNodeVec) + maximumNumberNodes * CBoostedTreeNode::estimateMemoryUsage(
                                                     m_Loss->numberParameters()))};
    std::size_t foldRoundLossMemoryUsage{
        m_NumberFolds.value() * m_Hyperparameters.numberRounds() * sizeof(TOptionalDouble)};
    // The leaves' row masks memory is accounted for here because it's proportional
    // to the log2(number of nodes). The compressed bit vector representation uses
    // roughly log2(E[run length]) / E[run length] bytes per bit. As we grow the
    // tree we partition the data and so the average run length (sequential unmasked
    // values) is just equal to the number of leaves. Putting this together, if there
    // are n rows and m leaves each leaf will use n * log2(m) / m and so their total
    // memory will be n * log2(m). In practice, we don't get the optimal compression,
    // a reasonable margin is a factor of 4.
    std::size_t rowMaskMemoryUsage{4 * numberRows *
                                   static_cast<std::size_t>(std::ceil(std::log2(
                                       static_cast<double>(maximumNumberLeaves))))};
    // We only maintain statistics for leaves we know we may possibly split this
    // halves the peak number of statistics we maintain.
    std::size_t leafNodeStatisticsMemoryUsage{
        rowMaskMemoryUsage + maximumNumberLeaves *
                                 CBoostedTreeLeafNodeStatisticsScratch::estimateMemoryUsage(
                                     maximumNumberFeatures, m_NumberSplitsPerFeature,
                                     m_Loss->numberParameters()) /
                                 2};
    std::size_t dataTypeMemoryUsage{maximumNumberFeatures * sizeof(CDataFrameUtils::SDataType)};
    std::size_t featureSampleProbabilitiesMemoryUsage{maximumNumberFeatures * sizeof(double)};
    std::size_t fixedCandidateSplitsMemoryUsage{maximumNumberFeatures * sizeof(TFloatVec)};
    // Assuming either many or few missing rows, we get good compression of the bit
    // vector. Specifically, we'll assume the average run length is 64 for which
    // we get a constant 8 / 64.
    std::size_t missingFeatureMaskMemoryUsage{8 * numberColumns * numberRows / 64};
    std::size_t trainTestMaskMemoryUsage{
        2 * m_NumberFolds.value() *
        static_cast<std::size_t>(std::ceil(std::min(m_TrainFractionPerFold.value(),
                                                    1.0 - m_TrainFractionPerFold.value()) *
                                           static_cast<double>(numberRows)))};

    std::size_t worstCaseMemoryUsage{
        sizeof(*this) + forestMemoryUsage + foldRoundLossMemoryUsage +
        hyperparametersMemoryUsage + leafNodeStatisticsMemoryUsage + dataTypeMemoryUsage +
        featureSampleProbabilitiesMemoryUsage + fixedCandidateSplitsMemoryUsage +
        missingFeatureMaskMemoryUsage + trainTestMaskMemoryUsage};

    return CBoostedTreeImpl::correctedMemoryUsage(static_cast<double>(worstCaseMemoryUsage));
}

std::size_t
CBoostedTreeImpl::estimateMemoryUsageTrainIncremental(std::size_t /*numberRows*/,
                                                      std::size_t /*numberColumns*/) const {

    // TODO https://github.com/elastic/ml-cpp/issues/1790.
    return 0;
}

std::size_t CBoostedTreeImpl::correctedMemoryUsage(double memoryUsageBytes) {
    // We use a piecewise linear function of the estimated memory usage to compute
    // the corrected value. The values are selected in a way to reduce over-estimation
    // and to improve the behaviour on the trial nodes in the cloud. The high level strategy
    // also ensures that corrected memory usage is a monotonic function of estimated memory
    // usage and any change to the approach should preserve this property.
    TDoubleVec estimatedMemoryUsageMB{0.0,    20.0,    1024.0, 4096.0,
                                      8192.0, 12288.0, 16384.0};
    TDoubleVec correctedMemoryUsageMB{0.0,   20.0,   179.2, 512.0,
                                      819.2, 1088.0, 1280.0};
    common::CSpline<> spline(common::CSplineTypes::E_Linear);
    spline.interpolate(estimatedMemoryUsageMB, correctedMemoryUsageMB,
                       common::CSplineTypes::E_ParabolicRunout);
    return static_cast<std::size_t>(spline.value(memoryUsageBytes / BYTES_IN_MB) * BYTES_IN_MB);
}

double CBoostedTreeImpl::expectedLossGapAfterTrainIncremental(double numberOldTrainingRows,
                                                              double numberNewTrainingRows) const {

    // There are two cases:
    //   1. We train repeatedly using the same holdout set,
    //   2. We train incrementally having originally trained via cross-validation.
    //
    // In the first case, we compare performance on the same data set throughout and so
    // the loss of each candidate model can be directly compared. In the second case,
    // we compare the loss for the retrained model with a model trained on all the
    // original data. Since the data summary comprises a subset of these data we are
    // in effect comparing training error on old data + validation error on new training
    // data for the original model with something closer to validation error on all data
    // for the retrained model. If we don't have much new data or the improvement we can
    // make on them is small this typically causes us to reject models which actually
    // perform better in test. To address this we record gap between the train and
    // validation loss on the old training data in train and add it on to the threshold
    // to accept adjusting for the proportion of old training data we have.
    return m_NumberFolds.value() == 1
               ? 0.0
               : numberOldTrainingRows * m_PreviousTrainLossGap /
                     (numberOldTrainingRows + numberNewTrainingRows);
}

bool CBoostedTreeImpl::canTrain() const {
    return std::accumulate(m_FeatureSampleProbabilities.begin(),
                           m_FeatureSampleProbabilities.end(), 0.0) > 0.0;
}

CBoostedTreeImpl::TDoubleDoublePr
CBoostedTreeImpl::gainAndCurvatureAtPercentile(double percentile, const TNodeVecVec& forest) {

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

    if (gains.empty()) {
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
    m_FoldRoundTestLosses.resize(m_NumberFolds.value());
    for (auto& losses : m_FoldRoundTestLosses) {
        losses.resize(m_Hyperparameters.numberRounds());
    }
}

void CBoostedTreeImpl::computeClassificationWeights(const core::CDataFrame& frame) {

    using TFloatStorageVec = std::vector<common::CFloatStorage>;

    if (m_Loss->type() == E_BinaryClassification || m_Loss->type() == E_MulticlassClassification) {

        std::size_t numberClasses{
            m_Loss->type() == E_BinaryClassification ? 2 : m_Loss->numberParameters()};
        TFloatStorageVec storage(2);

        switch (m_ClassAssignmentObjective) {
        case CBoostedTree::E_Accuracy:
            m_ClassificationWeights = TVector::Ones(numberClasses);
            break;
        case CBoostedTree::E_MinimumRecall:
            m_ClassificationWeights = CDataFrameUtils::maximumMinimumRecallClassWeights(
                m_NumberThreads, frame, this->allTrainingRowsMask(),
                numberClasses, m_DependentVariable,
                [storage, numberClasses, this](const TRowRef& row) mutable {
                    if (m_Loss->type() == E_BinaryClassification) {
                        // We predict the log-odds but this is expected to return
                        // the log of the predicted class probabilities.
                        TMemoryMappedFloatVector result{&storage[0], 2};
                        result.array() =
                            m_Loss
                                ->transform(readPrediction(row, m_ExtraColumns, numberClasses))
                                .array()
                                .log();
                        return result;
                    }
                    return readPrediction(row, m_ExtraColumns, numberClasses);
                });
            break;
        case CBoostedTree::E_Custom:
            if (m_ClassificationWeightsOverride != boost::none) {
                const auto& classes = frame.categoricalColumnValues()[m_DependentVariable];
                m_ClassificationWeights = TVector::Ones(numberClasses);
                for (std::size_t i = 0; i < classes.size(); ++i) {
                    auto j = std::find_if(m_ClassificationWeightsOverride->begin(),
                                          m_ClassificationWeightsOverride->end(),
                                          [&](const auto& weight) {
                                              return weight.first == classes[i];
                                          });
                    if (j != m_ClassificationWeightsOverride->end()) {
                        m_ClassificationWeights(i) = j->second;
                    } else {
                        LOG_WARN(<< "Missing weight for class '" << classes[i] << "'. Overrides = "
                                 << core::CContainerPrinter::print(m_ClassificationWeightsOverride)
                                 << ".");
                    }
                }
                LOG_TRACE(<< "classification weights = "
                          << m_ClassificationWeights.transpose());
            }
            break;
        }
    }
}

void CBoostedTreeImpl::initializeTreeShap(const core::CDataFrame& frame) {
    // Populate number samples reaching each node.
    CTreeShapFeatureImportance::computeNumberSamples(m_NumberThreads, frame,
                                                     *m_Encoder, m_BestForest);

    if (m_NumberTopShapValues > 0) {
        // Create the SHAP calculator.
        m_TreeShap = std::make_unique<CTreeShapFeatureImportance>(
            m_NumberThreads, frame, *m_Encoder, m_BestForest, m_NumberTopShapValues);
    } else {
        // TODO these are not currently written into the inference model
        // but they would be nice to expose since they provide good insight
        // into how the splits affect the target variable.
        // Set internal node values anyway.
        //CTreeShapFeatureImportance::computeInternalNodeValues(m_BestForest);
    }
}

void CBoostedTreeImpl::selectTreesToRetrain(const core::CDataFrame& frame) {

    if (m_TreesToRetrain.empty() == false) {
        return;
    }

    TDoubleVec probabilities{retrainTreeSelectionProbabilities(
        m_NumberThreads, frame, m_ExtraColumns, m_DependentVariable, *m_Encoder,
        this->allTrainingRowsMask(), *m_Loss, m_BestForest)};

    std::size_t numberToRetrain{static_cast<std::size_t>(
        std::max(m_RetrainFraction * static_cast<double>(m_BestForest.size()), 1.0) + 0.5)};
    common::CSampling::categoricalSampleWithoutReplacement(
        m_Rng, probabilities, numberToRetrain, m_TreesToRetrain);
}

template<typename F>
CBoostedTreeImpl::SCrossValidationResult
CBoostedTreeImpl::crossValidateForest(core::CDataFrame& frame,
                                      std::size_t maximumNumberTrees,
                                      const F& trainForest) {

    // We want to ensure we evaluate on equal proportions for each fold.
    TSizeVec folds(m_NumberFolds.value());
    std::iota(folds.begin(), folds.end(), 0);
    common::CSampling::random_shuffle(m_Rng, folds.begin(), folds.end());

    auto stopCrossValidationEarly = [&](TMeanVarAccumulator testLossMoments) {
        // Always train on at least one fold and every fold for the first
        // "number folds" rounds. Exit cross-validation early if it's clear
        // that the test error is not close to the minimum test error. We use
        // the estimated test error for each remaining fold at two standard
        // deviations below the mean for this.
        if (m_StopCrossValidationEarly &&
            m_Hyperparameters.currentRound() >= m_NumberFolds.value() &&
            folds.size() < m_NumberFolds.value()) {
            for (const auto& testLoss : this->estimateMissingTestLosses(folds)) {
                testLossMoments.add(common::CBasicStatistics::mean(testLoss) -
                                    2.0 * std::sqrt(common::CBasicStatistics::maximumLikelihoodVariance(
                                              testLoss)));
            }
            return common::CBasicStatistics::mean(testLossMoments) >
                   this->minimumTestLoss();
        }
        return false;
    };

    TNodeVecVec forest;
    TMeanVarAccumulator testLossMoments;
    TMeanAccumulator meanLossGap;
    TDoubleVec numberTrees;
    numberTrees.reserve(m_Hyperparameters.currentRound());
    TMeanAccumulator meanForestSizeAccumulator;

    while (folds.empty() == false && stopCrossValidationEarly(testLossMoments) == false) {
        std::size_t fold{folds.back()};
        folds.pop_back();
        double testLoss;
        double lossGap;
        TDoubleVec testLossValues;
        std::tie(forest, testLoss, lossGap, testLossValues) =
            trainForest(frame, m_TrainingRowMasks[fold], m_TestingRowMasks[fold], m_TrainingProgress)
                .asTuple();
        LOG_TRACE(<< "fold = " << fold << " forest size = " << forest.size()
                  << " test set loss = " << testLoss);
        testLossMoments.add(testLoss);
        meanLossGap.add(lossGap);
        m_FoldRoundTestLosses[fold][m_Hyperparameters.currentRound()] = testLoss;
        numberTrees.push_back(static_cast<double>(forest.size()));
        meanForestSizeAccumulator.add(numberForestNodes(forest));
        m_Instrumentation->lossValues(fold, std::move(testLossValues));
    }
    m_TrainingProgress.increment(maximumNumberTrees * folds.size());
    LOG_TRACE(<< "skipped " << folds.size() << " folds");

    std::sort(numberTrees.begin(), numberTrees.end());
    std::size_t medianNumberTrees{
        static_cast<std::size_t>(common::CBasicStatistics::median(numberTrees))};
    double meanForestSize{common::CBasicStatistics::mean(meanForestSizeAccumulator)};
    testLossMoments = this->correctTestLossMoments(folds, testLossMoments);
    LOG_TRACE(<< "test mean loss = " << common::CBasicStatistics::mean(testLossMoments)
              << ", sigma = " << std::sqrt(common::CBasicStatistics::mean(testLossMoments))
              << ", mean number nodes in forest = " << meanForestSize);

    m_Hyperparameters.addRoundStats(meanForestSizeAccumulator,
                                    common::CBasicStatistics::mean(testLossMoments));

    return {std::move(forest), testLossMoments,
            common::CBasicStatistics::mean(meanLossGap), medianNumberTrees, meanForestSize};
}

CBoostedTreeImpl::TNodeVec CBoostedTreeImpl::initializePredictionsAndLossDerivatives(
    core::CDataFrame& frame,
    const core::CPackedBitVector& trainingRowMask,
    const core::CPackedBitVector& testingRowMask) const {

    core::CPackedBitVector updateRowMask{trainingRowMask | testingRowMask};
    frame.writeColumns(
        m_NumberThreads, 0, frame.numberRows(),
        [this](const TRowItr& beginRows, const TRowItr& endRows) {
            std::size_t numberLossParameters{m_Loss->numberParameters()};
            for (auto row_ = beginRows; row_ != endRows; ++row_) {
                auto row = *row_;
                if (m_Hyperparameters.incrementalTraining()) {
                    writePrediction(row, m_ExtraColumns, numberLossParameters,
                                    readPreviousPrediction(row, m_ExtraColumns,
                                                           numberLossParameters));
                } else {
                    zeroPrediction(row, m_ExtraColumns, numberLossParameters);
                    zeroLossGradient(row, m_ExtraColumns, numberLossParameters);
                    zeroLossCurvature(row, m_ExtraColumns, numberLossParameters);
                }
            }
        },
        &updateRowMask);

    TNodeVec tree;
    if (m_Hyperparameters.incrementalTraining() == false) {
        // At the start we will centre the data w.r.t. the given loss function.
        tree.assign({CBoostedTreeNode{m_Loss->numberParameters()}});
        this->computeLeafValues(frame, trainingRowMask, *m_Loss, 1.0 /*eta*/,
                                0.0 /*lambda*/, tree);
        this->refreshPredictionsAndLossDerivatives(
            frame, trainingRowMask | testingRowMask, *m_Loss,
            [&](const TRowRef& row, TMemoryMappedFloatVector& prediction) {
                prediction += root(tree).value(m_Encoder->encode(row), tree);
            });
    }

    return tree;
}

CBoostedTreeImpl::STrainForestResult
CBoostedTreeImpl::trainForest(core::CDataFrame& frame,
                              const core::CPackedBitVector& trainingRowMask,
                              const core::CPackedBitVector& testingRowMask,
                              core::CLoopProgress& trainingProgress) const {

    LOG_TRACE(<< "Training one forest...");

    // We always advance the rng a fixed number of steps training one forest.
    // This ensures even if decisions change for a single forest then we produce
    // the same sequence of random numbers next time round. We advance the rng
    // enough so that the sequences for different calls won't overlap.
    CResetAndJumpOnExit resetAndJumpOnExit{m_Rng};

    auto makeRootLeafNodeStatistics =
        [&](const TFloatVecVec& candidateSplits, const TSizeVec& treeFeatureBag,
            const TSizeVec& nodeFeatureBag,
            const core::CPackedBitVector& trainingRowMask_, TWorkspace& workspace) {
            return std::make_shared<CBoostedTreeLeafNodeStatisticsScratch>(
                rootIndex(), m_ExtraColumns, m_Loss->numberParameters(), frame,
                m_Hyperparameters, candidateSplits, treeFeatureBag,
                nodeFeatureBag, 0 /*depth*/, trainingRowMask_, workspace);
        };

    std::size_t maximumNumberInternalNodes{maximumTreeSize(trainingRowMask)};

    TNodeVecVec forest{this->initializePredictionsAndLossDerivatives(
        frame, trainingRowMask, testingRowMask)};
    forest.reserve(m_Hyperparameters.maximumNumberTrees().value());

    CScopeRecordMemoryUsage scopeMemoryUsage{forest, m_Instrumentation->memoryUsageCallback()};

    double eta{m_Hyperparameters.eta().value()};

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
    scopeMemoryUsage.add(downsampledRowMask);
    auto candidateSplits = this->candidateSplits(frame, downsampledRowMask);
    // We compute and cache row splits once upfront for features using fixed splits.
    TBoolVec featuresToRefresh(m_FixedCandidateSplits.size());
    for (std::size_t i = 0; i < m_FixedCandidateSplits.size(); ++i) {
        featuresToRefresh[i] = m_FeatureSampleProbabilities[i] > 0.0 &&
                               m_FixedCandidateSplits[i].empty();
    }
    this->refreshSplitsCache(frame, candidateSplits, featuresToRefresh, trainingRowMask);
    scopeMemoryUsage.add(candidateSplits);

    std::size_t retries{0};

    std::size_t deployedSize{0};
    TDoubleVec testLosses;
    testLosses.reserve(m_Hyperparameters.maximumNumberTrees().value());
    scopeMemoryUsage.add(testLosses);

    CTrainForestStoppingCondition stoppingCondition{
        m_Hyperparameters.maximumNumberTrees().value()};
    TWorkspace workspace{m_Loss->numberParameters()};

    // For each iteration:
    //  1. Periodically compute weighted quantiles for features F and candidate
    //     splits S from F.
    //  2. Build one tree on S.
    //  3. Update predictions and loss derivatives.

    do {
        auto tree = this->trainTree(frame, downsampledRowMask, candidateSplits,
                                    maximumNumberInternalNodes,
                                    makeRootLeafNodeStatistics, workspace);

        retries = tree.size() == 1 ? retries + 1 : 0;

        // Simply break out of training if the model exceeds the size limit. We
        // don't try and anticipate this limit: cross-validation error will direct
        // hyperparameter tuning to select the best parameters subject to this
        // constraint.
        if (retries == m_MaximumAttemptsToAddTree ||
            deployedSize > this->maximumTrainedModelSize()) {
            break;
        }

        bool forceRefreshSplits{tree.size() <= 1};

        if (tree.size() > 1) {
            scopeMemoryUsage.add(tree);
            this->computeLeafValues(
                frame, trainingRowMask, *m_Loss, eta,
                m_Hyperparameters.leafWeightPenaltyMultiplier().value(), tree);
            this->refreshPredictionsAndLossDerivatives(
                frame, trainingRowMask | testingRowMask, *m_Loss,
                [&](const TRowRef& row, TMemoryMappedFloatVector& prediction) {
                    prediction += root(tree).value(m_Encoder->encode(row), tree);
                });
            forest.push_back(std::move(tree));
            eta = std::min(1.0, m_Hyperparameters.etaGrowthRatePerTree().value() * eta);
            retries = 0;
            deployedSize += std::accumulate(forest.back().begin(), forest.back().end(),
                                            0, [](auto size, const auto& node) {
                                                return size + node.deployedSize();
                                            });
            trainingProgress.increment();
        }

        downsampledRowMask = this->downsample(trainingRowMask);
        // The memory variation in the row mask from sample to sample is too
        // small to bother to track.

        if (forceRefreshSplits || forest.size() == nextTreeCountToRefreshSplits) {
            scopeMemoryUsage.remove(candidateSplits);
            candidateSplits = this->candidateSplits(frame, downsampledRowMask);
            this->refreshSplitsCache(frame, candidateSplits, featuresToRefresh, trainingRowMask);
            scopeMemoryUsage.add(candidateSplits);
            nextTreeCountToRefreshSplits += static_cast<std::size_t>(
                std::max(0.5 / eta, MINIMUM_SPLIT_REFRESH_INTERVAL));
        }
    } while (stoppingCondition.shouldStop(forest.size(), [&] {
        double trainLoss{this->meanLoss(frame, trainingRowMask)};
        double testLoss{this->meanLoss(frame, testingRowMask)};
        testLosses.push_back(testLoss);
        return std::make_pair(trainLoss, testLoss);
    }) == false);

    LOG_TRACE(<< "Stopped at " << forest.size() - 1 << "/"
              << m_Hyperparameters.maximumNumberTrees().print());

    trainingProgress.increment(
        std::max(m_Hyperparameters.maximumNumberTrees().value(), forest.size()) -
        forest.size());

    forest.resize(stoppingCondition.bestSize());

    LOG_TRACE(<< "Trained one forest");

    return {forest, stoppingCondition.bestTestLoss(),
            stoppingCondition.lossGap(), std::move(testLosses)};
}

CBoostedTreeImpl::STrainForestResult
CBoostedTreeImpl::updateForest(core::CDataFrame& frame,
                               const core::CPackedBitVector& trainingRowMask,
                               const core::CPackedBitVector& testingRowMask,
                               core::CLoopProgress& trainingProgress) const {

    LOG_TRACE(<< "Incrementally training one forest...");

    if (m_TreesToRetrain.empty()) {
        return {{}, INF, 0.0, {}};
    }

    // We always advance the rng a fixed number of steps updating one forest.
    // This ensures even if decisions change for a single forest then we produce
    // the same sequence of random numbers next time round. We advance the rng
    // enough so that the sequences for different calls won't overlap.
    CResetAndJumpOnExit resetAndJumpOnExit{m_Rng};

    auto makeRootLeafNodeStatistics =
        [&](const TFloatVecVec& candidateSplits, const TSizeVec& treeFeatureBag,
            const TSizeVec& nodeFeatureBag,
            const core::CPackedBitVector& trainingRowMask_, TWorkspace& workspace) {
            return std::make_shared<CBoostedTreeLeafNodeStatisticsIncremental>(
                rootIndex(), m_ExtraColumns, m_Loss->numberParameters(), frame,
                m_Hyperparameters, candidateSplits, treeFeatureBag,
                nodeFeatureBag, 0 /*depth*/, trainingRowMask_, workspace);
        };

    std::size_t maximumNumberInternalNodes{maximumTreeSize(trainingRowMask)};

    TNodeVecVec retrainedTrees;
    retrainedTrees.reserve(m_TreesToRetrain.size() + 1);
    this->initializePredictionsAndLossDerivatives(frame, trainingRowMask, testingRowMask);

    CScopeRecordMemoryUsage scopeMemoryUsage{
        retrainedTrees, m_Instrumentation->memoryUsageCallback()};

    std::size_t nextTreeCountToRefreshSplits{1};

    core::CPackedBitVector oldTrainingRowMask{trainingRowMask & ~m_NewTrainingRowMask};
    core::CPackedBitVector newTrainingRowMask{trainingRowMask & m_NewTrainingRowMask};
    auto oldDownsampledRowMask = this->downsample(oldTrainingRowMask);
    auto newDownsampledRowMask = this->downsample(newTrainingRowMask);
    auto downsampledRowMask = oldDownsampledRowMask | newDownsampledRowMask;
    scopeMemoryUsage.add(oldDownsampledRowMask);
    scopeMemoryUsage.add(newDownsampledRowMask);
    scopeMemoryUsage.add(downsampledRowMask);
    TFloatVecVec candidateSplits;
    // We compute and cache row splits once upfront for features using fixed splits.
    TBoolVec featuresToRefresh(m_FixedCandidateSplits.size());
    for (std::size_t i = 0; i < m_FixedCandidateSplits.size(); ++i) {
        featuresToRefresh[i] = m_FeatureSampleProbabilities[i] > 0.0 &&
                               m_FixedCandidateSplits[i].empty();
    }

    TDoubleVec testLosses;
    testLosses.reserve(m_TreesToRetrain.size());
    scopeMemoryUsage.add(testLosses);

    TWorkspace workspace{m_Loss->numberParameters()};

    // The exact sequence of operations in this loop is important. For each
    // iteration:
    //   1. Remove tree to be retrained predictions and add *previous* retrained
    //      tree predictions and refresh loss derivatives.
    //   2. Periodically compute weighted quantiles for features F and candidate
    //      splits S from F.
    //   3. Build one tree on S.

    retrainedTrees.emplace_back();
    for (const auto& index : m_TreesToRetrain) {

        LOG_TRACE(<< "Retraining(" << index
                  << ") =" << root(m_BestForest[index]).print(m_BestForest[index]));

        const auto& treeToRetrain = m_BestForest[index];
        const auto& treeWhichWasRetrained = retrainedTrees.back();

        double eta{index < m_BestForest.size() - m_MaximumNumberNewTrees
                       ? m_Hyperparameters.retrainedTreeEta().value()
                       : m_Hyperparameters.etaForTreeAtPosition(index)};
        LOG_TRACE(<< "eta = " << eta);

        workspace.retraining(treeToRetrain);

        auto loss = m_Loss->incremental(
            eta, m_Hyperparameters.predictionChangeCost().value(), treeToRetrain);

        this->refreshPredictionsAndLossDerivatives(
            frame, trainingRowMask, *loss,
            [&](const TRowRef& row, TMemoryMappedFloatVector& prediction) {
                auto encodedRow = m_Encoder->encode(row);
                prediction -= root(treeToRetrain).value(encodedRow, treeToRetrain);
                if (treeWhichWasRetrained.empty() == false) {
                    prediction += root(treeWhichWasRetrained).value(encodedRow, treeWhichWasRetrained);
                }
            });

        if (retrainedTrees.size() == nextTreeCountToRefreshSplits) {
            scopeMemoryUsage.remove(candidateSplits);
            candidateSplits = this->candidateSplits(frame, downsampledRowMask);
            this->refreshSplitsCache(frame, candidateSplits, featuresToRefresh, trainingRowMask);
            scopeMemoryUsage.add(candidateSplits);
            nextTreeCountToRefreshSplits += static_cast<std::size_t>(
                std::max(0.5 / eta, MINIMUM_SPLIT_REFRESH_INTERVAL));
        }

        auto tree = this->trainTree(frame, downsampledRowMask, candidateSplits,
                                    maximumNumberInternalNodes,
                                    makeRootLeafNodeStatistics, workspace);
        this->computeLeafValues(frame, trainingRowMask, *loss, eta,
                                m_Hyperparameters.leafWeightPenaltyMultiplier().value(),
                                tree);
        LOG_TRACE(<< "retrained = " << root(tree).print(tree));

        // We delay updating the test row predictions until we have the new
        // tree in order to correctly estimate the validation loss.
        this->refreshPredictions(
            frame, testingRowMask, *loss,
            [&](const TRowRef& row, TMemoryMappedFloatVector& prediction) {
                auto encodedRow = m_Encoder->encode(row);
                prediction -= root(treeToRetrain).value(encodedRow, treeToRetrain);
                if (tree.empty() == false) {
                    prediction += root(tree).value(encodedRow, tree);
                }
            });

        scopeMemoryUsage.add(tree);
        retrainedTrees.push_back(std::move(tree));
        trainingProgress.increment();

        oldDownsampledRowMask = this->downsample(oldTrainingRowMask);
        newDownsampledRowMask = this->downsample(newTrainingRowMask);
        downsampledRowMask = oldDownsampledRowMask | newDownsampledRowMask;
        // The memory variation in the row mask from sample to sample is too
        // small to bother to track.

        testLosses.push_back(this->meanChangePenalisedLoss(frame, testingRowMask));
    }
    retrainedTrees.erase(retrainedTrees.begin());

    auto bestLoss = static_cast<std::size_t>(
        std::min_element(testLosses.begin(), testLosses.end()) - testLosses.begin());
    retrainedTrees.resize(bestLoss + 1);
    LOG_TRACE(<< "# retrained trees = " << retrainedTrees.size());

    return {std::move(retrainedTrees), testLosses[bestLoss], 0.0, std::move(testLosses)};
}

core::CPackedBitVector
CBoostedTreeImpl::downsample(const core::CPackedBitVector& trainingRowMask) const {
    // We compute a stochastic version of the candidate splits, gradients and
    // curvatures for each tree we train. The sampling scheme should minimize
    // the correlation with previous trees for fixed sample size so randomly
    // sampling without replacement is appropriate.
    if (trainingRowMask.manhattan() == 0.0) {
        return trainingRowMask;
    }

    core::CPackedBitVector result;
    do {
        result = core::CPackedBitVector{};
        for (auto i = trainingRowMask.beginOneBits();
             i != trainingRowMask.endOneBits(); ++i) {
            if (common::CSampling::uniformSample(m_Rng, 0.0, 1.0) <
                m_Hyperparameters.downsampleFactor().value()) {
                result.extend(false, *i - result.size());
                result.extend(true);
            }
        }
    } while (result.manhattan() == 0.0);
    result.extend(false, trainingRowMask.size() - result.size());
    return result;
}

void CBoostedTreeImpl::initializeFixedCandidateSplits(core::CDataFrame& frame) {

    using TDoubleUSet = boost::unordered_set<double>;
    using TDoubleUSetVec = std::vector<TDoubleUSet>;

    TSizeVec features;
    candidateRegressorFeatures(m_FeatureSampleProbabilities, features);

    m_FixedCandidateSplits.clear();
    m_FixedCandidateSplits.resize(this->numberFeatures());

    for (auto i : features) {
        if (m_Encoder->isBinary(i)) {
            m_FixedCandidateSplits[i] = TFloatVec{0.5F};
            LOG_TRACE(<< "feature '" << i << "' splits = "
                      << core::CContainerPrinter::print(m_FixedCandidateSplits[i]));
        }
    }

    features.erase(std::remove_if(features.begin(), features.end(),
                                  [this](std::size_t index) {
                                      return m_Encoder->isBinary(index);
                                  }),
                   features.end());
    LOG_TRACE(<< "candidate features = " << core::CContainerPrinter::print(features));

    auto allTrainingRowsMask = this->allTrainingRowsMask();
    auto result = frame.readRows(
        m_NumberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](TDoubleUSetVec& state, const TRowItr& beginRows, const TRowItr& endRows) {
                for (auto& set : state) {
                    set.reserve(m_NumberSplitsPerFeature + 2);
                }
                for (auto row = beginRows; row != endRows; ++row) {
                    for (std::size_t i = 0; i < features.size(); ++i) {
                        if (state[i].size() <= m_NumberSplitsPerFeature + 1) {
                            state[i].insert(m_Encoder->encode(*row)[features[i]]);
                        }
                    }
                }
            },
            TDoubleUSetVec(features.size())),
        &allTrainingRowsMask);
    auto sets = result.first;

    TDoubleUSetVec uniques{std::move(sets[0].s_FunctionState)};
    for (std::size_t i = 1; i < sets.size(); ++i) {
        for (std::size_t j = 0; j < uniques.size(); ++j) {
            const auto& uniques_ = sets[i].s_FunctionState[j];
            uniques[j].insert(uniques_.begin(), uniques_.end());
        }
    }

    TDoubleVec values;
    for (std::size_t i = 0; i < features.size(); ++i) {
        if (uniques[i].size() <= m_NumberSplitsPerFeature) {
            values.assign(uniques[i].begin(), uniques[i].end());
            std::sort(values.begin(), values.end());
            auto& featureCandidateSplits = m_FixedCandidateSplits[features[i]];
            featureCandidateSplits.reserve(values.size() - 1);
            for (std::size_t j = 1; j < values.size(); ++j) {
                featureCandidateSplits.emplace_back(0.5 * (values[j] + values[j - 1]));
            }
            LOG_TRACE(<< "feature '" << features[i] << "' splits = "
                      << core::CContainerPrinter::print(featureCandidateSplits));
        }
    }

    TBoolVec featuresToRefresh(m_FixedCandidateSplits.size());
    for (std::size_t i = 0; i < m_FixedCandidateSplits.size(); ++i) {
        featuresToRefresh[i] = (m_FixedCandidateSplits[i].empty() == false);
    }

    this->refreshSplitsCache(frame, m_FixedCandidateSplits, featuresToRefresh,
                             allTrainingRowsMask);
}

CBoostedTreeImpl::TFloatVecVec
CBoostedTreeImpl::candidateSplits(const core::CDataFrame& frame,
                                  const core::CPackedBitVector& trainingRowMask) const {

    TFloatVecVec candidateSplits{m_FixedCandidateSplits};

    TSizeVec features;
    candidateRegressorFeatures(m_FeatureSampleProbabilities, features);
    features.erase(std::remove_if(features.begin(), features.end(),
                                  [this](std::size_t index) {
                                      return m_FixedCandidateSplits[index].empty() == false;
                                  }),
                   features.end());
    LOG_TRACE(<< "candidate features = " << core::CContainerPrinter::print(features));

    if (features.empty()) {
        return candidateSplits;
    }

    auto featureQuantiles =
        CDataFrameUtils::columnQuantiles(
            m_NumberThreads, frame, trainingRowMask, features,
            common::CFastQuantileSketch{
                common::CFastQuantileSketch::E_Linear,
                std::max(m_NumberSplitsPerFeature, std::size_t{50}), m_Rng},
            m_Encoder.get(),
            [this](const TRowRef& row) {
                std::size_t numberLossParameters{m_Loss->numberParameters()};
                return trace(numberLossParameters,
                             readLossCurvature(row, m_ExtraColumns, numberLossParameters));
            })
            .first;

    for (std::size_t i = 0; i < features.size(); ++i) {

        auto& featureCandidateSplits = candidateSplits[features[i]];

        // Because we compute candidate splits for downsamples of the rows it's
        // possible that all values are missing for a particular feature. In this
        // case, we can happily initialize the candidate splits to an empty set
        // since we'll only be choosing how to assign missing values.
        if (featureQuantiles[i].count() > 0.0) {
            featureCandidateSplits.reserve(m_NumberSplitsPerFeature - 1);
            for (std::size_t j = 1; j < m_NumberSplitsPerFeature; ++j) {
                double rank{100.0 * static_cast<double>(j) /
                                static_cast<double>(m_NumberSplitsPerFeature) +
                            common::CSampling::uniformSample(m_Rng, -0.1, 0.1)};
                double q;
                if (featureQuantiles[i].quantile(rank, q)) {
                    featureCandidateSplits.emplace_back(q);
                } else {
                    LOG_WARN(<< "Failed to compute quantile " << rank << ": ignoring split");
                }
            }
            std::sort(featureCandidateSplits.begin(), featureCandidateSplits.end());
        }

        const auto& dataType = m_FeatureDataTypes[features[i]];

        if (dataType.s_IsInteger) {
            // The key point here is that we know that if two distinct splits fall
            // between two consecutive integers they must produce identical partitions
            // of the data and so always have the same loss. We only need to retain
            // one such split for training. We achieve this by snapping to the midpoint
            // and subsequently deduplicating.
            std::for_each(featureCandidateSplits.begin(), featureCandidateSplits.end(),
                          [](common::CFloatStorage& split) {
                              split = std::floor(split) + 0.5;
                          });
        }
        featureCandidateSplits.erase(std::unique(featureCandidateSplits.begin(),
                                                 featureCandidateSplits.end()),
                                     featureCandidateSplits.end());
        featureCandidateSplits.erase(std::remove_if(featureCandidateSplits.begin(),
                                                    featureCandidateSplits.end(),
                                                    [&dataType](double split) {
                                                        return split < dataType.s_Min ||
                                                               split > dataType.s_Max;
                                                    }),
                                     featureCandidateSplits.end());
    }

    return candidateSplits;
}

void CBoostedTreeImpl::refreshSplitsCache(core::CDataFrame& frame,
                                          const TFloatVecVec& candidateSplits,
                                          const TBoolVec& featureMask,
                                          const core::CPackedBitVector& trainingRowMask) const {
    if (std::none_of(featureMask.begin(), featureMask.end(),
                     [](bool refresh) { return refresh; })) {
        return;
    }

    frame.writeColumns(
        m_NumberThreads, 0, frame.numberRows(),
        [&](const TRowItr& beginRows, const TRowItr& endRows) {
            for (auto row_ = beginRows; row_ != endRows; ++row_) {
                auto row{*row_};
                auto encodedRow = m_Encoder->encode(row);
                auto* splits = beginSplits(row, m_ExtraColumns);
                for (std::size_t i = 0; i < encodedRow.numberColumns(); ++splits) {
                    CPackedUInt8Decorator::TUInt8Ary packedSplits{
                        CPackedUInt8Decorator{*splits}.readBytes()};
                    for (std::size_t j = 0;
                         j < packedSplits.size() && i < encodedRow.numberColumns();
                         ++i, ++j) {
                        if (featureMask[i]) {
                            double feature{encodedRow[i]};
                            packedSplits[j] =
                                CDataFrameUtils::isMissing(feature)
                                    ? static_cast<std::uint8_t>(
                                          missingSplit(candidateSplits[i]))
                                    : static_cast<std::uint8_t>(
                                          std::upper_bound(candidateSplits[i].begin(),
                                                           candidateSplits[i].end(), feature) -
                                          candidateSplits[i].begin());
                        }
                    }
                    *splits = CPackedUInt8Decorator{packedSplits};
                }
            }
        },
        &trainingRowMask);
}

CBoostedTreeImpl::TNodeVec
CBoostedTreeImpl::trainTree(core::CDataFrame& frame,
                            const core::CPackedBitVector& trainingRowMask,
                            const TFloatVecVec& candidateSplits,
                            std::size_t maximumNumberInternalNodes,
                            const TMakeRootLeafNodeStatistics& makeRootLeafNodeStatistics,
                            TWorkspace& workspace) const {

    LOG_TRACE(<< "Training one tree...");

    using TLeafNodeStatisticsPtrQueue = boost::circular_buffer<TLeafNodeStatisticsPtr>;

    workspace.reinitialize(m_NumberThreads, candidateSplits);
    TSizeVec featuresToInclude{workspace.featuresToInclude()};
    LOG_TRACE(<< "features to include = "
              << core::CContainerPrinter::print(featuresToInclude));

    TNodeVec tree(1);
    // Since number of leaves in a perfect binary tree is (numberInternalNodes+1)
    // the total number of nodes in a tree is (2*numberInternalNodes+1).
    tree.reserve(2 * maximumNumberInternalNodes + 1);

    // Sampling transforms the probabilities. We use a placeholder outside
    // the loop adding nodes so we only allocate the vector once.
    TDoubleVec featureSampleProbabilities{m_FeatureSampleProbabilities};
    TSizeVec treeFeatureBag;
    TSizeVec nodeFeatureBag;
    this->treeFeatureBag(featureSampleProbabilities, treeFeatureBag);
    treeFeatureBag = merge(featuresToInclude, std::move(treeFeatureBag));
    LOG_TRACE(<< "tree bag = " << core::CContainerPrinter::print(treeFeatureBag));

    featureSampleProbabilities = m_FeatureSampleProbabilities;
    this->nodeFeatureBag(treeFeatureBag, featureSampleProbabilities, nodeFeatureBag);
    nodeFeatureBag = merge(featuresToInclude, std::move(nodeFeatureBag));

    TLeafNodeStatisticsPtrQueue splittableLeaves(maximumNumberInternalNodes / 2 + 3);
    splittableLeaves.push_back(makeRootLeafNodeStatistics(
        candidateSplits, treeFeatureBag, nodeFeatureBag, trainingRowMask, workspace));

    // We update local variables because the callback can be expensive if it
    // requires accessing atomics.
    struct SMemoryStats {
        std::int64_t s_Current = 0;
        std::int64_t s_Max = 0;
    } memory;
    TMemoryUsageCallback localRecordMemoryUsage{[&](std::int64_t delta) {
        memory.s_Current += delta;
        memory.s_Max = std::max(memory.s_Max, memory.s_Current);
    }};
    CScopeRecordMemoryUsage scopeMemoryUsage{splittableLeaves,
                                             std::move(localRecordMemoryUsage)};
    scopeMemoryUsage.add(workspace);

    // For each iteration we:
    //   1. Find the leaf with the greatest decrease in loss
    //   2. If no split (significantly) reduced the loss we terminate
    //   3. Otherwise we split that leaf

    double totalGain{0.0};

    common::COrderings::SLess less;

    for (std::size_t i = 0; i < maximumNumberInternalNodes; ++i) {

        if (splittableLeaves.empty()) {
            break;
        }

        auto leaf = splittableLeaves.back();
        splittableLeaves.pop_back();

        scopeMemoryUsage.remove(leaf);

        if (leaf->gain() < MINIMUM_RELATIVE_GAIN_PER_SPLIT * totalGain) {
            break;
        }

        totalGain += leaf->gain();
        workspace.minimumGain(MINIMUM_RELATIVE_GAIN_PER_SPLIT * totalGain);
        LOG_TRACE(<< "splitting " << leaf->id() << " leaf gain = " << leaf->gain()
                  << " total gain = " << totalGain);

        std::size_t splitFeature;
        double splitValue;
        std::tie(splitFeature, splitValue) = leaf->bestSplit();

        bool assignMissingToLeft{leaf->assignMissingToLeft()};

        // Add the left and right children to the tree.
        std::size_t leftChildId;
        std::size_t rightChildId;
        std::tie(leftChildId, rightChildId) = tree[leaf->id()].split(
            candidateSplits, splitFeature, splitValue, assignMissingToLeft,
            leaf->gain(), leaf->gainVariance(), leaf->curvature(), tree);

        featureSampleProbabilities = m_FeatureSampleProbabilities;
        this->nodeFeatureBag(treeFeatureBag, featureSampleProbabilities, nodeFeatureBag);
        nodeFeatureBag = merge(featuresToInclude, std::move(nodeFeatureBag));

        std::size_t numberSplittableLeaves{splittableLeaves.size()};
        std::size_t currentNumberInternalNodes{(tree.size() - 1) / 2};
        auto smallestCurrentCandidateGainIndex =
            static_cast<std::ptrdiff_t>(numberSplittableLeaves) -
            static_cast<std::ptrdiff_t>(maximumNumberInternalNodes - currentNumberInternalNodes);
        double smallestCandidateGain{
            smallestCurrentCandidateGainIndex >= 0
                ? splittableLeaves[static_cast<std::size_t>(smallestCurrentCandidateGainIndex)]
                      ->gain()
                : 0.0};

        TLeafNodeStatisticsPtr leftChild;
        TLeafNodeStatisticsPtr rightChild;
        std::tie(leftChild, rightChild) = leaf->split(
            leftChildId, rightChildId, smallestCandidateGain, frame, m_Hyperparameters,
            treeFeatureBag, nodeFeatureBag, tree[leaf->id()], workspace);

        // Need gain to be computed to compare here.
        if (leftChild != nullptr && rightChild != nullptr && less(rightChild, leftChild)) {
            std::swap(leftChild, rightChild);
        }

        if (leftChild != nullptr &&
            leftChild->gain() >= MINIMUM_RELATIVE_GAIN_PER_SPLIT * totalGain) {
            scopeMemoryUsage.add(leftChild);
            splittableLeaves.push_back(std::move(leftChild));
        }
        if (rightChild != nullptr &&
            rightChild->gain() >= MINIMUM_RELATIVE_GAIN_PER_SPLIT * totalGain) {
            scopeMemoryUsage.add(rightChild);
            splittableLeaves.push_back(std::move(rightChild));
        }
        std::inplace_merge(splittableLeaves.begin(),
                           splittableLeaves.begin() + numberSplittableLeaves,
                           splittableLeaves.end(), less);

        // Drop any leaves which can't possibly be split.
        while (splittableLeaves.size() + i + 1 > maximumNumberInternalNodes) {
            scopeMemoryUsage.remove(splittableLeaves.front());
            workspace.minimumGain(splittableLeaves.front()->gain());
            splittableLeaves.pop_front();
        }
    }

    tree.shrink_to_fit();

    // Flush the maximum memory used by the leaf statistics to the callback.
    m_Instrumentation->updateMemoryUsage(memory.s_Max);
    m_Instrumentation->updateMemoryUsage(-memory.s_Max);

    LOG_TRACE(<< "Trained one tree. # nodes = " << tree.size());

    return tree;
}

void CBoostedTreeImpl::scaleRegularizationMultipliers(double scale) {
    if (m_Hyperparameters.scalingDisabled() == false) {
        if (m_Hyperparameters.depthPenaltyMultiplier().fixed() == false) {
            m_Hyperparameters.depthPenaltyMultiplier().scale(scale);
        }
        if (m_Hyperparameters.treeSizePenaltyMultiplier().fixed() == false) {
            m_Hyperparameters.treeSizePenaltyMultiplier().scale(scale);
        }
        if (m_Hyperparameters.leafWeightPenaltyMultiplier().fixed() == false) {
            m_Hyperparameters.leafWeightPenaltyMultiplier().scale(scale);
        }
        if (m_Hyperparameters.treeTopologyChangePenalty().fixed() == false) {
            m_Hyperparameters.treeTopologyChangePenalty().scale(scale);
        }
    }
}

double CBoostedTreeImpl::minimumTestLoss() const {
    using TMinAccumulator = common::CBasicStatistics::SMin<double>::TAccumulator;
    TMinAccumulator minimumTestLoss;
    for (std::size_t round = 0; round + 1 < m_Hyperparameters.currentRound(); ++round) {
        TMeanVarAccumulator roundLossMoments;
        for (std::size_t fold = 0; fold < m_NumberFolds.value(); ++fold) {
            if (m_FoldRoundTestLosses[fold][round] != boost::none) {
                roundLossMoments.add(*m_FoldRoundTestLosses[fold][round]);
            }
        }
        if (static_cast<std::size_t>(common::CBasicStatistics::count(roundLossMoments)) ==
            m_NumberFolds.value()) {
            minimumTestLoss.add(common::CBasicStatistics::mean(roundLossMoments));
        }
    }
    return minimumTestLoss[0];
}

CBoostedTreeImpl::TMeanVarAccumulator
CBoostedTreeImpl::correctTestLossMoments(const TSizeVec& missing,
                                         TMeanVarAccumulator testLossMoments) const {
    if (missing.empty()) {
        return testLossMoments;
    }
    for (const auto& testLoss : this->estimateMissingTestLosses(missing)) {
        testLossMoments += testLoss;
    }
    return testLossMoments;
}

CBoostedTreeImpl::TMeanVarAccumulatorVec
CBoostedTreeImpl::estimateMissingTestLosses(const TSizeVec& missing) const {

    // We have a subset of folds for which we've computed test loss. We want to
    // estimate the test loss we'll see for the remaining folds to decide if it
    // is worthwhile to continue training with these parameters and to correct
    // the loss value supplied to Bayesian Optimisation to account for the folds
    // we haven't trained on. We tackle this problem as follows:
    //   1. Find all previous rounds R which share at least one fold with the
    //      current round, i.e. one fold for which we've computed the actual
    //      loss for the current round parameters.
    //   2. For each fold f_i for which we haven't estimated the loss in the
    //      current round fit an OLS model m_i to R to predict the loss of f_i.
    //   3. Compute the predicted value for the test loss on each f_i given
    //      the test losses we've computed so far the current round using m_i.
    //   4. Estimate the uncertainty from the variance of the residuals from
    //      fitting the model m_i to R.
    //
    // The feature vector we use is defined as:
    //
    //   |   calculated fold error 1  |
    //   |   calculated fold error 2  |
    //   |             ...            |
    //   | 1{fold error 1 is present} |
    //   | 1{fold error 2 is present} |
    //   |             ...            |
    //
    // where the indices range over the folds for which we have errors in the
    // current round.

    TSizeVec present(m_NumberFolds.value());
    std::iota(present.begin(), present.end(), 0);
    TSizeVec ordered{missing};
    std::sort(ordered.begin(), ordered.end());
    common::CSetTools::inplace_set_difference(present, ordered.begin(), ordered.end());
    LOG_TRACE(<< "present = " << core::CContainerPrinter::print(present));

    // Get the current round feature vector. Fixed so computed outside the loop.
    TVector x(2 * present.size());
    for (std::size_t col = 0; col < present.size(); ++col) {
        x(col) = *m_FoldRoundTestLosses[present[col]][m_Hyperparameters.currentRound()];
        x(present.size() + col) = 0.0;
    }

    TMeanVarAccumulatorVec predictedTestLosses;
    predictedTestLosses.reserve(missing.size());

    for (std::size_t target : missing) {
        // Extract the training mask.
        TSizeVec trainingMask;
        trainingMask.reserve(m_Hyperparameters.currentRound());
        for (std::size_t round = 0; round < m_Hyperparameters.currentRound(); ++round) {
            if (m_FoldRoundTestLosses[target][round] &&
                std::find_if(present.begin(), present.end(), [&](std::size_t fold) {
                    return m_FoldRoundTestLosses[fold][round];
                }) != present.end()) {
                trainingMask.push_back(round);
            }
        }

        // Fit the OLS regression.
        common::CDenseMatrix<double> A(trainingMask.size(), 2 * present.size());
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
            common::CBasicStatistics::maximumLikelihoodVariance(residualMoments)};
        LOG_TRACE(<< "prediction(x = " << x.transpose() << ", fold = " << target
                  << ") = (mean = " << predictedTestLoss
                  << ", variance = " << predictedTestLossVariance << ")");

        predictedTestLosses.push_back(common::CBasicStatistics::momentsAccumulator(
            1.0, predictedTestLoss, predictedTestLossVariance));
    }

    return predictedTestLosses;
}

std::size_t CBoostedTreeImpl::rowsPerFeature(std::size_t numberRows) const {
    // For small data sets (fewer than 1k examples) we allow ourselves to use
    // more features than implied by m_RowsPerFeature. Since we remove nuisance
    // features which carry little information about the target this is fine
    // from an accuracy perspective. From a runtime perspective we always train
    // fast for such small data sets.
    return std::max(std::min(m_RowsPerFeature, numberRows / 20), std::size_t{1});
}

std::size_t CBoostedTreeImpl::numberFeatures() const {
    return m_Encoder->numberEncodedColumns();
}

std::size_t CBoostedTreeImpl::featureBagSize(double fraction) const {
    return static_cast<std::size_t>(std::max(
        std::ceil(std::min(fraction, 1.0) * static_cast<double>(this->numberFeatures())), 1.0));
}

void CBoostedTreeImpl::treeFeatureBag(TDoubleVec& probabilities, TSizeVec& treeFeatureBag) const {

    std::size_t size{
        this->featureBagSize(1.25 * m_Hyperparameters.featureBagFraction().value())};

    candidateRegressorFeatures(probabilities, treeFeatureBag);
    if (size >= treeFeatureBag.size()) {
        return;
    }

    common::CSampling::categoricalSampleWithoutReplacement(m_Rng, probabilities,
                                                           size, treeFeatureBag);
    std::sort(treeFeatureBag.begin(), treeFeatureBag.end());
}

void CBoostedTreeImpl::nodeFeatureBag(const TSizeVec& treeFeatureBag,
                                      TDoubleVec& probabilities,
                                      TSizeVec& nodeFeatureBag) const {

    std::size_t size{this->featureBagSize(m_Hyperparameters.featureBagFraction().value())};

    if (size >= treeFeatureBag.size()) {
        // Since we don't include features with zero probability of being sampled
        // in the bag we can just copy this collection over as the candidates.
        nodeFeatureBag = treeFeatureBag;
        return;
    }

    // We have P(i in S) = P(i in S | i in B) P(i in B) for sample S and bag B.
    // We'd ideally like to preserve P(i in S) by arranging for P(i in S | i in B)
    // to be equal to P(i in S) / P(i in B). P(i in B) is the chance of sampling
    // item i without replacement for weights w. There is no closed form solution
    // for this probability. We can simply sample many bags and use the relative
    // frequency to get this quantity to arbitrary precision, but this isn't
    // worthwhile. There are two limits, for |B| / |F| -> 0 then P(i in B) -> w_i
    // and we want to sample uniformly at random from B and for |B| -> |F| then
    // P(i in B) -> 1 and we want to sample according to w so we reweight by
    // 1 / (w_i + |B| / |F| (1 - w_i)).

    double fraction{static_cast<double>(treeFeatureBag.size()) /
                    static_cast<double>(std::count_if(
                        probabilities.begin(), probabilities.end(),
                        [](auto probability) { return probability > 0.0; }))};
    LOG_TRACE(<< "fraction = " << fraction);

    for (std::size_t i = 0; i < treeFeatureBag.size(); ++i) {
        probabilities[i] = probabilities[treeFeatureBag[i]];
    }
    probabilities.resize(treeFeatureBag.size());
    double Z{std::accumulate(probabilities.begin(), probabilities.end(), 0.0)};
    for (auto& probability : probabilities) {
        probability /= Z;
        probability /= probability + fraction * (1.0 - probability);
    }

    common::CSampling::categoricalSampleWithoutReplacement(m_Rng, probabilities,
                                                           size, nodeFeatureBag);
    for (auto& i : nodeFeatureBag) {
        i = treeFeatureBag[i];
    }
    std::sort(nodeFeatureBag.begin(), nodeFeatureBag.end());
}

void CBoostedTreeImpl::candidateRegressorFeatures(const TDoubleVec& probabilities,
                                                  TSizeVec& features) {
    features.clear();
    features.reserve(probabilities.size());
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        if (probabilities[i] > 0.0) {
            features.push_back(i);
        }
    }
}

void CBoostedTreeImpl::computeLeafValues(core::CDataFrame& frame,
                                         const core::CPackedBitVector& trainingRowMask,
                                         const TLossFunction& loss,
                                         double eta,
                                         double lambda,
                                         TNodeVec& tree) const {

    if (tree.empty()) {
        return;
    }

    TSizeVec leafMap(tree.size());
    std::size_t numberLeaves{0};
    for (std::size_t i = 0; i < tree.size(); ++i) {
        if (tree[i].isLeaf()) {
            leafMap[i] = numberLeaves++;
        }
    }

    TArgMinLossVec leafValues;
    auto nextPass = [&] {
        bool done{true};
        for (const auto& value : leafValues) {
            done &= (value.nextPass() == false);
        }
        return done == false;
    };

    leafValues.resize(numberLeaves, loss.minimizer(lambda, m_Rng));
    do {
        TArgMinLossVecVec result(m_NumberThreads, leafValues);
        this->minimumLossLeafValues(false /*new example*/, frame,
                                    trainingRowMask & ~m_NewTrainingRowMask,
                                    loss, leafMap, tree, result);
        this->minimumLossLeafValues(true /*new example*/, frame,
                                    trainingRowMask & m_NewTrainingRowMask,
                                    loss, leafMap, tree, result);
        leafValues = std::move(result[0]);
        for (std::size_t i = 1; i < result.size(); ++i) {
            for (std::size_t j = 0; j < leafValues.size(); ++j) {
                leafValues[j].merge(result[i][j]);
            }
        }
    } while (nextPass());

    core::parallel_for_each(0, tree.size(), [&](std::size_t i) {
        if (tree[i].isLeaf()) {
            tree[i].value(eta * leafValues[leafMap[i]].value());
        }
    });

    LOG_TRACE(<< "tree = " << root(tree).print(tree));
}

void CBoostedTreeImpl::minimumLossLeafValues(bool newExample,
                                             const core::CDataFrame& frame,
                                             const core::CPackedBitVector& rowMask,
                                             const TLossFunction& loss,
                                             const TSizeVec& leafMap,
                                             const TNodeVec& tree,
                                             TArgMinLossVecVec& result) const {

    core::CDataFrame::TRowFuncVec minimizers;
    minimizers.reserve(result.size());
    for (auto& leafValues : result) {
        minimizers.push_back([&](const TRowItr& beginRows, const TRowItr& endRows) {
            std::size_t numberLossParameters{loss.numberParameters()};
            const auto& rootNode = root(tree);
            for (auto row_ = beginRows; row_ != endRows; ++row_) {
                auto row = *row_;
                auto encodedRow = m_Encoder->encode(row);
                auto prediction = readPrediction(row, m_ExtraColumns, numberLossParameters);
                double actual{readActual(row, m_DependentVariable)};
                double weight{readExampleWeight(row, m_ExtraColumns)};
                std::size_t index{rootNode.leafIndex(row, m_ExtraColumns, tree)};
                leafValues[leafMap[index]].add(encodedRow, newExample,
                                               prediction, actual, weight);
            }
        });
    }

    frame.readRows(0, frame.numberRows(), minimizers, &rowMask);
}

void CBoostedTreeImpl::refreshPredictionsAndLossDerivatives(
    core::CDataFrame& frame,
    const core::CPackedBitVector& rowMask,
    const TLossFunction& loss,
    const TUpdateRowPrediction& updateRowPrediction) const {
    this->refreshPredictionsAndLossDerivatives(false /*new example*/, frame,
                                               rowMask & ~m_NewTrainingRowMask,
                                               loss, updateRowPrediction);
    this->refreshPredictionsAndLossDerivatives(true /*new example*/, frame,
                                               rowMask & m_NewTrainingRowMask,
                                               loss, updateRowPrediction);
}

void CBoostedTreeImpl::refreshPredictionsAndLossDerivatives(
    bool newExample,
    core::CDataFrame& frame,
    const core::CPackedBitVector& rowMask,
    const TLossFunction& loss,
    const TUpdateRowPrediction& updateRowPrediction) const {
    frame.writeColumns(
        m_NumberThreads, 0, frame.numberRows(),
        [&](const TRowItr& beginRows, const TRowItr& endRows) {
            std::size_t numberLossParameters{loss.numberParameters()};
            for (auto row_ = beginRows; row_ != endRows; ++row_) {
                auto row = *row_;
                auto encodedRow = m_Encoder->encode(row);
                auto prediction = readPrediction(row, m_ExtraColumns, numberLossParameters);
                double actual{readActual(row, m_DependentVariable)};
                double weight{readExampleWeight(row, m_ExtraColumns)};
                updateRowPrediction(row, prediction);
                writeLossGradient(row, encodedRow, newExample, m_ExtraColumns,
                                  loss, prediction, actual, weight);
                writeLossCurvature(row, encodedRow, newExample, m_ExtraColumns,
                                   loss, prediction, actual, weight);
            }
        },
        &rowMask);
}

void CBoostedTreeImpl::refreshPredictions(core::CDataFrame& frame,
                                          const core::CPackedBitVector& rowMask,
                                          const TLossFunction& loss,
                                          const TUpdateRowPrediction& updateRowPrediction) const {
    frame.writeColumns(m_NumberThreads, 0, frame.numberRows(),
                       [&](const TRowItr& beginRows, const TRowItr& endRows) {
                           std::size_t numberLossParameters{loss.numberParameters()};
                           for (auto row_ = beginRows; row_ != endRows; ++row_) {
                               auto row = *row_;
                               auto prediction = readPrediction(
                                   row, m_ExtraColumns, numberLossParameters);
                               updateRowPrediction(row, prediction);
                           }
                       },
                       &rowMask);
}

double CBoostedTreeImpl::meanLoss(const core::CDataFrame& frame,
                                  const core::CPackedBitVector& rowMask) const {

    auto results = frame.readRows(
        m_NumberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](TMeanAccumulator& loss, const TRowItr& beginRows, const TRowItr& endRows) {
                std::size_t numberLossParameters{m_Loss->numberParameters()};
                for (auto row_ = beginRows; row_ != endRows; ++row_) {
                    auto row = *row_;
                    auto prediction = readPrediction(row, m_ExtraColumns, numberLossParameters);
                    double actual{readActual(row, m_DependentVariable)};
                    double weight{readExampleWeight(row, m_ExtraColumns)};
                    loss.add(m_Loss->value(prediction, actual), weight);
                }
            },
            TMeanAccumulator{}),
        &rowMask);

    TMeanAccumulator loss;
    for (const auto& result : results.first) {
        loss += result.s_FunctionState;
    }

    LOG_TRACE(<< "loss = " << common::CBasicStatistics::mean(loss));

    return common::CBasicStatistics::mean(loss);
}

double CBoostedTreeImpl::meanChangePenalisedLoss(const core::CDataFrame& frame,
                                                 const core::CPackedBitVector& rowMask) const {

    // Add on 0.01 times the difference in the old predictions to encourage us
    // to choose more similar forests if accuracy is similar.

    core::CPackedBitVector oldRowMask{rowMask & ~m_NewTrainingRowMask};
    auto results = frame.readRows(
        m_NumberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](TMeanAccumulator& loss, const TRowItr& beginRows, const TRowItr& endRows) {
                std::size_t numberLossParameters{m_Loss->numberParameters()};
                for (auto row_ = beginRows; row_ != endRows; ++row_) {
                    auto row = *row_;
                    auto prediction = readPrediction(row, m_ExtraColumns, numberLossParameters);
                    auto previousPrediction = readPreviousPrediction(
                        row, m_ExtraColumns, numberLossParameters);
                    double weight{readExampleWeight(row, m_ExtraColumns)};
                    loss.add(m_Loss->difference(prediction, previousPrediction, 0.01), weight);
                }
            },
            TMeanAccumulator{}),
        &oldRowMask);

    TMeanAccumulator lossAdjustment;
    for (const auto& result : results.first) {
        lossAdjustment += result.s_FunctionState;
    }

    double adjustedLoss{this->meanLoss(frame, rowMask) +
                        oldRowMask.manhattan() / rowMask.manhattan() *
                            common::CBasicStatistics::mean(lossAdjustment)};

    LOG_TRACE(<< "adjusted loss = " << adjustedLoss);

    return adjustedLoss;
}

double CBoostedTreeImpl::betweenFoldTestLossVariance() const {
    TMeanVarAccumulator result;
    for (const auto& testLosses : m_FoldRoundTestLosses) {
        TMeanAccumulator meanTestLoss;
        for (std::size_t i = 0; i <= m_Hyperparameters.currentRound(); ++i) {
            if (testLosses[i] != boost::none) {
                meanTestLoss.add(*testLosses[i]);
            }
        }
        result.add(common::CBasicStatistics::mean(meanTestLoss));
    }
    return common::CBasicStatistics::maximumLikelihoodVariance(result);
}

CBoostedTreeImpl::TVector CBoostedTreeImpl::predictRow(const CEncodedDataFrameRowRef& row) const {
    TVector result{TVector::Zero(m_Loss->numberParameters())};
    for (const auto& tree : m_BestForest) {
        result += root(tree).value(row, tree);
    }
    return result;
}

std::size_t CBoostedTreeImpl::maximumTreeSize(const core::CPackedBitVector& trainingRowMask) {
    return maximumTreeSize(static_cast<std::size_t>(trainingRowMask.manhattan()));
}

std::size_t CBoostedTreeImpl::maximumTreeSize(std::size_t numberRows) {
    return static_cast<std::size_t>(
        std::ceil(10.0 * std::sqrt(static_cast<double>(numberRows))));
}

std::size_t CBoostedTreeImpl::numberTreesToRetrain() const {
    return m_TreesToRetrain.size();
}

std::size_t CBoostedTreeImpl::maximumTrainedModelSize() const {
    return static_cast<std::size_t>(0.95 * static_cast<double>(m_MaximumDeployedSize) + 0.5);
}

void CBoostedTreeImpl::recordHyperparameters() {
    m_Instrumentation->hyperparameters().s_ClassAssignmentObjective = m_ClassAssignmentObjective;
    m_Instrumentation->hyperparameters().s_MaxAttemptsToAddTree = m_MaximumAttemptsToAddTree;
    m_Instrumentation->hyperparameters().s_NumFolds = m_NumberFolds.value();
    m_Instrumentation->hyperparameters().s_NumSplitsPerFeature = m_NumberSplitsPerFeature;
    m_Hyperparameters.recordHyperparameters(*m_Instrumentation);
}

void CBoostedTreeImpl::startProgressMonitoringFineTuneHyperparameters() {

    // This costs "number folds" * "maximum number trees per forest" units
    // per round.

    m_Instrumentation->startNewProgressMonitoredTask(CBoostedTreeFactory::FINE_TUNING_PARAMETERS);

    std::size_t totalNumberSteps{m_Hyperparameters.numberRounds() *
                                 m_Hyperparameters.maximumNumberTrees().value() *
                                 m_NumberFolds.value()};
    LOG_TRACE(<< "main loop total number steps = " << totalNumberSteps);
    m_TrainingProgress = core::CLoopProgress{
        totalNumberSteps, m_Instrumentation->progressCallback(), 1.0, 1024};

    // Make sure progress starts where it left off.
    m_TrainingProgress.increment(m_Hyperparameters.currentRound() *
                                 m_Hyperparameters.maximumNumberTrees().value() *
                                 m_NumberFolds.value());
}

void CBoostedTreeImpl::startProgressMonitoringFinalTrain() {

    // The final model training uses more data so it's monitored separately.

    m_Instrumentation->startNewProgressMonitoredTask(CBoostedTreeFactory::FINAL_TRAINING);
    m_TrainingProgress =
        core::CLoopProgress{m_Hyperparameters.maximumNumberTrees().value(),
                            m_Instrumentation->progressCallback(), 1.0, 1024};
}

void CBoostedTreeImpl::skipProgressMonitoringFinalTrain() {
    m_Instrumentation->startNewProgressMonitoredTask(CBoostedTreeFactory::FINAL_TRAINING);
}

void CBoostedTreeImpl::startProgressMonitoringTrainIncremental() {

    // This costs "number folds" * "maximum number retrained trees" units
    // per round.

    m_Instrumentation->startNewProgressMonitoredTask(CBoostedTreeFactory::INCREMENTAL_TRAIN);

    std::size_t totalNumberSteps{m_Hyperparameters.numberRounds() *
                                 this->numberTreesToRetrain() * m_NumberFolds.value()};
    LOG_TRACE(<< "main loop total number steps = " << totalNumberSteps);
    m_TrainingProgress = core::CLoopProgress{
        totalNumberSteps, m_Instrumentation->progressCallback(), 1.0, 1024};

    // Make sure progress starts where it left off.
    m_TrainingProgress.increment(m_Hyperparameters.currentRound() *
                                 this->numberTreesToRetrain() * m_NumberFolds.value());
}

namespace {
const std::string VERSION_8_2_TAG{"8.2"};
const TStrVec SUPPORTED_VERSIONS{VERSION_8_2_TAG};

const std::string BEST_FOREST_TAG{"best_forest"};
const std::string CLASSIFICATION_WEIGHTS_OVERRIDE_TAG{"classification_weights_tag"};
const std::string DEPENDENT_VARIABLE_TAG{"dependent_variable"};
const std::string ENCODER_TAG{"encoder"};
const std::string FEATURE_DATA_TYPES_TAG{"feature_data_types"};
const std::string FEATURE_SAMPLE_PROBABILITIES_TAG{"feature_sample_probabilities"};
const std::string FOLD_ROUND_TEST_LOSSES_TAG{"fold_round_test_losses"};
const std::string FORCE_ACCEPT_INCREMENTAL_TRAINING_TAG{"force_accept_incremental_training"};
const std::string HYPERPARAMETERS_TAG{"hyperparameters"};
const std::string INITIALIZATION_STAGE_TAG{"initialization_progress"};
const std::string LOSS_TAG{"loss"};
const std::string MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG{"maximum_attempts_to_add_tree"};
const std::string MAXIMUM_NUMBER_NEW_TREES_TAG{"maximum_number_new_trees"};
const std::string MISSING_FEATURE_ROW_MASKS_TAG{"missing_feature_row_masks"};
const std::string NEW_TRAINING_ROW_MASK_TAG{"new_training_row_mask_tag"};
const std::string NUMBER_FOLDS_TAG{"number_folds"};
const std::string NUMBER_SPLITS_PER_FEATURE_TAG{"number_splits_per_feature"};
const std::string NUMBER_THREADS_TAG{"number_threads"};
const std::string PREVIOUS_TRAIN_LOSS_GAP_TAG{"previous_train_loss_gap"};
const std::string PREVIOUS_TRAIN_NUMBER_ROWS_TAG{"previous_train_number_rows"};
const std::string RANDOM_NUMBER_GENERATOR_TAG{"random_number_generator"};
const std::string RETRAIN_FRACTION_TAG{"retrain_fraction"};
const std::string ROWS_PER_FEATURE_TAG{"rows_per_feature"};
const std::string SEED_TAG{"seed"};
const std::string STOP_CROSS_VALIDATION_EARLY_TAG{"stop_cross_validation_early"};
const std::string TESTING_ROW_MASKS_TAG{"testing_row_masks"};
const std::string TRAINING_ROW_MASKS_TAG{"training_row_masks"};
const std::string TRAIN_FRACTION_PER_FOLD_TAG{"train_fraction_per_folds"};
const std::string TREES_TO_RETRAIN_TAG{"trees_to_retrain"};
const std::string NUMBER_TOP_SHAP_VALUES_TAG{"top_shap_values"};
const std::string DATA_SUMMARIZATION_FRACTION_TAG{"data_summarization_fraction"};
}

void CBoostedTreeImpl::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(VERSION_8_2_TAG, "", inserter);
    core::CPersistUtils::persist(BEST_FOREST_TAG, m_BestForest, inserter);
    core::CPersistUtils::persistIfNotNull(CLASSIFICATION_WEIGHTS_OVERRIDE_TAG,
                                          m_ClassificationWeightsOverride, inserter);
    core::CPersistUtils::persist(DATA_SUMMARIZATION_FRACTION_TAG,
                                 m_DataSummarizationFraction, inserter);
    core::CPersistUtils::persist(DEPENDENT_VARIABLE_TAG, m_DependentVariable, inserter);
    core::CPersistUtils::persistIfNotNull(ENCODER_TAG, m_Encoder, inserter);
    core::CPersistUtils::persist(FEATURE_DATA_TYPES_TAG, m_FeatureDataTypes, inserter);
    core::CPersistUtils::persist(FEATURE_SAMPLE_PROBABILITIES_TAG,
                                 m_FeatureSampleProbabilities, inserter);
    core::CPersistUtils::persist(FOLD_ROUND_TEST_LOSSES_TAG, m_FoldRoundTestLosses, inserter);
    core::CPersistUtils::persist(FORCE_ACCEPT_INCREMENTAL_TRAINING_TAG,
                                 m_ForceAcceptIncrementalTraining, inserter);
    core::CPersistUtils::persist(HYPERPARAMETERS_TAG, m_Hyperparameters, inserter);
    core::CPersistUtils::persist(INITIALIZATION_STAGE_TAG,
                                 static_cast<int>(m_InitializationStage), inserter);
    if (m_Loss != nullptr) {
        inserter.insertLevel(LOSS_TAG, [this](core::CStatePersistInserter& inserter_) {
            m_Loss->persistLoss(inserter_);
        });
    }
    core::CPersistUtils::persist(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                                 m_MaximumAttemptsToAddTree, inserter);
    core::CPersistUtils::persist(MAXIMUM_NUMBER_NEW_TREES_TAG,
                                 m_MaximumNumberNewTrees, inserter);
    core::CPersistUtils::persist(MISSING_FEATURE_ROW_MASKS_TAG,
                                 m_MissingFeatureRowMasks, inserter);
    core::CPersistUtils::persist(NEW_TRAINING_ROW_MASK_TAG, m_NewTrainingRowMask, inserter);
    core::CPersistUtils::persist(NUMBER_FOLDS_TAG, m_NumberFolds, inserter);
    core::CPersistUtils::persist(NUMBER_SPLITS_PER_FEATURE_TAG,
                                 m_NumberSplitsPerFeature, inserter);
    core::CPersistUtils::persist(NUMBER_THREADS_TAG, m_NumberThreads, inserter);
    core::CPersistUtils::persist(NUMBER_TOP_SHAP_VALUES_TAG, m_NumberTopShapValues, inserter);
    core::CPersistUtils::persist(PREVIOUS_TRAIN_LOSS_GAP_TAG, m_PreviousTrainLossGap, inserter);
    core::CPersistUtils::persist(PREVIOUS_TRAIN_NUMBER_ROWS_TAG,
                                 m_PreviousTrainNumberRows, inserter);
    inserter.insertValue(RANDOM_NUMBER_GENERATOR_TAG, m_Rng.toString());
    core::CPersistUtils::persist(RETRAIN_FRACTION_TAG, m_RetrainFraction, inserter);
    core::CPersistUtils::persist(ROWS_PER_FEATURE_TAG, m_RowsPerFeature, inserter);
    core::CPersistUtils::persist(SEED_TAG, m_Seed, inserter);
    core::CPersistUtils::persist(STOP_CROSS_VALIDATION_EARLY_TAG,
                                 m_StopCrossValidationEarly, inserter);
    core::CPersistUtils::persist(TESTING_ROW_MASKS_TAG, m_TestingRowMasks, inserter);
    core::CPersistUtils::persist(TRAINING_ROW_MASKS_TAG, m_TrainingRowMasks, inserter);
    core::CPersistUtils::persist(TRAIN_FRACTION_PER_FOLD_TAG, m_TrainFractionPerFold, inserter);
    core::CPersistUtils::persist(TREES_TO_RETRAIN_TAG, m_TreesToRetrain, inserter);
    // Extra column information is recreated when training state is restored.
}

bool CBoostedTreeImpl::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    if (std::find(SUPPORTED_VERSIONS.begin(), SUPPORTED_VERSIONS.end(),
                  traverser.name()) == SUPPORTED_VERSIONS.end()) {
        LOG_ERROR(<< "Input error: unsupported state serialization version. "
                  << "Currently supported versions: "
                  << core::CContainerPrinter::print(SUPPORTED_VERSIONS) << ".");
        return false;
    }

    auto restoreLoss = [this](core::CStateRestoreTraverser& traverser_) {
        m_Loss = CLoss::restoreLoss(traverser_);
        return m_Loss != nullptr;
    };

    int initializationStage{static_cast<int>(E_FullyInitialized)};

    do {
        const std::string& name{traverser.name()};
        RESTORE(BEST_FOREST_TAG,
                core::CPersistUtils::restore(BEST_FOREST_TAG, m_BestForest, traverser))
        RESTORE_SETUP_TEARDOWN(
            CLASSIFICATION_WEIGHTS_OVERRIDE_TAG,
            m_ClassificationWeightsOverride = TStrDoublePrVec{},
            core::CPersistUtils::restore(CLASSIFICATION_WEIGHTS_OVERRIDE_TAG,
                                         *m_ClassificationWeightsOverride, traverser),
            /*no-op*/)
        RESTORE(DATA_SUMMARIZATION_FRACTION_TAG,
                core::CPersistUtils::restore(DATA_SUMMARIZATION_FRACTION_TAG,
                                             m_DataSummarizationFraction, traverser))
        RESTORE(DEPENDENT_VARIABLE_TAG,
                core::CPersistUtils::restore(DEPENDENT_VARIABLE_TAG,
                                             m_DependentVariable, traverser))
        RESTORE_NO_ERROR(ENCODER_TAG,
                         m_Encoder = std::make_unique<CDataFrameCategoryEncoder>(traverser))
        RESTORE(FEATURE_DATA_TYPES_TAG,
                core::CPersistUtils::restore(FEATURE_DATA_TYPES_TAG,
                                             m_FeatureDataTypes, traverser));
        RESTORE(FEATURE_SAMPLE_PROBABILITIES_TAG,
                core::CPersistUtils::restore(FEATURE_SAMPLE_PROBABILITIES_TAG,
                                             m_FeatureSampleProbabilities, traverser))
        RESTORE(FOLD_ROUND_TEST_LOSSES_TAG,
                core::CPersistUtils::restore(FOLD_ROUND_TEST_LOSSES_TAG,
                                             m_FoldRoundTestLosses, traverser))
        RESTORE(FORCE_ACCEPT_INCREMENTAL_TRAINING_TAG,
                core::CPersistUtils::restore(FORCE_ACCEPT_INCREMENTAL_TRAINING_TAG,
                                             m_ForceAcceptIncrementalTraining, traverser))
        RESTORE(HYPERPARAMETERS_TAG,
                core::CPersistUtils::restore(HYPERPARAMETERS_TAG, m_Hyperparameters, traverser))
        RESTORE(INITIALIZATION_STAGE_TAG,
                core::CPersistUtils::restore(INITIALIZATION_STAGE_TAG,
                                             initializationStage, traverser))
        RESTORE(LOSS_TAG, traverser.traverseSubLevel(restoreLoss))
        RESTORE(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                core::CPersistUtils::restore(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                                             m_MaximumAttemptsToAddTree, traverser))
        RESTORE(MAXIMUM_NUMBER_NEW_TREES_TAG,
                core::CPersistUtils::restore(MAXIMUM_NUMBER_NEW_TREES_TAG,
                                             m_MaximumNumberNewTrees, traverser))
        RESTORE(MISSING_FEATURE_ROW_MASKS_TAG,
                core::CPersistUtils::restore(MISSING_FEATURE_ROW_MASKS_TAG,
                                             m_MissingFeatureRowMasks, traverser))
        RESTORE(NEW_TRAINING_ROW_MASK_TAG,
                core::CPersistUtils::restore(NEW_TRAINING_ROW_MASK_TAG,
                                             m_NewTrainingRowMask, traverser))
        RESTORE(NUMBER_FOLDS_TAG,
                core::CPersistUtils::restore(NUMBER_FOLDS_TAG, m_NumberFolds, traverser))
        RESTORE(NUMBER_SPLITS_PER_FEATURE_TAG,
                core::CPersistUtils::restore(NUMBER_SPLITS_PER_FEATURE_TAG,
                                             m_NumberSplitsPerFeature, traverser))
        RESTORE(NUMBER_THREADS_TAG,
                core::CPersistUtils::restore(NUMBER_THREADS_TAG, m_NumberThreads, traverser))
        RESTORE(NUMBER_TOP_SHAP_VALUES_TAG,
                core::CPersistUtils::restore(NUMBER_TOP_SHAP_VALUES_TAG,
                                             m_NumberTopShapValues, traverser))
        RESTORE(PREVIOUS_TRAIN_LOSS_GAP_TAG,
                core::CPersistUtils::restore(PREVIOUS_TRAIN_LOSS_GAP_TAG,
                                             m_PreviousTrainLossGap, traverser))
        RESTORE(PREVIOUS_TRAIN_NUMBER_ROWS_TAG,
                core::CPersistUtils::restore(PREVIOUS_TRAIN_NUMBER_ROWS_TAG,
                                             m_PreviousTrainNumberRows, traverser))
        RESTORE(RANDOM_NUMBER_GENERATOR_TAG, m_Rng.fromString(traverser.value()))
        RESTORE(RETRAIN_FRACTION_TAG,
                core::CPersistUtils::restore(RETRAIN_FRACTION_TAG, m_RetrainFraction, traverser))
        RESTORE(ROWS_PER_FEATURE_TAG,
                core::CPersistUtils::restore(ROWS_PER_FEATURE_TAG, m_RowsPerFeature, traverser))
        RESTORE(SEED_TAG, core::CPersistUtils::restore(SEED_TAG, m_Seed, traverser))
        RESTORE(STOP_CROSS_VALIDATION_EARLY_TAG,
                core::CPersistUtils::restore(STOP_CROSS_VALIDATION_EARLY_TAG,
                                             m_StopCrossValidationEarly, traverser))
        RESTORE(TESTING_ROW_MASKS_TAG,
                core::CPersistUtils::restore(TESTING_ROW_MASKS_TAG, m_TestingRowMasks, traverser))
        RESTORE(TRAINING_ROW_MASKS_TAG,
                core::CPersistUtils::restore(TRAINING_ROW_MASKS_TAG, m_TrainingRowMasks, traverser))
        RESTORE(TRAIN_FRACTION_PER_FOLD_TAG,
                core::CPersistUtils::restore(TRAIN_FRACTION_PER_FOLD_TAG,
                                             m_TrainFractionPerFold, traverser))
        RESTORE(TREES_TO_RETRAIN_TAG,
                core::CPersistUtils::restore(TREES_TO_RETRAIN_TAG, m_TreesToRetrain, traverser))
    } while (traverser.next());

    // Extra column information is recreated when training state is restored.

    m_InitializationStage = static_cast<EInitializationStage>(initializationStage);

    this->checkRestoredInvariants();

    return true;
}

void CBoostedTreeImpl::checkRestoredInvariants() const {

    VIOLATES_INVARIANT_NO_EVALUATION(m_Loss, ==, nullptr);
    VIOLATES_INVARIANT_NO_EVALUATION(m_Encoder, ==, nullptr);
    VIOLATES_INVARIANT_NO_EVALUATION(m_Instrumentation, ==, nullptr);
    VIOLATES_INVARIANT(m_FeatureDataTypes.size(), !=,
                       m_FeatureSampleProbabilities.size());
    VIOLATES_INVARIANT(m_DependentVariable, >=, m_MissingFeatureRowMasks.size());
    VIOLATES_INVARIANT(m_TrainingRowMasks.size(), !=, m_TestingRowMasks.size());
    for (std::size_t i = 0; i < m_TrainingRowMasks.size(); ++i) {
        VIOLATES_INVARIANT(m_TrainingRowMasks[i].size(), !=,
                           m_TestingRowMasks[i].size());
    }
    if (m_FoldRoundTestLosses.empty() == false) {
        VIOLATES_INVARIANT(m_FoldRoundTestLosses.size(), !=, m_NumberFolds.value());
        for (const auto& losses : m_FoldRoundTestLosses) {
            VIOLATES_INVARIANT(losses.size(), >, m_Hyperparameters.numberRounds());
        }
    }
    for (auto tree : m_TreesToRetrain) {
        VIOLATES_INVARIANT(tree, >=, m_BestForest.size());
    }
    m_Hyperparameters.checkRestoredInvariants(m_InitializationStage ==
                                              CBoostedTreeImpl::E_FullyInitialized);
}

void CBoostedTreeImpl::checkTrainInvariants(const core::CDataFrame& frame) const {
    if (m_DependentVariable >= frame.numberColumns()) {
        HANDLE_FATAL(<< "Internal error: dependent variable '" << m_DependentVariable
                     << "' was incorrectly initialized. Please report this problem.");
    }
    if (m_Loss == nullptr) {
        HANDLE_FATAL(<< "Internal error: must supply a loss function for training. "
                     << "Please report this problem.");
    }
    if (m_Encoder == nullptr) {
        HANDLE_FATAL(<< "Internal error: must supply an category encoder. "
                     << "Please report this problem.");
    }
    for (const auto& mask : m_MissingFeatureRowMasks) {
        if (mask.size() != frame.numberRows()) {
            HANDLE_FATAL(<< "Internal error: unexpected missing feature mask ("
                         << mask.size() << " !=  " << frame.numberRows()
                         << "). Please report this problem.");
        }
    }
    for (const auto& mask : m_TrainingRowMasks) {
        if (mask.size() != frame.numberRows()) {
            HANDLE_FATAL(<< "Internal error: unexpected train row mask ("
                         << mask.size() << " !=  " << frame.numberRows()
                         << "). Please report this problem.");
        }
    }
    for (const auto& mask : m_TestingRowMasks) {
        if (mask.size() != frame.numberRows()) {
            HANDLE_FATAL(<< "Internal error: unexpected test row mask ("
                         << mask.size() << " !=  " << frame.numberRows()
                         << "). Please report this problem.");
        }
    }
    m_Hyperparameters.checkSearchInvariants();
}

void CBoostedTreeImpl::checkIncrementalTrainInvariants(const core::CDataFrame& frame) const {
    if (m_BestForest.empty()) {
        HANDLE_FATAL(<< "Internal error: no model available to incrementally train."
                     << " Please report this problem.");
    }
    if (m_DependentVariable >= frame.numberColumns()) {
        HANDLE_FATAL(<< "Internal error: dependent variable '" << m_DependentVariable
                     << "' was incorrectly initialized. Please report this problem.");
    }
    if (m_Loss == nullptr) {
        HANDLE_FATAL(<< "Internal error: must supply a loss function for training. "
                     << "Please report this problem.");
    }
    if (m_Encoder == nullptr) {
        HANDLE_FATAL(<< "Internal error: must supply an category encoder. "
                     << "Please report this problem.");
    }
    for (const auto& mask : m_MissingFeatureRowMasks) {
        if (mask.size() != frame.numberRows()) {
            HANDLE_FATAL(<< "Internal error: unexpected missing feature mask ("
                         << mask.size() << " !=  " << frame.numberRows()
                         << "). Please report this problem.");
        }
    }
    for (const auto& mask : m_TrainingRowMasks) {
        if (mask.size() != frame.numberRows()) {
            HANDLE_FATAL(<< "Internal error: unexpected train row mask ("
                         << mask.size() << " !=  " << frame.numberRows()
                         << "). Please report this problem.");
        }
    }
    for (const auto& mask : m_TestingRowMasks) {
        if (mask.size() != frame.numberRows()) {
            HANDLE_FATAL(<< "Internal error: unexpected test row mask ("
                         << mask.size() << " !=  " << frame.numberRows()
                         << "). Please report this problem.");
        }
    }
    m_Hyperparameters.checkSearchInvariants();
}

std::size_t CBoostedTreeImpl::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_Loss)};
    mem += core::CMemory::dynamicSize(m_ExtraColumns);
    mem += core::CMemory::dynamicSize(m_Encoder);
    mem += core::CMemory::dynamicSize(m_FeatureDataTypes);
    mem += core::CMemory::dynamicSize(m_FeatureSampleProbabilities);
    mem += core::CMemory::dynamicSize(m_MissingFeatureRowMasks);
    mem += core::CMemory::dynamicSize(m_FixedCandidateSplits);
    mem += core::CMemory::dynamicSize(m_TrainingRowMasks);
    mem += core::CMemory::dynamicSize(m_TestingRowMasks);
    mem += core::CMemory::dynamicSize(m_NewTrainingRowMask);
    mem += core::CMemory::dynamicSize(m_FoldRoundTestLosses);
    mem += core::CMemory::dynamicSize(m_ClassificationWeightsOverride);
    mem += core::CMemory::dynamicSize(m_ClassificationWeights);
    mem += core::CMemory::dynamicSize(m_BestForest);
    mem += core::CMemory::dynamicSize(m_Hyperparameters);
    mem += core::CMemory::dynamicSize(m_TreeShap);
    mem += core::CMemory::dynamicSize(m_Instrumentation);
    mem += core::CMemory::dynamicSize(m_TreesToRetrain);
    return mem;
}

void CBoostedTreeImpl::accept(CBoostedTree::CVisitor& visitor) {
    m_Encoder->accept(visitor);
    for (const auto& tree : m_BestForest) {
        visitor.addTree();
        for (const auto& node : tree) {
            node.accept(visitor);
        }
    }
    visitor.addClassificationWeights(m_ClassificationWeights.to<TDoubleVec>());
    visitor.addLossFunction(this->loss());
}

const CBoostedTreeHyperparameters& CBoostedTreeImpl::hyperparameters() const {
    return m_Hyperparameters;
}

CBoostedTreeHyperparameters& CBoostedTreeImpl::hyperparameters() {
    return m_Hyperparameters;
}

CTreeShapFeatureImportance* CBoostedTreeImpl::shap() {
    return m_TreeShap.get();
}

core::CPackedBitVector CBoostedTreeImpl::dataSummarization(const core::CDataFrame& frame) const {

    // Note that if we are training on using a holdout set we include the holdout
    // set in the data we consider for summarisation. Typical usage in this case
    // is we're planning to train by query and so the summarisation fraction would
    // be one ensuring we retain the full holdout set.
    //
    // (I considered ensuring that the holdout set is always included in the data
    // summary in entirety, which is consistent with this usage, but in practice
    // it may be useful to be able to incrementally train a model with a cutdown
    // summary after having trained by query. When we come to implement train by
    // query it will be behind a new API which can ensure the data summarization
    // fraction is set appropriately.)

    core::CPackedBitVector allTrainingRowsMask{this->allTrainingRowsMask()};

    if (m_DataSummarizationFraction >= 1.0) {
        return allTrainingRowsMask;
    }

    std::size_t sampleSize{std::max(
        static_cast<std::size_t>(allTrainingRowsMask.manhattan() * m_DataSummarizationFraction),
        static_cast<std::size_t>(2))};
    core::CPackedBitVector rowMask{CDataFrameUtils::stratifiedSamplingRowMasks(
        m_NumberThreads, frame, m_DependentVariable, m_Rng, sampleSize, 10, allTrainingRowsMask)};

    return rowMask;
}

const CBoostedTreeImpl::TDoubleVec& CBoostedTreeImpl::featureSampleProbabilities() const {
    return m_FeatureSampleProbabilities;
}

const CBoostedTreeImpl::TOptionalDoubleVecVec& CBoostedTreeImpl::foldRoundTestLosses() const {
    return m_FoldRoundTestLosses;
}

const CDataFrameCategoryEncoder& CBoostedTreeImpl::encoder() const {
    return *m_Encoder;
}

const CBoostedTreeImpl::TNodeVecVec& CBoostedTreeImpl::trainedModel() const {
    return m_BestForest;
}

CBoostedTreeImpl::TLossFunction& CBoostedTreeImpl::loss() const {
    if (m_Loss == nullptr) {
        HANDLE_FATAL(<< "Internal error: loss function unavailable. "
                     << "Please report this problem.");
    }
    return *m_Loss;
}

std::size_t CBoostedTreeImpl::columnHoldingDependentVariable() const {
    return m_DependentVariable;
}

const core::CPackedBitVector& CBoostedTreeImpl::newTrainingRowMask() const {
    return m_NewTrainingRowMask;
}

const CBoostedTreeImpl::TSizeVec& CBoostedTreeImpl::extraColumns() const {
    return m_ExtraColumns;
}

const CBoostedTreeImpl::TVector& CBoostedTreeImpl::classificationWeights() const {
    return m_ClassificationWeights;
}

core::CPackedBitVector CBoostedTreeImpl::allTrainingRowsMask() const {
    return ~m_MissingFeatureRowMasks[m_DependentVariable];
}

double CBoostedTreeImpl::meanNumberTrainingRowsPerFold() const {
    TMeanAccumulator result;
    for (const auto& mask : m_TrainingRowMasks) {
        result.add(mask.manhattan());
    }
    return common::CBasicStatistics::mean(result);
}

const double CBoostedTreeImpl::MINIMUM_RELATIVE_GAIN_PER_SPLIT{1e-7};
}
}
}
