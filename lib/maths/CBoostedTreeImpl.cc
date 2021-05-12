/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeImpl.h>

#include <core/CContainerPrinter.h>
#include <core/CImmutableRadixSet.h>
#include <core/CLogger.h>
#include <core/CLoopProgress.h>
#include <core/CMemory.h>
#include <core/CPackedBitVector.h>
#include <core/CPersistUtils.h>
#include <core/CProgramCounters.h>
#include <core/CStopWatch.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CBayesianOptimisation.h>
#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CBoostedTreeLeafNodeStatistics.h>
#include <maths/CBoostedTreeLeafNodeStatisticsIncremental.h>
#include <maths/CBoostedTreeLeafNodeStatisticsScratch.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/CDataFrameAnalysisInstrumentationInterface.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CQuantileSketch.h>
#include <maths/CSampling.h>
#include <maths/CSetTools.h>
#include <maths/CSpline.h>
#include <maths/CTreeShapFeatureImportance.h>

#include <boost/circular_buffer.hpp>

#include <algorithm>
#include <limits>
#include <memory>

namespace ml {
namespace maths {
using namespace boosted_tree;
using namespace boosted_tree_detail;
using TStrVec = CBoostedTreeImpl::TStrVec;
using TRowItr = core::CDataFrame::TRowItr;
using TMeanVarAccumulator = CBoostedTreeImpl::TMeanVarAccumulator;
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

double lossAtNSigma(double n, const TMeanVarAccumulator& lossMoments) {
    return CBasicStatistics::mean(lossMoments) +
           n * std::sqrt(CBasicStatistics::variance(lossMoments));
}

double trace(std::size_t columns, const TMemoryMappedFloatVector& upperTriangle) {
    // This assumes the upper triangle of the matrix is stored row major.
    double result{0.0};
    for (int i = 0, j = static_cast<int>(columns);
         i < upperTriangle.size() && j > 0; i += j, --j) {
        result += upperTriangle(i);
    }
    return result;
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
      m_BestHyperparameters(m_Regularization,
                            m_DownsampleFactor,
                            m_Eta,
                            m_EtaGrowthRatePerTree,
                            m_MaximumNumberTrees,
                            m_FeatureBagFraction,
                            m_PredictionChangeCost),
      m_Instrumentation{instrumentation != nullptr ? instrumentation : &INSTRUMENTATION_STUB} {
}

CBoostedTreeImpl::CBoostedTreeImpl() = default;
CBoostedTreeImpl::~CBoostedTreeImpl() = default;
CBoostedTreeImpl::CBoostedTreeImpl(CBoostedTreeImpl&&) noexcept = default;
CBoostedTreeImpl& CBoostedTreeImpl::operator=(CBoostedTreeImpl&&) noexcept = default;

void CBoostedTreeImpl::train(core::CDataFrame& frame,
                             const TTrainingStateCallback& recordTrainStateCallback) {

    this->checkTrainInvariants(frame);

    if (m_Loss->isRegression()) {
        m_Instrumentation->type(CDataFrameTrainBoostedTreeInstrumentationInterface::E_Regression);
    } else {
        m_Instrumentation->type(CDataFrameTrainBoostedTreeInstrumentationInterface::E_Classification);
    }

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
        m_BestForestTestLoss = this->meanLoss(frame, allTrainingRowsMask);
        LOG_TRACE(<< "Test loss = " << m_BestForestTestLoss);

    } else if (m_CurrentRound < m_NumberRounds || m_BestForest.empty()) {
        TMeanVarAccumulator timeAccumulator;
        core::CStopWatch stopWatch;
        stopWatch.start();
        std::uint64_t lastLap{stopWatch.lap()};

        // Hyperparameter optimisation loop.

        this->initializePerFoldTestLosses();
        this->initializeHyperparameterSamples();

        while (m_CurrentRound < m_NumberRounds) {

            LOG_TRACE(<< "Optimisation round = " << m_CurrentRound + 1);
            m_Instrumentation->iteration(m_CurrentRound + 1);

            this->recordHyperparameters();

            TMeanVarAccumulator lossMoments;
            std::size_t maximumNumberTrees;
            double numberNodes;
            std::tie(lossMoments, maximumNumberTrees, numberNodes) = this->crossValidateForest(
                frame, m_MaximumNumberTrees,
                [this](core::CDataFrame& frame_, const core::CPackedBitVector& trainingRowMask,
                       const core::CPackedBitVector& testingRowMask,
                       core::CLoopProgress& trainingProgress) {
                    return this->trainForest(frame_, trainingRowMask,
                                             testingRowMask, trainingProgress);
                });

            this->captureBestHyperparameters(lossMoments, maximumNumberTrees,
                                             0.0 /*no kept nodes*/, numberNodes);

            if (this->selectNextHyperparameters(lossMoments, *m_BayesianOptimization) == false) {
                LOG_INFO(<< "Exiting hyperparameter optimisation loop early");
                break;
            }

            std::int64_t memoryUsage(this->memoryUsage());
            m_Instrumentation->updateMemoryUsage(memoryUsage - lastMemoryUsage);
            lastMemoryUsage = memoryUsage;

            // Store the training state after each hyperparameter search step.
            m_CurrentRound += 1;
            LOG_TRACE(<< "Round " << m_CurrentRound << " state recording started");
            this->recordState(recordTrainStateCallback);
            LOG_TRACE(<< "Round " << m_CurrentRound << " state recording finished");

            std::uint64_t currentLap{stopWatch.lap()};
            std::uint64_t delta{currentLap - lastLap};
            m_Instrumentation->iterationTime(delta);

            timeAccumulator.add(static_cast<double>(delta));
            lastLap = currentLap;
            m_Instrumentation->flush(HYPERPARAMETER_OPTIMIZATION_ROUND +
                                     std::to_string(m_CurrentRound));
        }

        LOG_TRACE(<< "Test loss = " << m_BestForestTestLoss);

        this->restoreBestHyperparameters();
        this->scaleRegularizers(allTrainingRowsMask.manhattan() /
                                m_TrainingRowMasks[0].manhattan());
        this->startProgressMonitoringFinalTrain();
        // reinitialize random number generator for reproducible results
        // TODO #1866 introduce accept randomize_seed configuration parameter
        m_Rng = CPRNG::CXorOShiro128Plus{};
        if (m_BestForest.empty()) {
            std::tie(m_BestForest, std::ignore, std::ignore) = this->trainForest(
                frame, allTrainingRowsMask, allTrainingRowsMask, m_TrainingProgress);

            this->recordState(recordTrainStateCallback);
        }
        m_Instrumentation->iteration(m_CurrentRound);
        m_Instrumentation->flush(TRAIN_FINAL_FOREST);

        timeAccumulator.add(static_cast<double>(stopWatch.stop() - lastLap));

        LOG_TRACE(<< "Training finished after " << m_CurrentRound << " iterations. "
                  << "Time per iteration in ms mean: "
                  << CBasicStatistics::mean(timeAccumulator) << " std. dev:  "
                  << std::sqrt(CBasicStatistics::variance(timeAccumulator)));

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

    this->selectTreesToRetrain(frame);

    std::int64_t lastMemoryUsage(this->memoryUsage());

    this->startProgressMonitoringTrainIncremental();

    double initialLoss{lossAtNSigma(1.0, [&] {
        TMeanVarAccumulator lossMoments;
        for (const auto& mask : m_TestingRowMasks) {
            lossMoments.add(this->meanAdjustedLoss(frame, mask));
        }
        return lossMoments;
    }())};

    double retrainedNumberNodes{0.0};
    for (const auto& i : m_TreesToRetrain) {
        retrainedNumberNodes += static_cast<double>(m_BestForest[i].size());
    }
    double numberKeptNodes{numberForestNodes(m_BestForest) - retrainedNumberNodes};

    // Hyperparameter optimisation loop.

    this->initializePerFoldTestLosses();
    this->initializeHyperparameterSamples();

    std::size_t maximumNumberTrees{this->numberTreesToRetrain()};
    TMeanVarAccumulator timeAccumulator;
    core::CStopWatch stopWatch;
    stopWatch.start();
    std::uint64_t lastLap{stopWatch.lap()};

    while (m_CurrentRound < m_NumberRounds) {

        LOG_TRACE(<< "Optimisation round = " << m_CurrentRound + 1);
        m_Instrumentation->iteration(m_CurrentRound + 1);

        this->recordHyperparameters();

        TMeanVarAccumulator lossMoments;
        double numberRetrainedNodes;
        std::tie(lossMoments, std::ignore, numberRetrainedNodes) = this->crossValidateForest(
            frame, maximumNumberTrees,
            [this](core::CDataFrame& frame_, const core::CPackedBitVector& trainingRowMask,
                   const core::CPackedBitVector& testingRowMask,
                   core::CLoopProgress& trainingProgress) {
                return this->updateForest(frame_, trainingRowMask,
                                          testingRowMask, trainingProgress);
            });

        this->captureBestHyperparameters(lossMoments, m_BestForest.size(),
                                         numberKeptNodes, numberRetrainedNodes);

        if (this->selectNextHyperparameters(lossMoments, *m_BayesianOptimization) == false) {
            LOG_INFO(<< "Exiting hyperparameter optimisation loop early");
            break;
        }

        std::int64_t memoryUsage(this->memoryUsage());
        m_Instrumentation->updateMemoryUsage(memoryUsage - lastMemoryUsage);
        lastMemoryUsage = memoryUsage;

        m_CurrentRound += 1;
        LOG_TRACE(<< "Round " << m_CurrentRound << " state recording started");
        this->recordState(recordTrainStateCallback);
        LOG_TRACE(<< "Round " << m_CurrentRound << " state recording finished");

        std::uint64_t currentLap{stopWatch.lap()};
        std::uint64_t delta{currentLap - lastLap};
        m_Instrumentation->iterationTime(delta);

        timeAccumulator.add(static_cast<double>(delta));
        lastLap = currentLap;
        m_Instrumentation->flush(HYPERPARAMETER_OPTIMIZATION_ROUND +
                                 std::to_string(m_CurrentRound));
    }

    LOG_TRACE(<< "Training finished after " << m_CurrentRound << " iterations. "
              << "Time per iteration in ms mean: " << CBasicStatistics::mean(timeAccumulator)
              << " std. dev:  " << std::sqrt(CBasicStatistics::variance(timeAccumulator)));

    if (m_BestForestTestLoss <
        initialLoss + this->modelSizePenalty(numberKeptNodes, retrainedNumberNodes)) {
        this->restoreBestHyperparameters();
        core::CPackedBitVector allTrainingRowsMask{this->allTrainingRowsMask()};
        this->updateForest(frame, allTrainingRowsMask, allTrainingRowsMask, m_TrainingProgress);
    }

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
    std::size_t maximumNumberLeaves{this->maximumTreeSize(numberRows) + 1};
    std::size_t maximumNumberNodes{2 * maximumNumberLeaves - 1};
    std::size_t maximumNumberFeatures{std::min(numberColumns - 1, numberRows / m_RowsPerFeature)};
    std::size_t forestMemoryUsage{
        m_MaximumNumberTrees *
        (sizeof(TNodeVec) + maximumNumberNodes * CBoostedTreeNode::estimateMemoryUsage(
                                                     m_Loss->numberParameters()))};
    std::size_t foldRoundLossMemoryUsage{m_NumberFolds * m_NumberRounds *
                                         sizeof(TOptionalDouble)};
    std::size_t hyperparametersMemoryUsage{numberColumns * sizeof(double)};
    std::size_t tunableHyperparametersMemoryUsage{
        this->numberHyperparametersToTune() * sizeof(int)};
    std::size_t hyperparameterSamplesMemoryUsage{
        (m_NumberRounds / 3 + 1) * this->numberHyperparametersToTune() * sizeof(double)};
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
    std::size_t featureSampleProbabilities{maximumNumberFeatures * sizeof(double)};
    // Assuming either many or few missing rows, we get good compression of the bit
    // vector. Specifically, we'll assume the average run length is 64 for which
    // we get a constant 8 / 64.
    std::size_t missingFeatureMaskMemoryUsage{8 * numberColumns * numberRows / 64};
    std::size_t trainTestMaskMemoryUsage{
        2 * static_cast<std::size_t>(std::ceil(std::log2(static_cast<double>(m_NumberFolds)))) *
        numberRows};
    std::size_t bayesianOptimisationMemoryUsage{CBayesianOptimisation::estimateMemoryUsage(
        this->numberHyperparametersToTune(), m_NumberRounds)};
    std::size_t worstCaseMemoryUsage{
        sizeof(*this) + forestMemoryUsage + foldRoundLossMemoryUsage +
        hyperparametersMemoryUsage + tunableHyperparametersMemoryUsage +
        hyperparameterSamplesMemoryUsage + leafNodeStatisticsMemoryUsage +
        dataTypeMemoryUsage + featureSampleProbabilities + missingFeatureMaskMemoryUsage +
        trainTestMaskMemoryUsage + bayesianOptimisationMemoryUsage};

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
    maths::CSpline<> spline(maths::CSplineTypes::E_Linear);
    spline.interpolate(estimatedMemoryUsageMB, correctedMemoryUsageMB,
                       maths::CSplineTypes::E_ParabolicRunout);
    return static_cast<std::size_t>(spline.value(memoryUsageBytes / BYTES_IN_MB) * BYTES_IN_MB);
}

bool CBoostedTreeImpl::canTrain() const {
    return std::accumulate(m_FeatureSampleProbabilities.begin(),
                           m_FeatureSampleProbabilities.end(), 0.0) > 0.0;
}

core::CPackedBitVector CBoostedTreeImpl::allTrainingRowsMask() const {
    return ~m_MissingFeatureRowMasks[m_DependentVariable];
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
    m_FoldRoundTestLosses.resize(m_NumberFolds);
    for (auto& losses : m_FoldRoundTestLosses) {
        losses.resize(m_NumberRounds);
    }
}

void CBoostedTreeImpl::computeClassificationWeights(const core::CDataFrame& frame) {

    using TFloatStorageVec = std::vector<CFloatStorage>;

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
    CSampling::categoricalSampleWithoutReplacement(
        m_Rng, probabilities, numberToRetrain, m_TreesToRetrain);
}

template<typename F>
CBoostedTreeImpl::TMeanVarAccumulatorSizeDoubleTuple
CBoostedTreeImpl::crossValidateForest(core::CDataFrame& frame,
                                      std::size_t maximumNumberTrees,
                                      const F& trainForest) {

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
    TMeanAccumulator meanForestSizeAccumulator;

    while (folds.size() > 0 && stopCrossValidationEarly(lossMoments) == false) {
        std::size_t fold{folds.back()};
        folds.pop_back();
        TNodeVecVec forest;
        double loss;
        TDoubleVec lossValues;
        std::tie(forest, loss, lossValues) = trainForest(
            frame, m_TrainingRowMasks[fold], m_TestingRowMasks[fold], m_TrainingProgress);
        LOG_TRACE(<< "fold = " << fold << " forest size = " << forest.size()
                  << " test set loss = " << loss);
        lossMoments.add(loss);
        m_FoldRoundTestLosses[fold][m_CurrentRound] = loss;
        numberTrees.push_back(static_cast<double>(forest.size()));
        meanForestSizeAccumulator.add(numberForestNodes(forest));
        m_Instrumentation->lossValues(fold, std::move(lossValues));
    }
    m_TrainingProgress.increment(maximumNumberTrees * folds.size());
    LOG_TRACE(<< "skipped " << folds.size() << " folds");

    std::sort(numberTrees.begin(), numberTrees.end());
    std::size_t medianNumberTrees{
        static_cast<std::size_t>(CBasicStatistics::median(numberTrees))};
    double meanForestSize{CBasicStatistics::mean(meanForestSizeAccumulator)};
    lossMoments = this->correctTestLossMoments(folds, lossMoments);
    LOG_TRACE(<< "test mean loss = " << CBasicStatistics::mean(lossMoments)
              << ", sigma = " << std::sqrt(CBasicStatistics::mean(lossMoments))
              << ", mean number nodes in forest = " << meanForestSize);

    m_MeanForestSizeAccumulator += meanForestSizeAccumulator;
    m_MeanLossAccumulator.add(CBasicStatistics::mean(lossMoments));

    return {lossMoments, medianNumberTrees, meanForestSize};
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
                if (m_IncrementalTraining == false) {
                    zeroPrediction(row, m_ExtraColumns, numberLossParameters);
                } else {
                    readPrediction(row, m_ExtraColumns, numberLossParameters) =
                        readPreviousPrediction(row, m_ExtraColumns, numberLossParameters);
                }
                zeroLossGradient(row, m_ExtraColumns, numberLossParameters);
                zeroLossCurvature(row, m_ExtraColumns, numberLossParameters);
            }
        },
        &updateRowMask);

    TNodeVec tree;
    if (m_IncrementalTraining == false) {
        // At the start we will centre the data w.r.t. the given loss function.
        tree.assign({CBoostedTreeNode{m_Loss->numberParameters()}});
        this->refreshPredictionsAndLossDerivatives(frame, trainingRowMask,
                                                   testingRowMask, *m_Loss,
                                                   1.0 /*eta*/, 0.0 /*lambda*/, tree);
    }

    return tree;
}

CBoostedTreeImpl::TNodeVecVecDoubleDoubleVecTr
CBoostedTreeImpl::trainForest(core::CDataFrame& frame,
                              const core::CPackedBitVector& trainingRowMask,
                              const core::CPackedBitVector& testingRowMask,
                              core::CLoopProgress& trainingProgress) const {

    LOG_TRACE(<< "Training one forest...");

    auto makeRootLeafNodeStatistics =
        [&](const TImmutableRadixSetVec& candidateSplits,
            const TSizeVec& treeFeatureBag, const TSizeVec& nodeFeatureBag,
            const core::CPackedBitVector& trainingRowMask_, TWorkspace& workspace) {
            return std::make_shared<CBoostedTreeLeafNodeStatisticsScratch>(
                rootIndex(), m_ExtraColumns, m_Loss->numberParameters(), m_NumberThreads,
                frame, *m_Encoder, m_Regularization, candidateSplits, treeFeatureBag,
                nodeFeatureBag, 0 /*depth*/, trainingRowMask_, workspace);
        };

    std::size_t maximumNumberInternalNodes{maximumTreeSize(trainingRowMask)};

    TNodeVecVec forest{this->initializePredictionsAndLossDerivatives(
        frame, trainingRowMask, testingRowMask)};
    forest.reserve(m_MaximumNumberTrees);

    CScopeRecordMemoryUsage scopeMemoryUsage{forest, m_Instrumentation->memoryUsageCallback()};

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
    scopeMemoryUsage.add(downsampledRowMask);
    auto candidateSplits = this->candidateSplits(frame, downsampledRowMask);
    scopeMemoryUsage.add(candidateSplits);

    std::size_t retries{0};

    TDoubleVec losses;
    losses.reserve(m_MaximumNumberTrees);
    scopeMemoryUsage.add(losses);

    CTrainForestStoppingCondition stoppingCondition{m_MaximumNumberTrees};
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

        if (retries == m_MaximumAttemptsToAddTree) {
            break;
        }

        if (tree.size() > 1) {
            scopeMemoryUsage.add(tree);
            this->refreshPredictionsAndLossDerivatives(
                frame, trainingRowMask, testingRowMask, *m_Loss, eta,
                m_Regularization.leafWeightPenaltyMultiplier(), tree);
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
        double loss{this->meanLoss(frame, testingRowMask)};
        losses.push_back(loss);
        return loss;
    }) == false);

    LOG_TRACE(<< "Stopped at " << forest.size() - 1 << "/" << m_MaximumNumberTrees);

    trainingProgress.increment(std::max(m_MaximumNumberTrees, forest.size()) -
                               forest.size());

    forest.resize(stoppingCondition.bestSize());

    LOG_TRACE(<< "Trained one forest");

    return {forest, stoppingCondition.bestLoss(), std::move(losses)};
}

CBoostedTreeImpl::TNodeVecVecDoubleDoubleVecTr
CBoostedTreeImpl::updateForest(core::CDataFrame& frame,
                               const core::CPackedBitVector& trainingRowMask,
                               const core::CPackedBitVector& testingRowMask,
                               core::CLoopProgress& trainingProgress) const {

    LOG_TRACE(<< "Incrementally training one forest...");

    auto makeRootLeafNodeStatistics =
        [&](const TImmutableRadixSetVec& candidateSplits,
            const TSizeVec& treeFeatureBag, const TSizeVec& nodeFeatureBag,
            const core::CPackedBitVector& trainingRowMask_, TWorkspace& workspace) {
            return std::make_shared<CBoostedTreeLeafNodeStatisticsIncremental>(
                rootIndex(), m_ExtraColumns, m_Loss->numberParameters(), m_NumberThreads,
                frame, *m_Encoder, m_Regularization, candidateSplits, treeFeatureBag,
                nodeFeatureBag, 0 /*depth*/, trainingRowMask_, workspace);
        };

    std::size_t maximumNumberInternalNodes{maximumTreeSize(trainingRowMask)};

    TNodeVecVec retrainedTrees;
    retrainedTrees.reserve(m_TreesToRetrain.size());
    this->initializePredictionsAndLossDerivatives(frame, trainingRowMask, testingRowMask);

    CScopeRecordMemoryUsage scopeMemoryUsage{
        retrainedTrees, m_Instrumentation->memoryUsageCallback()};

    core::CPackedBitVector oldTrainingRowMask{trainingRowMask & ~m_NewTrainingRowMask};
    core::CPackedBitVector newTrainingRowMask{trainingRowMask & m_NewTrainingRowMask};
    auto downsampledRowMask = this->downsample(oldTrainingRowMask) |
                              this->downsample(newTrainingRowMask);
    scopeMemoryUsage.add(downsampledRowMask);
    auto candidateSplits = this->candidateSplits(frame, trainingRowMask);
    scopeMemoryUsage.add(candidateSplits);

    TDoubleVec losses;
    losses.reserve(m_TreesToRetrain.size());
    scopeMemoryUsage.add(losses);

    TWorkspace workspace{m_Loss->numberParameters()};

    // For each iteration:
    //  1. Rebuild one tree on fixed upfront candidate splits of features.
    //  2. Update predictions and loss derivatives.

    for (const auto& index : m_TreesToRetrain) {

        this->removePredictions(frame, trainingRowMask, testingRowMask, m_BestForest[index]);

        auto tree = this->trainTree(frame, downsampledRowMask, candidateSplits,
                                    maximumNumberInternalNodes,
                                    makeRootLeafNodeStatistics, workspace);

        scopeMemoryUsage.add(tree);
        double eta{this->etaForTreeAtPosition(index)};
        auto loss = m_Loss->incremental(eta, m_PredictionChangeCost, m_BestForest[index]);
        this->refreshPredictionsAndLossDerivatives(
            frame, trainingRowMask, testingRowMask, *loss, eta,
            m_Regularization.leafWeightPenaltyMultiplier(), tree);
        retrainedTrees.push_back(std::move(tree));
        trainingProgress.increment();

        downsampledRowMask = this->downsample(oldTrainingRowMask) |
                             this->downsample(newTrainingRowMask);

        losses.push_back(this->meanAdjustedLoss(frame, testingRowMask));
    }

    auto bestLoss = static_cast<std::size_t>(
        std::min_element(losses.begin(), losses.end()) - losses.begin());
    retrainedTrees.resize(bestLoss + 1);

    return {std::move(retrainedTrees), losses[bestLoss], std::move(losses)};
}

double CBoostedTreeImpl::etaForTreeAtPosition(std::size_t index) const {
    return std::min(m_Eta + CTools::stable(std::pow(m_EtaGrowthRatePerTree,
                                                    static_cast<double>(index))),
                    1.0);
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

    TSizeVec features;
    candidateRegressorFeatures(m_FeatureSampleProbabilities, features);
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
            m_Encoder.get(),
            [this](const TRowRef& row) {
                std::size_t numberLossParameters{m_Loss->numberParameters()};
                return trace(numberLossParameters,
                             readLossCurvature(row, m_ExtraColumns, numberLossParameters));
            })
            .first;

    TImmutableRadixSetVec candidateSplits(this->numberFeatures());

    for (auto i : binaryFeatures) {
        candidateSplits[i] = core::CImmutableRadixSet<double>{0.5};
        LOG_TRACE(<< "feature '" << i << "' splits = " << candidateSplits[i].print());
    }
    for (std::size_t i = 0; i < features.size(); ++i) {

        TDoubleVec featureSplits;

        // Because we compute candidate splits for downsamples of the rows it's
        // possible that all values are missing for a particular feature. In this
        // case, we can happily initialize the candidate splits to an empty set
        // since we'll only be choosing how to assign missing values.
        if (featureQuantiles[i].count() > 0.0) {
            featureSplits.reserve(m_NumberSplitsPerFeature - 1);
            for (std::size_t j = 1; j < m_NumberSplitsPerFeature; ++j) {
                double rank{100.0 * static_cast<double>(j) /
                                static_cast<double>(m_NumberSplitsPerFeature) +
                            CSampling::uniformSample(m_Rng, -0.1, 0.1)};
                double q;
                if (featureQuantiles[i].quantile(rank, q)) {
                    featureSplits.push_back(q);
                } else {
                    LOG_WARN(<< "Failed to compute quantile " << rank << ": ignoring split");
                }
            }
        }

        const auto& dataType = m_FeatureDataTypes[features[i]];

        if (dataType.s_IsInteger) {
            // The key point here is that we know that if two distinct splits fall
            // between two consecutive integers they must produce identical partitions
            // of the data and so always have the same loss. We only need to retain
            // one such split for training. We achieve this by snapping to the midpoint
            // and subsequently deduplicating.
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
                            std::size_t maximumNumberInternalNodes,
                            const TMakeRootLeafNodeStatistics& makeRootLeafNodeStatistics,
                            TWorkspace& workspace) const {

    LOG_TRACE(<< "Training one tree...");

    using TLeafNodeStatisticsPtrQueue = boost::circular_buffer<TLeafNodeStatisticsPtr>;

    workspace.reinitialize(m_NumberThreads, candidateSplits);

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

    featureSampleProbabilities = m_FeatureSampleProbabilities;
    this->nodeFeatureBag(treeFeatureBag, featureSampleProbabilities, nodeFeatureBag);

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

    COrderings::SLess less;

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

        // add the left and right children to the tree
        std::size_t leftChildId;
        std::size_t rightChildId;
        std::tie(leftChildId, rightChildId) =
            tree[leaf->id()].split(splitFeature, splitValue, assignMissingToLeft,
                                   leaf->gain(), leaf->curvature(), tree);

        featureSampleProbabilities = m_FeatureSampleProbabilities;
        this->nodeFeatureBag(treeFeatureBag, featureSampleProbabilities, nodeFeatureBag);

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
            leftChildId, rightChildId, m_NumberThreads, smallestCandidateGain,
            frame, *m_Encoder, m_Regularization, treeFeatureBag, nodeFeatureBag,
            tree[leaf->id()], workspace);

        // Need gain to be computed to compare here
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

    TSizeVec present(m_NumberFolds);
    std::iota(present.begin(), present.end(), 0);
    TSizeVec ordered{missing};
    std::sort(ordered.begin(), ordered.end());
    CSetTools::inplace_set_difference(present, ordered.begin(), ordered.end());
    LOG_TRACE(<< "present = " << core::CContainerPrinter::print(present));

    // Get the current round feature vector. Fixed so computed outside the loop.
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
        LOG_TRACE(<< "prediction(x = " << x.transpose() << ", fold = " << target
                  << ") = (mean = " << predictedTestLoss
                  << ", variance = " << predictedTestLossVariance << ")");

        predictedTestLosses.push_back(CBasicStatistics::momentsAccumulator(
            1.0, predictedTestLoss, predictedTestLossVariance));
    }

    return predictedTestLosses;
}

std::size_t CBoostedTreeImpl::numberFeatures() const {
    return m_Encoder->numberEncodedColumns();
}

std::size_t CBoostedTreeImpl::featureBagSize(double fraction) const {
    return static_cast<std::size_t>(std::max(
        std::ceil(std::min(fraction, 1.0) * static_cast<double>(this->numberFeatures())), 1.0));
}

void CBoostedTreeImpl::treeFeatureBag(TDoubleVec& probabilities, TSizeVec& treeFeatureBag) const {

    std::size_t size{this->featureBagSize(1.25 * m_FeatureBagFraction)};

    candidateRegressorFeatures(probabilities, treeFeatureBag);
    if (size >= treeFeatureBag.size()) {
        return;
    }

    CSampling::categoricalSampleWithoutReplacement(m_Rng, probabilities, size, treeFeatureBag);
    std::sort(treeFeatureBag.begin(), treeFeatureBag.end());
}

void CBoostedTreeImpl::nodeFeatureBag(const TSizeVec& treeFeatureBag,
                                      TDoubleVec& probabilities,
                                      TSizeVec& nodeFeatureBag) const {

    std::size_t size{this->featureBagSize(m_FeatureBagFraction)};

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

    CSampling::categoricalSampleWithoutReplacement(m_Rng, probabilities, size, nodeFeatureBag);
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

void CBoostedTreeImpl::removePredictions(core::CDataFrame& frame,
                                         const core::CPackedBitVector& trainingRowMask,
                                         const core::CPackedBitVector& testingRowMask,
                                         const TNodeVec& tree) const {

    core::CPackedBitVector rowMask{trainingRowMask | testingRowMask};

    frame.writeColumns(m_NumberThreads, 0, frame.numberRows(),
                       [&](const TRowItr& beginRows, const TRowItr& endRows) {
                           std::size_t numberLossParameters{m_Loss->numberParameters()};
                           const auto& rootNode = root(tree);
                           for (auto row_ = beginRows; row_ != endRows; ++row_) {
                               auto row = *row_;
                               readPrediction(row, m_ExtraColumns, numberLossParameters) -=
                                   rootNode.value(m_Encoder->encode(row), tree);
                           }
                       },
                       &rowMask);
}

void CBoostedTreeImpl::refreshPredictionsAndLossDerivatives(
    core::CDataFrame& frame,
    const core::CPackedBitVector& trainingRowMask,
    const core::CPackedBitVector& testingRowMask,
    const TLossFunction& loss,
    double eta,
    double lambda,
    TNodeVec& tree) const {

    TArgMinLossVec leafValues(tree.size(), loss.minimizer(lambda, m_Rng));
    auto nextPass = [&] {
        bool done{true};
        for (const auto& value : leafValues) {
            done &= (value.nextPass() == false);
        }
        return done == false;
    };

    do {
        TArgMinLossVecVec result(m_NumberThreads, leafValues);
        if (m_IncrementalTraining) {
            this->minimumLossLeafValues(false /*new example*/, frame,
                                        trainingRowMask & ~m_NewTrainingRowMask,
                                        loss, tree, result);
            this->minimumLossLeafValues(true /*new example*/, frame,
                                        trainingRowMask & m_NewTrainingRowMask,
                                        loss, tree, result);
        } else {
            this->minimumLossLeafValues(false /*new example*/, frame,
                                        trainingRowMask, loss, tree, result);
        }

        leafValues = std::move(result[0]);
        for (std::size_t i = 1; i < result.size(); ++i) {
            for (std::size_t j = 0; j < leafValues.size(); ++j) {
                leafValues[j].merge(result[i][j]);
            }
        }
    } while (nextPass());

    for (std::size_t i = 0; i < tree.size(); ++i) {
        if (tree[i].isLeaf()) {
            tree[i].value(eta * leafValues[i].value());
        }
    }

    LOG_TRACE(<< "tree =\n" << root(tree).print(tree));

    core::CPackedBitVector updateRowMask{trainingRowMask | testingRowMask};

    if (m_IncrementalTraining) {
        this->writeRowDerivatives(false /*new example*/, frame,
                                  updateRowMask & ~m_NewTrainingRowMask, loss, tree);
        this->writeRowDerivatives(true /*new example*/, frame,
                                  updateRowMask & m_NewTrainingRowMask, loss, tree);
    } else {
        this->writeRowDerivatives(false /*new example*/, frame, updateRowMask, loss, tree);
    }
}

void CBoostedTreeImpl::minimumLossLeafValues(bool newExample,
                                             const core::CDataFrame& frame,
                                             const core::CPackedBitVector& rowMask,
                                             const TLossFunction& loss,
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
                leafValues[rootNode.leafIndex(encodedRow, tree)].add(
                    encodedRow, newExample, prediction, actual, weight);
            }
        });
    }

    frame.readRows(0, frame.numberRows(), minimizers, &rowMask);
}

void CBoostedTreeImpl::writeRowDerivatives(bool newExample,
                                           core::CDataFrame& frame,
                                           const core::CPackedBitVector& rowMask,
                                           const TLossFunction& loss,
                                           const TNodeVec& tree) const {
    frame.writeColumns(
        m_NumberThreads, 0, frame.numberRows(),
        [&](const TRowItr& beginRows, const TRowItr& endRows) {
            std::size_t numberLossParameters{loss.numberParameters()};
            const auto& rootNode = root(tree);
            for (auto row_ = beginRows; row_ != endRows; ++row_) {
                auto row = *row_;
                auto prediction = readPrediction(row, m_ExtraColumns, numberLossParameters);
                double actual{readActual(row, m_DependentVariable)};
                double weight{readExampleWeight(row, m_ExtraColumns)};
                prediction += rootNode.value(m_Encoder->encode(row), tree);
                writeLossGradient(row, newExample, m_ExtraColumns, *m_Encoder,
                                  loss, prediction, actual, weight);
                writeLossCurvature(row, newExample, m_ExtraColumns, *m_Encoder,
                                   loss, prediction, actual, weight);
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
                    loss.add(m_Loss->value(prediction, actual, weight));
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

double CBoostedTreeImpl::meanAdjustedLoss(const core::CDataFrame& frame,
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
                    double weight{0.01 * readExampleWeight(row, m_ExtraColumns)};
                    loss.add(m_Loss->difference(prediction, previousPrediction, weight));
                }
            },
            TMeanAccumulator{}),
        &oldRowMask);

    TMeanAccumulator loss;
    for (const auto& result : results.first) {
        loss += result.s_FunctionState;
    }

    return this->meanLoss(frame, rowMask) + CBasicStatistics::mean(loss);
}

CBoostedTreeImpl::TVector CBoostedTreeImpl::predictRow(const CEncodedDataFrameRowRef& row) const {
    TVector result{TVector::Zero(m_Loss->numberParameters())};
    for (const auto& tree : m_BestForest) {
        result += root(tree).value(row, tree);
    }
    return result;
}

bool CBoostedTreeImpl::selectNextHyperparameters(const TMeanVarAccumulator& lossMoments,
                                                 CBayesianOptimisation& bopt) {

    TVector parameters{m_TunableHyperparameters.size()};

    TVector minBoundary;
    TVector maxBoundary;
    std::tie(minBoundary, maxBoundary) = bopt.boundingBox();

    // Downsampling directly affects the loss terms: it multiplies the sums over
    // gradients and Hessians in expectation by the downsample factor. To preserve
    // the same effect for regularisers we need to scale these terms by the same
    // multiplier.
    double scale{1.0};
    if (m_DownsampleFactorOverride == boost::none) {
        auto i = std::distance(m_TunableHyperparameters.begin(),
                               std::find(m_TunableHyperparameters.begin(),
                                         m_TunableHyperparameters.end(), E_DownsampleFactor));
        if (static_cast<std::size_t>(i) < m_TunableHyperparameters.size()) {
            scale = std::min(1.0, 2.0 * m_DownsampleFactor /
                                      (CTools::stableExp(minBoundary(i)) +
                                       CTools::stableExp(maxBoundary(i))));
        }
    }

    // Read parameters for last round.
    for (std::size_t i = 0; i < m_TunableHyperparameters.size(); ++i) {
        switch (m_TunableHyperparameters[i]) {
        case E_Alpha:
            parameters(i) =
                CTools::stableLog(m_Regularization.depthPenaltyMultiplier() / scale);
            break;
        case E_DownsampleFactor:
            parameters(i) = CTools::stableLog(m_DownsampleFactor);
            break;
        case E_Eta:
            parameters(i) = CTools::stableLog(m_Eta);
            break;
        case E_EtaGrowthRatePerTree:
            parameters(i) = m_EtaGrowthRatePerTree;
            break;
        case E_FeatureBagFraction:
            parameters(i) = m_FeatureBagFraction;
            break;
        case E_MaximumNumberTrees:
            parameters(i) = static_cast<double>(m_MaximumNumberTrees);
            break;
        case E_Gamma:
            parameters(i) = CTools::stableLog(
                m_Regularization.treeSizePenaltyMultiplier() / scale);
            break;
        case E_Lambda:
            parameters(i) = CTools::stableLog(
                m_Regularization.leafWeightPenaltyMultiplier() / scale);
            break;
        case E_SoftTreeDepthLimit:
            parameters(i) = m_Regularization.softTreeDepthLimit();
            break;
        case E_SoftTreeDepthTolerance:
            parameters(i) = m_Regularization.softTreeDepthTolerance();
            break;
        case E_PredictionChangeCost:
            parameters(i) = CTools::stableLog(m_PredictionChangeCost);
            break;
        case E_TreeTopologyChangePenalty:
            parameters(i) = CTools::stableLog(m_Regularization.treeTopologyChangePenalty());
            break;
        }
    }

    double meanLoss{CBasicStatistics::mean(lossMoments)};
    double lossVariance{CBasicStatistics::variance(lossMoments)};

    LOG_TRACE(<< "round = " << m_CurrentRound << " loss = " << meanLoss << " variance = "
              << lossVariance << ": regularization = " << m_Regularization.print()
              << ", downsample factor = " << m_DownsampleFactor << ", eta = " << m_Eta
              << ", eta growth rate per tree = " << m_EtaGrowthRatePerTree
              << ", feature bag fraction = " << m_FeatureBagFraction);

    bopt.add(parameters, meanLoss, lossVariance);
    if (m_CurrentRound < m_HyperparameterSamples.size()) {
        std::copy(m_HyperparameterSamples[m_CurrentRound].begin(),
                  m_HyperparameterSamples[m_CurrentRound].end(), parameters.data());
        parameters = minBoundary + parameters.cwiseProduct(maxBoundary - minBoundary);
    } else {
        if (m_StopHyperparameterOptimizationEarly &&
            m_BayesianOptimization->anovaTotalVariance() < 1e-9) {
            return false;
        }
        std::tie(parameters, std::ignore) = bopt.maximumExpectedImprovement();
    }

    // Write parameters for next round.
    if (m_DownsampleFactorOverride == boost::none) {
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
            m_Regularization.depthPenaltyMultiplier(
                scale * CTools::stableExp(parameters(i)));
            break;
        case E_DownsampleFactor:
            m_DownsampleFactor = CTools::stableExp(parameters(i));
            break;
        case E_Eta:
            m_Eta = CTools::stableExp(parameters(i));
            break;
        case E_EtaGrowthRatePerTree:
            m_EtaGrowthRatePerTree = parameters(i);
            break;
        case E_FeatureBagFraction:
            m_FeatureBagFraction = parameters(i);
            break;
        case E_MaximumNumberTrees:
            m_MaximumNumberTrees = static_cast<std::size_t>(std::ceil(parameters(i)));
            break;
        case E_Gamma:
            m_Regularization.treeSizePenaltyMultiplier(
                scale * CTools::stableExp(parameters(i)));
            break;
        case E_Lambda:
            m_Regularization.leafWeightPenaltyMultiplier(
                scale * CTools::stableExp(parameters(i)));
            break;
        case E_SoftTreeDepthLimit:
            m_Regularization.softTreeDepthLimit(std::max(parameters(i), 2.0));
            break;
        case E_SoftTreeDepthTolerance:
            m_Regularization.softTreeDepthTolerance(parameters(i));
            break;
        case E_PredictionChangeCost:
            m_PredictionChangeCost = CTools::stableExp(parameters(i));
            break;
        case E_TreeTopologyChangePenalty:
            m_Regularization.treeTopologyChangePenalty(CTools::stableExp(parameters(i)));
            break;
        }
    }

    return true;
}

void CBoostedTreeImpl::captureBestHyperparameters(const TMeanVarAccumulator& lossMoments,
                                                  std::size_t maximumNumberTrees,
                                                  double numberKeptNodes,
                                                  double numberRetrainedNodes) {
    // We capture the parameters with the lowest error at one standard
    // deviation above the mean. If the mean error improvement is marginal
    // we prefer the solution with the least variation across the folds.

    double loss{lossAtNSigma(1.0, lossMoments) +
                this->modelSizePenalty(numberKeptNodes, numberRetrainedNodes)};
    if (loss < m_BestForestTestLoss) {
        m_BestForestTestLoss = loss;
        m_BestHyperparameters = CBoostedTreeHyperparameters(
            m_Regularization, m_DownsampleFactor, m_Eta, m_EtaGrowthRatePerTree,
            maximumNumberTrees, m_FeatureBagFraction, m_PredictionChangeCost);
    }
}

double CBoostedTreeImpl::modelSizePenalty(double numberKeptNodes,
                                          double numberRetrainedNodes) const {
    // 0.01 * "forest number nodes" * E[GP] / "average forest number nodes" to meanLoss.
    return 0.01 * (numberKeptNodes + numberRetrainedNodes) /
           (numberKeptNodes + CBasicStatistics::mean(m_MeanForestSizeAccumulator)) *
           CBasicStatistics::mean(m_MeanLossAccumulator);
}

void CBoostedTreeImpl::restoreBestHyperparameters() {
    m_Regularization = m_BestHyperparameters.regularization();
    m_DownsampleFactor = m_BestHyperparameters.downsampleFactor();
    m_Eta = m_BestHyperparameters.eta();
    m_EtaGrowthRatePerTree = m_BestHyperparameters.etaGrowthRatePerTree();
    m_MaximumNumberTrees = m_BestHyperparameters.maximumNumberTrees();
    m_FeatureBagFraction = m_BestHyperparameters.featureBagFraction();
    m_PredictionChangeCost = m_BestHyperparameters.predictionChangeCost();
    LOG_TRACE(<< "loss* = " << m_BestForestTestLoss
              << ", regularization* = " << m_Regularization.print()
              << ", downsample factor* = " << m_DownsampleFactor << ", eta* = " << m_Eta
              << ", eta growth rate per tree* = " << m_EtaGrowthRatePerTree
              << ", maximum number trees* = " << m_MaximumNumberTrees
              << ", feature bag fraction* = " << m_FeatureBagFraction
              << ", prediction change cost* = " << m_PredictionChangeCost);
}

void CBoostedTreeImpl::scaleRegularizers(double scale) {
    if (m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        m_Regularization.depthPenaltyMultiplier(
            scale * m_Regularization.depthPenaltyMultiplier());
    }
    if (m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        m_Regularization.treeSizePenaltyMultiplier(
            scale * m_Regularization.treeSizePenaltyMultiplier());
    }
    if (m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        m_Regularization.leafWeightPenaltyMultiplier(
            scale * m_Regularization.leafWeightPenaltyMultiplier());
    }
}

std::size_t CBoostedTreeImpl::numberHyperparametersToTune() const {
    return m_RegularizationOverride.countNotSetForTrain() +
           (m_DownsampleFactorOverride != boost::none ? 0 : 1) +
           (m_EtaOverride != boost::none
                ? 0
                : (m_EtaGrowthRatePerTreeOverride != boost::none ? 1 : 2)) +
           (m_FeatureBagFractionOverride != boost::none ? 0 : 1);
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

void CBoostedTreeImpl::recordHyperparameters() {
    m_Instrumentation->hyperparameters().s_Eta = m_Eta;
    m_Instrumentation->hyperparameters().s_ClassAssignmentObjective = m_ClassAssignmentObjective;
    m_Instrumentation->hyperparameters().s_DepthPenaltyMultiplier =
        m_Regularization.depthPenaltyMultiplier();
    m_Instrumentation->hyperparameters().s_SoftTreeDepthLimit =
        m_Regularization.softTreeDepthLimit();
    m_Instrumentation->hyperparameters().s_SoftTreeDepthTolerance =
        m_Regularization.softTreeDepthTolerance();
    m_Instrumentation->hyperparameters().s_TreeSizePenaltyMultiplier =
        m_Regularization.treeSizePenaltyMultiplier();
    m_Instrumentation->hyperparameters().s_LeafWeightPenaltyMultiplier =
        m_Regularization.leafWeightPenaltyMultiplier();
    m_Instrumentation->hyperparameters().s_TreeTopologyChangePenalty =
        m_Regularization.treeTopologyChangePenalty();
    m_Instrumentation->hyperparameters().s_DownsampleFactor = m_DownsampleFactor;
    m_Instrumentation->hyperparameters().s_NumFolds = m_NumberFolds;
    m_Instrumentation->hyperparameters().s_MaxTrees = m_MaximumNumberTrees;
    m_Instrumentation->hyperparameters().s_FeatureBagFraction = m_FeatureBagFraction;
    m_Instrumentation->hyperparameters().s_PredictionChangeCost = m_PredictionChangeCost;
    m_Instrumentation->hyperparameters().s_EtaGrowthRatePerTree = m_EtaGrowthRatePerTree;
    m_Instrumentation->hyperparameters().s_MaxAttemptsToAddTree = m_MaximumAttemptsToAddTree;
    m_Instrumentation->hyperparameters().s_NumSplitsPerFeature = m_NumberSplitsPerFeature;
    m_Instrumentation->hyperparameters().s_MaxOptimizationRoundsPerHyperparameter =
        m_MaximumOptimisationRoundsPerHyperparameter;
}

void CBoostedTreeImpl::initializeTunableHyperparameters() {
    m_TunableHyperparameters.clear();
    for (int i = 0; i < static_cast<int>(NUMBER_HYPERPARAMETERS); ++i) {
        switch (static_cast<EHyperparameter>(i)) {
        case E_DownsampleFactor:
            if (m_IncrementalTraining == false && m_DownsampleFactorOverride == boost::none) {
                m_TunableHyperparameters.push_back(E_DownsampleFactor);
            }
            break;
        case E_Alpha:
            if (m_IncrementalTraining == false &&
                m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
                m_TunableHyperparameters.push_back(E_Alpha);
            }
            break;
        case E_Lambda:
            if (m_IncrementalTraining == false &&
                m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
                m_TunableHyperparameters.push_back(E_Lambda);
            }
            break;
        case E_Gamma:
            if (m_IncrementalTraining == false &&
                m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
                m_TunableHyperparameters.push_back(E_Gamma);
            }
            break;
        case E_SoftTreeDepthLimit:
            if (m_IncrementalTraining == false &&
                m_RegularizationOverride.softTreeDepthLimit() == boost::none) {
                m_TunableHyperparameters.push_back(E_SoftTreeDepthLimit);
            }
            break;
        case E_SoftTreeDepthTolerance:
            if (m_IncrementalTraining == false &&
                m_RegularizationOverride.softTreeDepthTolerance() == boost::none) {
                m_TunableHyperparameters.push_back(E_SoftTreeDepthTolerance);
            }
            break;
        case E_Eta:
            if (m_IncrementalTraining == false && m_EtaOverride == boost::none) {
                m_TunableHyperparameters.push_back(E_Eta);
            }
            break;
        case E_EtaGrowthRatePerTree:
            if (m_IncrementalTraining == false && m_EtaOverride == boost::none &&
                m_EtaGrowthRatePerTreeOverride == boost::none) {
                m_TunableHyperparameters.push_back(E_EtaGrowthRatePerTree);
            }
            break;
        case E_FeatureBagFraction:
            if (m_IncrementalTraining == false && m_FeatureBagFractionOverride == boost::none) {
                m_TunableHyperparameters.push_back(E_FeatureBagFraction);
            }
            break;
        case E_PredictionChangeCost:
            if (m_IncrementalTraining == false &&
                m_PredictionChangeCostOverride == boost::none) {
                m_TunableHyperparameters.push_back(E_PredictionChangeCost);
            }
            break;
        case E_TreeTopologyChangePenalty:
            if (m_IncrementalTraining &&
                m_RegularizationOverride.treeTopologyChangePenalty() == boost::none) {
                m_TunableHyperparameters.push_back(E_TreeTopologyChangePenalty);
            }
            break;
        case E_MaximumNumberTrees:
            // maximum number trees is not a tunable parameter
            break;
        }
    }
}

void CBoostedTreeImpl::initializeHyperparameterSamples() {
    std::size_t dim{m_TunableHyperparameters.size()};
    std::size_t n{m_NumberRounds / 3 + 1};
    CSampling::sobolSequenceSample(dim, n, m_HyperparameterSamples);
}

void CBoostedTreeImpl::startProgressMonitoringFineTuneHyperparameters() {

    // This costs "number folds" * "maximum number trees per forest" units
    // per round.

    m_Instrumentation->startNewProgressMonitoredTask(CBoostedTreeFactory::FINE_TUNING_PARAMETERS);

    std::size_t totalNumberSteps{m_NumberRounds * m_MaximumNumberTrees * m_NumberFolds};
    LOG_TRACE(<< "main loop total number steps = " << totalNumberSteps);
    m_TrainingProgress = core::CLoopProgress{
        totalNumberSteps, m_Instrumentation->progressCallback(), 1.0, 1024};

    // Make sure progress starts where it left off.
    m_TrainingProgress.increment(m_CurrentRound * m_MaximumNumberTrees * m_NumberFolds);
}

void CBoostedTreeImpl::startProgressMonitoringFinalTrain() {

    // The final model training uses more data so it's monitored separately.

    m_Instrumentation->startNewProgressMonitoredTask(CBoostedTreeFactory::FINAL_TRAINING);
    m_TrainingProgress = core::CLoopProgress{
        m_MaximumNumberTrees, m_Instrumentation->progressCallback(), 1.0, 1024};
}

void CBoostedTreeImpl::skipProgressMonitoringFinalTrain() {
    m_Instrumentation->startNewProgressMonitoredTask(CBoostedTreeFactory::FINAL_TRAINING);
}

void CBoostedTreeImpl::startProgressMonitoringTrainIncremental() {

    // This costs "number folds" * "maximum number retrained trees" units
    // per round.

    m_Instrumentation->startNewProgressMonitoredTask(CBoostedTreeFactory::INCREMENTAL_TRAIN);

    std::size_t totalNumberSteps{m_NumberRounds * this->numberTreesToRetrain() * m_NumberFolds};
    LOG_TRACE(<< "main loop total number steps = " << totalNumberSteps);
    m_TrainingProgress = core::CLoopProgress{
        totalNumberSteps, m_Instrumentation->progressCallback(), 1.0, 1024};

    // Make sure progress starts where it left off.
    m_TrainingProgress.increment(m_CurrentRound * this->numberTreesToRetrain() * m_NumberFolds);
}

namespace {
// TODO Can we upgrade state after introducing incremental training or do we
// need a new version tag?
const std::string VERSION_7_11_TAG{"7.11"};
const std::string VERSION_7_8_TAG{"7.8"};
const TStrVec SUPPORTED_VERSIONS{VERSION_7_8_TAG, VERSION_7_11_TAG};

const std::string BAYESIAN_OPTIMIZATION_TAG{"bayesian_optimization"};
const std::string BEST_FOREST_TAG{"best_forest"};
const std::string BEST_FOREST_TEST_LOSS_TAG{"best_forest_test_loss"};
const std::string BEST_HYPERPARAMETERS_TAG{"best_hyperparameters"};
const std::string CLASSIFICATION_WEIGHTS_OVERRIDE_TAG{"classification_weights_tag"};
const std::string CURRENT_ROUND_TAG{"current_round"};
const std::string DEPENDENT_VARIABLE_TAG{"dependent_variable"};
const std::string DOWNSAMPLE_FACTOR_OVERRIDE_TAG{"downsample_factor_override"};
const std::string DOWNSAMPLE_FACTOR_TAG{"downsample_factor"};
const std::string ENCODER_TAG{"encoder"};
const std::string ETA_GROWTH_RATE_PER_TREE_TAG{"eta_growth_rate_per_tree"};
const std::string ETA_GROWTH_RATE_PER_TREE_OVERRIDE_TAG{"eta_growth_rate_per_tree_override"};
const std::string ETA_OVERRIDE_TAG{"eta_override"};
const std::string ETA_TAG{"eta"};
const std::string FEATURE_BAG_FRACTION_OVERRIDE_TAG{"feature_bag_fraction_override"};
const std::string FEATURE_BAG_FRACTION_TAG{"feature_bag_fraction"};
const std::string FEATURE_DATA_TYPES_TAG{"feature_data_types"};
const std::string FEATURE_SAMPLE_PROBABILITIES_TAG{"feature_sample_probabilities"};
const std::string FOLD_ROUND_TEST_LOSSES_TAG{"fold_round_test_losses"};
const std::string INITIALIZATION_STAGE_TAG{"initialization_progress"};
const std::string INCREMENTAL_TRAINING_TAG{"incremental_training"};
const std::string LOSS_TAG{"loss"};
const std::string LOSS_NAME_TAG{"loss_name"};
const std::string MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG{"maximum_attempts_to_add_tree"};
const std::string MAXIMUM_NUMBER_TREES_OVERRIDE_TAG{"maximum_number_trees_override"};
const std::string MAXIMUM_NUMBER_TREES_TAG{"maximum_number_trees"};
const std::string MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG{
    "maximum_optimisation_rounds_per_hyperparameter"};
const std::string MEAN_FOREST_SIZE_ACCUMULATOR_TAG{"mean_forest_size"};
const std::string MEAN_LOSS_ACCUMULATOR_TAG{"mean_loss"};
const std::string MISSING_FEATURE_ROW_MASKS_TAG{"missing_feature_row_masks"};
const std::string NUMBER_FOLDS_TAG{"number_folds"};
const std::string NUMBER_FOLDS_OVERRIDE_TAG{"number_folds_override"};
const std::string NUMBER_ROUNDS_TAG{"number_rounds"};
const std::string NUMBER_SPLITS_PER_FEATURE_TAG{"number_splits_per_feature"};
const std::string NUMBER_THREADS_TAG{"number_threads"};
const std::string PREDICTION_CHANGE_COST_TAG{"prediction_change_cost"};
const std::string PREDICTION_CHANGE_COST_OVERRIDE_TAG{"prediction_change_cost_override"};
const std::string RANDOM_NUMBER_GENERATOR_TAG{"random_number_generator"};
const std::string REGULARIZATION_TAG{"regularization"};
const std::string REGULARIZATION_OVERRIDE_TAG{"regularization_override"};
const std::string RETRAIN_FRACTION_TAG{"retrain_fraction"};
const std::string ROWS_PER_FEATURE_TAG{"rows_per_feature"};
const std::string STOP_CROSS_VALIDATION_EARLY_TAG{"stop_cross_validation_eraly"};
const std::string STOP_HYPERPARAMETER_OPTIMIZATION_EARLY_TAG{"stop_hyperparameter_optimization_early"};
const std::string TESTING_ROW_MASKS_TAG{"testing_row_masks"};
const std::string TRAINING_ROW_MASKS_TAG{"training_row_masks"};
const std::string TREES_TO_RETRAIN_TAG{"trees_to_retrain"};
const std::string NUMBER_TOP_SHAP_VALUES_TAG{"top_shap_values"};
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
    core::CPersistUtils::persist(VERSION_7_11_TAG, "", inserter);
    core::CPersistUtils::persistIfNotNull(BAYESIAN_OPTIMIZATION_TAG,
                                          m_BayesianOptimization, inserter);
    core::CPersistUtils::persist(BEST_FOREST_TEST_LOSS_TAG, m_BestForestTestLoss, inserter);
    core::CPersistUtils::persist(BEST_FOREST_TAG, m_BestForest, inserter);
    core::CPersistUtils::persist(BEST_HYPERPARAMETERS_TAG, m_BestHyperparameters, inserter);
    core::CPersistUtils::persistIfNotNull(CLASSIFICATION_WEIGHTS_OVERRIDE_TAG,
                                          m_ClassificationWeightsOverride, inserter);
    core::CPersistUtils::persist(CURRENT_ROUND_TAG, m_CurrentRound, inserter);
    core::CPersistUtils::persist(DEPENDENT_VARIABLE_TAG, m_DependentVariable, inserter);
    core::CPersistUtils::persist(DOWNSAMPLE_FACTOR_OVERRIDE_TAG,
                                 m_DownsampleFactorOverride, inserter);
    core::CPersistUtils::persist(DOWNSAMPLE_FACTOR_TAG, m_DownsampleFactor, inserter);
    core::CPersistUtils::persistIfNotNull(ENCODER_TAG, m_Encoder, inserter);
    core::CPersistUtils::persist(ETA_GROWTH_RATE_PER_TREE_TAG,
                                 m_EtaGrowthRatePerTree, inserter);
    core::CPersistUtils::persist(ETA_GROWTH_RATE_PER_TREE_OVERRIDE_TAG,
                                 m_EtaGrowthRatePerTreeOverride, inserter);
    core::CPersistUtils::persist(ETA_TAG, m_Eta, inserter);
    core::CPersistUtils::persist(ETA_OVERRIDE_TAG, m_EtaOverride, inserter);
    core::CPersistUtils::persist(FEATURE_BAG_FRACTION_TAG, m_FeatureBagFraction, inserter);
    core::CPersistUtils::persist(FEATURE_BAG_FRACTION_OVERRIDE_TAG,
                                 m_FeatureBagFractionOverride, inserter);
    core::CPersistUtils::persist(FEATURE_DATA_TYPES_TAG, m_FeatureDataTypes, inserter);
    core::CPersistUtils::persist(FEATURE_SAMPLE_PROBABILITIES_TAG,
                                 m_FeatureSampleProbabilities, inserter);
    core::CPersistUtils::persist(FOLD_ROUND_TEST_LOSSES_TAG, m_FoldRoundTestLosses, inserter);
    core::CPersistUtils::persist(INCREMENTAL_TRAINING_TAG, m_IncrementalTraining, inserter);
    core::CPersistUtils::persist(INITIALIZATION_STAGE_TAG,
                                 static_cast<int>(m_InitializationStage), inserter);
    if (m_Loss != nullptr) {
        inserter.insertLevel(LOSS_TAG, [this](core::CStatePersistInserter& inserter_) {
            m_Loss->persistLoss(inserter_);
        });
    }
    core::CPersistUtils::persist(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                                 m_MaximumAttemptsToAddTree, inserter);
    core::CPersistUtils::persist(MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                                 m_MaximumOptimisationRoundsPerHyperparameter, inserter);
    core::CPersistUtils::persist(MAXIMUM_NUMBER_TREES_TAG, m_MaximumNumberTrees, inserter);
    core::CPersistUtils::persist(MAXIMUM_NUMBER_TREES_OVERRIDE_TAG,
                                 m_MaximumNumberTreesOverride, inserter);
    core::CPersistUtils::persist(MEAN_FOREST_SIZE_ACCUMULATOR_TAG,
                                 m_MeanForestSizeAccumulator, inserter);
    core::CPersistUtils::persist(MEAN_LOSS_ACCUMULATOR_TAG, m_MeanLossAccumulator, inserter);
    core::CPersistUtils::persist(MISSING_FEATURE_ROW_MASKS_TAG,
                                 m_MissingFeatureRowMasks, inserter);
    core::CPersistUtils::persist(NUMBER_FOLDS_TAG, m_NumberFolds, inserter);
    core::CPersistUtils::persist(NUMBER_FOLDS_OVERRIDE_TAG, m_NumberFoldsOverride, inserter);
    core::CPersistUtils::persist(NUMBER_ROUNDS_TAG, m_NumberRounds, inserter);
    core::CPersistUtils::persist(NUMBER_SPLITS_PER_FEATURE_TAG,
                                 m_NumberSplitsPerFeature, inserter);
    core::CPersistUtils::persist(NUMBER_THREADS_TAG, m_NumberThreads, inserter);
    core::CPersistUtils::persist(NUMBER_TOP_SHAP_VALUES_TAG, m_NumberTopShapValues, inserter);
    core::CPersistUtils::persist(PREDICTION_CHANGE_COST_TAG, m_PredictionChangeCost, inserter);
    core::CPersistUtils::persist(PREDICTION_CHANGE_COST_OVERRIDE_TAG,
                                 m_PredictionChangeCostOverride, inserter);
    inserter.insertValue(RANDOM_NUMBER_GENERATOR_TAG, m_Rng.toString());
    core::CPersistUtils::persist(REGULARIZATION_OVERRIDE_TAG,
                                 m_RegularizationOverride, inserter);
    core::CPersistUtils::persist(REGULARIZATION_TAG, m_Regularization, inserter);
    core::CPersistUtils::persist(RETRAIN_FRACTION_TAG, m_RetrainFraction, inserter);
    core::CPersistUtils::persist(ROWS_PER_FEATURE_TAG, m_RowsPerFeature, inserter);
    core::CPersistUtils::persist(STOP_CROSS_VALIDATION_EARLY_TAG,
                                 m_StopCrossValidationEarly, inserter);
    core::CPersistUtils::persist(TESTING_ROW_MASKS_TAG, m_TestingRowMasks, inserter);
    core::CPersistUtils::persist(TRAINING_ROW_MASKS_TAG, m_TrainingRowMasks, inserter);
    core::CPersistUtils::persist(TREES_TO_RETRAIN_TAG, m_TreesToRetrain, inserter);
    core::CPersistUtils::persist(STOP_HYPERPARAMETER_OPTIMIZATION_EARLY_TAG,
                                 m_StopHyperparameterOptimizationEarly, inserter);
    // m_TunableHyperparameters is not persisted explicitly, it is re-generated
    // from overriden hyperparameters.
    // m_HyperparameterSamples is not persisted explicitly, it is re-generated.
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

    if (traverser.name() != VERSION_7_11_TAG) {
        m_StopHyperparameterOptimizationEarly = false;
    }

    do {
        const std::string& name = traverser.name();
        RESTORE_NO_ERROR(BAYESIAN_OPTIMIZATION_TAG,
                         m_BayesianOptimization =
                             std::make_unique<CBayesianOptimisation>(traverser))
        RESTORE(BEST_FOREST_TAG,
                core::CPersistUtils::restore(BEST_FOREST_TAG, m_BestForest, traverser))
        RESTORE(BEST_FOREST_TEST_LOSS_TAG,
                core::CPersistUtils::restore(BEST_FOREST_TEST_LOSS_TAG,
                                             m_BestForestTestLoss, traverser))
        RESTORE(BEST_HYPERPARAMETERS_TAG,
                core::CPersistUtils::restore(BEST_HYPERPARAMETERS_TAG,
                                             m_BestHyperparameters, traverser))
        RESTORE_SETUP_TEARDOWN(
            CLASSIFICATION_WEIGHTS_OVERRIDE_TAG,
            m_ClassificationWeightsOverride = TStrDoublePrVec{},
            core::CPersistUtils::restore(CLASSIFICATION_WEIGHTS_OVERRIDE_TAG,
                                         *m_ClassificationWeightsOverride, traverser),
            /*no-op*/)
        RESTORE(CURRENT_ROUND_TAG,
                core::CPersistUtils::restore(CURRENT_ROUND_TAG, m_CurrentRound, traverser))
        RESTORE(DEPENDENT_VARIABLE_TAG,
                core::CPersistUtils::restore(DEPENDENT_VARIABLE_TAG,
                                             m_DependentVariable, traverser))
        RESTORE(DOWNSAMPLE_FACTOR_OVERRIDE_TAG,
                core::CPersistUtils::restore(DOWNSAMPLE_FACTOR_OVERRIDE_TAG,
                                             m_DownsampleFactorOverride, traverser))
        RESTORE(DOWNSAMPLE_FACTOR_TAG,
                core::CPersistUtils::restore(DOWNSAMPLE_FACTOR_TAG, m_DownsampleFactor, traverser))
        RESTORE_NO_ERROR(ENCODER_TAG,
                         m_Encoder = std::make_unique<CDataFrameCategoryEncoder>(traverser))
        RESTORE(ETA_GROWTH_RATE_PER_TREE_TAG,
                core::CPersistUtils::restore(ETA_GROWTH_RATE_PER_TREE_TAG,
                                             m_EtaGrowthRatePerTree, traverser))
        RESTORE(ETA_GROWTH_RATE_PER_TREE_OVERRIDE_TAG,
                core::CPersistUtils::restore(ETA_GROWTH_RATE_PER_TREE_OVERRIDE_TAG,
                                             m_EtaGrowthRatePerTreeOverride, traverser))
        RESTORE(ETA_OVERRIDE_TAG,
                core::CPersistUtils::restore(ETA_OVERRIDE_TAG, m_EtaOverride, traverser))
        RESTORE(ETA_TAG, core::CPersistUtils::restore(ETA_TAG, m_Eta, traverser))
        RESTORE(FEATURE_BAG_FRACTION_OVERRIDE_TAG,
                core::CPersistUtils::restore(FEATURE_BAG_FRACTION_OVERRIDE_TAG,
                                             m_FeatureBagFractionOverride, traverser))
        RESTORE(FEATURE_BAG_FRACTION_TAG,
                core::CPersistUtils::restore(FEATURE_BAG_FRACTION_TAG,
                                             m_FeatureBagFraction, traverser))
        RESTORE(FEATURE_DATA_TYPES_TAG,
                core::CPersistUtils::restore(FEATURE_DATA_TYPES_TAG,
                                             m_FeatureDataTypes, traverser));
        RESTORE(FEATURE_SAMPLE_PROBABILITIES_TAG,
                core::CPersistUtils::restore(FEATURE_SAMPLE_PROBABILITIES_TAG,
                                             m_FeatureSampleProbabilities, traverser))
        RESTORE(FOLD_ROUND_TEST_LOSSES_TAG,
                core::CPersistUtils::restore(FOLD_ROUND_TEST_LOSSES_TAG,
                                             m_FoldRoundTestLosses, traverser))
        RESTORE(INCREMENTAL_TRAINING_TAG,
                core::CPersistUtils::restore(INCREMENTAL_TRAINING_TAG,
                                             m_IncrementalTraining, traverser))
        RESTORE(INITIALIZATION_STAGE_TAG,
                core::CPersistUtils::restore(INITIALIZATION_STAGE_TAG,
                                             initializationStage, traverser))
        RESTORE(LOSS_TAG, traverser.traverseSubLevel(restoreLoss))
        RESTORE(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                core::CPersistUtils::restore(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                                             m_MaximumAttemptsToAddTree, traverser))
        RESTORE(MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                core::CPersistUtils::restore(
                    MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                    m_MaximumOptimisationRoundsPerHyperparameter, traverser))
        RESTORE(MAXIMUM_NUMBER_TREES_OVERRIDE_TAG,
                core::CPersistUtils::restore(MAXIMUM_NUMBER_TREES_OVERRIDE_TAG,
                                             m_MaximumNumberTreesOverride, traverser))
        RESTORE(MAXIMUM_NUMBER_TREES_TAG,
                core::CPersistUtils::restore(MAXIMUM_NUMBER_TREES_TAG,
                                             m_MaximumNumberTrees, traverser))
        RESTORE(MEAN_FOREST_SIZE_ACCUMULATOR_TAG,
                core::CPersistUtils::restore(MEAN_FOREST_SIZE_ACCUMULATOR_TAG,
                                             m_MeanForestSizeAccumulator, traverser))
        RESTORE(MEAN_LOSS_ACCUMULATOR_TAG,
                core::CPersistUtils::restore(MEAN_LOSS_ACCUMULATOR_TAG,
                                             m_MeanLossAccumulator, traverser))
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
        RESTORE(NUMBER_TOP_SHAP_VALUES_TAG,
                core::CPersistUtils::restore(NUMBER_TOP_SHAP_VALUES_TAG,
                                             m_NumberTopShapValues, traverser))
        RESTORE(PREDICTION_CHANGE_COST_TAG,
                core::CPersistUtils::restore(PREDICTION_CHANGE_COST_TAG,
                                             m_PredictionChangeCost, traverser))
        RESTORE(PREDICTION_CHANGE_COST_OVERRIDE_TAG,
                core::CPersistUtils::restore(PREDICTION_CHANGE_COST_OVERRIDE_TAG,
                                             m_PredictionChangeCostOverride, traverser))
        RESTORE(RANDOM_NUMBER_GENERATOR_TAG, m_Rng.fromString(traverser.value()))
        RESTORE(REGULARIZATION_TAG,
                core::CPersistUtils::restore(REGULARIZATION_TAG, m_Regularization, traverser))
        RESTORE(REGULARIZATION_OVERRIDE_TAG,
                core::CPersistUtils::restore(REGULARIZATION_OVERRIDE_TAG,
                                             m_RegularizationOverride, traverser))
        RESTORE(RETRAIN_FRACTION_TAG,
                core::CPersistUtils::restore(RETRAIN_FRACTION_TAG, m_RetrainFraction, traverser))
        RESTORE(ROWS_PER_FEATURE_TAG,
                core::CPersistUtils::restore(ROWS_PER_FEATURE_TAG, m_RowsPerFeature, traverser))
        RESTORE(STOP_CROSS_VALIDATION_EARLY_TAG,
                core::CPersistUtils::restore(STOP_CROSS_VALIDATION_EARLY_TAG,
                                             m_StopCrossValidationEarly, traverser))
        RESTORE(TESTING_ROW_MASKS_TAG,
                core::CPersistUtils::restore(TESTING_ROW_MASKS_TAG, m_TestingRowMasks, traverser))
        RESTORE(TRAINING_ROW_MASKS_TAG,
                core::CPersistUtils::restore(TRAINING_ROW_MASKS_TAG, m_TrainingRowMasks, traverser))
        RESTORE(TREES_TO_RETRAIN_TAG,
                core::CPersistUtils::restore(TREES_TO_RETRAIN_TAG, m_TreesToRetrain, traverser))
        RESTORE(STOP_HYPERPARAMETER_OPTIMIZATION_EARLY_TAG,
                core::CPersistUtils::restore(STOP_HYPERPARAMETER_OPTIMIZATION_EARLY_TAG,
                                             m_StopHyperparameterOptimizationEarly, traverser))
        // m_TunableHyperparameters is not restored explicitly it is re-generated
        // from overriden hyperparameters.
        // m_HyperparameterSamples is not restored explicitly it is re-generated.
    } while (traverser.next());

    this->initializeTunableHyperparameters();
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
    if (m_InitializationStage == CBoostedTreeImpl::E_FullyInitialized) {
        VIOLATES_INVARIANT_NO_EVALUATION(m_BayesianOptimization, ==, nullptr);
    }
    VIOLATES_INVARIANT(m_CurrentRound, >, m_NumberRounds);
    for (const auto& samples : m_HyperparameterSamples) {
        VIOLATES_INVARIANT(m_TunableHyperparameters.size(), !=, samples.size());
    }
    if (m_FoldRoundTestLosses.empty() == false) {
        VIOLATES_INVARIANT(m_FoldRoundTestLosses.size(), !=, m_NumberFolds);
        for (const auto& losses : m_FoldRoundTestLosses) {
            VIOLATES_INVARIANT(losses.size(), >, m_NumberRounds);
        }
    }
    for (auto tree : m_TreesToRetrain) {
        VIOLATES_INVARIANT(tree, >=, m_BestForest.size());
    }
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
    if (m_BayesianOptimization == nullptr) {
        HANDLE_FATAL(<< "Internal error: must supply an optimizer. Please report this problem.");
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
            HANDLE_FATAL(<< "Internal error: unexpected missing training mask ("
                         << mask.size() << " !=  " << frame.numberRows()
                         << "). Please report this problem.");
        }
    }
    for (const auto& mask : m_TestingRowMasks) {
        if (mask.size() != frame.numberRows()) {
            HANDLE_FATAL(<< "Internal error: unexpected missing testing mask ("
                         << mask.size() << " !=  " << frame.numberRows()
                         << "). Please report this problem.");
        }
    }
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
    if (m_BayesianOptimization == nullptr) {
        HANDLE_FATAL(<< "Internal error: must supply an optimizer. Please report this problem.");
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
            HANDLE_FATAL(<< "Internal error: unexpected missing training mask ("
                         << mask.size() << " !=  " << frame.numberRows()
                         << "). Please report this problem.");
        }
    }
    for (const auto& mask : m_TestingRowMasks) {
        if (mask.size() != frame.numberRows()) {
            HANDLE_FATAL(<< "Internal error: unexpected missing testing mask ("
                         << mask.size() << " !=  " << frame.numberRows()
                         << "). Please report this problem.");
        }
    }
}

std::size_t CBoostedTreeImpl::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_Loss)};
    mem += core::CMemory::dynamicSize(m_Encoder);
    mem += core::CMemory::dynamicSize(m_FeatureDataTypes);
    mem += core::CMemory::dynamicSize(m_FeatureSampleProbabilities);
    mem += core::CMemory::dynamicSize(m_MissingFeatureRowMasks);
    mem += core::CMemory::dynamicSize(m_TrainingRowMasks);
    mem += core::CMemory::dynamicSize(m_TestingRowMasks);
    mem += core::CMemory::dynamicSize(m_FoldRoundTestLosses);
    mem += core::CMemory::dynamicSize(m_BestForest);
    mem += core::CMemory::dynamicSize(m_BayesianOptimization);
    mem += core::CMemory::dynamicSize(m_Instrumentation);
    mem += core::CMemory::dynamicSize(m_HyperparameterSamples);
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

const CBoostedTreeHyperparameters& CBoostedTreeImpl::bestHyperparameters() const {
    return m_BestHyperparameters;
}

CTreeShapFeatureImportance* CBoostedTreeImpl::shap() {
    return m_TreeShap.get();
}

core::CPackedBitVector CBoostedTreeImpl::dataSummarization(const core::CDataFrame& frame) const {
    // get row mask for sampling
    // TODO #1834 implement a data summarization strategy.
    core::CPackedBitVector rowMask{};
    std::size_t sampleSize(std::min(
        frame.numberRows(), static_cast<std::size_t>(std::max(
                                static_cast<double>(frame.numberRows()) * 0.1, 100.0))));
    for (std::size_t i = 0; i < sampleSize; ++i) {
        rowMask.extend(true);
    }
    rowMask.extend(false, frame.numberRows() - rowMask.size());

    return rowMask;
}

CBoostedTreeImpl::THyperparameterImportanceVec
CBoostedTreeImpl::hyperparameterImportance() const {
    THyperparameterImportanceVec hyperparameterImportances;
    hyperparameterImportances.reserve(m_TunableHyperparameters.size());
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
            hyperparameterValue = m_Regularization.depthPenaltyMultiplier();
            break;
        case E_DownsampleFactor:
            hyperparameterValue = m_DownsampleFactor;
            break;
        case E_Eta:
            hyperparameterValue = m_Eta;
            break;
        case E_EtaGrowthRatePerTree:
            hyperparameterValue = m_EtaGrowthRatePerTree;
            break;
        case E_FeatureBagFraction:
            hyperparameterValue = m_FeatureBagFraction;
            break;
        case E_MaximumNumberTrees:
            hyperparameterValue = static_cast<double>(m_MaximumNumberTrees);
            hyperparameterType = boosted_tree_detail::SHyperparameterImportance::E_Uint64;
            break;
        case E_Gamma:
            hyperparameterValue = m_Regularization.treeSizePenaltyMultiplier();
            break;
        case E_Lambda:
            hyperparameterValue = m_Regularization.leafWeightPenaltyMultiplier();
            break;
        case E_SoftTreeDepthLimit:
            hyperparameterValue = m_Regularization.softTreeDepthLimit();
            break;
        case E_SoftTreeDepthTolerance:
            hyperparameterValue = m_Regularization.softTreeDepthTolerance();
            break;
        case E_PredictionChangeCost:
            hyperparameterValue = m_PredictionChangeCost;
            break;
        case E_TreeTopologyChangePenalty:
            hyperparameterValue = m_Regularization.treeTopologyChangePenalty();
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
        hyperparameterImportances.push_back(
            {static_cast<EHyperparameter>(i), hyperparameterValue,
             absoluteImportance, relativeImportance, supplied, hyperparameterType});
    }
    return hyperparameterImportances;
}

const CBoostedTreeImpl::TDoubleVec& CBoostedTreeImpl::featureSampleProbabilities() const {
    return m_FeatureSampleProbabilities;
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

const CBoostedTreeImpl::TSizeVec& CBoostedTreeImpl::extraColumns() const {
    return m_ExtraColumns;
}

const CBoostedTreeImpl::TVector& CBoostedTreeImpl::classificationWeights() const {
    return m_ClassificationWeights;
}

const double CBoostedTreeImpl::MINIMUM_RELATIVE_GAIN_PER_SPLIT{1e-7};
}
}
