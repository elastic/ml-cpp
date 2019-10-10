/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeImpl.h>

#include <core/CLoopProgress.h>
#include <core/CPersistUtils.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CBayesianOptimisation.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CQuantileSketch.h>
#include <maths/CSampling.h>
#include <maths/CSetTools.h>

namespace ml {
namespace maths {
using namespace boosted_tree;
using namespace boosted_tree_detail;

namespace {
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

std::size_t lossGradientColumn(std::size_t numberColumns) {
    return numberColumns - 2;
}

std::size_t lossCurvatureColumn(std::size_t numberColumns) {
    return numberColumns - 1;
}

double readPrediction(const TRowRef& row) {
    return row[predictionColumn(row.numberColumns())];
}

double readLossGradient(const TRowRef& row) {
    return row[lossGradientColumn(row.numberColumns())];
}

double readLossCurvature(const TRowRef& row) {
    return row[lossCurvatureColumn(row.numberColumns())];
}

double readActual(const TRowRef& row, std::size_t dependentVariable) {
    return row[dependentVariable];
}

const std::size_t ASSIGN_MISSING_TO_LEFT{0};
const std::size_t ASSIGN_MISSING_TO_RIGHT{1};
}

CBoostedTreeImpl::CLeafNodeStatistics::CLeafNodeStatistics(std::size_t id,
                                                           const CLeafNodeStatistics& parent,
                                                           const CLeafNodeStatistics& sibling,
                                                           core::CPackedBitVector rowMask)
    : m_Id{id}, m_Regularization{sibling.m_Regularization},
      m_CandidateSplits{sibling.m_CandidateSplits}, m_Depth{sibling.m_Depth},
      m_FeatureBag{sibling.m_FeatureBag}, m_RowMask{std::move(rowMask)} {

    LOG_TRACE(<< "row mask = " << m_RowMask);
    LOG_TRACE(<< "feature bag = " << core::CContainerPrinter::print(m_FeatureBag));

    m_Gradients.resize(m_CandidateSplits.size());
    m_Curvatures.resize(m_CandidateSplits.size());
    m_MissingGradients.resize(m_CandidateSplits.size(), 0.0);
    m_MissingCurvatures.resize(m_CandidateSplits.size(), 0.0);

    for (std::size_t i = 0; i < m_CandidateSplits.size(); ++i) {
        std::size_t numberSplits{m_CandidateSplits[i].size() + 1};
        m_Gradients[i].resize(numberSplits);
        m_Curvatures[i].resize(numberSplits);
        for (std::size_t j = 0; j < numberSplits; ++j) {
            m_Gradients[i][j] = parent.m_Gradients[i][j] - sibling.m_Gradients[i][j];
            m_Curvatures[i][j] = parent.m_Curvatures[i][j] - sibling.m_Curvatures[i][j];
        }
        m_MissingGradients[i] = parent.m_MissingGradients[i] -
                                sibling.m_MissingGradients[i];
        m_MissingCurvatures[i] = parent.m_MissingCurvatures[i] -
                                 sibling.m_MissingCurvatures[i];
    }

    LOG_TRACE(<< "gradients = " << core::CContainerPrinter::print(m_Gradients));
    LOG_TRACE(<< "curvatures = " << core::CContainerPrinter::print(m_Curvatures));
    LOG_TRACE(<< "missing gradients = " << core::CContainerPrinter::print(m_MissingGradients));
    LOG_TRACE(<< "missing curvatures = "
              << core::CContainerPrinter::print(m_MissingCurvatures));
}

void CBoostedTreeImpl::CLeafNodeStatistics::addRowDerivatives(const CEncodedDataFrameRowRef& row,
                                                              SDerivatives& derivatives) const {

    const TRowRef& unencodedRow{row.unencodedRow()};
    double gradient{readLossGradient(unencodedRow)};
    double curvature{readLossCurvature(unencodedRow)};

    for (std::size_t i = 0; i < m_CandidateSplits.size(); ++i) {
        double featureValue{row[i]};
        if (CDataFrameUtils::isMissing(featureValue)) {
            derivatives.s_MissingGradients[i] += gradient;
            derivatives.s_MissingCurvatures[i] += curvature;
        } else {
            const auto& featureCandidateSplits = m_CandidateSplits[i];
            auto j = std::upper_bound(featureCandidateSplits.begin(),
                                      featureCandidateSplits.end(), featureValue) -
                     featureCandidateSplits.begin();
            derivatives.s_Gradients[i][j] += gradient;
            derivatives.s_Curvatures[i][j] += curvature;
        }
    }
}

CBoostedTreeImpl::CLeafNodeStatistics::SSplitStatistics
CBoostedTreeImpl::CLeafNodeStatistics::computeBestSplitStatistics() const {

    // We have three possible regularization terms we'll use:
    //   1. Tree size: gamma * "node count"
    //   2. Sum square weights: lambda * sum{"leaf weight" ^ 2)}
    //   3. Tree depth: alpha * sum{exp(("depth" / "target depth" - 1.0) / "tolerance")}

    SSplitStatistics result{-INF, 0.0, m_FeatureBag.size(), INF, true};

    for (auto i : m_FeatureBag) {
        double g{std::accumulate(m_Gradients[i].begin(), m_Gradients[i].end(), 0.0) +
                 m_MissingGradients[i]};
        double h{std::accumulate(m_Curvatures[i].begin(), m_Curvatures[i].end(), 0.0) +
                 m_MissingCurvatures[i]};
        double gl[]{m_MissingGradients[i], 0.0};
        double hl[]{m_MissingCurvatures[i], 0.0};

        double maximumGain{-INF};
        double splitAt{-INF};
        bool assignMissingToLeft{true};

        for (std::size_t j = 0; j + 1 < m_Gradients[i].size(); ++j) {
            gl[ASSIGN_MISSING_TO_LEFT] += m_Gradients[i][j];
            hl[ASSIGN_MISSING_TO_LEFT] += m_Curvatures[i][j];
            gl[ASSIGN_MISSING_TO_RIGHT] += m_Gradients[i][j];
            hl[ASSIGN_MISSING_TO_RIGHT] += m_Curvatures[i][j];

            double gain[]{CTools::pow2(gl[ASSIGN_MISSING_TO_LEFT]) /
                                  (hl[ASSIGN_MISSING_TO_LEFT] +
                                   m_Regularization.leafWeightPenaltyMultiplier()) +
                              CTools::pow2(g - gl[ASSIGN_MISSING_TO_LEFT]) /
                                  (h - hl[ASSIGN_MISSING_TO_LEFT] +
                                   m_Regularization.leafWeightPenaltyMultiplier()),
                          CTools::pow2(gl[ASSIGN_MISSING_TO_RIGHT]) /
                                  (hl[ASSIGN_MISSING_TO_RIGHT] +
                                   m_Regularization.leafWeightPenaltyMultiplier()) +
                              CTools::pow2(g - gl[ASSIGN_MISSING_TO_RIGHT]) /
                                  (h - hl[ASSIGN_MISSING_TO_RIGHT] +
                                   m_Regularization.leafWeightPenaltyMultiplier())};

            if (gain[ASSIGN_MISSING_TO_LEFT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_LEFT];
                splitAt = m_CandidateSplits[i][j];
                assignMissingToLeft = true;
            }
            if (gain[ASSIGN_MISSING_TO_RIGHT] > maximumGain) {
                maximumGain = gain[ASSIGN_MISSING_TO_RIGHT];
                splitAt = m_CandidateSplits[i][j];
                assignMissingToLeft = false;
            }
        }

        double penaltyForDepth{m_Regularization.penaltyForDepth(m_Depth)};
        double penaltyForDepthPlusOne{m_Regularization.penaltyForDepth(m_Depth + 1)};
        double gain{0.5 * (maximumGain - CTools::pow2(g) / (h + m_Regularization.leafWeightPenaltyMultiplier())) -
                    m_Regularization.treeSizePenaltyMultiplier() -
                    m_Regularization.depthPenaltyMultiplier() *
                        (2.0 * penaltyForDepthPlusOne - penaltyForDepth)};

        SSplitStatistics candidate{gain, h, i, splitAt, assignMissingToLeft};
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
      m_BestHyperparameters{m_Regularization, m_Eta, m_EtaGrowthRatePerTree, m_FeatureBagFraction} {
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

    std::uint64_t lastMemoryUsage(this->memoryUsage());
    recordMemoryUsage(lastMemoryUsage);

    if (this->canTrain() == false) {
        // Fallback to using the constant predictor which minimises the loss.

        core::CPackedBitVector trainingRowMask{this->allTrainingRowsMask()};
        m_BestForest.assign(1, this->initializePredictionsAndLossDerivatives(frame, trainingRowMask));
        m_BestForestTestLoss = this->meanLoss(frame, trainingRowMask, m_BestForest);
        LOG_TRACE(<< "Test loss = " << m_BestForestTestLoss);

    } else {
        // Hyperparameter optimisation loop.

        while (m_CurrentRound < m_NumberRounds) {

            LOG_TRACE(<< "Optimisation round = " << m_CurrentRound + 1);

            TMeanVarAccumulator lossMoments{this->crossValidateForest(frame, recordMemoryUsage)};

            this->captureBestHyperparameters(lossMoments);

            // Trap the case that the dependent variable is (effectively) constant.
            // There is no point adjusting hyperparameters in this case - and we run
            // into numerical issues trying - since any forest will do.
            if (std::sqrt(CBasicStatistics::variance(lossMoments)) <
                1e-10 * std::fabs(CBasicStatistics::mean(lossMoments))) {
                break;
            }
            if (this->selectNextHyperparameters(lossMoments, *m_BayesianOptimization) == false) {
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
        }

        LOG_TRACE(<< "Test loss = " << m_BestForestTestLoss);

        this->restoreBestHyperparameters();

        m_BestForest = this->trainForest(frame, this->allTrainingRowsMask(), recordMemoryUsage);
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

void CBoostedTreeImpl::write(core::CRapidJsonConcurrentLineWriter& /*writer*/) const {
    // TODO
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

std::size_t CBoostedTreeImpl::numberExtraColumnsForTrain() {
    // We store the gradient and curvature of the loss function and the predicted
    // value for the dependent variable of the regression in the data frame.
    return 3;
}

std::size_t CBoostedTreeImpl::estimateMemoryUsage(std::size_t numberRows,
                                                  std::size_t numberColumns) const {
    // The maximum tree size is defined is the maximum number of leaves minus one.
    // A binary tree with n + 1 leaves has 2n + 1 nodes in total.
    std::size_t maximumNumberNodes{2 * this->maximumTreeSize(numberRows) + 1};
    std::size_t forestMemoryUsage{
        m_MaximumNumberTrees *
        (sizeof(TNodeVec) + maximumNumberNodes * sizeof(CBoostedTreeNode))};
    std::size_t extraColumnsMemoryUsage{this->numberExtraColumnsForTrain() *
                                        numberRows * sizeof(CFloatStorage)};
    std::size_t hyperparametersMemoryUsage{numberColumns * sizeof(double)};
    std::size_t leafNodeStatisticsMemoryUsage{
        maximumNumberNodes * CLeafNodeStatistics::estimateMemoryUsage(
                                 numberRows, numberColumns, m_FeatureBagFraction,
                                 m_NumberSplitsPerFeature)};
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

CBoostedTreeImpl::TMeanVarAccumulator
CBoostedTreeImpl::crossValidateForest(core::CDataFrame& frame,
                                      const TMemoryUsageCallback& recordMemoryUsage) const {
    TMeanVarAccumulator lossMoments;
    for (std::size_t i = 0; i < m_NumberFolds; ++i) {
        TNodeVecVec forest(this->trainForest(frame, m_TrainingRowMasks[i], recordMemoryUsage));
        double loss{this->meanLoss(frame, m_TestingRowMasks[i], forest)};
        lossMoments.add(loss);
        LOG_TRACE(<< "fold = " << i << " forest size = " << forest.size()
                  << " test set loss = " << loss);
    }
    LOG_TRACE(<< "test mean loss = " << CBasicStatistics::mean(lossMoments)
              << ", sigma = " << std::sqrt(CBasicStatistics::mean(lossMoments)));
    return lossMoments;
}

CBoostedTreeImpl::TNodeVec CBoostedTreeImpl::initializePredictionsAndLossDerivatives(
    core::CDataFrame& frame,
    const core::CPackedBitVector& trainingRowMask) const {

    frame.writeColumns(m_NumberThreads, 0, frame.numberRows(),
                       [](TRowItr beginRows, TRowItr endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               std::size_t numberColumns{row->numberColumns()};
                               row->writeColumn(predictionColumn(numberColumns), 0.0);
                               row->writeColumn(lossGradientColumn(numberColumns), 0.0);
                               row->writeColumn(lossCurvatureColumn(numberColumns), 0.0);
                           }
                       },
                       &trainingRowMask);

    // At the start we will centre the data w.r.t. the given loss function.
    TNodeVec tree(1);
    this->refreshPredictionsAndLossDerivatives(frame, trainingRowMask, 1.0, tree);

    return tree;
}

CBoostedTreeImpl::TNodeVecVec
CBoostedTreeImpl::trainForest(core::CDataFrame& frame,
                              const core::CPackedBitVector& trainingRowMask,
                              const TMemoryUsageCallback& recordMemoryUsage) const {

    LOG_TRACE(<< "Training one forest...");

    std::size_t maximumTreeSize{this->maximumTreeSize(trainingRowMask)};

    TNodeVecVec forest{this->initializePredictionsAndLossDerivatives(frame, trainingRowMask)};
    forest.reserve(m_MaximumNumberTrees);

    CScopeRecordMemoryUsage scopeMemoryUsage{forest, recordMemoryUsage};

    // For each iteration:
    //  1. Compute weighted quantiles for features F
    //  2. Compute candidate split set S from quantiles of F
    //  3. Build one tree on (F, S)
    //  4. Update predictions and loss derivatives

    double eta{m_Eta};
    double oneMinusBias{eta};

    TDoubleVecVec candidateSplits(this->candidateSplits(frame, trainingRowMask));
    scopeMemoryUsage.add(candidateSplits);

    std::size_t retries = 0;
    do {
        auto tree = this->trainTree(frame, trainingRowMask, candidateSplits,
                                    maximumTreeSize, recordMemoryUsage);

        retries = tree.size() == 1 ? retries + 1 : 0;

        if (oneMinusBias > 0.9 && retries == m_MaximumAttemptsToAddTree) {
            break;
        }

        if (tree.size() > 1) {
            scopeMemoryUsage.add(tree);
            this->refreshPredictionsAndLossDerivatives(frame, trainingRowMask, eta, tree);
            forest.push_back(std::move(tree));
            eta = std::min(1.0, m_EtaGrowthRatePerTree * eta);
            oneMinusBias += eta * (1.0 - oneMinusBias);
            retries = 0;
        } else if (oneMinusBias < 1.0) {
            scopeMemoryUsage.add(tree);
            this->refreshPredictionsAndLossDerivatives(frame, trainingRowMask, 1.0, tree);
            oneMinusBias = 1.0;
            forest.push_back(std::move(tree));
        }
        LOG_TRACE(<< "bias = " << (1.0 - oneMinusBias));

        if (m_Loss->isCurvatureConstant() == false) {
            candidateSplits = this->candidateSplits(frame, trainingRowMask);
        }
    } while (forest.size() < m_MaximumNumberTrees);

    LOG_TRACE(<< "Trained one forest");

    m_TrainingProgress.increment();

    return forest;
}

CBoostedTreeImpl::TDoubleVecVec
CBoostedTreeImpl::candidateSplits(const core::CDataFrame& frame,
                                  const core::CPackedBitVector& trainingRowMask) const {

    using TQuantileSketchVec = std::vector<CQuantileSketch>;

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

    TQuantileSketchVec featureQuantiles;
    CDataFrameUtils::columnQuantiles(
        m_NumberThreads, frame, trainingRowMask, features,
        CQuantileSketch{CQuantileSketch::E_Linear,
                        std::max(2 * m_NumberSplitsPerFeature, std::size_t{50})},
        featureQuantiles, m_Encoder.get(), readLossCurvature);

    TDoubleVecVec candidateSplits(this->numberFeatures());

    for (auto i : binaryFeatures) {
        candidateSplits[i] = TDoubleVec{0.5};
        LOG_TRACE(<< "feature '" << i << "' splits = "
                  << core::CContainerPrinter::print(candidateSplits[i]));
    }
    for (std::size_t i = 0; i < features.size(); ++i) {

        TDoubleVec featureSplits;
        featureSplits.reserve(m_NumberSplitsPerFeature - 1);

        for (std::size_t j = 1; j < m_NumberSplitsPerFeature; ++j) {
            double rank{100.0 * static_cast<double>(j) /
                        static_cast<double>(m_NumberSplitsPerFeature)};
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
        candidateSplits[features[i]] = std::move(featureSplits);

        LOG_TRACE(<< "feature '" << features[i] << "' splits = "
                  << core::CContainerPrinter::print(candidateSplits[features[i]]));
    }

    LOG_TRACE(<< "candidate splits = " << core::CContainerPrinter::print(candidateSplits));

    return candidateSplits;
}

CBoostedTreeImpl::TNodeVec
CBoostedTreeImpl::trainTree(core::CDataFrame& frame,
                            const core::CPackedBitVector& trainingRowMask,
                            const TDoubleVecVec& candidateSplits,
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
        candidateSplits, 0 /*depth*/, this->featureBag(), trainingRowMask));

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

        bool assignMissingToLeft{leaf->assignMissingToLeft()};

        std::size_t leftChildId, rightChildId;
        std::tie(leftChildId, rightChildId) =
            tree[leaf->id()].split(splitFeature, splitValue, assignMissingToLeft,
                                   leaf->gain(), leaf->curvature(), tree);

        TSizeVec featureBag{this->featureBag()};

        core::CPackedBitVector leftChildRowMask;
        core::CPackedBitVector rightChildRowMask;
        bool leftChildHasFewerRows;
        std::tie(leftChildRowMask, rightChildRowMask, leftChildHasFewerRows) =
            tree[leaf->id()].childrenRowMasks(m_NumberThreads, frame, *m_Encoder,
                                              std::move(leaf->rowMask()));

        TLeafNodeStatisticsPtr leftChild;
        TLeafNodeStatisticsPtr rightChild;
        std::tie(leftChild, rightChild) = leaf->split(
            leftChildId, rightChildId, m_NumberThreads, frame, *m_Encoder, m_Regularization,
            candidateSplits, std::move(featureBag), std::move(leftChildRowMask),
            std::move(rightChildRowMask), leftChildHasFewerRows);

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
                    for (auto itr = beginRows; itr != endRows; ++itr) {
                        const TRowRef& row{*itr};
                        double prediction{readPrediction(row)};
                        double actual{readActual(row, m_DependentVariable)};
                        leafValues_[root(tree).leafIndex(m_Encoder->encode(row), tree)]
                            .add(prediction, actual);
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

    auto results = frame.writeColumns(
        m_NumberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](double& loss, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    double prediction{readPrediction(*row) +
                                      root(tree).value(m_Encoder->encode(*row), tree)};
                    double actual{readActual(*row, m_DependentVariable)};

                    std::size_t numberColumns{row->numberColumns()};
                    row->writeColumn(predictionColumn(numberColumns), prediction);
                    row->writeColumn(lossGradientColumn(numberColumns),
                                     m_Loss->gradient(prediction, actual));
                    row->writeColumn(lossCurvatureColumn(numberColumns),
                                     m_Loss->curvature(prediction, actual));

                    loss += m_Loss->value(prediction, actual);
                }
            },
            0.0 /*total loss*/),
        &trainingRowMask);

    double totalLoss{0.0};
    for (const auto& loss : results.first) {
        totalLoss += loss.s_FunctionState;
    }
    LOG_TRACE(<< "training set loss = " << totalLoss);
}

double CBoostedTreeImpl::meanLoss(const core::CDataFrame& frame,
                                  const core::CPackedBitVector& rowMask,
                                  const TNodeVecVec& forest) const {

    auto results = frame.readRows(
        m_NumberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](TMeanAccumulator& loss, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    double prediction{predictRow(m_Encoder->encode(*row), forest)};
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
              << ": regularization = " << m_Regularization.print() << ", eta = " << m_Eta
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
        parameters = bopt.maximumExpectedImprovement();
    }

    // Write parameters for next round.
    i = 0;
    if (m_RegularizationOverride.depthPenaltyMultiplier() == boost::none) {
        m_Regularization.depthPenaltyMultiplier(std::exp(parameters(i++)));
    }
    if (m_RegularizationOverride.leafWeightPenaltyMultiplier() == boost::none) {
        m_Regularization.leafWeightPenaltyMultiplier(std::exp(parameters(i++)));
    }
    if (m_RegularizationOverride.treeSizePenaltyMultiplier() == boost::none) {
        m_Regularization.treeSizePenaltyMultiplier(std::exp(parameters(i++)));
    }
    if (m_RegularizationOverride.softTreeDepthLimit() == boost::none) {
        m_Regularization.softTreeDepthLimit(parameters(i++));
    }
    if (m_RegularizationOverride.softTreeDepthTolerance() == boost::none) {
        m_Regularization.softTreeDepthTolerance(parameters(i++));
    }
    if (m_EtaOverride == boost::none) {
        m_Eta = std::exp(parameters(i++));
        m_EtaGrowthRatePerTree = parameters(i++);
    }
    if (m_FeatureBagFractionOverride == boost::none) {
        m_FeatureBagFraction = parameters(i++);
    }

    return true;
}

void CBoostedTreeImpl::captureBestHyperparameters(const TMeanVarAccumulator& lossMoments) {
    // We capture the parameters with the lowest error at one standard
    // deviation above the mean. If the mean error improvement is marginal
    // we prefer the solution with the least variation across the folds.
    double loss{CBasicStatistics::mean(lossMoments) +
                std::sqrt(CBasicStatistics::variance(lossMoments))};
    if (loss < m_BestForestTestLoss) {
        m_BestForestTestLoss = loss;
        m_BestHyperparameters = SHyperparameters{
            m_Regularization, m_Eta, m_EtaGrowthRatePerTree, m_FeatureBagFraction};
    }
}

void CBoostedTreeImpl::restoreBestHyperparameters() {
    m_Regularization = m_BestHyperparameters.s_Regularization;
    m_Eta = m_BestHyperparameters.s_Eta;
    m_EtaGrowthRatePerTree = m_BestHyperparameters.s_EtaGrowthRatePerTree;
    m_FeatureBagFraction = m_BestHyperparameters.s_FeatureBagFraction;
    LOG_TRACE(<< "regularization* = " << m_Regularization.print() << ", eta* = " << m_Eta
              << ", eta growth rate per tree* = " << m_EtaGrowthRatePerTree
              << ", feature bag fraction* = " << m_FeatureBagFraction);
}

std::size_t CBoostedTreeImpl::numberHyperparametersToTune() const {
    return m_RegularizationOverride.countNotSet() +
           (m_EtaOverride != boost::none ? 0 : 2) +
           (m_FeatureBagFractionOverride != boost::none ? 0 : 1);
}

std::size_t CBoostedTreeImpl::maximumTreeSize(const core::CPackedBitVector& trainingRowMask) const {
    return this->maximumTreeSize(static_cast<std::size_t>(trainingRowMask.manhattan()));
}

std::size_t CBoostedTreeImpl::maximumTreeSize(std::size_t numberRows) const {
    return static_cast<std::size_t>(std::ceil(
        m_MaximumTreeSizeMultiplier * std::sqrt(static_cast<double>(numberRows))));
}

const std::size_t CBoostedTreeImpl::PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE{256};

namespace {
const std::string VERSION_7_5_TAG{"7.5"};

const std::string BAYESIAN_OPTIMIZATION_TAG{"bayesian_optimization"};
const std::string BEST_FOREST_TAG{"best_forest"};
const std::string BEST_FOREST_TEST_LOSS_TAG{"best_forest_test_loss"};
const std::string BEST_HYPERPARAMETERS_TAG{"best_hyperparameters"};
const std::string CURRENT_ROUND_TAG{"current_round"};
const std::string DEPENDENT_VARIABLE_TAG{"dependent_variable"};
const std::string ENCODER_TAG{"encoder"};
const std::string ETA_GROWTH_RATE_PER_TREE_TAG{"eta_growth_rate_per_tree"};
const std::string ETA_OVERRIDE_TAG{"eta_override"};
const std::string ETA_TAG{"eta"};
const std::string FEATURE_BAG_FRACTION_OVERRIDE_TAG{"feature_bag_fraction_override"};
const std::string FEATURE_BAG_FRACTION_TAG{"feature_bag_fraction"};
const std::string FEATURE_DATA_TYPES_TAG{"feature_data_types"};
const std::string FEATURE_SAMPLE_PROBABILITIES_TAG{"feature_sample_probabilities"};
const std::string LOSS_TAG{"loss"};
const std::string MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG{"maximum_attempts_to_add_tree"};
const std::string MAXIMUM_NUMBER_TREES_OVERRIDE_TAG{"maximum_number_trees_override"};
const std::string MAXIMUM_NUMBER_TREES_TAG{"maximum_number_trees"};
const std::string MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG{
    "maximum_optimisation_rounds_per_hyperparameter"};
const std::string MAXIMUM_TREE_SIZE_MULTIPLIER_TAG{"maximum_tree_size_multiplier"};
const std::string MISSING_FEATURE_ROW_MASKS_TAG{"missing_feature_row_masks"};
const std::string NUMBER_FOLDS_TAG{"number_folds"};
const std::string NUMBER_ROUNDS_TAG{"number_rounds"};
const std::string NUMBER_SPLITS_PER_FEATURE_TAG{"number_splits_per_feature"};
const std::string NUMBER_THREADS_TAG{"number_threads"};
const std::string RANDOM_NUMBER_GENERATOR_TAG{"random_number_generator"};
const std::string REGULARIZATION_TAG{"regularization"};
const std::string REGULARIZATION_OVERRIDE_TAG{"regularization_override"};
const std::string ROWS_PER_FEATURE_TAG{"rows_per_feature"};
const std::string TESTING_ROW_MASKS_TAG{"testing_row_masks"};
const std::string TRAINING_ROW_MASKS_TAG{"training_row_masks"};
const std::string TRAINING_PROGRESS_TAG{"training_progress"};

const std::string REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG{"regularization_depth_penalty_multiplier"};
const std::string REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG{
    "regularization_tree_size_penalty_multiplier"};
const std::string REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG{
    "regularization_leaf_weight_penalty_multiplier"};
const std::string REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG{"regularization_soft_tree_depth_limit"};
const std::string REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG{
    "regularization_soft_tree_depth_tolerance"};

const std::string HYPERPARAM_ETA_TAG{"hyperparam_eta"};
const std::string HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG{"hyperparam_eta_growth_rate_per_tree"};
const std::string HYPERPARAM_FEATURE_BAG_FRACTION_TAG{"hyperparam_feature_bag_fraction"};
const std::string HYPERPARAM_REGULARIZATION_TAG{"hyperparam_regularization"};
}

const std::string& CBoostedTreeImpl::bestHyperparametersName() {
    return BEST_HYPERPARAMETERS_TAG;
}

const std::string& CBoostedTreeImpl::bestRegularizationHyperparametersName() {
    return HYPERPARAM_REGULARIZATION_TAG;
}

CBoostedTreeImpl::TStrVec CBoostedTreeImpl::bestHyperparameterNames() {
    return {HYPERPARAM_ETA_TAG,
            HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
            HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
            REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG,
            REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG,
            REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
            REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG,
            REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG};
}

template<typename T>
void CBoostedTreeImpl::CRegularization<T>::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG,
                                 m_DepthPenaltyMultiplier, inserter);
    core::CPersistUtils::persist(REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG,
                                 m_TreeSizePenaltyMultiplier, inserter);
    core::CPersistUtils::persist(REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
                                 m_LeafWeightPenaltyMultiplier, inserter);
    core::CPersistUtils::persist(REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG,
                                 m_SoftTreeDepthLimit, inserter);
    core::CPersistUtils::persist(REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG,
                                 m_SoftTreeDepthTolerance, inserter);
}

void CBoostedTreeImpl::SHyperparameters::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(HYPERPARAM_ETA_TAG, s_Eta, inserter);
    core::CPersistUtils::persist(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                                 s_EtaGrowthRatePerTree, inserter);
    core::CPersistUtils::persist(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                                 s_FeatureBagFraction, inserter);
    core::CPersistUtils::persist(HYPERPARAM_REGULARIZATION_TAG, s_Regularization, inserter);
}

void CBoostedTreeImpl::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(VERSION_7_5_TAG, "", inserter);
    core::CPersistUtils::persist(BAYESIAN_OPTIMIZATION_TAG, *m_BayesianOptimization, inserter);
    core::CPersistUtils::persist(BEST_FOREST_TEST_LOSS_TAG, m_BestForestTestLoss, inserter);
    core::CPersistUtils::persist(CURRENT_ROUND_TAG, m_CurrentRound, inserter);
    core::CPersistUtils::persist(DEPENDENT_VARIABLE_TAG, m_DependentVariable, inserter);
    core::CPersistUtils::persist(ENCODER_TAG, *m_Encoder, inserter);
    core::CPersistUtils::persist(ETA_GROWTH_RATE_PER_TREE_TAG,
                                 m_EtaGrowthRatePerTree, inserter);
    core::CPersistUtils::persist(ETA_TAG, m_Eta, inserter);
    core::CPersistUtils::persist(FEATURE_BAG_FRACTION_TAG, m_FeatureBagFraction, inserter);
    core::CPersistUtils::persist(FEATURE_DATA_TYPES_TAG, m_FeatureDataTypes, inserter);
    core::CPersistUtils::persist(FEATURE_SAMPLE_PROBABILITIES_TAG,
                                 m_FeatureSampleProbabilities, inserter);
    core::CPersistUtils::persist(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                                 m_MaximumAttemptsToAddTree, inserter);
    core::CPersistUtils::persist(MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                                 m_MaximumOptimisationRoundsPerHyperparameter, inserter);
    core::CPersistUtils::persist(MAXIMUM_TREE_SIZE_MULTIPLIER_TAG,
                                 m_MaximumTreeSizeMultiplier, inserter);
    core::CPersistUtils::persist(MISSING_FEATURE_ROW_MASKS_TAG,
                                 m_MissingFeatureRowMasks, inserter);
    core::CPersistUtils::persist(NUMBER_FOLDS_TAG, m_NumberFolds, inserter);
    core::CPersistUtils::persist(NUMBER_ROUNDS_TAG, m_NumberRounds, inserter);
    core::CPersistUtils::persist(NUMBER_SPLITS_PER_FEATURE_TAG,
                                 m_NumberSplitsPerFeature, inserter);
    core::CPersistUtils::persist(NUMBER_THREADS_TAG, m_NumberThreads, inserter);
    inserter.insertValue(RANDOM_NUMBER_GENERATOR_TAG, m_Rng.toString());
    core::CPersistUtils::persist(REGULARIZATION_OVERRIDE_TAG,
                                 m_RegularizationOverride, inserter);
    core::CPersistUtils::persist(REGULARIZATION_TAG, m_Regularization, inserter);
    core::CPersistUtils::persist(ROWS_PER_FEATURE_TAG, m_RowsPerFeature, inserter);
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
}

template<typename T>
bool CBoostedTreeImpl::CRegularization<T>::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG,
                core::CPersistUtils::restore(REGULARIZATION_DEPTH_PENALTY_MULTIPLIER_TAG,
                                             m_DepthPenaltyMultiplier, traverser))
        RESTORE(REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG,
                core::CPersistUtils::restore(REGULARIZATION_TREE_SIZE_PENALTY_MULTIPLIER_TAG,
                                             m_TreeSizePenaltyMultiplier, traverser))
        RESTORE(REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
                core::CPersistUtils::restore(REGULARIZATION_LEAF_WEIGHT_PENALTY_MULTIPLIER_TAG,
                                             m_LeafWeightPenaltyMultiplier, traverser))
        RESTORE(REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG,
                core::CPersistUtils::restore(REGULARIZATION_SOFT_TREE_DEPTH_LIMIT_TAG,
                                             m_SoftTreeDepthLimit, traverser))
        RESTORE(REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG,
                core::CPersistUtils::restore(REGULARIZATION_SOFT_TREE_DEPTH_TOLERANCE_TAG,
                                             m_SoftTreeDepthTolerance, traverser))
    } while (traverser.next());
    return true;
}

bool CBoostedTreeImpl::SHyperparameters::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(HYPERPARAM_ETA_TAG,
                core::CPersistUtils::restore(HYPERPARAM_ETA_TAG, s_Eta, traverser))
        RESTORE(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                core::CPersistUtils::restore(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                                             s_EtaGrowthRatePerTree, traverser))
        RESTORE(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                core::CPersistUtils::restore(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                                             s_FeatureBagFraction, traverser))
        RESTORE(HYPERPARAM_REGULARIZATION_TAG,
                core::CPersistUtils::restore(HYPERPARAM_REGULARIZATION_TAG,
                                             s_Regularization, traverser))
    } while (traverser.next());
    return true;
}

bool CBoostedTreeImpl::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    if (traverser.name() == VERSION_7_5_TAG) {
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
            RESTORE(MAXIMUM_TREE_SIZE_MULTIPLIER_TAG,
                    core::CPersistUtils::restore(MAXIMUM_TREE_SIZE_MULTIPLIER_TAG,
                                                 m_MaximumTreeSizeMultiplier, traverser))
            RESTORE(MISSING_FEATURE_ROW_MASKS_TAG,
                    core::CPersistUtils::restore(MISSING_FEATURE_ROW_MASKS_TAG,
                                                 m_MissingFeatureRowMasks, traverser))
            RESTORE(NUMBER_FOLDS_TAG,
                    core::CPersistUtils::restore(NUMBER_FOLDS_TAG, m_NumberFolds, traverser))
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
            RESTORE(TESTING_ROW_MASKS_TAG,
                    core::CPersistUtils::restore(TESTING_ROW_MASKS_TAG,
                                                 m_TestingRowMasks, traverser))
            RESTORE(MAXIMUM_NUMBER_TREES_TAG,
                    core::CPersistUtils::restore(MAXIMUM_NUMBER_TREES_TAG,
                                                 m_MaximumNumberTrees, traverser))
            RESTORE(TRAINING_ROW_MASKS_TAG,
                    core::CPersistUtils::restore(TRAINING_ROW_MASKS_TAG,
                                                 m_TrainingRowMasks, traverser))
            RESTORE(TRAINING_PROGRESS_TAG,
                    core::CPersistUtils::restore(TRAINING_PROGRESS_TAG,
                                                 m_TrainingProgress, traverser))
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
        } while (traverser.next());
        return true;
    }
    LOG_ERROR(<< "Input error: unsupported state serialization version. Currently supported version: "
              << VERSION_7_5_TAG);
    return false;
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
    mem += core::CMemory::dynamicSize(m_BestForest);
    mem += core::CMemory::dynamicSize(m_BayesianOptimization);
    return mem;
}

const double CBoostedTreeImpl::MINIMUM_RELATIVE_GAIN_PER_SPLIT{1e-7};
const double CBoostedTreeImpl::INF{std::numeric_limits<double>::max()};
}
}
