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
}

void CBoostedTreeImpl::CLeafNodeStatistics::addRowDerivatives(const CEncodedDataFrameRowRef& row,
                                                              SDerivatives& derivatives) const {

    const TRowRef& unencodedRow{row.unencodedRow()};

    for (std::size_t i = 0; i < m_CandidateSplits.size(); ++i) {
        double featureValue{row[i]};
        if (CDataFrameUtils::isMissing(featureValue)) {
            derivatives.s_MissingGradients[i] += readLossGradient(unencodedRow);
            derivatives.s_MissingCurvatures[i] += readLossCurvature(unencodedRow);
        } else {
            auto j = std::upper_bound(m_CandidateSplits[i].begin(),
                                      m_CandidateSplits[i].end(), featureValue) -
                     m_CandidateSplits[i].begin();
            derivatives.s_Gradients[i][j] += readLossGradient(unencodedRow);
            derivatives.s_Curvatures[i][j] += readLossCurvature(unencodedRow);
        }
    }
}

CBoostedTreeImpl::CBoostedTreeImpl(std::size_t numberThreads, CBoostedTree::TLossFunctionUPtr loss)
    : m_NumberThreads{numberThreads}, m_Loss{std::move(loss)},
      m_BestHyperparameters{m_Lambda, m_Gamma, m_Eta, m_EtaGrowthRatePerTree, m_FeatureBagFraction, m_FeatureSampleProbabilities} {
}

CBoostedTreeImpl::CBoostedTreeImpl() = default;

CBoostedTreeImpl::~CBoostedTreeImpl() = default;

CBoostedTreeImpl& CBoostedTreeImpl::operator=(CBoostedTreeImpl&&) = default;

void CBoostedTreeImpl::train(core::CDataFrame& frame,
                             const TProgressCallback& recordProgress,
                             const TMemoryUsageCallback& recordMemoryUsage) {

    if (m_DependentVariable >= frame.numberColumns()) {
        HANDLE_FATAL(<< "Internal error: dependent variable '" << m_DependentVariable
                     << "' was incorrectly initialized. Please report this problem.");
        return;
    }

    LOG_TRACE(<< "Main training loop...");

    // We account for cost of setup as one round. The main optimisation loop runs
    // for "m_NumberRounds + 1" rounds and training on the choosen hyperparameter
    // values is counted as one round. This gives a total of m_NumberRounds + 3.
    core::CLoopProgress progress{m_NumberRounds + 3 - m_CurrentRound, recordProgress};
    progress.increment();

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

        do {
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

            progress.increment();

            std::int64_t memoryUsage(this->memoryUsage());
            recordMemoryUsage(memoryUsage - lastMemoryUsage);
            lastMemoryUsage = memoryUsage;

        } while (m_CurrentRound++ < m_NumberRounds);

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

std::size_t CBoostedTreeImpl::columnHoldingDependentVariable() const {
    return m_DependentVariable;
}

std::size_t CBoostedTreeImpl::numberExtraColumnsForTrain() {
    return 3;
}

std::size_t CBoostedTreeImpl::estimateMemoryUsage(std::size_t numberRows,
                                                  std::size_t numberColumns) const {
    // The maximum tree size is defined is the maximum number of leaves minus one.
    // A binary tree with n + 1 leaves has 2n + 1 nodes in total.
    std::size_t maximumNumberNodes{2 * this->maximumTreeSize(numberRows) + 1};
    std::size_t forestMemoryUsage{
        m_MaximumNumberTrees * (sizeof(TNodeVec) + maximumNumberNodes * sizeof(CNode))};
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

CBoostedTreeImpl::TDoubleDoubleDoubleTr
CBoostedTreeImpl::regularisedLoss(const core::CDataFrame& frame,
                                  const core::CPackedBitVector& trainingRowMask,
                                  const TNodeVecVec& forest) const {

    auto results = frame.readRows(
        m_NumberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](double& loss, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    loss += m_Loss->value(readPrediction(*row),
                                          readActual(*row, m_DependentVariable));
                }
            },
            0.0),
        &trainingRowMask);

    double loss{0.0};
    for (const auto& result : results.first) {
        loss += result.s_FunctionState;
    }

    double leafCount{0.0};
    double sumSquareLeafWeights{0.0};
    for (const auto& tree : forest) {
        for (const auto& node : tree) {
            if (node.isLeaf()) {
                leafCount += 1.0;
                sumSquareLeafWeights += CTools::pow2(node.value());
            }
        }
    }

    return {loss, leafCount, 0.5 * sumSquareLeafWeights};
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
        auto tree = this->trainTree(frame, trainingRowMask, candidateSplits, recordMemoryUsage);

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
                            const TMemoryUsageCallback& recordMemoryUsage) const {

    LOG_TRACE(<< "Training one tree...");

    using TLeafNodeStatisticsPtr = std::shared_ptr<CLeafNodeStatistics>;
    using TLeafNodeStatisticsPtrQueue =
        std::priority_queue<TLeafNodeStatisticsPtr, std::vector<TLeafNodeStatisticsPtr>, COrderings::SLess>;

    std::size_t maximumTreeSize{this->maximumTreeSize(frame)};

    TNodeVec tree(1);
    tree.reserve(2 * maximumTreeSize + 1);

    TLeafNodeStatisticsPtrQueue leaves;
    leaves.push(std::make_shared<CLeafNodeStatistics>(
        0 /*root*/, m_NumberThreads, frame, *m_Encoder, m_Lambda, m_Gamma,
        candidateSplits, this->featureBag(), trainingRowMask));

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
        std::tie(leftChildId, rightChildId) = tree[leaf->id()].split(
            splitFeature, splitValue, assignMissingToLeft, tree);

        TSizeVec featureBag{this->featureBag()};

        core::CPackedBitVector leftChildRowMask;
        core::CPackedBitVector rightChildRowMask;
        std::tie(leftChildRowMask, rightChildRowMask) = tree[leaf->id()].rowMasks(
            m_NumberThreads, frame, *m_Encoder, std::move(leaf->rowMask()));

        TLeafNodeStatisticsPtr leftChild;
        TLeafNodeStatisticsPtr rightChild;
        std::tie(leftChild, rightChild) = leaf->split(
            leftChildId, rightChildId, m_NumberThreads, frame, *m_Encoder,
            m_Lambda, m_Gamma, candidateSplits, std::move(featureBag),
            std::move(leftChildRowMask), std::move(rightChildRowMask));

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
    return m_Encoder->numberFeatures();
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

    using TArgMinLossUPtrVec = std::vector<CLoss::TArgMinLossUPtr>;

    TArgMinLossUPtrVec leafValues;
    leafValues.reserve(tree.size());
    for (std::size_t i = 0; i < tree.size(); ++i) {
        leafValues.push_back(m_Loss->minimizer());
    }

    frame.readRows(
        1, 0, frame.numberRows(),
        [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                double prediction{readPrediction(*row)};
                double actual{readActual(*row, m_DependentVariable)};
                leafValues[root(tree).leafIndex(m_Encoder->encode(*row), tree)]->add(
                    prediction, actual);
            }
        },
        &trainingRowMask);

    for (std::size_t i = 0; i < tree.size(); ++i) {
        tree[i].value(eta * leafValues[i]->value());
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
    for (const auto& result : results.first) {
        totalLoss += result.s_FunctionState;
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

const CBoostedTreeImpl::CNode& CBoostedTreeImpl::root(const TNodeVec& tree) {
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
    if (m_LambdaOverride == boost::none) {
        parameters(i++) = std::log(m_Lambda);
    }
    if (m_GammaOverride == boost::none) {
        parameters(i++) = std::log(m_Gamma);
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
    if (m_LambdaOverride == boost::none) {
        m_Lambda = std::exp(parameters(i++));
    }
    if (m_GammaOverride == boost::none) {
        m_Gamma = std::exp(parameters(i++));
    }
    if (m_EtaOverride == boost::none) {
        m_Eta = std::exp(parameters(i++));
        m_EtaGrowthRatePerTree = parameters(i++);
    }
    if (m_FeatureBagFractionOverride == boost::none) {
        m_FeatureBagFraction = parameters(i++);
    }

    LOG_TRACE(<< "lambda = " << m_Lambda << ", gamma = " << m_Gamma << ", eta = " << m_Eta
              << ", eta growth rate per tree = " << m_EtaGrowthRatePerTree
              << ", feature bag fraction = " << m_FeatureBagFraction);
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
            m_Lambda, m_Gamma, m_Eta, m_EtaGrowthRatePerTree, m_FeatureBagFraction, m_FeatureSampleProbabilities};
    }
}

void CBoostedTreeImpl::restoreBestHyperparameters() {
    m_Lambda = m_BestHyperparameters.s_Lambda;
    m_Gamma = m_BestHyperparameters.s_Gamma;
    m_Eta = m_BestHyperparameters.s_Eta;
    m_EtaGrowthRatePerTree = m_BestHyperparameters.s_EtaGrowthRatePerTree;
    m_FeatureBagFraction = m_BestHyperparameters.s_FeatureBagFraction;
    m_FeatureSampleProbabilities = m_BestHyperparameters.s_FeatureSampleProbabilities;
    LOG_TRACE(<< "lambda* = " << m_Lambda << ", gamma* = " << m_Gamma
              << ", eta* = " << m_Eta << ", eta growth rate per tree* = " << m_EtaGrowthRatePerTree
              << ", feature bag fraction* = " << m_FeatureBagFraction);
}

std::size_t CBoostedTreeImpl::numberHyperparametersToTune() const {
    return (m_LambdaOverride ? 0 : 1) + (m_GammaOverride ? 0 : 1) +
           (m_EtaOverride ? 0 : 2) + (m_FeatureBagFractionOverride ? 0 : 1);
}

std::size_t CBoostedTreeImpl::maximumTreeSize(const core::CDataFrame& frame) const {
    return this->maximumTreeSize(frame.numberRows());
}

std::size_t CBoostedTreeImpl::maximumTreeSize(std::size_t numberRows) const {
    return static_cast<std::size_t>(std::ceil(
        m_MaximumTreeSizeMultiplier * std::sqrt(static_cast<double>(numberRows))));
}

const std::size_t CBoostedTreeImpl::PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE{256};

namespace {
const std::string BAYESIAN_OPTIMIZATION_TAG{"bayesian_optimization"};
const std::string BEST_FOREST_TAG{"best_forest"};
const std::string BEST_FOREST_TEST_LOSS_TAG{"best_forest_test_loss"};
const std::string BEST_HYPERPARAMETERS_TAG{"best_hyperparameters"};
const std::string CURRENT_ROUND_TAG{"current_round"};
const std::string DEPENDENT_VARIABLE_TAG{"dependent_variable"};
const std::string ENCODER_TAG{"encoder_tag"};
const std::string ETA_GROWTH_RATE_PER_TREE_TAG{"eta_growth_rate_per_tree"};
const std::string ETA_OVERRIDE_TAG{"eta_override"};
const std::string ETA_TAG{"eta"};
const std::string FEATURE_BAG_FRACTION_OVERRIDE_TAG{"feature_bag_fraction_override"};
const std::string FEATURE_BAG_FRACTION_TAG{"feature_bag_fraction"};
const std::string FEATURE_DATA_TYPES_TAG{"feature_data_types"};
const std::string FEATURE_SAMPLE_PROBABILITIES_TAG{"feature_sample_probabilities"};
const std::string GAMMA_OVERRIDE_TAG{"gamma_override"};
const std::string GAMMA_TAG{"gamma"};
const std::string LAMBDA_OVERRIDE_TAG{"lambda_override"};
const std::string LAMBDA_TAG{"lambda"};
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
const std::string ROWS_PER_FEATURE_TAG{"rows_per_feature"};
const std::string TESTING_ROW_MASKS_TAG{"testing_row_masks"};
const std::string TRAINING_ROW_MASKS_TAG{"training_row_masks"};

const std::string LEFT_CHILD_TAG{"left_child"};
const std::string RIGHT_CHILD_TAG{"right_child"};
const std::string SPLIT_FEATURE_TAG{"split_feature"};
const std::string ASSIGN_MISSING_TO_LEFT_TAG{"assign_missing_to_left "};
const std::string NODE_VALUE_TAG{"node_value"};
const std::string SPLIT_VALUE_TAG{"split_value"};

const std::string HYPERPARAM_LAMBDA_TAG{"hyperparam_lambda"};
const std::string HYPERPARAM_GAMMA_TAG{"hyperparam_gamma"};
const std::string HYPERPARAM_ETA_TAG{"hyperparam_eta"};
const std::string HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG{"hyperparam_eta_growth_rate_per_tree"};
const std::string HYPERPARAM_FEATURE_BAG_FRACTION_TAG{"hyperparam_feature_bag_fraction"};
const std::string HYPERPARAM_FEATURE_SAMPLE_PROBABILITIES_TAG{"hyperparam_feature_sample_probabilities"};
}

void CBoostedTreeImpl::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
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
    core::CPersistUtils::persist(GAMMA_TAG, m_Gamma, inserter);
    core::CPersistUtils::persist(LAMBDA_TAG, m_Lambda, inserter);
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
    core::CPersistUtils::persist(ROWS_PER_FEATURE_TAG, m_RowsPerFeature, inserter);
    core::CPersistUtils::persist(TESTING_ROW_MASKS_TAG, m_TestingRowMasks, inserter);
    core::CPersistUtils::persist(MAXIMUM_NUMBER_TREES_TAG, m_MaximumNumberTrees, inserter);
    core::CPersistUtils::persist(TRAINING_ROW_MASKS_TAG, m_TrainingRowMasks, inserter);
    core::CPersistUtils::persist(BEST_FOREST_TAG, m_BestForest, inserter);
    core::CPersistUtils::persist(BEST_HYPERPARAMETERS_TAG, m_BestHyperparameters, inserter);
    core::CPersistUtils::persist(ETA_OVERRIDE_TAG, m_EtaOverride, inserter);
    core::CPersistUtils::persist(FEATURE_BAG_FRACTION_OVERRIDE_TAG,
                                 m_FeatureBagFractionOverride, inserter);
    core::CPersistUtils::persist(GAMMA_OVERRIDE_TAG, m_GammaOverride, inserter);
    core::CPersistUtils::persist(LAMBDA_OVERRIDE_TAG, m_LambdaOverride, inserter);
    core::CPersistUtils::persist(MAXIMUM_NUMBER_TREES_OVERRIDE_TAG,
                                 m_MaximumNumberTreesOverride, inserter);
    inserter.insertValue(LOSS_TAG, m_Loss->name());
}

void CBoostedTreeImpl::CNode::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(LEFT_CHILD_TAG, m_LeftChild, inserter);
    core::CPersistUtils::persist(RIGHT_CHILD_TAG, m_RightChild, inserter);
    core::CPersistUtils::persist(SPLIT_FEATURE_TAG, m_SplitFeature, inserter);
    core::CPersistUtils::persist(ASSIGN_MISSING_TO_LEFT_TAG, m_AssignMissingToLeft, inserter);
    core::CPersistUtils::persist(NODE_VALUE_TAG, m_NodeValue, inserter);
    core::CPersistUtils::persist(SPLIT_VALUE_TAG, m_SplitValue, inserter);
}

void CBoostedTreeImpl::SHyperparameters::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(HYPERPARAM_LAMBDA_TAG, s_Lambda, inserter);
    core::CPersistUtils::persist(HYPERPARAM_GAMMA_TAG, s_Gamma, inserter);
    core::CPersistUtils::persist(HYPERPARAM_ETA_TAG, s_Eta, inserter);
    core::CPersistUtils::persist(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                                 s_EtaGrowthRatePerTree, inserter);
    core::CPersistUtils::persist(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                                 s_FeatureBagFraction, inserter);
    core::CPersistUtils::persist(HYPERPARAM_FEATURE_SAMPLE_PROBABILITIES_TAG,
                                 s_FeatureSampleProbabilities, inserter);
}

bool CBoostedTreeImpl::SHyperparameters::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(HYPERPARAM_LAMBDA_TAG,
                core::CPersistUtils::restore(HYPERPARAM_LAMBDA_TAG, s_Lambda, traverser))
        RESTORE(HYPERPARAM_GAMMA_TAG,
                core::CPersistUtils::restore(HYPERPARAM_GAMMA_TAG, s_Gamma, traverser))
        RESTORE(HYPERPARAM_ETA_TAG,
                core::CPersistUtils::restore(HYPERPARAM_ETA_TAG, s_Eta, traverser))
        RESTORE(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                core::CPersistUtils::restore(HYPERPARAM_ETA_GROWTH_RATE_PER_TREE_TAG,
                                             s_EtaGrowthRatePerTree, traverser))
        RESTORE(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                core::CPersistUtils::restore(HYPERPARAM_FEATURE_BAG_FRACTION_TAG,
                                             s_FeatureBagFraction, traverser))
        RESTORE(HYPERPARAM_FEATURE_SAMPLE_PROBABILITIES_TAG,
                core::CPersistUtils::restore(HYPERPARAM_FEATURE_SAMPLE_PROBABILITIES_TAG,
                                             s_FeatureSampleProbabilities, traverser))
    } while (traverser.next());
    return true;
}

bool CBoostedTreeImpl::CNode::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(LEFT_CHILD_TAG,
                core::CPersistUtils::restore(LEFT_CHILD_TAG, m_LeftChild, traverser))
        RESTORE(RIGHT_CHILD_TAG,
                core::CPersistUtils::restore(RIGHT_CHILD_TAG, m_RightChild, traverser))
        RESTORE(SPLIT_FEATURE_TAG,
                core::CPersistUtils::restore(SPLIT_FEATURE_TAG, m_SplitFeature, traverser))
        RESTORE(ASSIGN_MISSING_TO_LEFT_TAG,
                core::CPersistUtils::restore(ASSIGN_MISSING_TO_LEFT_TAG,
                                             m_AssignMissingToLeft, traverser))
        RESTORE(NODE_VALUE_TAG,
                core::CPersistUtils::restore(NODE_VALUE_TAG, m_NodeValue, traverser))
        RESTORE(SPLIT_VALUE_TAG,
                core::CPersistUtils::restore(SPLIT_VALUE_TAG, m_SplitValue, traverser))
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

bool CBoostedTreeImpl::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
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
        RESTORE(GAMMA_TAG, core::CPersistUtils::restore(GAMMA_TAG, m_Gamma, traverser))
        RESTORE(LAMBDA_TAG, core::CPersistUtils::restore(LAMBDA_TAG, m_Lambda, traverser))
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
        RESTORE(ROWS_PER_FEATURE_TAG,
                core::CPersistUtils::restore(ROWS_PER_FEATURE_TAG, m_RowsPerFeature, traverser))
        RESTORE(TESTING_ROW_MASKS_TAG,
                core::CPersistUtils::restore(TESTING_ROW_MASKS_TAG, m_TestingRowMasks, traverser))
        RESTORE(MAXIMUM_NUMBER_TREES_TAG,
                core::CPersistUtils::restore(MAXIMUM_NUMBER_TREES_TAG,
                                             m_MaximumNumberTrees, traverser))
        RESTORE(TRAINING_ROW_MASKS_TAG,
                core::CPersistUtils::restore(TRAINING_ROW_MASKS_TAG, m_TrainingRowMasks, traverser))
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
        RESTORE(GAMMA_OVERRIDE_TAG,
                core::CPersistUtils::restore(GAMMA_OVERRIDE_TAG, m_GammaOverride, traverser))
        RESTORE(LAMBDA_OVERRIDE_TAG,
                core::CPersistUtils::restore(LAMBDA_OVERRIDE_TAG, m_LambdaOverride, traverser))
        RESTORE(MAXIMUM_NUMBER_TREES_OVERRIDE_TAG,
                core::CPersistUtils::restore(MAXIMUM_NUMBER_TREES_OVERRIDE_TAG,
                                             m_MaximumNumberTreesOverride, traverser))
        RESTORE(LOSS_TAG, restoreLoss(m_Loss, traverser))
    } while (traverser.next());
    return true;
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
