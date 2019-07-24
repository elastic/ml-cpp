/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeImpl.h>

#include <maths/CQuantileSketch.h>
#include <maths/CSampling.h>

namespace ml {
namespace maths {
using namespace boosted_tree;
using namespace boosted_tree_detail;

namespace {
std::size_t lossGradientColumn(std::size_t numberColumns) {
    return numberColumns - 2;
}
std::size_t lossCurvatureColumn(std::size_t numberColumns) {
    return numberColumns - 1;
}
}

void CBoostedTreeImpl::CLeafNodeStatistics::addRowDerivatives(const TRowRef& row,
                                                              SDerivatives& derivatives) const {

    std::size_t numberColumns{row.numberColumns()};
    std::size_t gradientColumn{lossGradientColumn(numberColumns)};
    std::size_t curvatureColumn{lossCurvatureColumn(numberColumns)};

    for (std::size_t i = 0; i < m_CandidateSplits.size(); ++i) {
        double featureValue{row[i]};
        if (CDataFrameUtils::isMissing(featureValue)) {
            derivatives.s_MissingGradients[i] += row[gradientColumn];
            derivatives.s_MissingCurvatures[i] += row[curvatureColumn];
        } else {
            auto j = std::upper_bound(m_CandidateSplits[i].begin(),
                                      m_CandidateSplits[i].end(), featureValue) -
                     m_CandidateSplits[i].begin();
            derivatives.s_Gradients[i][j] += row[gradientColumn];
            derivatives.s_Curvatures[i][j] += row[curvatureColumn];
        }
    }
}

CBoostedTreeImpl::CBoostedTreeImpl(std::size_t numberThreads,
                                   std::size_t dependentVariable,
                                   CBoostedTree::TLossFunctionUPtr loss)
    : m_NumberThreads{numberThreads}, m_DependentVariable{dependentVariable}, m_Loss{std::move(loss)} {
}

void CBoostedTreeImpl::numberFolds(std::size_t numberFolds) {
    if (numberFolds < 2) {
        LOG_WARN(<< "Must use at least two-folds for cross validation");
        numberFolds = 2;
    }
    m_NumberFolds = numberFolds;
}

void CBoostedTreeImpl::lambda(double lambda) {
    if (lambda < 0.0) {
        LOG_WARN(<< "Lambda must be non-negative");
        lambda = 0.0;
    }
    m_LambdaOverride = lambda;
}

void CBoostedTreeImpl::gamma(double gamma) {
    if (gamma < 0.0) {
        LOG_WARN(<< "Gamma must be non-negative");
        gamma = 0.0;
    }
    m_GammaOverride = gamma;
}

void CBoostedTreeImpl::eta(double eta) {
    if (eta < MINIMUM_ETA) {
        LOG_WARN(<< "Truncating supplied learning rate " << eta
                 << " which must be no smaller than " << MINIMUM_ETA);
        eta = std::max(eta, MINIMUM_ETA);
    }
    if (eta > 1.0) {
        LOG_WARN(<< "Using a learning rate greater than one doesn't make sense");
        eta = 1.0;
    }
    m_EtaOverride = eta;
}

void CBoostedTreeImpl::maximumNumberTrees(std::size_t maximumNumberTrees) {
    if (maximumNumberTrees == 0) {
        LOG_WARN(<< "Forest must have at least one tree");
        maximumNumberTrees = 1;
    }
    if (maximumNumberTrees > MAXIMUM_NUMBER_TREES) {
        LOG_WARN(<< "Truncating supplied maximum number of trees " << maximumNumberTrees
                 << " which must be no larger than " << MAXIMUM_NUMBER_TREES);
        maximumNumberTrees = std::min(maximumNumberTrees, MAXIMUM_NUMBER_TREES);
    }
    m_MaximumNumberTreesOverride = maximumNumberTrees;
}

void CBoostedTreeImpl::featureBagFraction(double featureBagFraction) {
    if (featureBagFraction < 0.0 || featureBagFraction > 1.0) {
        LOG_WARN(<< "Truncating supplied feature bag fraction " << featureBagFraction
                 << " which must be positive and not more than one");
        featureBagFraction = CTools::truncate(featureBagFraction, 0.0, 1.0);
    }
    m_FeatureBagFractionOverride = featureBagFraction;
}

void CBoostedTreeImpl::maximumOptimisationRoundsPerHyperparameter(std::size_t rounds) {
    m_MaximumOptimisationRoundsPerHyperparameter = rounds;
}

void CBoostedTreeImpl::rowsPerFeature(std::size_t rowsPerFeature) {
    if (m_RowsPerFeature == 0) {
        LOG_WARN(<< "Must have at least one training example per feature");
        rowsPerFeature = 1;
    }
    m_RowsPerFeature = rowsPerFeature;
}

void CBoostedTreeImpl::train(core::CDataFrame& frame,
                             CBoostedTree::TProgressCallback recordProgress) {
    LOG_TRACE(<< "Main training loop...");

    do {
        LOG_TRACE(<< "Optimisation round = " << m_CurrentRound + 1);

        TMeanVarAccumulator lossMoments{this->crossValidateForest(
            frame, m_TrainingRowMasks, m_TestingRowMasks, recordProgress)};

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
    } while (m_CurrentRound++ < m_NumberRounds);

    LOG_TRACE(<< "Test loss = " << m_BestForestTestLoss);

    this->restoreBestHyperparameters();
    m_BestForest = this->trainForest(
        frame, core::CPackedBitVector{frame.numberRows(), true}, recordProgress);
}

void CBoostedTreeImpl::predict(core::CDataFrame& frame,
                               CBoostedTree::TProgressCallback /*recordProgress*/) const {
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
                                 predictRow(*row, m_BestForest));
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

std::size_t CBoostedTreeImpl::numberExtraColumnsForTrain() {
    return 3;
}

CBoostedTree::TDoubleVec CBoostedTreeImpl::featureWeights() const {
    return m_FeatureSampleProbabilities;
}

std::size_t CBoostedTreeImpl::estimateMemoryUsage(std::size_t numberRows,
                                                  std::size_t numberColumns) const {
    std::size_t maximumNumberNodes{this->maximumTreeSize(numberRows)};
    std::size_t forestMemoryUsage{m_MaximumNumberTrees * maximumNumberNodes * sizeof(CNode)};
    std::size_t extraColumnsMemoryUsage{this->numberExtraColumnsForTrain() *
                                        numberRows * sizeof(CFloatStorage)};
    std::size_t hyperparametersMemoryUsage{sizeof(SHyperparameters) +
                                           numberColumns * sizeof(double)};
    std::size_t leafNodeStatisticsMemoryUsage{
        maximumNumberNodes * CLeafNodeStatistics::estimateMemoryUsage(
                                 numberRows, numberColumns, m_FeatureBagFraction,
                                 m_NumberSplitsPerFeature)};
    return forestMemoryUsage + extraColumnsMemoryUsage +
           hyperparametersMemoryUsage + leafNodeStatisticsMemoryUsage;
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
                    std::size_t numberColumns{row->numberColumns()};
                    loss += m_Loss->value((*row)[predictionColumn(numberColumns)],
                                          (*row)[m_DependentVariable]);
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

    return {loss, leafCount, sumSquareLeafWeights};
}

CBoostedTreeImpl::TMeanVarAccumulator
CBoostedTreeImpl::crossValidateForest(core::CDataFrame& frame,
                                      const TPackedBitVectorVec& trainingRowMasks,
                                      const TPackedBitVectorVec& testingRowMasks,
                                      CBoostedTree::TProgressCallback recordProgress) const {
    TMeanVarAccumulator lossMoments;
    for (std::size_t i = 0; i < m_NumberFolds; ++i) {
        TNodeVecVec forest(this->trainForest(frame, trainingRowMasks[i], recordProgress));
        double loss{this->meanLoss(frame, testingRowMasks[i], forest)};
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

    TNodeVec tree(1);
    refreshPredictionsAndLossDerivatives(frame, trainingRowMask, m_Eta, tree);

    return tree;
}

CBoostedTreeImpl::TNodeVecVec
CBoostedTreeImpl::trainForest(core::CDataFrame& frame,
                              const core::CPackedBitVector& trainingRowMask,
                              CBoostedTree::TProgressCallback /*recordProgress*/) const {

    LOG_TRACE(<< "Training one forest...");

    TNodeVecVec forest{this->initializePredictionsAndLossDerivatives(frame, trainingRowMask)};
    forest.reserve(m_MaximumNumberTrees);

    // For each iteration:
    //  1. Compute weighted quantiles for features F
    //  2. Compute candidate split set S from quantiles of F
    //  3. Build one tree on (F, S)
    //  4. Update predictions and loss derivatives

    double eta{m_Eta};
    double sumEta{eta};

    for (std::size_t retries = 0; forest.size() < m_MaximumNumberTrees; /**/) {

        TDoubleVecVec candidateSplits(this->candidateSplits(frame, trainingRowMask));

        auto tree = this->trainTree(frame, trainingRowMask, candidateSplits);

        retries = tree.size() == 1 ? retries + 1 : 0;

        if (sumEta > 1.0 && retries == m_MaximumAttemptsToAddTree) {
            break;
        }

        if (sumEta < 1.0 || retries == 0) {
            this->refreshPredictionsAndLossDerivatives(frame, trainingRowMask, eta, tree);
            forest.push_back(std::move(tree));
            eta = std::min(1.0, m_EtaGrowthRatePerTree * eta);
            sumEta += eta;
            retries = 0;
        }
    }

    LOG_TRACE(<< "Trained one forest");

    return forest;
}

CBoostedTreeImpl::TDoubleVecVec
CBoostedTreeImpl::candidateSplits(const core::CDataFrame& frame,
                                  const core::CPackedBitVector& trainingRowMask) const {

    using TQuantileSketchVec = std::vector<CQuantileSketch>;

    TSizeVec features{this->candidateFeatures()};
    LOG_TRACE(<< "candidate features = " << core::CContainerPrinter::print(features));

    TQuantileSketchVec columnQuantiles;
    CDataFrameUtils::columnQuantiles(
        m_NumberThreads, frame, trainingRowMask, features,
        CQuantileSketch{CQuantileSketch::E_Linear, 100}, columnQuantiles,
        [](const TRowRef& row) {
            return row[lossCurvatureColumn(row.numberColumns())];
        });

    TDoubleVecVec result(numberFeatures(frame));

    for (std::size_t i = 0; i < features.size(); ++i) {

        TDoubleVec columnSplits;
        columnSplits.reserve(m_NumberSplitsPerFeature - 1);

        for (std::size_t j = 1; j < m_NumberSplitsPerFeature; ++j) {
            double rank{100.0 * static_cast<double>(j) /
                        static_cast<double>(m_NumberSplitsPerFeature)};
            double q;
            if (columnQuantiles[i].quantile(rank, q)) {
                columnSplits.push_back(q);
            } else {
                LOG_WARN(<< "Failed to compute quantile " << rank << ": ignoring split");
            }
        }

        columnSplits.erase(std::unique(columnSplits.begin(), columnSplits.end()),
                           columnSplits.end());
        result[features[i]] = std::move(columnSplits);

        LOG_TRACE(<< "feature '" << features[i] << "' splits = "
                  << core::CContainerPrinter::print(result[features[i]]));
    }

    return result;
}

CBoostedTreeImpl::TNodeVec
CBoostedTreeImpl::trainTree(core::CDataFrame& frame,
                            const core::CPackedBitVector& trainingRowMask,
                            const TDoubleVecVec& candidateSplits) const {

    // TODO improve categorical regressor treatment

    LOG_TRACE(<< "Training one tree...");

    using TLeafNodeStatisticsPtr = std::shared_ptr<CLeafNodeStatistics>;
    using TLeafNodeStatisticsPtrQueue =
        std::priority_queue<TLeafNodeStatisticsPtr, std::vector<TLeafNodeStatisticsPtr>, COrderings::SLess>;

    std::size_t maximumTreeSize{this->maximumTreeSize(frame)};

    TNodeVec tree(1);
    tree.reserve(maximumTreeSize);

    TLeafNodeStatisticsPtrQueue leaves;
    leaves.push(std::make_shared<CLeafNodeStatistics>(
        0 /*root*/, m_NumberThreads, frame, m_Lambda, m_Gamma, candidateSplits,
        this->featureBag(frame), trainingRowMask));

    // For each iteration we:
    //   1. Find the leaf with the greatest decrease in loss
    //   2. If no split (significantly) reduced the loss we terminate
    //   3. Otherwise we split that leaf

    double totalGain{0.0};

    for (std::size_t i = 0; i < maximumTreeSize; ++i) {

        auto leaf = leaves.top();
        leaves.pop();

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

        TSizeVec featureBag{this->featureBag(frame)};

        core::CPackedBitVector leftChildRowMask;
        core::CPackedBitVector rightChildRowMask;
        std::tie(leftChildRowMask, rightChildRowMask) = tree[leaf->id()].rowMasks(
            m_NumberThreads, frame, std::move(leaf->rowMask()));

        TLeafNodeStatisticsPtr leftChild;
        TLeafNodeStatisticsPtr rightChild;
        std::tie(leftChild, rightChild) =
            leaf->split(leftChildId, rightChildId, m_NumberThreads, frame,
                        m_Lambda, m_Gamma, candidateSplits, std::move(featureBag),
                        std::move(leftChildRowMask), std::move(rightChildRowMask));

        leaves.push(std::move(leftChild));
        leaves.push(std::move(rightChild));
    }

    LOG_TRACE(<< "Trained one tree");

    return tree;
}

std::size_t CBoostedTreeImpl::featureBagSize(const core::CDataFrame& frame) const {
    return static_cast<std::size_t>(std::max(
        std::ceil(m_FeatureBagFraction * static_cast<double>(numberFeatures(frame))), 1.0));
}

CBoostedTreeImpl::TSizeVec CBoostedTreeImpl::featureBag(const core::CDataFrame& frame) const {

    std::size_t size{this->featureBagSize(frame)};

    TSizeVec features{this->candidateFeatures()};
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

    frame.readRows(1, 0, frame.numberRows(),
                   [&](TRowItr beginRows, TRowItr endRows) {
                       for (auto row = beginRows; row != endRows; ++row) {
                           std::size_t numberColumns{row->numberColumns()};
                           double prediction{(*row)[predictionColumn(numberColumns)]};
                           double actual{(*row)[m_DependentVariable]};
                           leafValues[root(tree).leafIndex(*row, tree)]->add(prediction, actual);
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
                    std::size_t numberColumns{row->numberColumns()};
                    double actual{(*row)[m_DependentVariable]};
                    double prediction{(*row)[predictionColumn(numberColumns)]};

                    prediction += root(tree).value(*row, tree);

                    row->writeColumn(predictionColumn(numberColumns), prediction);
                    row->writeColumn(lossGradientColumn(numberColumns),
                                     m_Loss->gradient(prediction, actual));
                    row->writeColumn(lossCurvatureColumn(numberColumns),
                                     m_Loss->curvature(prediction, actual));

                    loss += m_Loss->value(prediction, actual);
                }
            },
            0.0),
        &trainingRowMask);

    double loss{0.0};
    for (const auto& result : results.first) {
        loss += result.s_FunctionState;
    }
    LOG_TRACE(<< "training set loss = " << loss);
}

double CBoostedTreeImpl::meanLoss(const core::CDataFrame& frame,
                                  const core::CPackedBitVector& rowMask,
                                  const TNodeVecVec& forest) const {

    auto results = frame.readRows(
        m_NumberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](TMeanAccumulator& loss, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    double prediction{predictRow(*row, forest)};
                    double actual{(*row)[m_DependentVariable]};
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

CBoostedTreeImpl::TSizeVec CBoostedTreeImpl::candidateFeatures() const {
    TSizeVec result;
    result.reserve(m_FeatureSampleProbabilities.size());
    for (std::size_t i = 0; i < m_FeatureSampleProbabilities.size(); ++i) {
        if (m_FeatureSampleProbabilities[i] > 0.0) {
            result.push_back(i);
        }
    }
    return result;
}

const CBoostedTreeImpl::CNode& CBoostedTreeImpl::root(const CBoostedTreeImpl::TNodeVec& tree) {
    return tree[0];
}

double CBoostedTreeImpl::predictRow(const CBoostedTreeImpl::TRowRef& row,
                                    const CBoostedTreeImpl::TNodeVecVec& forest) {
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
    std::size_t i{0};
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
    parameters = bopt.maximumExpectedImprovement();

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
    return maximumTreeSize(frame.numberRows());
}

std::size_t CBoostedTreeImpl::maximumTreeSize(std::size_t numberRows) const {
    return static_cast<std::size_t>(std::ceil(
        m_MaximumTreeSizeFraction * std::sqrt(static_cast<double>(numberRows))));
}

const double CBoostedTreeImpl::MINIMUM_ETA{1e-3};
const std::size_t CBoostedTreeImpl::MAXIMUM_NUMBER_TREES{
    static_cast<std::size_t>(2.0 / MINIMUM_ETA + 0.5)};
const double CBoostedTreeImpl::MINIMUM_RELATIVE_GAIN_PER_SPLIT{1e-7};
const double CBoostedTreeImpl::INF{std::numeric_limits<double>::max()};
}
}
