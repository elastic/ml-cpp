/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeImpl.h>

#include <core/CPersistUtils.h>

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

    if (this->canTrain() == false) {
        // Fallback to using the constant predictor which minimises the loss.

        core::CPackedBitVector trainingRowMask{this->allTrainingRowsMask()};

        double eta{1.0};
        std::swap(eta, m_Eta);
        m_BestForest.assign(1, this->initializePredictionsAndLossDerivatives(frame, trainingRowMask));
        std::swap(eta, m_Eta);

        m_BestForestTestLoss = this->meanLoss(frame, trainingRowMask, m_BestForest);
        LOG_TRACE(<< "Test loss = " << m_BestForestTestLoss);

    } else {
        // Hyperparameter optimisation loop.

        do {
            LOG_TRACE(<< "Optimisation round = " << m_CurrentRound + 1);

            TMeanVarAccumulator lossMoments{this->crossValidateForest(frame, recordProgress)};

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
        m_BestForest = this->trainForest(frame, this->allTrainingRowsMask(), recordProgress);
    }
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
                                      CBoostedTree::TProgressCallback recordProgress) const {
    TMeanVarAccumulator lossMoments;
    for (std::size_t i = 0; i < m_NumberFolds; ++i) {
        TNodeVecVec forest(this->trainForest(frame, m_TrainingRowMasks[i], recordProgress));
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

    TNodeVec tree(1);
    this->refreshPredictionsAndLossDerivatives(frame, trainingRowMask, m_Eta, tree);

    return tree;
}

CBoostedTreeImpl::TNodeVecVec
CBoostedTreeImpl::trainForest(core::CDataFrame& frame,
                              const core::CPackedBitVector& trainingRowMask,
                              CBoostedTree::TProgressCallback  /*recordProgress*/) const {

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

namespace {
const std::string BAYESIAN_OPTIMIZATION_TAG{"bayesian_optimization"};
const std::string BEST_FOREST_TAG{"best_forest"};
const std::string BEST_FOREST_TEST_LOSS_TAG{"best_forest_test_loss"};
const std::string BEST_HYPERPARAMETERS_TAG{"best_hyperparameters"};
const std::string CURRENT_ROUND_TAG{"current_round"};
const std::string DEPENDENT_VARIABLE_TAG{"dependent_variable"};
const std::string ETA_GROWTH_RATE_PER_TREE_TAG{"eta_growth_rate_per_tree"};
const std::string ETA_OVERRIDE_TAG{"eta_override"};
const std::string ETA_TAG{"eta"};
const std::string FEATURE_BAG_FRACTION_OVERRIDE_TAG{"feature_bag_fraction_override"};
const std::string FEATURE_BAG_FRACTION_TAG{"feature_bag_fraction"};
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
const std::string MAXIMUM_TREE_SIZE_FRACTION_TAG{"maximum_tree_size_fraction"};
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
    core::CPersistUtils::persist(ETA_GROWTH_RATE_PER_TREE_TAG,
                                 m_EtaGrowthRatePerTree, inserter);
    core::CPersistUtils::persist(ETA_TAG, m_Eta, inserter);
    core::CPersistUtils::persist(FEATURE_BAG_FRACTION_TAG, m_FeatureBagFraction, inserter);
    core::CPersistUtils::persist(FEATURE_SAMPLE_PROBABILITIES_TAG,
                                 m_FeatureSampleProbabilities, inserter);
    core::CPersistUtils::persist(GAMMA_TAG, m_Gamma, inserter);
    core::CPersistUtils::persist(LAMBDA_TAG, m_Lambda, inserter);
    core::CPersistUtils::persist(MAXIMUM_ATTEMPTS_TO_ADD_TREE_TAG,
                                 m_MaximumAttemptsToAddTree, inserter);
    core::CPersistUtils::persist(MAXIMUM_OPTIMISATION_ROUNDS_PER_HYPERPARAMETER_TAG,
                                 m_MaximumOptimisationRoundsPerHyperparameter, inserter);
    core::CPersistUtils::persist(MAXIMUM_TREE_SIZE_FRACTION_TAG,
                                 m_MaximumTreeSizeFraction, inserter);
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
    core::CPersistUtils::persist(LOSS_TAG, m_Loss->name(), inserter);
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
    try {
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
            else {
                LOG_ERROR(<< "Unexpected name for restoring hyperparameters: "
                          << traverser.name());
                return false;
            }
        } while (traverser.next());
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
        return false;
    }

    return true;
}

bool CBoostedTreeImpl::CNode::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    try {
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
            else {
                LOG_ERROR(<< "Unexpected name for restoring node class: "
                          << traverser.name());
                return false;
            }
        } while (traverser.next());
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
        return false;
    }

    return true;
}

bool CBoostedTreeImpl::restoreLoss(CBoostedTree::TLossFunctionUPtr& loss,
                                   core::CStateRestoreTraverser& traverser) {
    std::string lossFunctionName;
    if (core::CPersistUtils::restore(LOSS_TAG, lossFunctionName, traverser)) {
        if (lossFunctionName == CMse::NAME) {
            loss = std::make_unique<CMse>();
            return true;
        } else {
            LOG_ERROR(<< "Error restoring loss function. Unknown loss function type "
                      << lossFunctionName);
        }
    }
    return false;
}

bool CBoostedTreeImpl::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    try {
        m_BayesianOptimization = std::make_unique<CBayesianOptimisation>();
        do {
            const std::string& name = traverser.name();
            RESTORE(BAYESIAN_OPTIMIZATION_TAG,
                    core::CPersistUtils::restore(BAYESIAN_OPTIMIZATION_TAG,
                                                 *m_BayesianOptimization, traverser))
            RESTORE(BEST_FOREST_TEST_LOSS_TAG,
                    core::CPersistUtils::restore(BEST_FOREST_TEST_LOSS_TAG,
                                                 m_BestForestTestLoss, traverser))
            RESTORE(CURRENT_ROUND_TAG,
                    core::CPersistUtils::restore(CURRENT_ROUND_TAG, m_CurrentRound, traverser))
            RESTORE(DEPENDENT_VARIABLE_TAG,
                    core::CPersistUtils::restore(DEPENDENT_VARIABLE_TAG,
                                                 m_DependentVariable, traverser))
            RESTORE(ETA_GROWTH_RATE_PER_TREE_TAG,
                    core::CPersistUtils::restore(ETA_GROWTH_RATE_PER_TREE_TAG,
                                                 m_EtaGrowthRatePerTree, traverser))
            RESTORE(ETA_TAG, core::CPersistUtils::restore(ETA_TAG, m_Eta, traverser))
            RESTORE(FEATURE_BAG_FRACTION_TAG,
                    core::CPersistUtils::restore(FEATURE_BAG_FRACTION_TAG,
                                                 m_FeatureBagFraction, traverser))
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
            RESTORE(MAXIMUM_TREE_SIZE_FRACTION_TAG,
                    core::CPersistUtils::restore(MAXIMUM_TREE_SIZE_FRACTION_TAG,
                                                 m_MaximumTreeSizeFraction, traverser))
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
                    core::CPersistUtils::restore(TESTING_ROW_MASKS_TAG,
                                                 m_TestingRowMasks, traverser))
            RESTORE(MAXIMUM_NUMBER_TREES_TAG,
                    core::CPersistUtils::restore(MAXIMUM_NUMBER_TREES_TAG,
                                                 m_MaximumNumberTrees, traverser))
            RESTORE(TRAINING_ROW_MASKS_TAG,
                    core::CPersistUtils::restore(TRAINING_ROW_MASKS_TAG,
                                                 m_TrainingRowMasks, traverser))
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
            else {
                LOG_ERROR(<< "Unexpected name for restoring boosted tree implementation: "
                          << traverser.name());
                return false;
            }
        } while (traverser.next());
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to restore state! " << e.what());
        return false;
    }

    return true;
}

const double CBoostedTreeImpl::MINIMUM_ETA{1e-3};
const std::size_t CBoostedTreeImpl::MAXIMUM_NUMBER_TREES{
    static_cast<std::size_t>(2.0 / MINIMUM_ETA + 0.5)};
const double CBoostedTreeImpl::MINIMUM_RELATIVE_GAIN_PER_SPLIT{1e-7};
const double CBoostedTreeImpl::INF{std::numeric_limits<double>::max()};
}
}
