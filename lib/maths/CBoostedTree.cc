/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTree.h>

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CPackedBitVector.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBayesianOptimisation.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CQuantileSketch.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>

#include <boost/operators.hpp>
#include <boost/optional.hpp>

#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace {
using namespace boosted_tree;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeDoublePr = std::pair<std::size_t, double>;
using TDoubleDoubleDoubleTr = std::tuple<double, double, double>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;
using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;

const double INF{std::numeric_limits<double>::max()};

std::size_t predictionColumn(std::size_t numberColumns) {
    return numberColumns - 3;
}

std::size_t lossGradientColumn(std::size_t numberColumns) {
    return numberColumns - 2;
}

std::size_t lossCurvatureColumn(std::size_t numberColumns) {
    return numberColumns - 1;
}

std::size_t numberFeatures(const core::CDataFrame& frame) {
    return frame.numberColumns() - 3;
}

class CNode;
using TNodeVec = std::vector<CNode>;
using TNodeVecVec = std::vector<TNodeVec>;

//! \brief A node of a regression tree.
//!
//! DESCRIPTION:\n
//! This defines a tree structure on a vector of nodes (maintaining the parent
//! child relationships as indexes into the vector). It holds the (binary)
//! splitting criterion (feature and value) and the tree's prediction at each
//! leaf. The intervals are open above so the left node contains feature vectors
//! for which the feature value is _strictly_ less than the split value.
//!
//! During training row masks are maintained for each node (so the data can be
//! efficiently traversed). This supports extracting the left and right child
//! node bit masks from the node's bit mask.
class CNode final {
public:
    //! Check if this is a leaf node.
    bool isLeaf() const { return m_LeftChild < 0; }

    //! Get the leaf index for \p row.
    std::size_t leafIndex(const TRowRef& row, const TNodeVec& tree, std::int32_t index = 0) const {
        if (this->isLeaf()) {
            return index;
        }
        return row[m_SplitFeature] < m_SplitValue
                   ? tree[m_LeftChild].leafIndex(row, tree, m_LeftChild)
                   : tree[m_RightChild].leafIndex(row, tree, m_RightChild);
    }

    //! Get the value predicted by \p tree for the feature vector \p row.
    double value(const TRowRef& row, const TNodeVec& tree) const {
        return tree[this->leafIndex(row, tree)].m_NodeValue;
    }

    //! Get the value of this node.
    double value() const { return m_NodeValue; }

    //! Set the node value to \p value.
    void value(double value) { m_NodeValue = value; }

    //! Split this node and add its child nodes to \p tree.
    std::pair<std::size_t, std::size_t>
    split(std::size_t splitFeature, double splitValue, TNodeVec& tree) {
        m_SplitFeature = splitFeature;
        m_SplitValue = splitValue;
        m_LeftChild = static_cast<std::int32_t>(tree.size());
        m_RightChild = static_cast<std::int32_t>(tree.size() + 1);
        tree.resize(tree.size() + 2);
        return {m_LeftChild, m_RightChild};
    }

    //! Get the row masks of the left and right children of this node.
    auto rowMasks(std::size_t numberThreads,
                  const core::CDataFrame& frame,
                  core::CPackedBitVector rowMask,
                  bool assignMissingToLeft) const {

        LOG_TRACE(<< "Splitting feature '" << m_SplitFeature << "' @ " << m_SplitValue);
        LOG_TRACE(<< "row mask = " << rowMask);

        auto result = frame.readRows(
            numberThreads, 0, frame.numberRows(),
            core::bindRetrievableState(
                [&](core::CPackedBitVector& leftRowMask, TRowItr beginRows, TRowItr endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        std::size_t index{row->index()};
                        double value{(*row)[m_SplitFeature]};
                        bool missing{CDataFrameUtils::isMissing(value)};
                        if ((missing && assignMissingToLeft) ||
                            (missing == false && value < m_SplitValue)) {
                            leftRowMask.extend(false, index - leftRowMask.size());
                            leftRowMask.extend(true);
                        }
                    }
                },
                core::CPackedBitVector{}),
            &rowMask);

        for (auto& mask : result.first) {
            mask.s_FunctionState.extend(false, rowMask.size() -
                                                   mask.s_FunctionState.size());
        }

        core::CPackedBitVector leftRowMask{std::move(result.first[0].s_FunctionState)};
        for (std::size_t i = 1; i < result.first.size(); ++i) {
            leftRowMask |= result.first[i].s_FunctionState;
        }
        LOG_TRACE(<< "# rows in left node = " << leftRowMask.manhattan());
        LOG_TRACE(<< "left row mask = " << leftRowMask);

        core::CPackedBitVector rightRowMask{std::move(rowMask)};
        rightRowMask ^= leftRowMask;
        LOG_TRACE(<< "# rows in right node = " << rightRowMask.manhattan());
        LOG_TRACE(<< "left row mask = " << rightRowMask);

        return std::make_pair(std::move(leftRowMask), std::move(rightRowMask));
    }

    //! Get a human readable description of this tree.
    std::string print(const TNodeVec& tree) const {
        std::ostringstream result;
        return this->doPrint("", tree, result).str();
    }

private:
    std::ostringstream&
    doPrint(std::string pad, const TNodeVec& tree, std::ostringstream& result) const {
        result << "\n" << pad;
        if (this->isLeaf()) {
            result << m_NodeValue;
        } else {
            result << "split '" << m_SplitFeature << "' @ " << m_SplitValue;
            tree[m_LeftChild].doPrint(pad + "  ", tree, result);
            tree[m_RightChild].doPrint(pad + "  ", tree, result);
        }
        return result;
    }

private:
    std::size_t m_SplitFeature = 0;
    double m_SplitValue = 0.0;
    std::int32_t m_LeftChild = -1;
    std::int32_t m_RightChild = -1;
    double m_NodeValue = 0.0;
};

//! \brief Maintains a collection of statistics about a leaf of the regression
//! tree as it is built.
//!
//! DESCRIPTION:\N
//! The regression tree is grown top down by greedily selecting the split with
//! the maximum gain (in the loss). This finds and scores the maximum gain split
//! of a single leaf of the tree.
class CLeafNodeStatistics final {
public:
    CLeafNodeStatistics(std::size_t id,
                        std::size_t numberThreads,
                        const core::CDataFrame& frame,
                        double lambda,
                        double gamma,
                        const TDoubleVecVec& candidateSplits,
                        TSizeVec featureBag,
                        core::CPackedBitVector rowMask)
        : m_Id{id}, m_Lambda{lambda}, m_Gamma{gamma}, m_CandidateSplits{candidateSplits},
          m_FeatureBag{std::move(featureBag)}, m_RowMask{std::move(rowMask)} {

        std::sort(m_FeatureBag.begin(), m_FeatureBag.end());
        LOG_TRACE(<< "row mask = " << m_RowMask);
        LOG_TRACE(<< "feature bag = " << core::CContainerPrinter::print(m_FeatureBag));

        this->computeAggregateGradientsAndCurvatures(numberThreads, frame);
    }

    CLeafNodeStatistics(const CLeafNodeStatistics&) = delete;
    CLeafNodeStatistics& operator=(const CLeafNodeStatistics&) = delete;
    CLeafNodeStatistics(CLeafNodeStatistics&&) = default;
    CLeafNodeStatistics& operator=(CLeafNodeStatistics&&) = default;

    //! Order two leaves by decreasing gain in splitting them.
    bool operator<(const CLeafNodeStatistics& rhs) const {
        return this->bestSplitStatistics() < rhs.bestSplitStatistics();
    }

    //! Get the gain in loss of the best split of this leaf.
    double gain() const { return this->bestSplitStatistics().s_Gain; }

    //! Get the best (feature, feature value) split.
    TSizeDoublePr bestSplit() const {
        const auto& split = this->bestSplitStatistics();
        return {split.s_Feature, split.s_SplitAt};
    }

    //! Check if we should assign the missing feature rows to the left child
    //! of the split.
    bool assignMissingToLeft() const {
        return this->bestSplitStatistics().s_AssignMissingToLeft;
    }

    //! Get the node's identifier.
    std::size_t id() const { return m_Id; }

    //! Get the row mask for this leaf node.
    core::CPackedBitVector& rowMask() { return m_RowMask; }

private:
    //! \brief Statistics relating to a split of the node.
    struct SSplitStatistics : private boost::less_than_comparable<SSplitStatistics> {
        SSplitStatistics(double gain, std::size_t feature, double splitAt, bool assignMissingToLeft)
            : s_Gain{gain}, s_Feature{feature}, s_SplitAt{splitAt}, s_AssignMissingToLeft{assignMissingToLeft} {
        }

        bool operator<(const SSplitStatistics& rhs) const {
            return COrderings::lexicographical_compare(s_Gain, s_Feature,
                                                       rhs.s_Gain, rhs.s_Feature);
        }

        std::string print() const {
            std::ostringstream result;
            result << "split feature '" << s_Feature << "' @ = " << s_SplitAt
                   << ", gain = " << s_Gain;
            return result.str();
        }

        double s_Gain;
        std::size_t s_Feature;
        double s_SplitAt;
        bool s_AssignMissingToLeft;
    };

    //! \brief A collection of aggregate derivatives.
    struct SDerivatives {
        SDerivatives(const TSizeVec& featureBag, const TDoubleVecVec& candidateSplits)
            : s_Gradients(featureBag.size()), s_Curvatures(featureBag.size()),
              s_MissingGradients(featureBag.size(), 0.0),
              s_MissingCurvatures(featureBag.size(), 0.0) {

            for (std::size_t i = 0; i < featureBag.size(); ++i) {
                s_Gradients[i].resize(candidateSplits[featureBag[i]].size() + 1);
                s_Curvatures[i].resize(candidateSplits[featureBag[i]].size() + 1);
            }
        }

        TDoubleVecVec s_Gradients;
        TDoubleVecVec s_Curvatures;
        TDoubleVec s_MissingGradients;
        TDoubleVec s_MissingCurvatures;
    };

private:
    void computeAggregateGradientsAndCurvatures(std::size_t numberThreads,
                                                const core::CDataFrame& frame) {

        auto result = frame.readRows(
            numberThreads, 0, frame.numberRows(),
            core::bindRetrievableState(
                [&](SDerivatives& state, TRowItr beginRows, TRowItr endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        this->addRowDerivatives(*row, state);
                    }
                },
                SDerivatives{m_FeatureBag, m_CandidateSplits}),
            &m_RowMask);

        auto& results = result.first;

        m_Gradients = std::move(results[0].s_FunctionState.s_Gradients);
        m_Curvatures = std::move(results[0].s_FunctionState.s_Curvatures);
        m_MissingGradients = std::move(results[0].s_FunctionState.s_MissingGradients);
        m_MissingCurvatures = std::move(results[0].s_FunctionState.s_MissingCurvatures);

        for (std::size_t k = 1; k < results.size(); ++k) {
            const auto& derivatives = results[k].s_FunctionState;
            for (std::size_t i = 0; i < m_FeatureBag.size(); ++i) {
                std::size_t numberSplits{m_CandidateSplits[m_FeatureBag[i]].size()};
                for (std::size_t j = 0; j <= numberSplits; ++j) {
                    m_Gradients[i][j] += derivatives.s_Gradients[i][j];
                    m_Curvatures[i][j] += derivatives.s_Curvatures[i][j];
                }
                m_MissingGradients[i] += derivatives.s_MissingGradients[i];
                m_MissingCurvatures[i] += derivatives.s_MissingCurvatures[i];
            }
        }

        LOG_TRACE(<< "gradients = " << core::CContainerPrinter::print(m_Gradients));
        LOG_TRACE(<< "curvatures = " << core::CContainerPrinter::print(m_Curvatures));
        LOG_TRACE(<< "missing gradients = "
                  << core::CContainerPrinter::print(m_MissingGradients));
        LOG_TRACE(<< "missing curvatures = "
                  << core::CContainerPrinter::print(m_MissingCurvatures));
    }

    void addRowDerivatives(const TRowRef& row, SDerivatives& derivatives) const {

        std::size_t numberColumns{row.numberColumns()};
        std::size_t derivativeColumn{lossGradientColumn(numberColumns)};
        std::size_t curvatureColumn{lossCurvatureColumn(numberColumns)};

        for (std::size_t i = 0; i < m_FeatureBag.size(); ++i) {
            std::size_t column{m_FeatureBag[i]};
            if (CDataFrameUtils::isMissing(row[column])) {
                derivatives.s_MissingGradients[i] += row[derivativeColumn];
                derivatives.s_MissingCurvatures[i] += row[curvatureColumn];
            } else {
                auto j = std::lower_bound(m_CandidateSplits[column].begin(),
                                          m_CandidateSplits[column].end(), row[column]) -
                         m_CandidateSplits[column].begin();
                derivatives.s_Gradients[i][j] += row[derivativeColumn];
                derivatives.s_Curvatures[i][j] += row[curvatureColumn];
            }
        }
    }

    const SSplitStatistics& bestSplitStatistics() const {
        if (m_BestSplit == boost::none) {
            m_BestSplit = this->computeBestSplitStatistics();
        }
        return *m_BestSplit;
    }

    SSplitStatistics computeBestSplitStatistics() const {

        static const std::size_t ASSIGN_MISSING_TO_LEFT{0};
        static const std::size_t ASSIGN_MISSING_TO_RIGHT{1};

        SSplitStatistics result{-INF, m_FeatureBag.size(), INF, true};

        for (std::size_t i = 0; i < m_FeatureBag.size(); ++i) {
            double g{std::accumulate(m_Gradients[i].begin(), m_Gradients[i].end(), 0.0) +
                     m_MissingGradients[i]};
            double h{std::accumulate(m_Curvatures[i].begin(), m_Curvatures[i].end(), 0.0) +
                     m_MissingCurvatures[i]};
            double gl[]{m_MissingGradients[i], 0.0};
            double hl[]{m_MissingCurvatures[i], 0.0};

            double maximumGain{-INF};
            std::size_t splitAt{m_FeatureBag.size()};
            std::size_t assignMissingTo{ASSIGN_MISSING_TO_LEFT};

            for (std::size_t j = 0; j + 1 < m_Gradients[i].size(); ++j) {
                gl[ASSIGN_MISSING_TO_LEFT] += m_Gradients[i][j];
                hl[ASSIGN_MISSING_TO_LEFT] += m_Curvatures[i][j];
                gl[ASSIGN_MISSING_TO_RIGHT] += m_Gradients[i][j];
                hl[ASSIGN_MISSING_TO_RIGHT] += m_Curvatures[i][j];

                double gain[]{CTools::pow2(gl[ASSIGN_MISSING_TO_LEFT]) /
                                      (hl[ASSIGN_MISSING_TO_LEFT] + m_Lambda) +
                                  CTools::pow2(g - gl[ASSIGN_MISSING_TO_LEFT]) /
                                      (h - hl[ASSIGN_MISSING_TO_LEFT] + m_Lambda),
                              CTools::pow2(gl[ASSIGN_MISSING_TO_RIGHT]) /
                                      (hl[ASSIGN_MISSING_TO_RIGHT] + m_Lambda) +
                                  CTools::pow2(g - gl[ASSIGN_MISSING_TO_RIGHT]) /
                                      (h - hl[ASSIGN_MISSING_TO_RIGHT] + m_Lambda)};

                if (gain[ASSIGN_MISSING_TO_LEFT] > maximumGain) {
                    maximumGain = gain[ASSIGN_MISSING_TO_LEFT];
                    splitAt = j;
                    assignMissingTo = ASSIGN_MISSING_TO_LEFT;
                }
                if (gain[ASSIGN_MISSING_TO_RIGHT] > maximumGain) {
                    maximumGain = gain[ASSIGN_MISSING_TO_RIGHT];
                    splitAt = j;
                    assignMissingTo = ASSIGN_MISSING_TO_RIGHT;
                }
            }

            double gain{0.5 * (maximumGain - CTools::pow2(g) / (h + m_Lambda)) - m_Gamma};

            SSplitStatistics candidate{gain, m_FeatureBag[i],
                                       m_CandidateSplits[m_FeatureBag[i]][splitAt],
                                       assignMissingTo == ASSIGN_MISSING_TO_LEFT};
            LOG_TRACE(<< "candidate split: " << candidate.print());

            if (candidate > result) {
                result = candidate;
            }
        }

        LOG_TRACE(<< "best split: " << result.print());

        return result;
    }

private:
    std::size_t m_Id;
    double m_Lambda;
    double m_Gamma;
    const TDoubleVecVec& m_CandidateSplits;
    TSizeVec m_FeatureBag;
    core::CPackedBitVector m_RowMask;
    TDoubleVecVec m_Gradients;
    TDoubleVecVec m_Curvatures;
    TDoubleVec m_MissingGradients;
    TDoubleVec m_MissingCurvatures;
    mutable boost::optional<SSplitStatistics> m_BestSplit;
};
}

namespace boosted_tree {

void CArgMinLoss::add(double prediction, double actual) {
    std::unique_lock<std::mutex> lock{m_Mutex};
    this->addImpl(prediction, actual);
}

double CMse::value(double prediction, double actual) const {
    return CTools::pow2(prediction - actual);
}

double CMse::gradient(double prediction, double actual) const {
    return prediction - actual;
}

double CMse::curvature(double /*prediction*/, double /*actual*/) const {
    return 2.0;
}

void CArgMinMse::addImpl(double prediction, double actual) {
    m_MeanError.add(actual - prediction);
}

CMse::TArgMinLossUPtr CMse::minimizer() const {
    return std::make_unique<CArgMinMse>();
}

double CArgMinMse::value() const {
    return CBasicStatistics::mean(m_MeanError);
}
}

class CBoostedTree::CImpl final {
public:
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

public:
    static const double MINIMUM_RELATIVE_GAIN_PER_SPLIT;

public:
    CImpl(std::size_t numberThreads, std::size_t dependentVariable, TLossFunctionUPtr loss)
        : m_NumberThreads{numberThreads},
          m_DependentVariable{dependentVariable}, m_Loss{std::move(loss)} {}

    void numberFolds(std::size_t numberFolds) { m_NumberFolds = numberFolds; }

    void lambda(double lambda) { m_LambdaOverride = lambda; }

    void gamma(double gamma) { m_GammaOverride = gamma; }

    void eta(double eta) { m_EtaOverride = eta; }

    void maximumNumberTrees(std::size_t maximumNumberTrees) {
        m_MaximumNumberTreesOverride = maximumNumberTrees;
    }

    void featureBagFraction(double featureBagFraction) {
        m_FeatureBagFractionOverride = featureBagFraction;
    }

    void maximumHyperparameterOptimisationRounds(std::size_t rounds) {
        m_MaximumHyperparameterOptimisationRounds = rounds;
    }

    //! Train the model on the values in \p frame.
    void train(core::CDataFrame& frame, TProgressCallback recordProgress) {

        if (m_Loss == nullptr) {
            HANDLE_FATAL(<< "Internal error: no loss function defined for regression."
                         << " Please report this problem.");
            return;
        }
        if (m_DependentVariable >= frame.numberColumns()) {
            HANDLE_FATAL(<< "Input error: invalid dependent variable for regression");
            return;
        }

        this->initializeMissingFeatureMasks(frame);

        TPackedBitVectorVec trainingRowMasks;
        TPackedBitVectorVec testingRowMasks;
        std::tie(trainingRowMasks, testingRowMasks) = this->crossValidationRowMasks();

        // We store the gradient and curvature of the loss function and the predicted
        // value for the dependent variable of the regression.
        frame.resizeColumns(m_NumberThreads, frame.numberColumns() + 3);

        this->initializeFeatureSampleDistribution(frame);
        this->initializeHyperparameters(frame, recordProgress);

        LOG_TRACE(<< "Main training loop...");

        CBayesianOptimisation bopt{this->hyperparameterBoundingBox()};

        for (std::size_t round = 0;
             round < m_MaximumHyperparameterOptimisationRounds; ++round) {
            LOG_TRACE(<< "Optimisation round = " << round + 1);

            TMeanVarAccumulator lossMoments{this->crossValidateForest(
                frame, trainingRowMasks, testingRowMasks, recordProgress)};

            this->captureBestHyperparameters(CBasicStatistics::mean(lossMoments));

            if (this->selectNextHyperparameters(lossMoments, bopt) == false) {
                break;
            }
        }

        this->restoreBestHyperparameters();
        m_BestForest = this->trainForest(
            frame, core::CPackedBitVector{frame.numberRows(), true}, recordProgress);
    }

    //! Write the predictions of the best trained model to \p frame.
    //!
    //! \note Must be called only if a trained model is available.
    void predict(core::CDataFrame& frame, TProgressCallback /*recordProgress*/) const {
        if (m_BestForestTestLoss == INF) {
            HANDLE_FATAL(<< "Internal error: no model available for prediction. "
                         << "Please report this problem.");
            return;
        }

        if (frame.writeColumns(m_NumberThreads, 0, frame.numberRows(), [&](TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    row->writeColumn(predictionColumn(row->numberColumns()),
                                     predict(*row, m_BestForest));
                }
            }) == false) {
            HANDLE_FATAL(<< "Internal error: failed model inference. "
                         << "Please report this problem.");
        }
    }

    //! Write this model to \p writer.
    void write(core::CRapidJsonConcurrentLineWriter& /*writer*/) const {
        // TODO
    }

private:
    using TDoubleDoublePrVec = std::vector<std::pair<double, double>>;
    using TOptionalDouble = boost::optional<double>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TVector = CDenseVector<double>;

    //! \brief The algorithm parameters we'll directly optimise to improve test error.
    struct SHyperparameters {
        double s_Lambda;
        double s_Gamma;
        double s_Eta;
        double s_FeatureBagFraction;
        TDoubleVec s_FeatureSampleProbabilities;
    };

private:
    //! Setup the missing feature row masks.
    void initializeMissingFeatureMasks(const core::CDataFrame& frame) {

        m_MissingFeatureRowMasks.resize(frame.numberColumns());

        auto result = frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                    double value{(*row)[i]};
                    if (CDataFrameUtils::isMissing(value)) {
                        m_MissingFeatureRowMasks[i].extend(
                            false, row->index() - m_MissingFeatureRowMasks[i].size());
                        m_MissingFeatureRowMasks[i].extend(true);
                    }
                }
            }
        });

        for (auto& mask : m_MissingFeatureRowMasks) {
            mask.extend(false, frame.numberRows() - mask.size());
            LOG_TRACE(<< "# missing = " << mask.manhattan());
        }
    }

    //! Get the row masks to use for the training and testing sets for k-fold
    //! cross validation estimates of the generalisation error.
    std::pair<TPackedBitVectorVec, TPackedBitVectorVec> crossValidationRowMasks() const {

        core::CPackedBitVector mask{~m_MissingFeatureRowMasks[m_DependentVariable]};

        TPackedBitVectorVec trainingRowMasks(m_NumberFolds);

        for (auto row = mask.beginOneBits(); row != mask.endOneBits(); ++row) {
            std::size_t fold{CSampling::uniformSample(m_Rng, 0, m_NumberFolds)};
            trainingRowMasks[fold].extend(true, *row - trainingRowMasks[fold].size());
            trainingRowMasks[fold].extend(false);
        }

        for (auto& fold : trainingRowMasks) {
            fold.extend(true, mask.size() - fold.size());
            LOG_TRACE(<< "# training = " << fold.manhattan());
        }

        TPackedBitVectorVec testingRowMasks(
            m_NumberFolds, core::CPackedBitVector{mask.size(), true});
        for (std::size_t i = 0; i < m_NumberFolds; ++i) {
            testingRowMasks[i] ^= trainingRowMasks[i];
            LOG_TRACE(<< "# testing = " << testingRowMasks[i].manhattan());
        }

        return {trainingRowMasks, testingRowMasks};
    }

    //! Initialize the regressors sample distribution.
    void initializeFeatureSampleDistribution(const core::CDataFrame& frame) {

        // Exclude all constant features by zeroing their probabilities.

        std::size_t n{numberFeatures(frame)};

        TSizeVec regressors(n);
        std::iota(regressors.begin(), regressors.end(), 0);
        regressors.erase(regressors.begin() + m_DependentVariable);

        TDoubleVecVec distinct(n);
        frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t i = 0; i < regressors.size(); ++i) {
                    double value{(*row)[regressors[i]]};
                    if (distinct[i].size() == 2) {
                        continue;
                    }
                    if (distinct[i].empty()) {
                        distinct[i].push_back(value);
                    }
                    if (value != distinct[i][0]) {
                        distinct[i].push_back(value);
                    }
                }
            }
        });

        regressors.erase(
            std::remove_if(regressors.begin(), regressors.end(),
                           [&](std::size_t i) { return distinct[i].size() < 2; }),
            regressors.end());
        LOG_TRACE(<< "regressors = " << core::CContainerPrinter::print(regressors));

        // TODO consider "correlation" with target variable.

        m_FeatureSampleProbabilities.assign(n, 0.0);
        if (regressors.empty()) {
            HANDLE_FATAL(<< "Input error: all features constant.");
        } else {
            double p{1.0 / static_cast<double>(regressors.size())};
            for (auto feature : regressors) {
                m_FeatureSampleProbabilities[feature] = p;
            }
        }
        LOG_TRACE(<< "P(sample) = "
                  << core::CContainerPrinter::print(m_FeatureSampleProbabilities));
    }

    //! Read overrides for hyperparameters and if necessary estimate the initial
    //! values for \f$\lambda\f$ and \f$\gamma\f$ which match the gain from an
    //! overfit tree.
    void initializeHyperparameters(core::CDataFrame& frame, TProgressCallback recordProgress) {

        m_Lambda = m_LambdaOverride.value_or(0.0);
        m_Gamma = m_GammaOverride.value_or(0.0);
        if (m_EtaOverride) {
            m_Eta = *m_EtaOverride;
        }
        if (m_MaximumNumberTreesOverride) {
            m_MaximumNumberTrees = *m_MaximumNumberTreesOverride;
        }
        if (m_FeatureBagFractionOverride) {
            m_FeatureBagFraction = *m_FeatureBagFractionOverride;
        }

        if (m_LambdaOverride && m_GammaOverride) {
            // Fall through.
        } else {
            core::CPackedBitVector trainingRowMask{frame.numberRows(), true};

            auto tree = this->initializePredictionsGradientsAndCurvatures(frame, trainingRowMask);

            double L[2];
            double T[2];
            double W[2];

            std::tie(L[0], T[0], W[0]) =
                this->regularisedLoss(frame, trainingRowMask, {std::move(tree)});
            LOG_TRACE(<< "loss = " << L[0] << ", # leaves = " << T[0]
                      << ", sum square weights = " << W[0]);

            auto forest = this->trainForest(frame, trainingRowMask, recordProgress);

            std::tie(L[1], T[1], W[1]) = this->regularisedLoss(frame, trainingRowMask, forest);
            LOG_TRACE(<< "loss = " << L[1] << ", # leaves = " << T[1]
                      << ", sum square weights = " << W[1]);

            double scale{static_cast<double>(m_NumberFolds - 1) /
                         static_cast<double>(m_NumberFolds)};
            double lambda{scale * std::max((L[0] - L[1]) / (W[1] - W[0]), 0.0) / 5.0};
            double gamma{scale * std::max((L[0] - L[1]) / (T[1] - T[0]), 0.0) / 5.0};

            if (m_LambdaOverride == boost::none) {
                m_Lambda = m_GammaOverride ? lambda : 0.5 * lambda;
            }
            if (m_GammaOverride == boost::none) {
                m_Gamma = m_LambdaOverride ? gamma : 0.5 * gamma;
            }
            LOG_TRACE(<< "lambda(initial) = " << m_Lambda << " gamma(initial) = " << m_Gamma);
        }

        m_MaximumTreeSizeFraction = 10.0;

        // We allow a large number of trees by default in the main parameter
        // optimisation loop. In practice, we should use many fewer if they
        // don't significantly improve test error.
        m_MaximumNumberTrees = 500;
    }

    //! Compute the sum loss for the predictions from \p frame and the leaf
    //! count and squared weight sum from \p forest.
    TDoubleDoubleDoubleTr regularisedLoss(const core::CDataFrame& frame,
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

    //! Train the forest and compute loss moments on each fold.
    TMeanVarAccumulator crossValidateForest(core::CDataFrame& frame,
                                            const TPackedBitVectorVec& trainingRowMasks,
                                            const TPackedBitVectorVec& testingRowMasks,
                                            TProgressCallback recordProgress) const {
        TMeanVarAccumulator lossMoments;
        for (std::size_t i = 0; i < m_NumberFolds; ++i) {
            TNodeVecVec forest(this->trainForest(frame, trainingRowMasks[i], recordProgress));
            double loss{this->meanLoss(frame, testingRowMasks[i], forest)};
            lossMoments.add(loss);
            LOG_TRACE(<< "fold = " << i << " test set loss = " << loss);
        }
        LOG_TRACE(<< "mean test set loss = " << CBasicStatistics::mean(lossMoments) << ", standard deviation = "
                  << std::sqrt(CBasicStatistics::mean(lossMoments)));
        return lossMoments;
    }

    //! Train one forest on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVecVec trainForest(core::CDataFrame& frame,
                            const core::CPackedBitVector& trainingRowMask,
                            TProgressCallback /*recordProgress*/) const {

        LOG_TRACE(<< "Training one forest...");

        TNodeVecVec forest{this->initializePredictionsGradientsAndCurvatures(frame, trainingRowMask)};
        forest.reserve(m_MaximumNumberTrees);

        // For each iteration:
        //  1. Compute weighted quantiles for features F
        //  2. Compute candidate split set S from quantiles of F
        //  3. Build one tree on (F, S)
        //  4. Shrink weights and update predictions and loss derivatives

        double shrinkage{1.0};

        for (std::size_t retries = 0; forest.size() < m_MaximumNumberTrees; /**/) {

            TDoubleVecVec candidateSplits(this->candidateSplits(frame, trainingRowMask));

            auto tree = this->trainTree(frame, trainingRowMask, candidateSplits);

            retries = tree.size() == 1 ? retries + 1 : 0;

            if (retries == m_MaximumAttemptsToAddTree) {
                break;
            } else if (retries == 0) {
                this->refreshPredictionsGradientsAndCurvatures(frame, trainingRowMask,
                                                               shrinkage, tree);
                forest.push_back(std::move(tree));
                shrinkage *= m_Eta;
            }
        }

        LOG_TRACE(<< "Trained one forest");

        return forest;
    }

    //! Get the candidate splits values for each feature.
    TDoubleVecVec candidateSplits(const core::CDataFrame& frame,
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

    //! Get the maximum number of nodes to use in a tree.
    //!
    //! \note This number will only be used if the regularised loss says its
    //! a good idea.
    std::size_t maximumTreeSize(const core::CDataFrame& frame) const {
        return static_cast<std::size_t>(
            std::ceil(m_MaximumTreeSizeFraction *
                      std::sqrt(static_cast<double>(frame.numberRows()))));
    }

    //! Train one tree on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVec trainTree(core::CDataFrame& frame,
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
            0 /*root*/, m_NumberThreads, frame, m_Lambda, m_Gamma,
            candidateSplits, this->featureBag(frame), trainingRowMask));

        // For each iteration we:
        //   1. Find the leaf with the greatest decrease in loss
        //   2. If no split (significantly) reduced the loss we terminate
        //   3. Otherwise we split that leaf

        double totalGain{0.0};

        for (std::size_t i = 0; i < maximumTreeSize; ++i) {

            auto split = leaves.top();
            leaves.pop();

            if (split->gain() < MINIMUM_RELATIVE_GAIN_PER_SPLIT * totalGain) {
                break;
            }

            totalGain += split->gain();
            LOG_TRACE(<< "total gain = " << totalGain);

            std::size_t splitFeature;
            double splitValue;
            std::tie(splitFeature, splitValue) = split->bestSplit();

            bool assignMissingToLeft{split->assignMissingToLeft()};

            std::size_t leftChildId, rightChildId;
            std::tie(leftChildId, rightChildId) =
                tree[split->id()].split(splitFeature, splitValue, tree);

            TSizeVec featureBag{this->featureBag(frame)};

            core::CPackedBitVector leftChildRowMask;
            core::CPackedBitVector rightChildRowMask;
            std::tie(leftChildRowMask, rightChildRowMask) = tree[split->id()].rowMasks(
                m_NumberThreads, frame, std::move(split->rowMask()), assignMissingToLeft);

            leaves.push(std::make_shared<CLeafNodeStatistics>(
                leftChildId, m_NumberThreads, frame, m_Lambda, m_Gamma,
                candidateSplits, featureBag, std::move(leftChildRowMask)));
            leaves.push(std::make_shared<CLeafNodeStatistics>(
                rightChildId, m_NumberThreads, frame, m_Lambda, m_Gamma,
                candidateSplits, featureBag, std::move(rightChildRowMask)));
        }

        LOG_TRACE(<< "Trained one tree");

        return tree;
    }

    //! Get the number of features to consider splitting on.
    std::size_t featureBagSize(const core::CDataFrame& frame) const {
        return static_cast<std::size_t>(std::ceil(
            m_FeatureBagFraction * static_cast<double>(numberFeatures(frame))));
    }

    //! Sample the features according to their categorical distribution.
    TSizeVec featureBag(const core::CDataFrame& frame) const {

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

    //! Initialize the predictions and loss function derivatives for the masked
    //! rows in \p frame.
    TNodeVec initializePredictionsGradientsAndCurvatures(core::CDataFrame& frame,
                                                         const core::CPackedBitVector& trainingRowMask) const {

        frame.writeColumns(
            m_NumberThreads, 0, frame.numberRows(),
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
        this->refreshPredictionsGradientsAndCurvatures(frame, trainingRowMask, 1.0, tree);

        return tree;
    }

    //! Refresh the predictions and loss function derivatives for the masked
    //! rows in \p frame with predictions of \p tree.
    void refreshPredictionsGradientsAndCurvatures(core::CDataFrame& frame,
                                                  const core::CPackedBitVector& trainingRowMask,
                                                  double shrinkage,
                                                  TNodeVec& tree) const {

        using TArgMinLossUPtrVec = std::vector<CLoss::TArgMinLossUPtr>;

        TArgMinLossUPtrVec leafValues;
        leafValues.reserve(tree.size());
        for (std::size_t i = 0; i < tree.size(); ++i) {
            leafValues.push_back(m_Loss->minimizer());
        }

        frame.readRows(
            m_NumberThreads, 0, frame.numberRows(),
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
            tree[i].value(shrinkage * leafValues[i]->value());
        }

        LOG_TRACE(<< "tree =\n" << root(tree).print(tree));

        frame.writeColumns(
            m_NumberThreads, 0, frame.numberRows(),
            [&](TRowItr beginRows, TRowItr endRows) {
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
                }
            },
            &trainingRowMask);
    }

    //! Compute the mean of the loss function on the masked rows of \p frame.
    double meanLoss(const core::CDataFrame& frame,
                    const core::CPackedBitVector& rowMask,
                    const TNodeVecVec& forest) const {

        auto results = frame.readRows(
            m_NumberThreads, 0, frame.numberRows(),
            core::bindRetrievableState(
                [&](TMeanAccumulator& loss, TRowItr beginRows, TRowItr endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        double prediction{predict(*row, forest)};
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

    //! Get a column mask of the suitable regressor features.
    TSizeVec candidateFeatures() const {
        TSizeVec result;
        result.reserve(m_FeatureSampleProbabilities.size());
        for (std::size_t i = 0; i < m_FeatureSampleProbabilities.size(); ++i) {
            if (m_FeatureSampleProbabilities[i] > 0.0) {
                result.push_back(i);
            }
        }
        return result;
    }

    //! Get the root node of \p tree.
    static const CNode& root(const TNodeVec& tree) { return tree[0]; }

    //! Get the forest's prediction for \p row.
    static double predict(const TRowRef& row, const TNodeVecVec& forest) {
        double result{0.0};
        for (const auto& tree : forest) {
            result += root(tree).value(row, tree);
        }
        return result;
    }
    //! Get the bounding box to use for hyperparameter optimisation.
    TDoubleDoublePrVec hyperparameterBoundingBox() const {
        TDoubleDoublePrVec result;
        if (m_LambdaOverride == boost::none) {
            result.emplace_back(std::log(m_Lambda / 10.0), std::log(10.0 * m_Lambda));
        }
        if (m_GammaOverride == boost::none) {
            result.emplace_back(std::log(m_Gamma / 10.0), std::log(10.0 * m_Gamma));
        }
        if (m_EtaOverride == boost::none) {
            result.emplace_back(0.8, 1.0);
        }
        if (m_FeatureBagFractionOverride == boost::none) {
            result.emplace_back(0.2, 0.8);
        }
        return result;
    }

    //! Select the next hyperparameters for which to train a model.
    bool selectNextHyperparameters(const TMeanVarAccumulator& lossMoments,
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
            parameters(i++) = m_Eta;
        }
        if (m_FeatureBagFractionOverride == boost::none) {
            parameters(i++) = m_FeatureBagFraction;
        }
        if (i == 0) {
            return false;
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
            m_Eta = parameters(i++);
        }
        if (m_FeatureBagFractionOverride == boost::none) {
            m_FeatureBagFraction = parameters(i++);
        }

        LOG_TRACE(<< "lambda = " << m_Lambda << ", gamma = " << m_Gamma << ", eta = " << m_Eta
                  << ", feature bag fraction = " << m_FeatureBagFraction);
        return true;
    }

    //! Capture the current hyperparameter values.
    void captureBestHyperparameters(double loss) {
        if (loss < m_BestForestTestLoss) {
            m_BestForestTestLoss = loss;
            m_BestHyperparameters = SHyperparameters{m_Lambda, m_Gamma, m_Eta, m_FeatureBagFraction,
                                                     m_FeatureSampleProbabilities};
        }
    }

    //! Set the hyperparamaters from the best recorded.
    void restoreBestHyperparameters() {
        m_Lambda = m_BestHyperparameters.s_Lambda;
        m_Gamma = m_BestHyperparameters.s_Gamma;
        m_Eta = m_BestHyperparameters.s_Eta;
        m_FeatureBagFraction = m_BestHyperparameters.s_FeatureBagFraction;
        m_FeatureSampleProbabilities = m_BestHyperparameters.s_FeatureSampleProbabilities;
    }

    //! Get the number of hyperparameters to tune.
    std::size_t numberHyperparametersToTune() const {
        return (m_LambdaOverride ? 0 : 1) + (m_GammaOverride ? 0 : 1) +
               (m_EtaOverride ? 0 : 1) + (m_FeatureBagFractionOverride ? 0 : 1);
    }

private:
    mutable CPRNG::CXorOShiro128Plus m_Rng;
    std::size_t m_NumberThreads;
    std::size_t m_DependentVariable;
    TLossFunctionUPtr m_Loss;
    TOptionalDouble m_LambdaOverride;
    TOptionalDouble m_GammaOverride;
    TOptionalDouble m_EtaOverride;
    TOptionalSize m_MaximumNumberTreesOverride;
    TOptionalDouble m_FeatureBagFractionOverride;
    double m_Lambda = 0.0;
    double m_Gamma = 0.0;
    std::size_t m_NumberFolds = 5;
    std::size_t m_MaximumNumberTrees = 5;
    std::size_t m_MaximumAttemptsToAddTree = 3;
    std::size_t m_NumberSplitsPerFeature = 40;
    std::size_t m_MaximumHyperparameterOptimisationRounds = 20;
    double m_FeatureBagFraction = 0.5;
    double m_MaximumTreeSizeFraction = 1.0;
    double m_Eta = 0.98;
    TDoubleVec m_FeatureSampleProbabilities;
    TPackedBitVectorVec m_MissingFeatureRowMasks;
    double m_BestForestTestLoss = INF;
    SHyperparameters m_BestHyperparameters;
    TNodeVecVec m_BestForest;
};

const double CBoostedTree::CImpl::MINIMUM_RELATIVE_GAIN_PER_SPLIT{1e-7};

CBoostedTree::CBoostedTree(std::size_t numberThreads, std::size_t dependentVariable, TLossFunctionUPtr loss)
    : m_Impl{std::make_unique<CImpl>(numberThreads, dependentVariable, std::move(loss))} {
}

CBoostedTree::~CBoostedTree() {
}

CBoostedTree& CBoostedTree::numberFolds(std::size_t folds) {
    m_Impl->numberFolds(folds);
    return *this;
}

CBoostedTree& CBoostedTree::lambda(double lambda) {
    m_Impl->lambda(lambda);
    return *this;
}

CBoostedTree& CBoostedTree::gamma(double gamma) {
    m_Impl->gamma(gamma);
    return *this;
}

CBoostedTree& CBoostedTree::eta(double eta) {
    m_Impl->eta(eta);
    return *this;
}

CBoostedTree& CBoostedTree::maximumNumberTrees(std::size_t maximumNumberTrees) {
    m_Impl->maximumNumberTrees(maximumNumberTrees);
    return *this;
}

CBoostedTree& CBoostedTree::featureBagFraction(double featureBagFraction) {
    m_Impl->featureBagFraction(featureBagFraction);
    return *this;
}

CBoostedTree& CBoostedTree::maximumHyperparameterOptimisationRounds(std::size_t rounds) {
    m_Impl->maximumHyperparameterOptimisationRounds(rounds);
    return *this;
}

void CBoostedTree::train(core::CDataFrame& frame, TProgressCallback recordProgress) {
    m_Impl->train(frame, recordProgress);
}

void CBoostedTree::predict(core::CDataFrame& frame, TProgressCallback recordProgress) const {
    m_Impl->predict(frame, recordProgress);
}

void CBoostedTree::write(core::CRapidJsonConcurrentLineWriter& writer) const {
    m_Impl->write(writer);
}

std::size_t CBoostedTree::numberExtraColumnsForTrain() const {
    return 3;
}

std::size_t CBoostedTree::columnHoldingPrediction(std::size_t numberColumns) const {
    return predictionColumn(numberColumns);
}
}
}
