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
        double value{row[m_SplitFeature]};
        bool missing{CDataFrameUtils::isMissing(value)};
        return (missing && m_AssignMissingToLeft) || (missing == false && value < m_SplitValue)
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
    split(std::size_t splitFeature, double splitValue, bool assignMissingToLeft, TNodeVec& tree) {
        m_SplitFeature = splitFeature;
        m_SplitValue = splitValue;
        m_AssignMissingToLeft = assignMissingToLeft;
        m_LeftChild = static_cast<std::int32_t>(tree.size());
        m_RightChild = static_cast<std::int32_t>(tree.size() + 1);
        tree.resize(tree.size() + 2);
        return {m_LeftChild, m_RightChild};
    }

    //! Get the row masks of the left and right children of this node.
    auto rowMasks(std::size_t numberThreads,
                  const core::CDataFrame& frame,
                  core::CPackedBitVector rowMask) const {

        LOG_TRACE(<< "Splitting feature '" << m_SplitFeature << "' @ " << m_SplitValue);
        LOG_TRACE(<< "# rows in node = " << rowMask.manhattan());
        LOG_TRACE(<< "row mask = " << rowMask);

        auto result = frame.readRows(
            numberThreads, 0, frame.numberRows(),
            core::bindRetrievableState(
                [&](core::CPackedBitVector& leftRowMask, TRowItr beginRows, TRowItr endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        std::size_t index{row->index()};
                        double value{(*row)[m_SplitFeature]};
                        bool missing{CDataFrameUtils::isMissing(value)};
                        if ((missing && m_AssignMissingToLeft) ||
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
            result << "split feature '" << m_SplitFeature << "' @ " << m_SplitValue;
            tree[m_LeftChild].doPrint(pad + "  ", tree, result);
            tree[m_RightChild].doPrint(pad + "  ", tree, result);
        }
        return result;
    }

private:
    std::size_t m_SplitFeature = 0;
    double m_SplitValue = 0.0;
    bool m_AssignMissingToLeft = true;
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

        this->computeAggregateLossDerivatives(numberThreads, frame);
    }
    //! This should only called by split but is public so it's accessible to make_shared.
    CLeafNodeStatistics(std::size_t id,
                        const CLeafNodeStatistics& parent,
                        const CLeafNodeStatistics& sibling,
                        core::CPackedBitVector rowMask)
        : m_Id{id}, m_Lambda{sibling.m_Lambda}, m_Gamma{sibling.m_Gamma},
          m_CandidateSplits{sibling.m_CandidateSplits},
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
                m_Curvatures[i][j] = parent.m_Curvatures[i][j] -
                                     sibling.m_Curvatures[i][j];
            }
            m_MissingGradients[i] = parent.m_MissingGradients[i] -
                                    sibling.m_MissingGradients[i];
            m_MissingCurvatures[i] = parent.m_MissingCurvatures[i] -
                                     sibling.m_MissingCurvatures[i];
        }

        LOG_TRACE(<< "gradients = " << core::CContainerPrinter::print(m_Gradients));
        LOG_TRACE(<< "curvatures = " << core::CContainerPrinter::print(m_Curvatures));
        LOG_TRACE(<< "missing gradients = "
                  << core::CContainerPrinter::print(m_MissingGradients));
        LOG_TRACE(<< "missing curvatures = "
                  << core::CContainerPrinter::print(m_MissingCurvatures));
    }

    CLeafNodeStatistics(const CLeafNodeStatistics&) = delete;
    CLeafNodeStatistics& operator=(const CLeafNodeStatistics&) = delete;
    CLeafNodeStatistics(CLeafNodeStatistics&&) = default;
    CLeafNodeStatistics& operator=(CLeafNodeStatistics&&) = default;

    //! Apply the split defined by (\p leftChildRowMask, \p rightChildRowMask).
    auto split(std::size_t leftChildId,
               std::size_t rightChildId,
               std::size_t numberThreads,
               const core::CDataFrame& frame,
               double lambda,
               double gamma,
               const TDoubleVecVec& candidateSplits,
               TSizeVec featureBag,
               core::CPackedBitVector leftChildRowMask,
               core::CPackedBitVector rightChildRowMask) {

        if (leftChildRowMask.manhattan() < rightChildRowMask.manhattan()) {
            auto leftChild = std::make_shared<CLeafNodeStatistics>(
                leftChildId, numberThreads, frame, lambda, gamma,
                candidateSplits, featureBag, std::move(leftChildRowMask));
            auto rightChild = std::make_shared<CLeafNodeStatistics>(
                rightChildId, *this, *leftChild, std::move(rightChildRowMask));

            return std::make_pair(leftChild, rightChild);
        }

        auto rightChild = std::make_shared<CLeafNodeStatistics>(
            rightChildId, numberThreads, frame, lambda, gamma, candidateSplits,
            featureBag, std::move(rightChildRowMask));
        auto leftChild = std::make_shared<CLeafNodeStatistics>(
            leftChildId, *this, *rightChild, std::move(leftChildRowMask));

        return std::make_pair(leftChild, rightChild);
    }

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

    //! Estimate maximum leaf statistics bookkeeping memory for training
    //! boosted trees on data frames with \p numberRows rows, \p numberCols columns
    //! with specified settings for \p featureBagFraction and \p numberSplitsPerFeature
    static size_t estimateMemoryUsage(std::size_t numberRows,
                                      std::size_t numberCols,
                                      double featureBagFraction,
                                      std::size_t numberSplitsPerFeature) {
        std::size_t featureBagSize{
            static_cast<std::size_t>(ceil(featureBagFraction) * (numberCols - 1)) *
            sizeof(std::size_t)};
        std::size_t rowMaskSize{numberRows / 32};
        std::size_t gradientsSize{(numberCols - 1) * numberSplitsPerFeature * sizeof(double)};
        std::size_t curvatureSize{gradientsSize};
        std::size_t missingGradientsSize{(numberCols - 1) * sizeof(double)};
        std::size_t missingCurvatureSize{missingGradientsSize};
        return featureBagSize + rowMaskSize + gradientsSize + curvatureSize +
               missingGradientsSize + missingCurvatureSize;
    }

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
            result << "split feature '" << s_Feature << "' @ " << s_SplitAt
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
        SDerivatives(const TDoubleVecVec& candidateSplits)
            : s_Gradients(candidateSplits.size()),
              s_Curvatures(candidateSplits.size()),
              s_MissingGradients(candidateSplits.size(), 0.0),
              s_MissingCurvatures(candidateSplits.size(), 0.0) {

            for (std::size_t i = 0; i < candidateSplits.size(); ++i) {
                std::size_t numberSplits{candidateSplits[i].size() + 1};
                s_Gradients[i].resize(numberSplits, 0.0);
                s_Curvatures[i].resize(numberSplits, 0.0);
            }
        }

        TDoubleVecVec s_Gradients;
        TDoubleVecVec s_Curvatures;
        TDoubleVec s_MissingGradients;
        TDoubleVec s_MissingCurvatures;
    };

private:
    void computeAggregateLossDerivatives(std::size_t numberThreads,
                                         const core::CDataFrame& frame) {

        auto result = frame.readRows(
            numberThreads, 0, frame.numberRows(),
            core::bindRetrievableState(
                [&](SDerivatives& state, TRowItr beginRows, TRowItr endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        this->addRowDerivatives(*row, state);
                    }
                },
                SDerivatives{m_CandidateSplits}),
            &m_RowMask);

        auto& results = result.first;

        m_Gradients = std::move(results[0].s_FunctionState.s_Gradients);
        m_Curvatures = std::move(results[0].s_FunctionState.s_Curvatures);
        m_MissingGradients = std::move(results[0].s_FunctionState.s_MissingGradients);
        m_MissingCurvatures = std::move(results[0].s_FunctionState.s_MissingCurvatures);

        for (std::size_t k = 1; k < results.size(); ++k) {
            const auto& derivatives = results[k].s_FunctionState;
            for (std::size_t i = 0; i < m_CandidateSplits.size(); ++i) {
                std::size_t numberSplits{m_CandidateSplits[i].size() + 1};
                for (std::size_t j = 0; j < numberSplits; ++j) {
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

        for (auto i : m_FeatureBag) {
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

            SSplitStatistics candidate{gain, i, m_CandidateSplits[i][splitAt],
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
    using TBayesinOptimizationUPtr = std::unique_ptr<maths::CBayesianOptimisation>;

public:
    static const double MINIMUM_ETA;
    static const std::size_t MAXIMUM_NUMBER_TREES;
    static const double MINIMUM_RELATIVE_GAIN_PER_SPLIT;

public:
    CImpl(std::size_t numberThreads, std::size_t dependentVariable, TLossFunctionUPtr loss)
        : m_NumberThreads{numberThreads},
          m_DependentVariable{dependentVariable}, m_Loss{std::move(loss)} {}

    void numberFolds(std::size_t numberFolds) {
        if (numberFolds < 2) {
            LOG_WARN(<< "Must use at least two-folds for cross validation");
            numberFolds = 2;
        }
        m_NumberFolds = numberFolds;
    }

    void lambda(double lambda) {
        if (lambda < 0.0) {
            LOG_WARN(<< "Lambda must be non-negative");
            lambda = 0.0;
        }
        m_LambdaOverride = lambda;
    }

    void gamma(double gamma) {
        if (gamma < 0.0) {
            LOG_WARN(<< "Gamma must be non-negative");
            gamma = 0.0;
        }
        m_GammaOverride = gamma;
    }

    void eta(double eta) {
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

    void maximumNumberTrees(std::size_t maximumNumberTrees) {
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

    void featureBagFraction(double featureBagFraction) {
        if (featureBagFraction < 0.0 || featureBagFraction > 1.0) {
            LOG_WARN(<< "Truncating supplied feature bag fraction " << featureBagFraction
                     << " which must be positive and not more than one");
            featureBagFraction = CTools::truncate(featureBagFraction, 0.0, 1.0);
        }
        m_FeatureBagFractionOverride = featureBagFraction;
    }

    void maximumOptimisationRoundsPerHyperparameter(std::size_t rounds) {
        m_MaximumOptimisationRoundsPerHyperparameter = rounds;
    }

    //! Train the model on the values in \p frame.
    void train(core::CDataFrame& frame, TProgressCallback recordProgress) {


        for (std::size_t round = 0; round < numberRounds; ++round) {
            LOG_TRACE(<< "Optimisation round = " << round + 1);

            TMeanVarAccumulator lossMoments{this->crossValidateForest(
                frame, trainingRowMasks, testingRowMasks, recordProgress)};

            this->captureBestHyperparameters(lossMoments);

            // Trap the case that the dependent variable is (effectively) constant.
            // There is no point adjusting hyperparameters in this case - and we run
            // into numerical issues trying - since any forest will do.
            if (std::sqrt(CBasicStatistics::variance(lossMoments)) <
                1e-10 * std::fabs(CBasicStatistics::mean(lossMoments))) {
                break;
            }
            if (this->selectNextHyperparameters(lossMoments, bopt) == false) {
                break;
            }
        }

        LOG_TRACE(<< "Test loss = " << m_BestForestTestLoss);

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
        bool successful;
        std::tie(std::ignore, successful) = frame.writeColumns(
            m_NumberThreads, 0, frame.numberRows(), [&](TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    row->writeColumn(predictionColumn(row->numberColumns()),
                                     predict(*row, m_BestForest));
                }
            });
        if (successful == false) {
            HANDLE_FATAL(<< "Internal error: failed model inference. "
                         << "Please report this problem.");
        }
    }

    //! Write this model to \p writer.
    void write(core::CRapidJsonConcurrentLineWriter& /*writer*/) const {
        // TODO
    }

    //! Get the number of columns training the model will add to the data frame.
    std::size_t numberExtraColumnsForTrain() const { return 3; }

    //! Get the feature sample probabilities.
    TDoubleVec featureWeights() const { return m_FeatureSampleProbabilities; }

    std::size_t estimateMemoryUsage(std::size_t numberRows, std::size_t numberColumns) const {
        std::size_t maximumNumberNodes{this->maximumTreeSize(numberRows)};
        std::size_t forestMemoryUsage{this->m_MaximumNumberTrees *
                                      maximumNumberNodes * sizeof(CNode)};
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
        double s_EtaGrowthRatePerTree;
        double s_FeatureBagFraction;
        TDoubleVec s_FeatureSampleProbabilities;
    };

private:


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
            LOG_TRACE(<< "fold = " << i << " forest size = " << forest.size()
                      << " test set loss = " << loss);
        }
        LOG_TRACE(<< "test mean loss = " << CBasicStatistics::mean(lossMoments)
                  << ", sigma = " << std::sqrt(CBasicStatistics::mean(lossMoments)));
        return lossMoments;
    }

    //! Train one forest on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVecVec trainForest(core::CDataFrame& frame,
                            const core::CPackedBitVector& trainingRowMask,
                            TProgressCallback /*recordProgress*/) const {

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
            std::tie(leftChild, rightChild) = leaf->split(
                leftChildId, rightChildId, m_NumberThreads, frame, m_Lambda,
                m_Gamma, candidateSplits, std::move(featureBag),
                std::move(leftChildRowMask), std::move(rightChildRowMask));

            leaves.push(std::move(leftChild));
            leaves.push(std::move(rightChild));
        }

        LOG_TRACE(<< "Trained one tree");

        return tree;
    }

    //! Get the number of features to consider splitting on.
    std::size_t featureBagSize(const core::CDataFrame& frame) const {
        return static_cast<std::size_t>(std::max(
            std::ceil(m_FeatureBagFraction * static_cast<double>(numberFeatures(frame))), 1.0));
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



    //! Refresh the predictions and loss function derivatives for the masked
    //! rows in \p frame with predictions of \p tree.
    void refreshPredictionsAndLossDerivatives(core::CDataFrame& frame,
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

        // We need sensible bounds for the region we'll search for optimal values.
        // For all parameters where we have initial estimates we use bounds of the
        // form a * initial and b * initial for a < 1 < b. For other parameters we
        // use a fixed range. Ideally, we'd use the smallest intervals that have a
        // high probability of containing good parameter values. We also parameterise
        // so the probability any subinterval contains a good value is proportional
        // to its length. For parameters whose difference is naturally measured as
        // a ratio, i.e. roughly speaking difference(p_1, p_0) = p_1 / p_0 for p_0
        // less than p_1, this translates to using log parameter values.

        TDoubleDoublePrVec result;
        if (m_LambdaOverride == boost::none) {
            result.emplace_back(std::log(m_Lambda / 10.0), std::log(10.0 * m_Lambda));
        }
        if (m_GammaOverride == boost::none) {
            result.emplace_back(std::log(m_Gamma / 10.0), std::log(10.0 * m_Gamma));
        }
        if (m_EtaOverride == boost::none) {
            double rate{m_EtaGrowthRatePerTree - 1.0};
            result.emplace_back(std::log(0.3 * m_Eta), std::log(3.0 * m_Eta));
            result.emplace_back(1.0 + rate / 2.0, 1.0 + 1.5 * rate);
        }
        if (m_FeatureBagFractionOverride == boost::none) {
            result.emplace_back(0.2, 0.8);
        }
        return result;
    }

    //! Get the number of hyperparameter tuning rounds to use.
    std::size_t numberHyperparameterTuningRounds() const {
        return std::max(m_MaximumOptimisationRoundsPerHyperparameter *
                            this->numberHyperparametersToTune(),
                        std::size_t{1});
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

        LOG_TRACE(<< "lambda = " << m_Lambda << ", gamma = " << m_Gamma
                  << ", eta = " << m_Eta << ", eta growth rate per tree = " << m_EtaGrowthRatePerTree
                  << ", feature bag fraction = " << m_FeatureBagFraction);
        return true;
    }

    //! Capture the current hyperparameter values.
    void captureBestHyperparameters(const TMeanVarAccumulator& lossMoments) {
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

    //! Set the hyperparamaters from the best recorded.
    void restoreBestHyperparameters() {
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

    //! Get the number of hyperparameters to tune.
    std::size_t numberHyperparametersToTune() const {
        return (m_LambdaOverride ? 0 : 1) + (m_GammaOverride ? 0 : 1) +
               (m_EtaOverride ? 0 : 2) + (m_FeatureBagFractionOverride ? 0 : 1);
    }

    //! Get the maximum number of nodes to use in a tree.
    //!
    //! \note This number will only be used if the regularised loss says its
    //! a good idea.
    std::size_t maximumTreeSize(const core::CDataFrame& frame) const {
        return maximumTreeSize(frame.numberRows());
    }

    //! Get the maximum number of nodes to use in a tree.
    //!
    //! \note This number will only be used if the regularised loss says its
    //! a good idea.
    std::size_t maximumTreeSize(std::size_t numberRows) const {
        return static_cast<std::size_t>(std::ceil(
            m_MaximumTreeSizeFraction * std::sqrt(static_cast<double>(numberRows))));
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
    double m_Eta = 0.1;
    double m_EtaGrowthRatePerTree = 1.05;
    std::size_t m_NumberFolds = 4;
    std::size_t m_MaximumNumberTrees = 20;
    std::size_t m_MaximumAttemptsToAddTree = 3;
    std::size_t m_NumberSplitsPerFeature = 40;
    std::size_t m_MaximumOptimisationRoundsPerHyperparameter = 5;
    double m_FeatureBagFraction = 0.5;
    double m_MaximumTreeSizeFraction = 1.0;
    TDoubleVec m_FeatureSampleProbabilities;
    TPackedBitVectorVec m_MissingFeatureRowMasks;
    double m_BestForestTestLoss = INF;
    SHyperparameters m_BestHyperparameters;
    TNodeVecVec m_BestForest;
    TBayesinOptimizationUPtr bayesianOptimizer;
    TNodeVecVec forest;
    std::size_t numberRounds;
    std::size_t currentRound;
};

const double CBoostedTree::CImpl::MINIMUM_ETA{1e-3};
const std::size_t CBoostedTree::CImpl::MAXIMUM_NUMBER_TREES{
    static_cast<std::size_t>(2.0 / MINIMUM_ETA + 0.5)};
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

CBoostedTree& CBoostedTree::maximumOptimisationRoundsPerHyperparameter(std::size_t rounds) {
    m_Impl->maximumOptimisationRoundsPerHyperparameter(rounds);
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

CBoostedTree::TDoubleVec CBoostedTree::featureWeights() const {
    return m_Impl->featureWeights();
}

std::size_t CBoostedTree::numberExtraColumnsForTrain() const {
    return m_Impl->numberExtraColumnsForTrain();
}

std::size_t CBoostedTree::columnHoldingPrediction(std::size_t numberColumns) const {
    return predictionColumn(numberColumns);
}
std::size_t CBoostedTree::estimateMemoryUsage(std::size_t numberRows,
                                              std::size_t numberColumns) const {
    return this->m_Impl->estimateMemoryUsage(numberRows, numberColumns);
}
}
}
