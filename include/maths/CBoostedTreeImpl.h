/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeImpl_h
#define INCLUDED_ml_maths_CBoostedTreeImpl_h

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CPackedBitVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTree.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPRNG.h>
#include <maths/CTools.h>
#include <maths/ImportExport.h>

#include <boost/operators.hpp>
#include <boost/optional.hpp>

#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
class CBayesianOptimisation;

namespace boosted_tree_detail {
inline std::size_t predictionColumn(std::size_t numberColumns) {
    return numberColumns - 3;
}
}

//! \brief Implementation of CBoostedTree.
class MATHS_EXPORT CBoostedTreeImpl final {
public:
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TBayesinOptimizationUPtr = std::unique_ptr<maths::CBayesianOptimisation>;
    using TProgressCallback = CBoostedTree::TProgressCallback;
    using TMemoryUsageCallback = CBoostedTree::TMemoryUsageCallback;
    using TTrainingStateCallback = CBoostedTree::TTrainingStateCallback;
    using TDoubleVec = std::vector<double>;

public:
    static const double MINIMUM_RELATIVE_GAIN_PER_SPLIT;

public:
    CBoostedTreeImpl(std::size_t numberThreads, CBoostedTree::TLossFunctionUPtr loss);

    ~CBoostedTreeImpl();

    CBoostedTreeImpl& operator=(const CBoostedTreeImpl&) = delete;
    CBoostedTreeImpl& operator=(CBoostedTreeImpl&&);

    //! Train the model on the values in \p frame.
    void train(core::CDataFrame& frame,
               const TProgressCallback& recordProgress,
               const TMemoryUsageCallback& recordMemoryUsage,
               const TTrainingStateCallback& recordTrainStateCallback);

    //! Write the predictions of the best trained model to \p frame.
    //!
    //! \note Must be called only if a trained model is available.
    void predict(core::CDataFrame& frame, const TProgressCallback& /*recordProgress*/) const;

    //! Write this model to \p writer.
    void write(core::CRapidJsonConcurrentLineWriter& /*writer*/) const;

    //! Get the feature sample probabilities.
    const TDoubleVec& featureWeights() const;

    //! Get the column containing the dependent variable.
    std::size_t columnHoldingDependentVariable() const;

    //! Get the number of columns training the model will add to the data frame.
    static std::size_t numberExtraColumnsForTrain();

    //! Estimate the maximum booking memory that training the boosted tree on a data
    //! frame with \p numberRows row and \p numberColumns columns will use.
    std::size_t estimateMemoryUsage(std::size_t numberRows, std::size_t numberColumns) const;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

private:
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TOptionalDouble = boost::optional<double>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TVector = CDenseVector<double>;
    using TRowItr = core::CDataFrame::TRowItr;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TDataFrameCategoryEncoderUPtr = std::unique_ptr<CDataFrameCategoryEncoder>;
    using TDataTypeVec = CDataFrameUtils::TDataTypeVec;

    class CNode;
    using TNodeVec = std::vector<CNode>;
    using TNodeVecVec = std::vector<TNodeVec>;

    //! \brief Holds the parameters associated with the different types of regularizer
    //! terms available.
    template<typename T>
    class CRegularization final {
    public:
        //! Set the multiplier of the tree depth penalty.
        CRegularization& alpha(double alpha) {
            m_Alpha = alpha;
            return *this;
        }

        //! Set the multiplier of the tree size penalty.
        CRegularization& gamma(double gamma) {
            m_Gamma = gamma;
            return *this;
        }

        //! Set the multiplier of the square leaf weight penalty.
        CRegularization& lambda(double lambda) {
            m_Lambda = lambda;
            return *this;
        }

        //! Set the maximum depth tree depth.
        CRegularization& maxTreeDepth(double maxTreeDepth) {
            m_MaxTreeDepth = maxTreeDepth;
            return *this;
        }

        //! Set the tolerance in the maximum depth tree depth.
        CRegularization& maxTreeDepthTolerance(double maxTreeDepthTolerance) {
            m_MaxTreeDepthTolerance = maxTreeDepthTolerance;
            return *this;
        }

        //! Count the number of parameters which have their default values.
        std::size_t countNotSet() const {
            return (m_Alpha == T{} ? 1 : 0) + (m_Gamma == T{} ? 1 : 0) +
                   (m_Lambda == T{} ? 1 : 0) + (m_MaxTreeDepth == T{} ? 1 : 0) +
                   (m_MaxTreeDepthTolerance == T{} ? 1 : 0);
        }

        //! Multiplier of the tree depth penalty.
        T alpha() const { return m_Alpha; }

        //! Multiplier of the tree size penalty.
        T gamma() const { return m_Gamma; }

        //! Multiplier of the square leaf weight penalty.
        T lambda() const { return m_Lambda; }

        //! Maximum depth tree depth.
        T maxTreeDepth() const { return m_MaxTreeDepth; }

        //! Maximum depth tree depth.
        T maxTreeDepthTolerance() const { return m_MaxTreeDepthTolerance; }

        //! Get the penalty which applies to a leaf at depth \p depth.
        T penaltyForDepth(std::size_t depth) const {
            return std::exp((static_cast<double>(depth) / m_MaxTreeDepth - 1.0) /
                            m_MaxTreeDepthTolerance);
        }

        //! Get description of the regularization parameters.
        std::string print() const {
            return "(alpha = " + toString(m_Alpha) +
                   ", max depth = " + toString(m_MaxTreeDepth) +
                   ", max depth tolerance = " + toString(m_MaxTreeDepthTolerance) +
                   ", gamma = " + toString(m_Gamma) +
                   ", lambda = " + toString(m_Lambda) + ")";
        }

        //! Persist by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Populate the object from serialized data.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    private:
        static std::string toString(double x) { return std::to_string(x); }
        static std::string toString(TOptionalDouble x) {
            return x != boost::none ? toString(*x) : "null";
        }

    private:
        T m_Alpha = T{};
        T m_Gamma = T{};
        T m_Lambda = T{};
        T m_MaxTreeDepth = T{};
        T m_MaxTreeDepthTolerance = T{};
    };

    using TRegularization = CRegularization<double>;
    using TRegularizationOverride = CRegularization<TOptionalDouble>;

    //! \brief The algorithm parameters we'll directly optimise to improve test error.
    struct SHyperparameters {
        //! The regularisation parameters.
        TRegularization s_Regularization;

        //! Shrinkage.
        double s_Eta;

        //! Rate of growth of shrinkage in the training loop.
        double s_EtaGrowthRatePerTree;

        //! The fraction of features we use per bag.
        double s_FeatureBagFraction;

        //! Persist by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Populate the object from serialized data.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
    };

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
        //! See core::CMemory.
        static bool dynamicSizeAlwaysZero() { return true; }

        //! Check if this is a leaf node.
        bool isLeaf() const { return m_LeftChild < 0; }

        //! Get the leaf index for \p row.
        std::size_t leafIndex(const CEncodedDataFrameRowRef& row,
                              const TNodeVec& tree,
                              std::int32_t index = 0) const {
            if (this->isLeaf()) {
                return index;
            }
            double value{row[m_SplitFeature]};
            bool missing{CDataFrameUtils::isMissing(value)};
            return (missing && m_AssignMissingToLeft) ||
                           (missing == false && value < m_SplitValue)
                       ? tree[m_LeftChild].leafIndex(row, tree, m_LeftChild)
                       : tree[m_RightChild].leafIndex(row, tree, m_RightChild);
        }

        //! Get the value predicted by \p tree for the feature vector \p row.
        double value(const CEncodedDataFrameRowRef& row, const TNodeVec& tree) const {
            return tree[this->leafIndex(row, tree)].m_NodeValue;
        }

        //! Get the value of this node.
        double value() const { return m_NodeValue; }

        //! Set the node value to \p value.
        void value(double value) { m_NodeValue = value; }

        //! Get the gain of the split.
        double gain() const { return m_Gain; }

        //! Get the total curvature at the rows below this node.
        double curvature() const { return m_Curvature; }

        //! Split this node and add its child nodes to \p tree.
        std::pair<std::size_t, std::size_t> split(std::size_t splitFeature,
                                                  double splitValue,
                                                  bool assignMissingToLeft,
                                                  double gain,
                                                  double curvature,
                                                  TNodeVec& tree) {
            m_SplitFeature = splitFeature;
            m_SplitValue = splitValue;
            m_AssignMissingToLeft = assignMissingToLeft;
            m_LeftChild = static_cast<std::int32_t>(tree.size());
            m_RightChild = static_cast<std::int32_t>(tree.size() + 1);
            m_Gain = gain;
            m_Curvature = curvature;
            tree.resize(tree.size() + 2);
            return {m_LeftChild, m_RightChild};
        }

        //! Get the row masks of the left and right children of this node.
        auto rowMasks(std::size_t numberThreads,
                      const core::CDataFrame& frame,
                      const CDataFrameCategoryEncoder& encoder,
                      core::CPackedBitVector rowMask) const {

            LOG_TRACE(<< "Splitting feature '" << m_SplitFeature << "' @ " << m_SplitValue);
            LOG_TRACE(<< "# rows in node = " << rowMask.manhattan());
            LOG_TRACE(<< "row mask = " << rowMask);

            auto result = frame.readRows(
                numberThreads, 0, frame.numberRows(),
                core::bindRetrievableState(
                    [&](auto& state, TRowItr beginRows, TRowItr endRows) {
                        core::CPackedBitVector& leftRowMask{std::get<0>(state)};
                        std::size_t& leftChildNumberRows{std::get<1>(state)};
                        std::size_t& rightChildNumberRows{std::get<2>(state)};
                        for (auto row = beginRows; row != endRows; ++row) {
                            std::size_t index{row->index()};
                            double value{encoder.encode(*row)[m_SplitFeature]};
                            bool missing{CDataFrameUtils::isMissing(value)};
                            if ((missing && m_AssignMissingToLeft) ||
                                (missing == false && value < m_SplitValue)) {
                                leftRowMask.extend(false, index - leftRowMask.size());
                                leftRowMask.extend(true);
                                ++leftChildNumberRows;
                            } else {
                                ++rightChildNumberRows;
                            }
                        }
                    },
                    std::make_tuple(core::CPackedBitVector{}, std::size_t{0}, std::size_t{0})),
                &rowMask);
            auto& masks = result.first;

            for (auto& mask_ : masks) {
                auto& mask = std::get<0>(mask_.s_FunctionState);
                mask.extend(false, rowMask.size() - mask.size());
            }

            core::CPackedBitVector leftRowMask;
            std::size_t leftChildNumberRows;
            std::size_t rightChildNumberRows;
            std::tie(leftRowMask, leftChildNumberRows, rightChildNumberRows) =
                std::move(masks[0].s_FunctionState);
            for (std::size_t i = 1; i < masks.size(); ++i) {
                leftRowMask |= std::get<0>(masks[i].s_FunctionState);
                leftChildNumberRows += std::get<1>(masks[i].s_FunctionState);
                rightChildNumberRows += std::get<2>(masks[i].s_FunctionState);
            }
            LOG_TRACE(<< "# rows in left node = " << leftRowMask.manhattan());
            LOG_TRACE(<< "left row mask = " << leftRowMask);

            core::CPackedBitVector rightRowMask{std::move(rowMask)};
            rightRowMask ^= leftRowMask;
            LOG_TRACE(<< "# rows in right node = " << rightRowMask.manhattan());
            LOG_TRACE(<< "left row mask = " << rightRowMask);

            return std::make_tuple(std::move(leftRowMask), std::move(rightRowMask),
                                   leftChildNumberRows < rightChildNumberRows);
        }

        //! Get a human readable description of this tree.
        std::string print(const TNodeVec& tree) const {
            std::ostringstream result;
            return this->doPrint("", tree, result).str();
        }

        //! Persist by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Populate the object from serialized data.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

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
        double m_Gain = 0.0;
        double m_Curvature = 0.0;
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
                            const CDataFrameCategoryEncoder& encoder,
                            const TRegularization& regularization,
                            const TDoubleVecVec& candidateSplits,
                            std::size_t depth,
                            TSizeVec featureBag,
                            core::CPackedBitVector rowMask)
            : m_Id{id}, m_Regularization{regularization},
              m_CandidateSplits{candidateSplits}, m_Depth{depth},
              m_FeatureBag{std::move(featureBag)}, m_RowMask{std::move(rowMask)} {

            std::sort(m_FeatureBag.begin(), m_FeatureBag.end());
            LOG_TRACE(<< "row mask = " << m_RowMask);
            LOG_TRACE(<< "feature bag = " << core::CContainerPrinter::print(m_FeatureBag));

            this->computeAggregateLossDerivatives(numberThreads, frame, encoder);
        }

        //! This should only called by split but is public so it's accessible to std::make_shared.
        CLeafNodeStatistics(std::size_t id,
                            const CLeafNodeStatistics& parent,
                            const CLeafNodeStatistics& sibling,
                            core::CPackedBitVector rowMask);

        CLeafNodeStatistics(const CLeafNodeStatistics&) = delete;

        CLeafNodeStatistics(CLeafNodeStatistics&&) = default;

        CLeafNodeStatistics& operator=(const CLeafNodeStatistics&) = delete;

        CLeafNodeStatistics& operator=(CLeafNodeStatistics&&) = default;

        //! Apply the split defined by (\p leftChildRowMask, \p rightChildRowMask).
        auto split(std::size_t leftChildId,
                   std::size_t rightChildId,
                   std::size_t numberThreads,
                   const core::CDataFrame& frame,
                   const CDataFrameCategoryEncoder& encoder,
                   const TRegularization& regularization,
                   const TDoubleVecVec& candidateSplits,
                   TSizeVec featureBag,
                   core::CPackedBitVector leftChildRowMask,
                   core::CPackedBitVector rightChildRowMask,
                   bool leftChildHasFewerRows) {

            if (leftChildHasFewerRows) {
                auto leftChild = std::make_shared<CLeafNodeStatistics>(
                    leftChildId, numberThreads, frame, encoder, regularization,
                    candidateSplits, m_Depth + 1, std::move(featureBag),
                    std::move(leftChildRowMask));
                auto rightChild = std::make_shared<CLeafNodeStatistics>(
                    rightChildId, *this, *leftChild, std::move(rightChildRowMask));

                return std::make_pair(leftChild, rightChild);
            }

            auto rightChild = std::make_shared<CLeafNodeStatistics>(
                rightChildId, numberThreads, frame, encoder, regularization, candidateSplits,
                m_Depth + 1, std::move(featureBag), std::move(rightChildRowMask));
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

        double curvature() const {
            return this->bestSplitStatistics().s_Curvature;
        }

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

        //! Get the memory used by this object.
        std::size_t memoryUsage() const {
            std::size_t mem{core::CMemory::dynamicSize(m_FeatureBag)};
            mem += core::CMemory::dynamicSize(m_RowMask);
            mem += core::CMemory::dynamicSize(m_Gradients);
            mem += core::CMemory::dynamicSize(m_Curvatures);
            mem += core::CMemory::dynamicSize(m_MissingGradients);
            mem += core::CMemory::dynamicSize(m_MissingCurvatures);
            return mem;
        }

        //! Estimate maximum leaf statistics bookkeeping memory for training
        //! boosted trees on data frames with \p numberRows rows, \p numberCols columns
        //! with specified settings for \p featureBagFraction and \p numberSplitsPerFeature
        static std::size_t estimateMemoryUsage(std::size_t numberRows,
                                               std::size_t numberCols,
                                               double featureBagFraction,
                                               std::size_t numberSplitsPerFeature) {
            std::size_t featureBagSize{
                static_cast<std::size_t>(std::ceil(
                    featureBagFraction * static_cast<double>(numberCols - 1))) *
                sizeof(std::size_t)};
            // We will typically get the close to the best compression for most of the
            // leaves when the set of splits becomes large, corresponding to the worst
            // case for memory usage. This is because the rows will be spread over many
            // rows so the masks will mainly contain 0 bits in this case.
            std::size_t rowMaskSize{numberRows / PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE};
            std::size_t gradientsSize{(numberCols - 1) *
                                      numberSplitsPerFeature * sizeof(double)};
            std::size_t curvatureSize{gradientsSize};
            std::size_t missingGradientsSize{(numberCols - 1) * sizeof(double)};
            std::size_t missingCurvatureSize{missingGradientsSize};
            return sizeof(CLeafNodeStatistics) + featureBagSize + rowMaskSize + gradientsSize +
                   curvatureSize + missingGradientsSize + missingCurvatureSize;
        }

    private:
        //! \brief Statistics relating to a split of the node.
        struct SSplitStatistics : private boost::less_than_comparable<SSplitStatistics> {
            SSplitStatistics(double gain, double curvature, std::size_t feature, double splitAt, bool assignMissingToLeft)
                : s_Gain{gain}, s_Curvature{curvature}, s_Feature{feature}, s_SplitAt{splitAt},
                  s_AssignMissingToLeft{assignMissingToLeft} {}

            bool operator<(const SSplitStatistics& rhs) const {
                return COrderings::lexicographical_compare(
                    s_Gain, s_Curvature, s_Feature, rhs.s_Gain, rhs.s_Curvature, rhs.s_Feature);
            }

            std::string print() const {
                std::ostringstream result;
                result << "split feature '" << s_Feature << "' @ " << s_SplitAt
                       << ", gain = " << s_Gain;
                return result.str();
            }

            double s_Gain;
            double s_Curvature;
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

            void merge(const SDerivatives& other) {
                for (std::size_t i = 0; i < s_Gradients.size(); ++i) {
                    for (std::size_t j = 0; j < s_Gradients[i].size(); ++j) {
                        s_Gradients[i][j] += other.s_Gradients[i][j];
                        s_Curvatures[i][j] += other.s_Curvatures[i][j];
                    }
                    s_MissingGradients[i] += other.s_MissingGradients[i];
                    s_MissingCurvatures[i] += other.s_MissingCurvatures[i];
                }
            }

            auto move() {
                return std::make_tuple(std::move(s_Gradients), std::move(s_Curvatures),
                                       std::move(s_MissingGradients),
                                       std::move(s_MissingCurvatures));
            }

            TDoubleVecVec s_Gradients;
            TDoubleVecVec s_Curvatures;
            TDoubleVec s_MissingGradients;
            TDoubleVec s_MissingCurvatures;
        };

    private:
        void computeAggregateLossDerivatives(std::size_t numberThreads,
                                             const core::CDataFrame& frame,
                                             const CDataFrameCategoryEncoder& encoder) {

            auto result = frame.readRows(
                numberThreads, 0, frame.numberRows(),
                core::bindRetrievableState(
                    [&](SDerivatives& derivatives, TRowItr beginRows, TRowItr endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            this->addRowDerivatives(encoder.encode(*row), derivatives);
                        }
                    },
                    SDerivatives{m_CandidateSplits}),
                &m_RowMask);

            SDerivatives derivatives{std::move(result.first[0].s_FunctionState)};
            for (std::size_t i = 1; i < result.first.size(); ++i) {
                derivatives.merge(result.first[i].s_FunctionState);
            }

            std::tie(m_Gradients, m_Curvatures, m_MissingGradients,
                     m_MissingCurvatures) = derivatives.move();

            LOG_TRACE(<< "gradients = " << core::CContainerPrinter::print(m_Gradients));
            LOG_TRACE(<< "curvatures = " << core::CContainerPrinter::print(m_Curvatures));
            LOG_TRACE(<< "missing gradients = "
                      << core::CContainerPrinter::print(m_MissingGradients));
            LOG_TRACE(<< "missing curvatures = "
                      << core::CContainerPrinter::print(m_MissingCurvatures));
        }

        void addRowDerivatives(const CEncodedDataFrameRowRef& row,
                               SDerivatives& derivatives) const;

        const SSplitStatistics& bestSplitStatistics() const {
            if (m_BestSplit == boost::none) {
                m_BestSplit = this->computeBestSplitStatistics();
            }
            return *m_BestSplit;
        }

        SSplitStatistics computeBestSplitStatistics() const;

    private:
        std::size_t m_Id;
        const TRegularization& m_Regularization;
        const TDoubleVecVec& m_CandidateSplits;
        std::size_t m_Depth;
        TSizeVec m_FeatureBag;
        core::CPackedBitVector m_RowMask;
        TDoubleVecVec m_Gradients;
        TDoubleVecVec m_Curvatures;
        TDoubleVec m_MissingGradients;
        TDoubleVec m_MissingCurvatures;
        mutable boost::optional<SSplitStatistics> m_BestSplit;
    };

private:
    // The maximum number of rows encoded by a single byte in the packed bit
    // vector assuming best compression.
    static const std::size_t PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE;

private:
    CBoostedTreeImpl();

    //! Check if we can train a model.
    bool canTrain() const;

    //! Get the full training set data mask, i.e. all rows which aren't missing
    //! the dependent variable.
    core::CPackedBitVector allTrainingRowsMask() const;

    //! Compute the \p percentile percentile gain per split and the sum of row
    //! curvatures per internal node of \p forest.
    TDoubleDoublePr gainAndCurvatureAtPercentile(double percentile,
                                                 const TNodeVecVec& forest) const;

    //! Train the forest and compute loss moments on each fold.
    TMeanVarAccumulator crossValidateForest(core::CDataFrame& frame,
                                            const TMemoryUsageCallback& recordMemoryUsage) const;

    //! Initialize the predictions and loss function derivatives for the masked
    //! rows in \p frame.
    TNodeVec initializePredictionsAndLossDerivatives(core::CDataFrame& frame,
                                                     const core::CPackedBitVector& trainingRowMask) const;

    //! Train one forest on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVecVec trainForest(core::CDataFrame& frame,
                            const core::CPackedBitVector& trainingRowMask,
                            const TMemoryUsageCallback& recordMemoryUsage) const;

    //! Get the candidate splits values for each feature.
    TDoubleVecVec candidateSplits(const core::CDataFrame& frame,
                                  const core::CPackedBitVector& trainingRowMask) const;

    //! Train one tree on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVec trainTree(core::CDataFrame& frame,
                       const core::CPackedBitVector& trainingRowMask,
                       const TDoubleVecVec& candidateSplits,
                       const std::size_t maximumTreeSize,
                       const TMemoryUsageCallback& recordMemoryUsage) const;

    //! Get the number of features including category encoding.
    std::size_t numberFeatures() const;

    //! Get the number of features to consider splitting on.
    std::size_t featureBagSize() const;

    //! Sample the features according to their categorical distribution.
    TSizeVec featureBag() const;

    //! Refresh the predictions and loss function derivatives for the masked
    //! rows in \p frame with predictions of \p tree.
    void refreshPredictionsAndLossDerivatives(core::CDataFrame& frame,
                                              const core::CPackedBitVector& trainingRowMask,
                                              double eta,
                                              TNodeVec& tree) const;

    //! Compute the mean of the loss function on the masked rows of \p frame.
    double meanLoss(const core::CDataFrame& frame,
                    const core::CPackedBitVector& rowMask,
                    const TNodeVecVec& forest) const;

    //! Get a column mask of the suitable regressor features.
    TSizeVec candidateRegressorFeatures() const;

    //! Get the root node of \p tree.
    static const CNode& root(const TNodeVec& tree);

    //! Get the forest's prediction for \p row.
    static double predictRow(const CEncodedDataFrameRowRef& row, const TNodeVecVec& forest);

    //! Select the next hyperparameters for which to train a model.
    bool selectNextHyperparameters(const TMeanVarAccumulator& lossMoments,
                                   CBayesianOptimisation& bopt);

    //! Capture the current hyperparameter values.
    void captureBestHyperparameters(const TMeanVarAccumulator& lossMoments);

    //! Set the hyperparamaters from the best recorded.
    void restoreBestHyperparameters();

    //! Get the number of hyperparameters to tune.
    std::size_t numberHyperparametersToTune() const;

    //! Get the maximum number of nodes to use in a tree.
    //!
    //! \note This number will only be used if the regularised loss says its
    //! a good idea.
    std::size_t maximumTreeSize(const core::CPackedBitVector& trainingRowMask) const;

    //! Get the maximum number of nodes to use in a tree.
    //!
    //! \note This number will only be used if the regularised loss says its
    //! a good idea.
    std::size_t maximumTreeSize(std::size_t numberRows) const;

    //! Restore \p loss function pointer from the \p traverser.
    static bool restoreLoss(CBoostedTree::TLossFunctionUPtr& loss,
                            core::CStateRestoreTraverser& traverser);

    //! Record the training state using the \p recordTrainState callback function
    void recordState(const TTrainingStateCallback& recordTrainState) const;

private:
    static const double INF;

private:
    mutable CPRNG::CXorOShiro128Plus m_Rng;
    std::size_t m_NumberThreads;
    std::size_t m_DependentVariable = std::numeric_limits<std::size_t>::max();
    CBoostedTree::TLossFunctionUPtr m_Loss;
    TRegularizationOverride m_RegularizationOverride;
    TOptionalDouble m_EtaOverride;
    TOptionalSize m_MaximumNumberTreesOverride;
    TOptionalDouble m_FeatureBagFractionOverride;
    TRegularization m_Regularization;
    double m_Eta = 0.1;
    double m_EtaGrowthRatePerTree = 1.05;
    std::size_t m_NumberFolds = 4;
    std::size_t m_MaximumNumberTrees = 20;
    std::size_t m_MaximumAttemptsToAddTree = 3;
    std::size_t m_NumberSplitsPerFeature = 75;
    std::size_t m_MaximumOptimisationRoundsPerHyperparameter = 3;
    std::size_t m_RowsPerFeature = 50;
    double m_FeatureBagFraction = 0.5;
    double m_MaximumTreeSizeMultiplier = 1.0;
    TDataTypeVec m_FeatureDataTypes;
    TDataFrameCategoryEncoderUPtr m_Encoder;
    TDoubleVec m_FeatureSampleProbabilities;
    TPackedBitVectorVec m_MissingFeatureRowMasks;
    TPackedBitVectorVec m_TrainingRowMasks;
    TPackedBitVectorVec m_TestingRowMasks;
    double m_BestForestTestLoss = INF;
    SHyperparameters m_BestHyperparameters;
    TNodeVecVec m_BestForest;
    TBayesinOptimizationUPtr m_BayesianOptimization;
    std::size_t m_NumberRounds = 1;
    std::size_t m_CurrentRound = 0;
    mutable core::CLoopProgress m_TrainingProgress;

    friend class CBoostedTreeFactory;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeImpl_h
