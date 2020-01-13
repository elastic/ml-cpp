/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeImpl_h
#define INCLUDED_ml_maths_CBoostedTreeImpl_h

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CImmutableRadixSet.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CPackedBitVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeHyperparameters.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPRNG.h>
#include <maths/CTools.h>
#include <maths/ImportExport.h>

#include <boost/operators.hpp>
#include <boost/optional.hpp>
#include <boost/range/irange.hpp>

#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace core {
template<typename>
class CImmutableRadixSet;
}
namespace maths {
class CBayesianOptimisation;

//! \brief Implementation of CBoostedTree.
class MATHS_EXPORT CBoostedTreeImpl final {
public:
    using TDoubleVec = std::vector<double>;
    using TStrVec = std::vector<std::string>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanVarAccumulatorSizePr = std::pair<TMeanVarAccumulator, std::size_t>;
    using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;
    using TBayesinOptimizationUPtr = std::unique_ptr<maths::CBayesianOptimisation>;
    using TNodeVec = CBoostedTree::TNodeVec;
    using TNodeVecVec = CBoostedTree::TNodeVecVec;
    using TLossFunctionUPtr = CBoostedTree::TLossFunctionUPtr;
    using TProgressCallback = CBoostedTree::TProgressCallback;
    using TMemoryUsageCallback = CBoostedTree::TMemoryUsageCallback;
    using TTrainingStateCallback = CBoostedTree::TTrainingStateCallback;
    using TOptionalDouble = boost::optional<double>;
    using TRegularization = CBoostedTreeRegularization<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeRange = boost::integer_range<std::size_t>;

public:
    static const double MINIMUM_RELATIVE_GAIN_PER_SPLIT;

public:
    CBoostedTreeImpl(std::size_t numberThreads, TLossFunctionUPtr loss);

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

    //! Compute SHAP values using the best trained model to \p frame.
    //!
    //! \note Must be called only if a trained model is available.
    void computeShapValues(core::CDataFrame& frame, const TProgressCallback&);

    //! Get the threshold on the predicted probability of class one at which to assign
    //! the row to class one.
    double decisionThreshold(const core::CDataFrame& frame) const;

    //! Get the feature sample probabilities.
    const TDoubleVec& featureSampleProbabilities() const;

    //! Get the model produced by training if it has been run.
    const TNodeVecVec& trainedModel() const;

    //! Get the column containing the dependent variable.
    std::size_t columnHoldingDependentVariable() const;

    //! Get the number of columns training the model will add to the data frame.
    constexpr static std::size_t numberExtraColumnsForTrain() {
        // We store as follows:
        //   1. The predicted value for the dependent variable
        //   2. The gradient of the loss function
        //   3. The curvature of the loss function
        //   4. The example's weight
        // In the last four rows of the data frame.
        return 4;
    }

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Estimate the maximum booking memory that training the boosted tree on a data
    //! frame with \p numberRows row and \p numberColumns columns will use.
    std::size_t estimateMemoryUsage(std::size_t numberRows, std::size_t numberColumns) const;

    //! The name of the object holding the best hyperaparameters in the state document.
    static const std::string& bestHyperparametersName();

    //! The name of the object holding the best regularisation hyperparameters in the
    //! state document.
    static const std::string& bestRegularizationHyperparametersName();

    //! A list of the names of the best individual hyperparameters in the state document.
    static TStrVec bestHyperparameterNames();

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Visit this tree trainer implementation.
    void accept(CBoostedTree::CVisitor& visitor);

    //! \return The best hyperparameters for validation error found so far.
    const CBoostedTreeHyperparameters& bestHyperparameters() const;

    //! Get the indices of the columns containing SHAP values.
    TSizeRange columnsHoldingShapValues() const;

    //! Get the number of largest SHAP values that will be returned for every row.
    std::size_t topShapValues() const;

    //! Get the number of columns in the original data frame.
    std::size_t numberInputColumns() const;

private:
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TOptionalDoubleVec = std::vector<TOptionalDouble>;
    using TOptionalDoubleVecVec = std::vector<TOptionalDoubleVec>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TImmutableRadixSetVec = std::vector<core::CImmutableRadixSet<double>>;
    using TVector = CDenseVector<double>;
    using TRowItr = core::CDataFrame::TRowItr;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TNodeVecVecDoublePr = std::pair<TNodeVecVec, double>;
    using TDataFrameCategoryEncoderUPtr = std::unique_ptr<CDataFrameCategoryEncoder>;
    using TDataTypeVec = CDataFrameUtils::TDataTypeVec;
    using TRegularizationOverride = CBoostedTreeRegularization<TOptionalDouble>;

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
                            const TImmutableRadixSetVec& candidateSplits,
                            const TSizeVec& featureBag,
                            std::size_t depth,
                            const core::CPackedBitVector& rowMask);

        //! Only called by split but is public so it's accessible to std::make_shared.
        CLeafNodeStatistics(std::size_t id,
                            std::size_t numberThreads,
                            const core::CDataFrame& frame,
                            const CDataFrameCategoryEncoder& encoder,
                            const TRegularization& regularization,
                            const TImmutableRadixSetVec& candidateSplits,
                            const TSizeVec& featureBag,
                            bool isLeftChild,
                            std::size_t depth,
                            const CBoostedTreeNode& split,
                            const core::CPackedBitVector& parentRowMask);
        //! Only called by split but is public so it's accessible to std::make_shared.
        CLeafNodeStatistics(std::size_t id,
                            const CLeafNodeStatistics& parent,
                            const CLeafNodeStatistics& sibling,
                            const TRegularization& regularization,
                            const TSizeVec& featureBag,
                            core::CPackedBitVector rowMask);

        CLeafNodeStatistics(const CLeafNodeStatistics&) = delete;

        // Move construction/assignment not possible due to const reference member

        CLeafNodeStatistics& operator=(const CLeafNodeStatistics&) = delete;

        //! Apply the split defined by \p split.
        //!
        //! \return Shared pointers to the left and right child node statistics.
        auto split(std::size_t leftChildId,
                   std::size_t rightChildId,
                   std::size_t numberThreads,
                   const core::CDataFrame& frame,
                   const CDataFrameCategoryEncoder& encoder,
                   const TRegularization& regularization,
                   const TImmutableRadixSetVec& candidateSplits,
                   const TSizeVec& featureBag,
                   const CBoostedTreeNode& split,
                   bool leftChildHasFewerRows);

        //! Order two leaves by decreasing gain in splitting them.
        bool operator<(const CLeafNodeStatistics& rhs) const {
            return m_BestSplit < rhs.m_BestSplit;
        }

        //! Get the gain in loss of the best split of this leaf.
        double gain() const { return m_BestSplit.s_Gain; }

        //! Get the total curvature of node.
        double curvature() const { return m_BestSplit.s_Curvature; }

        //! Get the best (feature, feature value) split.
        TSizeDoublePr bestSplit() const {
            return {m_BestSplit.s_Feature, m_BestSplit.s_SplitAt};
        }

        //! Check if the left child has fewer rows than the right child.
        bool leftChildHasFewerRows() const {
            return m_BestSplit.s_LeftChildHasFewerRows;
        }

        //! Check if we should assign the missing feature rows to the left child
        //! of the split.
        bool assignMissingToLeft() const {
            return m_BestSplit.s_AssignMissingToLeft;
        }

        //! Get the node's identifier.
        std::size_t id() const { return m_Id; }

        //! Get the row mask for this leaf node.
        core::CPackedBitVector& rowMask() { return m_RowMask; }

        //! Get the memory used by this object.
        std::size_t memoryUsage() const {
            std::size_t mem{core::CMemory::dynamicSize(m_RowMask)};
            mem += core::CMemory::dynamicSize(m_Derivatives);
            mem += core::CMemory::dynamicSize(m_MissingDerivatives);
            return mem;
        }

        //! Estimate the maximum leaf statistics' memory usage training on a data frame
        //! with \p numberRows rows and \p numberCols columns using \p featureBagFraction
        //! and \p numberSplitsPerFeature.
        static std::size_t estimateMemoryUsage(std::size_t numberRows,
                                               std::size_t numberCols,
                                               std::size_t numberSplitsPerFeature) {
            // We will typically get the close to the best compression for most of the
            // leaves when the set of splits becomes large, corresponding to the worst
            // case for memory usage. This is because the rows will be spread over many
            // rows so the masks will mainly contain 0 bits in this case.
            std::size_t rowMaskSize{numberRows / PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE};
            std::size_t derivativesSize{(numberCols - 1) * numberSplitsPerFeature *
                                        sizeof(SAggregateDerivatives)};
            std::size_t missingDerivativesSize{(numberCols - 1) * sizeof(SAggregateDerivatives)};
            return sizeof(CLeafNodeStatistics) + rowMaskSize + derivativesSize + missingDerivativesSize;
        }

    private:
        //! \brief Statistics relating to a split of the node.
        struct SSplitStatistics : private boost::less_than_comparable<SSplitStatistics> {
            SSplitStatistics() = default;
            SSplitStatistics(double gain,
                             double curvature,
                             std::size_t feature,
                             double splitAt,
                             bool leftChildHasFewerRows,
                             bool assignMissingToLeft)
                : s_Gain{gain}, s_Curvature{curvature}, s_Feature{feature}, s_SplitAt{splitAt},
                  s_LeftChildHasFewerRows{leftChildHasFewerRows}, s_AssignMissingToLeft{assignMissingToLeft} {
            }

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

            double s_Gain = -INF;
            double s_Curvature = 0.0;
            std::size_t s_Feature = -1;
            double s_SplitAt = INF;
            bool s_LeftChildHasFewerRows = true;
            bool s_AssignMissingToLeft = true;
        };

        //! \brief Aggregate derivatives.
        struct SAggregateDerivatives {
            void add(std::size_t count, double gradient, double curvature) {
                s_Count += count;
                s_Gradient += gradient;
                s_Curvature += curvature;
            }

            void merge(const SAggregateDerivatives& other) {
                this->add(other.s_Count, other.s_Gradient, other.s_Curvature);
            }

            std::string print() const {
                std::ostringstream result;
                result << "count = " << s_Count << ", gradient = " << s_Gradient
                       << ", curvature = " << s_Curvature;
                return result.str();
            }

            std::size_t s_Count = 0;
            double s_Gradient = 0.0;
            double s_Curvature = 0.0;
        };

        using TAggregateDerivativesVec = std::vector<SAggregateDerivatives>;
        using TAggregateDerivativesVecVec = std::vector<TAggregateDerivativesVec>;

        //! \brief A collection of aggregate derivatives for candidate feature splits.
        struct SSplitAggregateDerivatives {
            SSplitAggregateDerivatives(const TImmutableRadixSetVec& candidateSplits)
                : s_Derivatives(candidateSplits.size()),
                  s_MissingDerivatives(candidateSplits.size()) {
                for (std::size_t i = 0; i < candidateSplits.size(); ++i) {
                    s_Derivatives[i].resize(candidateSplits[i].size() + 1);
                }
            }

            void merge(const SSplitAggregateDerivatives& other) {
                for (std::size_t i = 0; i < s_Derivatives.size(); ++i) {
                    for (std::size_t j = 0; j < s_Derivatives[i].size(); ++j) {
                        s_Derivatives[i][j].merge(other.s_Derivatives[i][j]);
                    }
                    s_MissingDerivatives[i].merge(other.s_MissingDerivatives[i]);
                }
            }

            auto move() {
                return std::make_pair(std::move(s_Derivatives),
                                      std::move(s_MissingDerivatives));
            }

            TAggregateDerivativesVecVec s_Derivatives;
            TAggregateDerivativesVec s_MissingDerivatives;
        };

    private:
        void computeAggregateLossDerivatives(std::size_t numberThreads,
                                             const core::CDataFrame& frame,
                                             const CDataFrameCategoryEncoder& encoder);
        void computeRowMaskAndAggregateLossDerivatives(std::size_t numberThreads,
                                                       const core::CDataFrame& frame,
                                                       const CDataFrameCategoryEncoder& encoder,
                                                       bool isLeftChild,
                                                       const CBoostedTreeNode& split,
                                                       const core::CPackedBitVector& parentRowMask);

        void addRowDerivatives(const CEncodedDataFrameRowRef& row,
                               SSplitAggregateDerivatives& splitAggregateDerivatives) const;

        SSplitStatistics computeBestSplitStatistics(const TRegularization& regularization,
                                                    const TSizeVec& featureBag) const;

    private:
        std::size_t m_Id;
        const TImmutableRadixSetVec& m_CandidateSplits;
        std::size_t m_Depth;
        core::CPackedBitVector m_RowMask;
        TAggregateDerivativesVecVec m_Derivatives;
        TAggregateDerivativesVec m_MissingDerivatives;
        SSplitStatistics m_BestSplit;
    };

private:
    // The maximum number of rows encoded by a single byte in the packed bit
    // vector assuming best compression.
    static const std::size_t PACKED_BIT_VECTOR_MAXIMUM_ROWS_PER_BYTE;
    static const double INF;

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

    //! Presize the collection to hold the per fold test errors.
    void initializePerFoldTestLosses();

    //! Train the forest and compute loss moments on each fold.
    TMeanVarAccumulatorSizePr crossValidateForest(core::CDataFrame& frame,
                                                  const TMemoryUsageCallback& recordMemoryUsage);

    //! Initialize the predictions and loss function derivatives for the masked
    //! rows in \p frame.
    TNodeVec initializePredictionsAndLossDerivatives(core::CDataFrame& frame,
                                                     const core::CPackedBitVector& trainingRowMask,
                                                     const core::CPackedBitVector& testingRowMask) const;

    //! Train one forest on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVecVecDoublePr trainForest(core::CDataFrame& frame,
                                    const core::CPackedBitVector& trainingRowMask,
                                    const core::CPackedBitVector& testingRowMask,
                                    core::CLoopProgress& trainingProgress,
                                    const TMemoryUsageCallback& recordMemoryUsage) const;

    //! Randomly downsamples the training row mask by the downsample factor.
    core::CPackedBitVector downsample(const core::CPackedBitVector& trainingRowMask) const;

    //! Get the candidate splits values for each feature.
    TImmutableRadixSetVec candidateSplits(const core::CDataFrame& frame,
                                          const core::CPackedBitVector& trainingRowMask) const;

    //! Train one tree on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVec trainTree(core::CDataFrame& frame,
                       const core::CPackedBitVector& trainingRowMask,
                       const TImmutableRadixSetVec& candidateSplits,
                       const std::size_t maximumTreeSize,
                       const TMemoryUsageCallback& recordMemoryUsage) const;

    //! Compute the minimum mean test loss per fold for any round.
    double minimumTestLoss() const;

    //! Estimate the loss we'll get including the missing folds.
    TMeanVarAccumulator correctTestLossMoments(const TSizeVec& missing,
                                               TMeanVarAccumulator lossMoments) const;

    //! Estimate test losses for the \p missing folds.
    TMeanVarAccumulatorVec estimateMissingTestLosses(const TSizeVec& missing) const;

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
                                              const core::CPackedBitVector& testingRowMask,
                                              double eta,
                                              TNodeVec& tree) const;

    //! Compute the mean of the loss function on the masked rows of \p frame.
    double meanLoss(const core::CDataFrame& frame, const core::CPackedBitVector& rowMask) const;

    //! Get a column mask of the suitable regressor features.
    TSizeVec candidateRegressorFeatures() const;

    //! Get the root node of \p tree.
    static const CBoostedTreeNode& root(const TNodeVec& tree);

    //! Get the forest's prediction for \p row.
    static double predictRow(const CEncodedDataFrameRowRef& row, const TNodeVecVec& forest);

    //! Select the next hyperparameters for which to train a model.
    bool selectNextHyperparameters(const TMeanVarAccumulator& lossMoments,
                                   CBayesianOptimisation& bopt);

    //! Capture the current hyperparameter values.
    void captureBestHyperparameters(const TMeanVarAccumulator& lossMoments,
                                    std::size_t maximumNumberTrees);

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
    static bool restoreLoss(TLossFunctionUPtr& loss, core::CStateRestoreTraverser& traverser);

    //! Record the training state using the \p recordTrainState callback function
    void recordState(const TTrainingStateCallback& recordTrainState) const;

private:
    mutable CPRNG::CXorOShiro128Plus m_Rng;
    std::size_t m_NumberThreads;
    std::size_t m_DependentVariable = std::numeric_limits<std::size_t>::max();
    TLossFunctionUPtr m_Loss;
    CBoostedTree::EClassAssignmentObjective m_ClassAssignmentObjective;
    bool m_StopCrossValidationEarly = true;
    TRegularizationOverride m_RegularizationOverride;
    TOptionalDouble m_DownsampleFactorOverride;
    TOptionalDouble m_EtaOverride;
    TOptionalSize m_NumberFoldsOverride;
    TOptionalSize m_MaximumNumberTreesOverride;
    TOptionalDouble m_FeatureBagFractionOverride;
    TRegularization m_Regularization;
    double m_DownsampleFactor = 0.5;
    double m_Eta = 0.1;
    double m_EtaGrowthRatePerTree = 1.05;
    std::size_t m_NumberFolds = 4;
    std::size_t m_MaximumNumberTrees = 20;
    std::size_t m_MaximumAttemptsToAddTree = 3;
    std::size_t m_NumberSplitsPerFeature = 75;
    std::size_t m_MaximumOptimisationRoundsPerHyperparameter = 2;
    std::size_t m_RowsPerFeature = 50;
    double m_FeatureBagFraction = 0.5;
    TDataTypeVec m_FeatureDataTypes;
    TDataFrameCategoryEncoderUPtr m_Encoder;
    TDoubleVec m_FeatureSampleProbabilities;
    TPackedBitVectorVec m_MissingFeatureRowMasks;
    TPackedBitVectorVec m_TrainingRowMasks;
    TPackedBitVectorVec m_TestingRowMasks;
    double m_BestForestTestLoss = INF;
    TOptionalDoubleVecVec m_FoldRoundTestLosses;
    CBoostedTreeHyperparameters m_BestHyperparameters;
    TNodeVecVec m_BestForest;
    TBayesinOptimizationUPtr m_BayesianOptimization;
    std::size_t m_NumberRounds = 1;
    std::size_t m_CurrentRound = 0;
    core::CLoopProgress m_TrainingProgress;
    std::size_t m_TopShapValues = 0;
    std::size_t m_FirstShapColumnIndex = 0;
    std::size_t m_LastShapColumnIndex = 0;
    std::size_t m_NumberInputColumns = 0;

private:
    friend class CBoostedTreeFactory;
};

namespace boosted_tree_detail {
constexpr inline std::size_t predictionColumn(std::size_t numberColumns) {
    return numberColumns - CBoostedTreeImpl::numberExtraColumnsForTrain();
}

constexpr inline std::size_t lossGradientColumn(std::size_t numberColumns) {
    return predictionColumn(numberColumns) + 1;
}

constexpr inline std::size_t lossCurvatureColumn(std::size_t numberColumns) {
    return predictionColumn(numberColumns) + 2;
}

constexpr inline std::size_t exampleWeightColumn(std::size_t numberColumns) {
    return predictionColumn(numberColumns) + 3;
}
}
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeImpl_h
