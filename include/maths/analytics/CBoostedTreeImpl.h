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

#ifndef INCLUDED_ml_maths_analytics_CBoostedTreeImpl_h
#define INCLUDED_ml_maths_analytics_CBoostedTreeImpl_h

#include <core/CDataFrame.h>
#include <core/CMemory.h>
#include <core/CPackedBitVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CBoostedTreeHyperparameters.h>
#include <maths/analytics/CBoostedTreeLeafNodeStatistics.h>
#include <maths/analytics/CBoostedTreeLoss.h>
#include <maths/analytics/CBoostedTreeUtils.h>
#include <maths/analytics/CDataFrameAnalysisInstrumentationInterface.h>
#include <maths/analytics/CDataFrameCategoryEncoder.h>
#include <maths/analytics/CDataFrameUtils.h>
#include <maths/analytics/ImportExport.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CPRNG.h>

#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

class CBoostedTreeImplForTest;

namespace ml {
namespace maths {
namespace analytics {
class CTreeShapFeatureImportance;
namespace boosted_tree {
class CArgMinLoss;
}

//! \brief Implementation of CBoostedTree.
class MATHS_ANALYTICS_EXPORT CBoostedTreeImpl final {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TOptionalDouble = std::optional<double>;
    using TOptionalDoubleVec = std::vector<TOptionalDouble>;
    using TOptionalDoubleVecVec = std::vector<TOptionalDoubleVec>;
    using TStrDoublePrVec = std::vector<std::pair<std::string, double>>;
    using TOptionalStrDoublePrVec = std::optional<TStrDoublePrVec>;
    using TVector = common::CDenseVector<double>;
    using TMeanVarAccumulator = common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;
    using TNodeVec = CBoostedTree::TNodeVec;
    using TNodeVecVec = CBoostedTree::TNodeVecVec;
    using TLossFunction = boosted_tree::CLoss;
    using TLossFunctionUPtr = CBoostedTree::TLossFunctionUPtr;
    using TTrainingStateCallback = CBoostedTree::TTrainingStateCallback;
    using TRecordEncodersCallback = CBoostedTree::TRecordEncodersCallback;
    using TAnalysisInstrumentationPtr = CDataFrameTrainBoostedTreeInstrumentationInterface*;

public:
    static const double MINIMUM_RELATIVE_GAIN_PER_SPLIT;

public:
    CBoostedTreeImpl(std::size_t numberThreads,
                     TLossFunctionUPtr loss,
                     TAnalysisInstrumentationPtr instrumentation = nullptr);
    CBoostedTreeImpl(CBoostedTreeImpl&&) noexcept;

    ~CBoostedTreeImpl();

    CBoostedTreeImpl& operator=(const CBoostedTreeImpl&) = delete;
    CBoostedTreeImpl& operator=(CBoostedTreeImpl&&) noexcept;

    //! Train the model on the values in \p frame.
    void train(core::CDataFrame& frame, const TTrainingStateCallback& recordTrainStateCallback);

    //! Incrementally train the current model on the values in \p frame.
    //!
    //! \warning Assumes that train has already been called or a trained model has
    //! been reloaded.
    void trainIncremental(core::CDataFrame& frame,
                          const TTrainingStateCallback& recordTrainStateCallback);

    //! Write the predictions of the best trained model to \p frame.
    //!
    //! \warning Must be called only if a trained model is available.
    void predict(core::CDataFrame& frame) const;

    //! Write the predictions of the best trained model to the masked rows of \p frame.
    //!
    //! \warning Must be called only if a trained model is available.
    void predict(const core::CPackedBitVector& rowMask, core::CDataFrame& frame) const;

    //! Get the hyperparameters.
    const CBoostedTreeHyperparameters& hyperparameters() const;

    //! Get the writeable hyperparameters.
    CBoostedTreeHyperparameters& hyperparameters();

    //! Get the SHAP value calculator.
    //!
    //! \warning Will return a nullptr if a trained model isn't available.
    CTreeShapFeatureImportance* shap();

    //! Get the selected rows that summarize \p dataFrame.
    core::CPackedBitVector dataSummarization(const core::CDataFrame& frame) const;

    //! Get the data frame row encoder.
    const CDataFrameCategoryEncoder& encoder() const;

    //! Get the model produced by training if it has been run.
    const TNodeVecVec& trainedModel() const;

    //! Get the training loss function.
    TLossFunction& loss() const;

    //! Get the column containing the dependent variable.
    std::size_t columnHoldingDependentVariable() const;

    //! Get a mask for the new training data.
    const core::CPackedBitVector& newTrainingRowMask() const;

    //! Get start indices of the extra columns.
    const TSizeVec& extraColumns() const;

    //! Get the weights to apply to each class's predicted probability when
    //! assigning classes.
    const TVector& classificationWeights() const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Estimate the maximum booking memory that train on a data frame with
    //! \p numberRows rows and \p numberColumns columns will use.
    std::size_t estimateMemoryUsageForTrain(std::size_t numberRows,
                                            std::size_t numberColumns) const;

    //! Estimate the maximum booking memory that trainIncremental on a data frame
    //! with \p numberRows rows and \p numberColumns columns will use.
    std::size_t estimateMemoryUsageForTrainIncremental(std::size_t numberRows,
                                                       std::size_t numberColumns) const;

    //! Estimate the maximum booking memory that predict on a data frame with
    //! \p numberRows rows and \p numberColumns columns will use.
    std::size_t estimateMemoryUsageForPredict(std::size_t numberRows,
                                              std::size_t numberColumns) const;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Visit this tree trainer implementation.
    void accept(CBoostedTree::CVisitor& visitor);

    //! \return The full training set data mask, i.e. all rows which aren't missing
    //! the dependent variable.
    core::CPackedBitVector allTrainingRowMask() const;

    //! Get the mean number of training examples which are used in each fold.
    double meanNumberTrainingRowsPerFold() const;

    //!\ name Test Only
    //@{
    //! Get the threshold on the predicted probability of class one at which to
    //!
    //! Get the feature sample probabilities.
    const TDoubleVec& featureSampleProbabilities() const;

    //! Get the fold test losses for each round.
    const TOptionalDoubleVecVec& foldRoundTestLosses() const;
    //@}

private:
    using TBoolVec = std::vector<bool>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TFloatVec = std::vector<common::CFloatStorage>;
    using TFloatVecVec = std::vector<TFloatVec>;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TDoubleParameter = CBoostedTreeParameter<double>;
    using TSizeParameter = CBoostedTreeParameter<std::size_t>;
    using TDataFrameCategoryEncoderUPtr = std::unique_ptr<CDataFrameCategoryEncoder>;
    using TDataTypeVec = CDataFrameUtils::TDataTypeVec;
    using TTreeShapFeatureImportanceUPtr = std::unique_ptr<CTreeShapFeatureImportance>;
    using TLeafNodeStatisticsPtr = CBoostedTreeLeafNodeStatistics::TPtr;
    using TWorkspace = CBoostedTreeLeafNodeStatistics::CWorkspace;
    using TArgMinLossVec = std::vector<boosted_tree::CArgMinLoss>;
    using TArgMinLossVecVec = std::vector<TArgMinLossVec>;
    using TRowRef = boosted_tree_detail::TRowRef;
    using TMemoryMappedFloatVector = boosted_tree_detail::TMemoryMappedFloatVector;
    // clang-format off
    using TMakeRootLeafNodeStatistics =
        std::function<TLeafNodeStatisticsPtr (const TFloatVecVec&,
                                              const TSizeVec&,
                                              const TSizeVec&,
                                              const core::CPackedBitVector&,
                                              TWorkspace&)>;
    using TUpdateRowPrediction =
        std::function<void (const TRowRef&, TMemoryMappedFloatVector&)>;
    // clang-format on

    //! \brief The result of cross-validation.
    struct SCrossValidationResult {
        TNodeVecVec s_Forest;
        TMeanVarAccumulator s_TestLossMoments;
        double s_MeanLossGap{0.0};
        std::size_t s_NumberTrees{0};
        double s_NumberNodes{0.0};
    };

    //! \brief The result of training a single forest.
    struct STrainForestResult {
        std::tuple<TNodeVecVec, TMeanVarAccumulator, double, TDoubleVec> asTuple() {
            return {std::move(s_Forest), s_TestLoss, s_LossGap, std::move(s_TestLosses)};
        }
        TNodeVecVec s_Forest;
        TMeanVarAccumulator s_TestLoss{0.0};
        double s_LossGap{0.0};
        TDoubleVec s_TestLosses;
    };

    //! Tag progress through initialization.
    enum EInitializationStage {
        E_NotInitialized = 0,
        E_EncodingInitialized = 1,
        E_SoftTreeDepthLimitInitialized = 2,
        E_DepthPenaltyMultiplierInitialized = 3,
        E_TreeSizePenaltyMultiplierInitialized = 4,
        E_LeafWeightPenaltyMultiplierInitialized = 5,
        E_DownsampleFactorInitialized = 6,
        E_FeatureBagFractionInitialized = 7,
        E_EtaInitialized = 8,
        E_FullyInitialized = 9
    };

private:
    CBoostedTreeImpl();

    //! Get the loss gap we expect after incremental training.
    double expectedLossGapAfterTrainIncremental(double numberOldTrainingRows,
                                                double numberNewTrainingRows) const;

    //! Check if we can train a model.
    bool canTrain() const;

    //! Compute the \p percentile percentile gain per split and the sum of row
    //! curvatures per internal node of \p forest.
    static TDoubleDoublePr gainAndCurvatureAtPercentile(double percentile,
                                                        const TNodeVecVec& forest);

    //! Presize the collection to hold the per fold test errors.
    TDoubleVec initializePerFoldTestLosses();

    //! Compute the probability threshold at which to classify a row as class one.
    void computeClassificationWeights(const core::CDataFrame& frame);

    //! Prepare to calculate SHAP feature importances.
    void initializeTreeShap(const core::CDataFrame& frame);

    //! Select the trees of the best forest to retrain.
    void selectTreesToRetrain(const core::CDataFrame& frame);

    //! Train the forest and compute loss moments on each fold.
    template<typename F>
    SCrossValidationResult crossValidateForest(core::CDataFrame& frame,
                                               std::size_t maximumNumberTrees,
                                               const F& trainForest,
                                               TDoubleVec& minTestLosses);

    //! Initialize the predictions and loss function derivatives for the masked
    //! rows in \p frame.
    TNodeVec initializePredictionsAndLossDerivatives(core::CDataFrame& frame,
                                                     const core::CPackedBitVector& trainingRowMask,
                                                     const core::CPackedBitVector& testingRowMask) const;

    //! Train one forest on the rows of \p frame in the mask \p trainingRowMask.
    STrainForestResult
    trainForest(core::CDataFrame& frame,
                const core::CPackedBitVector& trainingRowMask,
                const core::CPackedBitVector& testingRowMask,
                core::CLoopProgress& trainingProgress,
                double minTestLoss = std::numeric_limits<double>::max()) const;

    //! Retrain a subset of the trees of one forest on the rows of \p frame in the
    //! mask \p trainingRowMask.
    STrainForestResult updateForest(core::CDataFrame& frame,
                                    const core::CPackedBitVector& trainingRowMask,
                                    const core::CPackedBitVector& testingRowMask,
                                    core::CLoopProgress& trainingProgress) const;

    //! Randomly downsamples the training row mask by the downsample factor.
    core::CPackedBitVector downsample(const core::CPackedBitVector& trainingRowMask) const;

    //! Set the candidate splits for low cardinality features which remain
    //! fixed for the duration of training.
    void initializeFixedCandidateSplits(core::CDataFrame& frame);

    //! Get the candidate splits values for each feature.
    TFloatVecVec candidateSplits(const core::CDataFrame& frame,
                                 const core::CPackedBitVector& trainingRowMask) const;

    //! Updates the row's cached splits if the candidate splits have changed.
    void refreshSplitsCache(core::CDataFrame& frame,
                            const TFloatVecVec& candidateSplits,
                            const TBoolVec& featureMask,
                            const core::CPackedBitVector& trainingRowMask) const;

    //! Train one tree on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVec trainTree(core::CDataFrame& frame,
                       const core::CPackedBitVector& trainingRowMask,
                       const TFloatVecVec& candidateSplits,
                       std::size_t maximumNumberInternalNodes,
                       const TMakeRootLeafNodeStatistics& makeRootLeafNodeStatistics,
                       TWorkspace& workspace) const;

    //! Scale the multipliers of the regularisation terms in the loss function to
    //! account for differences in training data set sizes.
    void scaleRegularizationMultipliers(double scale);

    //! Compute the minimum mean test loss per fold for any round.
    double minimumTestLoss() const;

    //! Estimate the loss we'll get including the missing folds.
    TMeanVarAccumulator correctTestLossMoments(const TSizeVec& missing,
                                               TMeanVarAccumulator testLossMoments) const;

    //! Estimate test losses for the \p missing folds.
    TMeanVarAccumulatorVec estimateMissingTestLosses(const TSizeVec& missing) const;

    //! Get the minimum number of rows we require per feature.
    std::size_t rowsPerFeature(std::size_t numberRows) const;

    //! Get the number of features including category encoding.
    std::size_t numberFeatures() const;

    //! Get the number of features to consider splitting on.
    std::size_t featureBagSize(double fractionMultiplier) const;

    //! Sample the features according to their categorical distribution.
    void treeFeatureBag(TDoubleVec& probabilities, TSizeVec& treeFeatureBag) const;

    //! Sample the features according to their categorical distribution.
    void nodeFeatureBag(const TSizeVec& treeFeatureBag,
                        TDoubleVec& probabilities,
                        TSizeVec& nodeFeatureBag) const;

    //! Get a column mask of the suitable regressor features.
    static void candidateRegressorFeatures(const TDoubleVec& probabilities, TSizeVec& features);

    //! Compute the leaf values to use for \p tree.
    void computeLeafValues(core::CDataFrame& frame,
                           const core::CPackedBitVector& trainingRowMask,
                           const TLossFunction& loss,
                           double eta,
                           double lambda,
                           TNodeVec& tree) const;

    //! Extract the leaf values for \p tree which minimize \p loss on \p rowMask
    //! rows of \p frame.
    void minimumLossLeafValues(bool newExample,
                               const core::CDataFrame& frame,
                               const core::CPackedBitVector& rowMask,
                               const TLossFunction& loss,
                               const TSizeVec& leafMap,
                               const TNodeVec& tree,
                               TArgMinLossVecVec& result) const;

    //! Update the predictions and the \p loss gradient and curvature for the
    //! \p rowMask rows of \p frame for all data.
    void refreshPredictionsAndLossDerivatives(core::CDataFrame& frame,
                                              const core::CPackedBitVector& rowMask,
                                              const TLossFunction& loss,
                                              const TUpdateRowPrediction& updateRowPrediction) const;

    //! Update the predictions and the \p loss gradient and curvature for the
    //! \p rowMask rows of \p frame for old or new data.
    void refreshPredictionsAndLossDerivatives(bool newExample,
                                              core::CDataFrame& frame,
                                              const core::CPackedBitVector& rowMask,
                                              const TLossFunction& loss,
                                              const TUpdateRowPrediction& updateRowPrediction) const;

    //! Update the predictions \p rowMask rows of \p frame.
    void refreshPredictions(core::CDataFrame& frame,
                            const core::CPackedBitVector& rowMask,
                            const TLossFunction& loss,
                            const TUpdateRowPrediction& updateRowPrediction) const;

    //! Compute the mean of the loss function on the masked rows of \p frame.
    TMeanVarAccumulator meanLoss(const core::CDataFrame& frame,
                                 const core::CPackedBitVector& rowMask) const;

    //! Compute the mean of the loss function on the masked rows of \p frame
    //! adjusted for incremental training.
    TMeanVarAccumulator meanChangePenalisedLoss(const core::CDataFrame& frame,
                                                const core::CPackedBitVector& rowMask) const;

    //! Compute the overall variance of the error we see between folds.
    double betweenFoldTestLossVariance() const;

    //! Get the best forest's prediction for \p row.
    TVector predictRow(const CEncodedDataFrameRowRef& row) const;

    //! Check invariants which are assumed to hold after restoring.
    void checkRestoredInvariants() const;

    //! Check invariants which are assumed to hold in order to train on \p frame.
    void checkTrainInvariants(const core::CDataFrame& frame) const;

    //! Check invariants which are assumed to hold in order to incrementally
    //! train on \p frame.
    void checkIncrementalTrainInvariants(const core::CDataFrame& frame) const;

    //! Get the maximum number of nodes to use in a tree.
    //!
    //! \note This number will only be used if the regularised loss says its
    //! a good idea.
    static std::size_t maximumTreeSize(const core::CPackedBitVector& trainingRowMask);

    //! Get the maximum number of nodes to use in a tree.
    //!
    //! \note This number will only be used if the regularised loss says its
    //! a good idea.
    static std::size_t maximumTreeSize(std::size_t numberRows);

    //! Get the number of trees to retrain.
    std::size_t numberTreesToRetrain() const;

    //! Get the maximum memory of any trained model we will produce.
    //!
    //! This is the largest model we will train if there is a limit on the
    //! deployed model.
    std::size_t maximumTrainedModelSize() const;

    //! Start monitoring fine tuning hyperparameters.
    void startProgressMonitoringFineTuneHyperparameters();

    //! Start monitoring the final model training.
    void startProgressMonitoringFinalTrain();

    //! Skip monitoring the final model training.
    void skipProgressMonitoringFinalTrain();

    //! Start progress monitoring incremental training.
    void startProgressMonitoringTrainIncremental();

    //! Record the training state using the \p recordTrainState callback function
    void recordState(const TTrainingStateCallback& recordTrainState) const;

    //! Record hyperparameters for instrumentation.
    void recordHyperparameters();

    //! Estimate the memory usage for training (either from scratch or incremental).
    std::size_t estimateMemoryUsageForTraining(std::size_t numberRows,
                                               std::size_t numberColumns,
                                               std::size_t numberTrees) const;

    //! Correct from worst case memory usage to a more realistic estimate.
    static std::size_t correctedMemoryUsageForTraining(double memoryUsageBytes);

private:
    //! \name Parameters
    //@{
    std::uint64_t m_Seed{0};
    mutable common::CPRNG::CXorOShiro128Plus m_Rng;
    std::size_t m_NumberThreads{1};
    std::size_t m_DependentVariable{std::numeric_limits<std::size_t>::max()};
    std::size_t m_MaximumDeployedSize{std::numeric_limits<std::size_t>::max()};
    TSizeVec m_ExtraColumns;
    TLossFunctionUPtr m_Loss;
    EInitializationStage m_InitializationStage{E_NotInitialized};
    std::size_t m_MaximumAttemptsToAddTree{3};
    CBoostedTreeHyperparameters m_Hyperparameters;
    //@}

    //! \name Cross-validation
    //@{
    TSizeParameter m_NumberFolds{4};
    TDoubleParameter m_TrainFractionPerFold{0.75};
    bool m_UserSuppliedHoldOutSet{false};
    bool m_StopCrossValidationEarly{true};
    TOptionalDoubleVecVec m_FoldRoundTestLosses;
    //@}

    //! \name Features
    //@{
    std::size_t m_NumberSplitsPerFeature{75};
    std::size_t m_RowsPerFeature{50};
    TDataFrameCategoryEncoderUPtr m_Encoder;
    TDataTypeVec m_FeatureDataTypes;
    TDoubleVec m_FeatureSampleProbabilities;
    TFloatVecVec m_FixedCandidateSplits;
    //@}

    //! \name Training Data
    //@{
    TPackedBitVectorVec m_MissingFeatureRowMasks;
    TPackedBitVectorVec m_TrainingRowMasks;
    TPackedBitVectorVec m_TestingRowMasks;
    core::CPackedBitVector m_NewTrainingRowMask;
    //@}

    //! \name Model
    //@{
    TNodeVecVec m_BestForest;
    CBoostedTree::EClassAssignmentObjective m_ClassAssignmentObjective{CBoostedTree::E_MinimumRecall};
    TOptionalStrDoublePrVec m_ClassificationWeightsOverride;
    TVector m_ClassificationWeights;
    //@}

    //! \name Feature Importance
    //@{
    std::size_t m_NumberTopShapValues{0};
    TTreeShapFeatureImportanceUPtr m_TreeShap;
    //@}

    //! \name Monitoring
    //@{
    TAnalysisInstrumentationPtr m_Instrumentation;
    core::CLoopProgress m_TrainingProgress;
    //@}

    //! \name Incremental Train
    //@{
    bool m_ForceAcceptIncrementalTraining{false};
    double m_DataSummarizationFraction{0.0};
    double m_RetrainFraction{0.1};
    double m_PreviousTrainLossGap{0.0};
    std::size_t m_PreviousTrainNumberRows{0};
    std::size_t m_MaximumNumberNewTrees{0};
    TSizeVec m_TreesToRetrain;
    //@}

private:
    friend class CBoostedTreeFactory;
    friend class CBoostedTreeHyperparameters;
    friend class ::CBoostedTreeImplForTest;
};
}
}
}

#endif // INCLUDED_ml_maths_analytics_CBoostedTreeImpl_h
