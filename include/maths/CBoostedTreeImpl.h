/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeImpl_h
#define INCLUDED_ml_maths_CBoostedTreeImpl_h

#include <core/CDataFrame.h>
#include <core/CImmutableRadixSet.h>
#include <core/CMemory.h>
#include <core/CPackedBitVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeHyperparameters.h>
#include <maths/CBoostedTreeLeafNodeStatistics.h>
#include <maths/CBoostedTreeLoss.h>
#include <maths/CBoostedTreeUtils.h>
#include <maths/CDataFrameAnalysisInstrumentationInterface.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>

#include <boost/optional.hpp>

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
class CBoostedTreeImplForTest;
class CTreeShapFeatureImportance;
namespace boosted_tree {
class CArgMinLoss;
}

//! \brief Implementation of CBoostedTree.
class MATHS_EXPORT CBoostedTreeImpl final {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TStrVec = std::vector<std::string>;
    using TOptionalDouble = boost::optional<double>;
    using TStrDoublePrVec = std::vector<std::pair<std::string, double>>;
    using TOptionalStrDoublePrVec = boost::optional<TStrDoublePrVec>;
    using TVector = CDenseVector<double>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanVarAccumulatorSizeDoubleTuple =
        std::tuple<TMeanVarAccumulator, std::size_t, double>;
    using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;
    using TBayesinOptimizationUPtr = std::unique_ptr<maths::CBayesianOptimisation>;
    using TNodeVec = CBoostedTree::TNodeVec;
    using TNodeVecVec = CBoostedTree::TNodeVecVec;
    using TLossFunction = boosted_tree::CLoss;
    using TLossFunctionUPtr = CBoostedTree::TLossFunctionUPtr;
    using TTrainingStateCallback = CBoostedTree::TTrainingStateCallback;
    using TRecordEncodersCallback = CBoostedTree::TRecordEncodersCallback;
    using TRegularization = CBoostedTreeRegularization<double>;
    using TAnalysisInstrumentationPtr = CDataFrameTrainBoostedTreeInstrumentationInterface*;
    using THyperparameterImportanceVec =
        std::vector<boosted_tree_detail::SHyperparameterImportance>;

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

    //! Get the SHAP value calculator.
    //!
    //! \warning Will return a nullptr if a trained model isn't available.
    CTreeShapFeatureImportance* shap();

    //! Get the vector of hyperparameter importances.
    THyperparameterImportanceVec hyperparameterImportance() const;

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

    //! Get start indices of the extra columns.
    const TSizeVec& extraColumns() const;

    //! Get the weights to apply to each class's predicted probability when
    //! assigning classes.
    const TVector& classificationWeights() const;

    //! Get the number of columns training the model will add to the data frame.
    static std::size_t numberExtraColumnsForTrain(std::size_t numberLossParameters) {
        // We store as follows:
        //   1. The predicted values for the dependent variable
        //   2. The gradient of the loss function
        //   3. The upper triangle of the hessian of the loss function
        //   4. The example's weight
        return numberLossParameters * (numberLossParameters + 5) / 2 + 1;
    }

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Estimate the maximum booking memory that training a boosted tree on a data
    //! frame with \p numberRows row and \p numberColumns columns will use.
    std::size_t estimateMemoryUsageTrain(std::size_t numberRows, std::size_t numberColumns) const;

    //! Estimate the maximum booking memory that incrementally training a boosted
    //! tree on a data frame with \p numberRows row and \p numberColumns columns
    //! will use.
    std::size_t estimateMemoryUsageTrainIncremental(std::size_t numberRows,
                                                    std::size_t numberColumns) const;

    //! Correct from worst case memory usage to a more realistic estimate.
    static std::size_t correctedMemoryUsage(double memoryUsageBytes);

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Visit this tree trainer implementation.
    void accept(CBoostedTree::CVisitor& visitor);

    //! \return The best hyperparameters for validation error found so far.
    const CBoostedTreeHyperparameters& bestHyperparameters() const;

    //!\ name Test Only
    //@{
    //! The name of the object holding the best hyperaparameters in the state document.
    static const std::string& bestHyperparametersName();

    //! The name of the object holding the best regularisation hyperparameters in the
    //! state document.
    static const std::string& bestRegularizationHyperparametersName();

    //! A list of the names of the best individual hyperparameters in the state document.
    static TStrVec bestHyperparameterNames();

    //! Get the threshold on the predicted probability of class one at which to
    //!
    //! Get the feature sample probabilities.
    const TDoubleVec& featureSampleProbabilities() const;
    //@}

private:
    using TDoubleDoublePr = std::pair<double, double>;
    using TOptionalDoubleVec = std::vector<TOptionalDouble>;
    using TOptionalDoubleVecVec = std::vector<TOptionalDoubleVec>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TImmutableRadixSetVec = std::vector<core::CImmutableRadixSet<double>>;
    using TNodeVecVecDoubleDoubleVecTr = std::tuple<TNodeVecVec, double, TDoubleVec>;
    using TDataFrameCategoryEncoderUPtr = std::unique_ptr<CDataFrameCategoryEncoder>;
    using TDataTypeVec = CDataFrameUtils::TDataTypeVec;
    using TRegularizationOverride = CBoostedTreeRegularization<TOptionalDouble>;
    using TTreeShapFeatureImportanceUPtr = std::unique_ptr<CTreeShapFeatureImportance>;
    using TLeafNodeStatisticsPtr = CBoostedTreeLeafNodeStatistics::TPtr;
    using TWorkspace = CBoostedTreeLeafNodeStatistics::CWorkspace;
    using TArgMinLossVec = std::vector<boosted_tree::CArgMinLoss>;
    using TArgMinLossVecVec = std::vector<TArgMinLossVec>;
    using THyperparametersVec = std::vector<boosted_tree_detail::EHyperparameter>;
    // clang-format off
    using TMakeRootLeafNodeStatistics =
        std::function<TLeafNodeStatisticsPtr (const TImmutableRadixSetVec&,
                                              const TSizeVec&,
                                              const TSizeVec&,
                                              const core::CPackedBitVector&,
                                              TWorkspace&)>;
    // clang-format on

    //! Tag progress through initialization.
    enum EInitializationStage {
        E_NotInitialized = 0,
        E_SoftTreeDepthLimitInitialized = 1,
        E_DepthPenaltyMultiplierInitialized = 2,
        E_TreeSizePenaltyMultiplierInitialized = 3,
        E_LeafWeightPenaltyMultiplierInitialized = 4,
        E_DownsampleFactorInitialized = 5,
        E_FeatureBagFractionInitialized = 6,
        E_EtaInitialized = 7,
        E_FullyInitialized = 8
    };

private:
    CBoostedTreeImpl();

    //! Check if we can train a model.
    bool canTrain() const;

    //! Get the full training set data mask, i.e. all rows which aren't missing
    //! the dependent variable.
    core::CPackedBitVector allTrainingRowsMask() const;

    //! Compute the \p percentile percentile gain per split and the sum of row
    //! curvatures per internal node of \p forest.
    static TDoubleDoublePr gainAndCurvatureAtPercentile(double percentile,
                                                        const TNodeVecVec& forest);

    //! Presize the collection to hold the per fold test errors.
    void initializePerFoldTestLosses();

    //! Compute the probability threshold at which to classify a row as class one.
    void computeClassificationWeights(const core::CDataFrame& frame);

    //! Prepare to calculate SHAP feature importances.
    void initializeTreeShap(const core::CDataFrame& frame);

    //! Select the trees of the best forest to retrain.
    void selectTreesToRetrain(const core::CDataFrame& frame);

    //! Train the forest and compute loss moments on each fold.
    template<typename F>
    TMeanVarAccumulatorSizeDoubleTuple crossValidateForest(core::CDataFrame& frame,
                                                           std::size_t maximumNumberTrees,
                                                           const F& trainForest);

    //! Initialize the predictions and loss function derivatives for the masked
    //! rows in \p frame.
    TNodeVec initializePredictionsAndLossDerivatives(core::CDataFrame& frame,
                                                     const core::CPackedBitVector& trainingRowMask,
                                                     const core::CPackedBitVector& testingRowMask) const;

    //! Train one forest on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVecVecDoubleDoubleVecTr trainForest(core::CDataFrame& frame,
                                             const core::CPackedBitVector& trainingRowMask,
                                             const core::CPackedBitVector& testingRowMask,
                                             core::CLoopProgress& trainingProgress) const;

    //! Retrain a subset of the trees of one forest on the rows of \p frame in the
    //! mask \p trainingRowMask.
    TNodeVecVecDoubleDoubleVecTr
    updateForest(core::CDataFrame& frame,
                 const core::CPackedBitVector& trainingRowMask,
                 const core::CPackedBitVector& testingRowMask,
                 core::CLoopProgress& trainingProgress) const;

    //! Compute the learn rate for the tree at \p index.
    double etaForTreeAtPosition(std::size_t index) const;

    //! Randomly downsamples the training row mask by the downsample factor.
    core::CPackedBitVector downsample(const core::CPackedBitVector& trainingRowMask) const;

    //! Get the candidate splits values for each feature.
    TImmutableRadixSetVec candidateSplits(const core::CDataFrame& frame,
                                          const core::CPackedBitVector& trainingRowMask) const;

    //! Train one tree on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVec trainTree(core::CDataFrame& frame,
                       const core::CPackedBitVector& trainingRowMask,
                       const TImmutableRadixSetVec& candidateSplits,
                       std::size_t maximumNumberInternalNodes,
                       const TMakeRootLeafNodeStatistics& makeRootLeafNodeStatistics,
                       TWorkspace& workspace) const;

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
    std::size_t featureBagSize(double fractionMultiplier) const;

    //! Sample the features according to their categorical distribution.
    void treeFeatureBag(TDoubleVec& probabilities, TSizeVec& treeFeatureBag) const;

    //! Sample the features according to their categorical distribution.
    void nodeFeatureBag(const TSizeVec& treeFeatureBag,
                        TDoubleVec& probabilities,
                        TSizeVec& nodeFeatureBag) const;

    //! Get a column mask of the suitable regressor features.
    static void candidateRegressorFeatures(const TDoubleVec& probabilities, TSizeVec& features);

    //! Remove the predictions of \p tree from \p frame for the masked rows.
    void removePredictions(core::CDataFrame& frame,
                           const core::CPackedBitVector& trainingRowMask,
                           const core::CPackedBitVector& testingRowMask,
                           const TNodeVec& tree) const;

    //! Refresh the predictions and loss function derivatives for the masked
    //! rows in \p frame with predictions of \p tree.
    void refreshPredictionsAndLossDerivatives(core::CDataFrame& frame,
                                              const core::CPackedBitVector& trainingRowMask,
                                              const core::CPackedBitVector& testingRowMask,
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
                               const TNodeVec& tree,
                               TArgMinLossVecVec& result) const;

    //! Write \p loss gradient and curvature for the \p rowMask rows of \p frame.
    void writeRowDerivatives(bool newExample,
                             core::CDataFrame& frame,
                             const core::CPackedBitVector& rowMask,
                             const TLossFunction& loss,
                             const TNodeVec& tree) const;

    //! Compute the mean of the loss function on the masked rows of \p frame.
    double meanLoss(const core::CDataFrame& frame, const core::CPackedBitVector& rowMask) const;

    //! Compute the mean of the loss function on the masked rows of \p frame
    //! adjusted for incremental training.
    double meanAdjustedLoss(const core::CDataFrame& frame,
                            const core::CPackedBitVector& rowMask) const;

    //! Get the best forest's prediction for \p row.
    TVector predictRow(const CEncodedDataFrameRowRef& row) const;

    //! Select the next hyperparameters for which to train a model.
    bool selectNextHyperparameters(const TMeanVarAccumulator& lossMoments,
                                   CBayesianOptimisation& bopt);

    //! Capture the current hyperparameter values.
    //!
    //! \param[in] numberKeptNodes If incrementally training the number of nodes
    //! in the retained portion of the forest.
    //! \param[in] numberRetrainedNodes The number of trees in the new (portion
    //! of the) forest.
    void captureBestHyperparameters(const TMeanVarAccumulator& lossMoments,
                                    std::size_t maximumNumberTrees,
                                    double numberKeptNodes,
                                    double numberRetrainedNodes);

    //! Compute the loss penalty for model size.
    double modelSizePenalty(double numberKeptNodes, double numberRetrainedNodes) const;

    //! Set the hyperparamaters from the best recorded.
    void restoreBestHyperparameters();

    //! Scale the regulariser multipliers by \p scale.
    void scaleRegularizers(double scale);

    //! Check invariants which are assumed to hold after restoring.
    void checkRestoredInvariants() const;

    //! Check invariants which are assumed to hold in order to train on \p frame.
    void checkTrainInvariants(const core::CDataFrame& frame) const;

    //! Check invariants which are assumed to hold in order to incrementally
    //! train on \p frame.
    void checkIncrementalTrainInvariants(const core::CDataFrame& frame) const;

    //! Get the number of hyperparameters to tune.
    std::size_t numberHyperparametersToTune() const;

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

    //! Populate the list of tunable hyperparameters.
    void initializeTunableHyperparameters();

    //! Use Sobol sampler for for random hyperparamers.
    void initializeHyperparameterSamples();

private:
    mutable CPRNG::CXorOShiro128Plus m_Rng;
    EInitializationStage m_InitializationStage = E_NotInitialized;
    std::size_t m_NumberThreads;
    std::size_t m_DependentVariable = std::numeric_limits<std::size_t>::max();
    TOptionalSize m_PaddedExtraColumns;
    TSizeVec m_ExtraColumns;
    TLossFunctionUPtr m_Loss;
    CBoostedTree::EClassAssignmentObjective m_ClassAssignmentObjective =
        CBoostedTree::E_MinimumRecall;
    bool m_IncrementalTraining = false;
    bool m_StopCrossValidationEarly = true;
    TRegularizationOverride m_RegularizationOverride;
    TOptionalDouble m_DownsampleFactorOverride;
    TOptionalDouble m_EtaOverride;
    TOptionalDouble m_EtaGrowthRatePerTreeOverride;
    TOptionalDouble m_PredictionChangeCostOverride;
    TOptionalSize m_NumberFoldsOverride;
    TOptionalSize m_MaximumNumberTreesOverride;
    TOptionalDouble m_FeatureBagFractionOverride;
    TOptionalStrDoublePrVec m_ClassificationWeightsOverride;
    TRegularization m_Regularization;
    TVector m_ClassificationWeights;
    double m_DownsampleFactor = 0.5;
    double m_Eta = 0.1;
    double m_EtaGrowthRatePerTree = 1.05;
    double m_PredictionChangeCost = 0.5;
    std::size_t m_NumberFolds = 4;
    std::size_t m_MaximumNumberTrees = 20;
    std::size_t m_MaximumAttemptsToAddTree = 3;
    std::size_t m_NumberSplitsPerFeature = 75;
    std::size_t m_MaximumOptimisationRoundsPerHyperparameter = 2;
    std::size_t m_RowsPerFeature = 50;
    double m_FeatureBagFraction = 0.5;
    double m_RetrainFraction = 0.1;
    TDataFrameCategoryEncoderUPtr m_Encoder;
    TDataTypeVec m_FeatureDataTypes;
    TDoubleVec m_FeatureSampleProbabilities;
    TSizeVec m_TreesToRetrain;
    TPackedBitVectorVec m_MissingFeatureRowMasks;
    TPackedBitVectorVec m_TrainingRowMasks;
    TPackedBitVectorVec m_TestingRowMasks;
    core::CPackedBitVector m_NewTrainingRowMask;
    double m_BestForestTestLoss = boosted_tree_detail::INF;
    TOptionalDoubleVecVec m_FoldRoundTestLosses;
    CBoostedTreeHyperparameters m_BestHyperparameters;
    TNodeVecVec m_BestForest;
    TBayesinOptimizationUPtr m_BayesianOptimization;
    std::size_t m_NumberRounds = 1;
    std::size_t m_CurrentRound = 0;
    core::CLoopProgress m_TrainingProgress;
    std::size_t m_NumberTopShapValues = 0;
    TTreeShapFeatureImportanceUPtr m_TreeShap;
    TAnalysisInstrumentationPtr m_Instrumentation;
    TMeanAccumulator m_MeanForestSizeAccumulator;
    TMeanAccumulator m_MeanLossAccumulator;
    THyperparametersVec m_TunableHyperparameters;
    TDoubleVecVec m_HyperparameterSamples;
    bool m_StopHyperparameterOptimizationEarly = true;

private:
    friend class CBoostedTreeFactory;
    friend class CBoostedTreeImplForTest;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeImpl_h
