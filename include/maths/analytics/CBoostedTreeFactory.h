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

#ifndef INCLUDED_ml_maths_analytics_CBoostedTreeFactory_h
#define INCLUDED_ml_maths_analytics_CBoostedTreeFactory_h

#include <core/CDataFrame.h>
#include <core/CDataSearcher.h>
#include <core/CNonCopyable.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CBoostedTreeUtils.h>
#include <maths/analytics/CDataFrameAnalysisInstrumentationInterface.h>
#include <maths/analytics/ImportExport.h>

#include <maths/common/CLinearAlgebra.h>

#include <boost/optional.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CPackedBitVector;
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
namespace analytics {
class CNode;
class CBoostedTree;
class CBoostedTreeImpl;

//! Factory for CBoostedTree objects.
class MATHS_ANALYTICS_EXPORT CBoostedTreeFactory final {
public:
    using TDoubleVec = std::vector<double>;
    using TStrDoublePrVec = std::vector<std::pair<std::string, double>>;
    using TBoostedTreeUPtr = std::unique_ptr<CBoostedTree>;
    using TTrainingStateCallback = CBoostedTree::TTrainingStateCallback;
    using TLossFunctionUPtr = CBoostedTree::TLossFunctionUPtr;
    using TEncoderUPtr = std::unique_ptr<CDataFrameCategoryEncoder>;
    using TStrSizeUMap = boost::unordered_map<std::string, std::size_t>;
    using TRestoreDataSummarizationFunc =
        std::function<std::pair<TEncoderUPtr, TStrSizeUMap>(core::CDataSearcher::TIStreamP, core::CDataFrame&)>;
    using TNode = CBoostedTreeNode;
    using TNodeVec = std::vector<TNode>;
    using TNodeVecVec = std::vector<TNodeVec>;
    using TNodeVecVecUPtr = std::unique_ptr<TNodeVecVec>;
    using TRestoreBestForestFunc =
        std::function<TNodeVecVecUPtr(core::CDataSearcher::TIStreamP, const TStrSizeUMap&)>;

public:
    //! \name Instrumentation Phases
    //@{
    static const std::string FEATURE_SELECTION;
    static const std::string COARSE_PARAMETER_SEARCH;
    static const std::string FINE_TUNING_PARAMETERS;
    static const std::string FINAL_TRAINING;
    static const std::string INCREMENTAL_TRAIN;
    //@}

public:
    //! Construct a boosted tree object from parameters.
    static CBoostedTreeFactory constructFromParameters(std::size_t numberThreads,
                                                       TLossFunctionUPtr loss);

    //! Construct a boosted tree object from its serialized version.
    //!
    //! \warning Throws runtime error on fail to restore.
    static CBoostedTreeFactory constructFromString(std::istream& jsonStream);

    //! Constructs a boosted tree object using data from the previously trained model.
    //!
    //! \warning Calls HANDLE_FATAL (which calls std::terminate unless overridden) on
    //! fail to restore.
    static CBoostedTreeFactory
    constructFromDefinition(std::size_t numberThreads,
                            TLossFunctionUPtr loss,
                            core::CDataSearcher& dataSearcher,
                            core::CDataFrame& frame,
                            const TRestoreDataSummarizationFunc& dataSummarizationRestoreCallback,
                            const TRestoreBestForestFunc& bestForestRestoreCallback);

    //! Construct from the supplied \p model.
    //!
    //! \note This can be used for preparing for incremental training.
    static CBoostedTreeFactory constructFromModel(TBoostedTreeUPtr model);

    //! Get the maximum number of rows we'll train on.
    static std::size_t maximumNumberRows();

    ~CBoostedTreeFactory();
    CBoostedTreeFactory(CBoostedTreeFactory&) = delete;
    CBoostedTreeFactory& operator=(CBoostedTreeFactory&) = delete;
    CBoostedTreeFactory(CBoostedTreeFactory&&) noexcept;
    CBoostedTreeFactory& operator=(CBoostedTreeFactory&&) noexcept;

    //! Set the random number generator seed.
    CBoostedTreeFactory& seed(std::uint64_t seed);
    //! Set the objective to use when choosing the class assignments.
    CBoostedTreeFactory&
    classAssignmentObjective(CBoostedTree::EClassAssignmentObjective objective);
    //! Set the class weights used for assigning classes from predicted probabilities.
    CBoostedTreeFactory& classificationWeights(TStrDoublePrVec weights);
    //! Set the column containing the row weights to use for training.
    CBoostedTreeFactory& rowWeightColumnName(std::string columnName);
    //! Set the minimum fraction with a category value to one-hot encode.
    CBoostedTreeFactory& minimumFrequencyToOneHotEncode(double frequency);
    //! Set the number of initial rows to use as a holdout set for evaluation.
    CBoostedTreeFactory& numberHoldoutRows(std::size_t numberHoldoutRows);
    //! Set the number of folds to use for estimating the generalisation error.
    CBoostedTreeFactory& numberFolds(std::size_t numberFolds);
    //! Set the fraction fold data to use for training.
    CBoostedTreeFactory& trainFractionPerFold(double fraction);
    //! Set the maximum number of rows to use for training when tuning hyperparameters.
    CBoostedTreeFactory& maximumNumberTrainRows(std::size_t rows);
    //! Stratify the cross-validation we do for regression.
    CBoostedTreeFactory& stratifyRegressionCrossValidation(bool stratify);
    //! Stop cross-validation early if the test loss is not promising.
    CBoostedTreeFactory& stopCrossValidationEarly(bool stopEarly);
    //! The number of rows per feature to sample in the initial downsample.
    CBoostedTreeFactory& initialDownsampleRowsPerFeature(double rowsPerFeature);
    //! The amount by which to downsample the data for stochastic gradient estimates.
    CBoostedTreeFactory& downsampleFactor(TDoubleVec factor);
    //! Set the sum of leaf depth penalties multiplier.
    CBoostedTreeFactory& depthPenaltyMultiplier(TDoubleVec multiplier);
    //! Set the tree size penalty multiplier.
    CBoostedTreeFactory& treeSizePenaltyMultiplier(TDoubleVec multiplier);
    //! Set the sum of weights squared multiplier.
    CBoostedTreeFactory& leafWeightPenaltyMultiplier(TDoubleVec multiplier);
    //! Set the penalty for changing the tree toppology when incrementally training.
    CBoostedTreeFactory& treeTopologyChangePenalty(TDoubleVec penalty);
    //! Set the soft tree depth limit.
    CBoostedTreeFactory& softTreeDepthLimit(TDoubleVec limit);
    //! Set the soft tree depth tolerance. This controls how hard we'll try to
    //! respect the soft tree depth limit.
    CBoostedTreeFactory& softTreeDepthTolerance(TDoubleVec tolerance);
    //! Set the fractional relative tolerance in the target maximum tree depth.
    CBoostedTreeFactory& maxTreeDepthTolerance(TDoubleVec tolerance);
    //! Set the amount we'll shrink the tree leaf weights we compute.
    CBoostedTreeFactory& eta(TDoubleVec eta);
    //! Set the amount we'll shrink the retrained tree leaf weights we compute.
    CBoostedTreeFactory& retrainedTreeEta(TDoubleVec eta);
    //! Set the amount we'll grow eta on each each iteration.
    CBoostedTreeFactory& etaGrowthRatePerTree(TDoubleVec growthRate);
    //! Set the maximum number of trees in the ensemble.
    CBoostedTreeFactory& maximumNumberTrees(std::size_t maximumNumberTrees);
    //! Set the maximum supported size for deploying a model.
    CBoostedTreeFactory& maximumDeployedSize(std::size_t maximumDeployedSize);
    //! Set the fraction of features we'll use in the bag to build a tree.
    CBoostedTreeFactory& featureBagFraction(TDoubleVec fraction);
    //! Set the relative weight to assign changing old predictions in the loss
    //! function for incremental training.
    CBoostedTreeFactory& predictionChangeCost(TDoubleVec cost);
    //! Set the maximum number of optimisation rounds we'll use for hyperparameter
    //! optimisation per parameter for fine tuning.
    CBoostedTreeFactory& maximumOptimisationRoundsPerHyperparameter(std::size_t rounds);
    //! Set the number of restarts to use in global probing for Bayesian Optimisation.
    CBoostedTreeFactory& bayesianOptimisationRestarts(std::size_t restarts);
    //! Set the number of training examples we need per feature we'll include.
    CBoostedTreeFactory& rowsPerFeature(std::size_t rowsPerFeature);
    //! Set the number of training examples we need per feature we'll include.
    CBoostedTreeFactory& numberTopShapValues(std::size_t numberTopShapValues);
    //! Set the flag to enable or disable early stopping.
    CBoostedTreeFactory& stopHyperparameterOptimizationEarly(bool enable);
    //! Set the fraction of data rows for data summarization in (0.0, 1.0].
    CBoostedTreeFactory& dataSummarizationFraction(double fraction);
    //! Set the row mask for new data with which we want to incrementally train.
    CBoostedTreeFactory& newTrainingRowMask(core::CPackedBitVector rowMask);
    //! Set the fraction of trees in the forest to retrain.
    CBoostedTreeFactory& retrainFraction(double fraction);
    //! Set the gap between the train and test loss for the last train run.
    CBoostedTreeFactory& previousTrainLossGap(double gap);
    //! Set the number of rows for the last train run.
    CBoostedTreeFactory& previousTrainNumberRows(std::size_t numberRows);
    //! Set the maximum number of trees that can be added during an incremental training step.
    CBoostedTreeFactory& maximumNumberNewTrees(std::size_t maximumNumberNewTrees);
    //! Set whether or not to always accept the result of incremental training.
    CBoostedTreeFactory& forceAcceptIncrementalTraining(bool force);
    //! Set whether or not to scale regularisation hyperaparameters when varying
    //! downsample factor.
    CBoostedTreeFactory& disableHyperparameterScaling(bool disabled);
    //! Set the data summarization information.
    CBoostedTreeFactory& featureEncoder(TEncoderUPtr encoder);
    //! Set the best forest from the previous training.
    CBoostedTreeFactory& bestForest(TNodeVecVec forest);

    //! Set pointer to the analysis instrumentation.
    CBoostedTreeFactory&
    analysisInstrumentation(CDataFrameTrainBoostedTreeInstrumentationInterface& instrumentation);
    //! Set the callback function for training state recording.
    CBoostedTreeFactory& trainingStateCallback(TTrainingStateCallback callback);

    //! Estimate the maximum booking memory used computing encodings for a
    //! frame with \p numberRows, \p numberColumns and \p numberCategoricalColumns.
    std::size_t estimateMemoryUsageForEncode(std::size_t numberRows,
                                             std::size_t numberColumns,
                                             std::size_t numberCategoricalColumns) const;
    //! Estimate the maximum booking memory used training a model on a data frame
    //! with \p numberRows and \p numberColumns.
    std::size_t estimateMemoryUsageForTrain(std::size_t numberRows,
                                            std::size_t numberColumns) const;
    //! Estimate the maximum booking memory used incrementally training a model
    //! on a data frame with \p numberRows and \p numberColumns.
    std::size_t estimateMemoryUsageForTrainIncremental(std::size_t numberRows,
                                                       std::size_t numberColumns) const;
    //! Estimate the maximum booking memory used when predicting a model on a data
    //! frame with \p numberRows and \p numberColumns.
    std::size_t estimateMemoryUsageForPredict(std::size_t numberRows,
                                              std::size_t numberColumns) const;
    //! Estimate the number of columns computing encodings will add to the data frame.
    static std::size_t estimateExtraColumnsForEncode();
    //! Estimate the number of columns training the model will add to the data frame.
    static std::size_t estimateExtraColumnsForTrain(std::size_t numberColumns,
                                                    std::size_t numberLossParameters);
    //! Estimate the number of columns updating the model will add to the data frame.
    static std::size_t estimateExtraColumnsForTrainIncremental(std::size_t numberColumns,
                                                               std::size_t numberLossParameters);
    //! Estimate the number of columns predicting the model will add to the data frame.
    static std::size_t estimateExtraColumnsForPredict(std::size_t numberLossParameters);

    //! Build a boosted tree object for encoding on \p frame.
    TBoostedTreeUPtr buildForEncode(core::CDataFrame& frame, std::size_t dependentVariable);

    //! Build a boosted tree object for training on \p frame.
    TBoostedTreeUPtr buildForTrain(core::CDataFrame& frame, std::size_t dependentVariable);

    //! Build a boosted tree object for prediction on \p frame.
    TBoostedTreeUPtr buildForPredict(core::CDataFrame& frame, std::size_t dependentVariable);

    //! Build a boosted tree object for incremental training on \p frame.
    TBoostedTreeUPtr buildForTrainIncremental(core::CDataFrame& frame,
                                              std::size_t dependentVariable);

    //! Restore a boosted tree object for training on \p frame.
    //!
    //! \warning A tree object can only be restored once.
    TBoostedTreeUPtr restoreFor(core::CDataFrame& frame, std::size_t dependentVariable);

private:
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TOptionalDouble = boost::optional<double>;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TBoostedTreeImplUPtr = std::unique_ptr<CBoostedTreeImpl>;

private:
    CBoostedTreeFactory(std::size_t numberThreads, TLossFunctionUPtr loss);

    //! Persist state writing to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore readining state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Compute the row masks for the missing values for each feature.
    void initializeMissingFeatureMasks(const core::CDataFrame& frame) const;

    //! Set up the number of folds we'll use for cross-validation.
    void initializeNumberFolds(core::CDataFrame& frame) const;

    //! Initialize data frame information required for encoding.
    void prepareDataFrameForEncode(core::CDataFrame& frame) const;

    //! Resize the data frame with the extra columns used by train.
    void prepareDataFrameForTrain(core::CDataFrame& frame) const;

    //! Resize the data frame with the extra columns used by incremental train.
    void prepareDataFrameForIncrementalTrain(core::CDataFrame& frame) const;

    //! Resize the data frame with the extra columns used by prediction.
    void prepareDataFrameForPredict(core::CDataFrame& frame) const;

    //! Set up cross validation.
    void initializeCrossValidation(core::CDataFrame& frame) const;

    //! Encode categorical fields and at the same time select the features to use.
    void selectFeaturesAndEncodeCategories(core::CDataFrame& frame) const;

    //! Initialize the cache used for storing row splits.
    void initializeSplitsCache(core::CDataFrame& frame) const;

    //! Determine the encoded feature types.
    void determineFeatureDataTypes(const core::CDataFrame& frame) const;

    //! Initialize the regressors sample distribution.
    void initializeFeatureSampleDistribution() const;

    //! Apply scaling to hyperparameter ranges based on the difference in the update
    //! data set and original train data set sizes.
    void initialHyperparameterScaling();

    //! Setup before setting initial values for hyperparameters.
    void initializeHyperparametersSetup(core::CDataFrame& frame);

    //! Set the initial values for hyperparameters.
    void initializeHyperparameters(core::CDataFrame& frame);

    //! Estimate a good initial value and bounding box to search for regularisation
    //! hyperparameters.
    void initializeUnsetRegularizationHyperparameters(core::CDataFrame& frame);

    //! Estimate a good initial value and range to search for the feature bag
    //! fraction.
    void initializeUnsetFeatureBagFraction(core::CDataFrame& frame);

    //! Estimate a good initial value and range to search for the downsample
    //! factor.
    void initializeUnsetDownsampleFactor(core::CDataFrame& frame);

    //! Estimate a good initial value and range to search for the learn rate.
    void initializeUnsetEta(core::CDataFrame& frame);

    //! Estimate a good initial value and range to search for the learn rate
    //! to use for retrained trees when training incrementally.
    void initializeUnsetRetrainedTreeEta();

    //! Estimate a good initial value and range to search for the cost of
    //! changing predictions when training incrementally.
    void initializeUnsetPredictionChangeCost();

    //! Estimate a good initial value and range to search for the for tree
    //! topology penalty when training incrementally.
    void initializeUnsetTreeTopologyPenalty(core::CDataFrame& frame);

    //! Estimate the reduction in gain from a split and the total curvature of
    //! the loss function at a split.
    TDoubleDoublePrVec estimateTreeGainAndCurvature(core::CDataFrame& frame,
                                                    const TDoubleVec& percentiles) const;

    //! Get the number of hyperparameter tuning rounds to use.
    std::size_t numberHyperparameterTuningRounds() const;

    //! Start progress monitoring feature selection.
    void startProgressMonitoringFeatureSelection();

    //! Start progress monitoring initializeHyperparameters.
    void startProgressMonitoringInitializeHyperparameters(const core::CDataFrame& frame);

    //! Get the number of progress iterations used for a line search.
    std::size_t lineSearchMaximumNumberIterations(const core::CDataFrame& frame,
                                                  double etaScale = 1.0) const;

    //! The maximum number of trees to use in the hyperparameter optimisation loop.
    std::size_t mainLoopMaximumNumberTrees(double eta) const;

    //! Check if we can skip \p f because initialization has passed \p stage.
    //!
    //! \return true if \p f was skipped.
    //! \note F must be a callable taking no arguments. Its return value is ignored.
    template<typename F>
    bool skipIfAfter(int stage, const F& f);

    //! Check if it is possible to skip \p f because initialization is at or after
    //! \p stage.
    //!
    //! \return True if \p f was skipped.
    //! \note If \p f is run then the state is checkpoint.
    //! \note F must be a callable taking no arguments. Its return value is ignored.
    template<typename F>
    bool skipCheckpointIfAtOrAfter(int stage, const F& f);

    //! Skip progress monitoring for feature selection if we've restarted part
    //! way through training.
    //!
    //! \note This makes sure we output that this task is complete.
    void skipProgressMonitoringFeatureSelection();

    //! Skip progress monitoring for initializeHyperparameters if we've restarted
    //! part way through training.
    //!
    //! \note This makes sure we output that this task is complete.
    void skipProgressMonitoringInitializeHyperparameters();

    //! Stubs out persistence.
    static void noopRecordTrainingState(CBoostedTree::TPersistFunc);

    //! Stubs out test loss adjustment.
    static double noopAdjustTestLoss(double, double, double testLoss);

private:
    TOptionalDouble m_MinimumFrequencyToOneHotEncode;
    std::size_t m_NumberHoldoutRows{0};
    bool m_StratifyRegressionCrossValidation{true};
    double m_InitialDownsampleRowsPerFeature{200.0};
    std::size_t m_MaximumNumberOfTrainRows{500000};
    double m_GainPerNode1stPercentile{0.0};
    double m_GainPerNode50thPercentile{0.0};
    double m_GainPerNode90thPercentile{0.0};
    double m_TotalCurvaturePerNode1stPercentile{0.0};
    double m_TotalCurvaturePerNode90thPercentile{0.0};
    double m_LossGap{0.0};
    std::size_t m_NumberTrees{0};
    std::size_t m_NumberThreads{1};
    std::string m_RowWeightColumnName;
    TBoostedTreeImplUPtr m_TreeImpl;
    mutable std::size_t m_PaddedExtraColumns{0};
    TTrainingStateCallback m_RecordTrainingState{noopRecordTrainingState};
};
}
}
}

#endif // INCLUDED_ml_maths_analytics_CBoostedTreeFactory_h
