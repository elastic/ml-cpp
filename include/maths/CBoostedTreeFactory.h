/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeFactory_h
#define INCLUDED_ml_maths_CBoostedTreeFactory_h

#include <core/CDataFrame.h>
#include <core/CDataSearcher.h>
#include <core/CNonCopyable.h>

#include <maths/CBoostedTree.h>
#include <maths/CDataFrameAnalysisInstrumentationInterface.h>
#include <maths/CLinearAlgebra.h>
#include <maths/ImportExport.h>

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

class CNode;
class CBoostedTree;
class CBoostedTreeImpl;

//! Factory for CBoostedTree objects.
class MATHS_EXPORT CBoostedTreeFactory final {
public:
    using TStrDoublePrVec = std::vector<std::pair<std::string, double>>;
    using TVector = CVectorNx1<double, 3>;
    using TBoostedTreeUPtr = std::unique_ptr<CBoostedTree>;
    using TTrainingStateCallback = CBoostedTree::TTrainingStateCallback;
    using TLossFunctionUPtr = CBoostedTree::TLossFunctionUPtr;
    using TAnalysisInstrumentationPtr = CDataFrameAnalysisInstrumentationInterface*;
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

    //! Set the objective to use when choosing the class assignments.
    CBoostedTreeFactory&
    classAssignmentObjective(CBoostedTree::EClassAssignmentObjective objective);
    //! Set the class weights used for assigning labels to classes from the
    //! predicted probabilities.
    CBoostedTreeFactory& classificationWeights(TStrDoublePrVec weights);
    //! Set the minimum fraction with a category value to one-hot encode.
    CBoostedTreeFactory& minimumFrequencyToOneHotEncode(double frequency);
    //! Set the number of folds to use for estimating the generalisation error.
    CBoostedTreeFactory& numberFolds(std::size_t numberFolds);
    //! Stratify the cross-validation we do for regression.
    CBoostedTreeFactory& stratifyRegressionCrossValidation(bool stratify);
    //! Stop cross-validation early if the test loss is not promising.
    CBoostedTreeFactory& stopCrossValidationEarly(bool stopEarly);
    //! The number of rows per feature to sample in the initial downsample.
    CBoostedTreeFactory& initialDownsampleRowsPerFeature(double rowsPerFeature);
    //! The amount by which to downsample the data for stochastic gradient estimates.
    CBoostedTreeFactory& downsampleFactor(double factor);
    //! Set the sum of leaf depth penalties multiplier.
    CBoostedTreeFactory& depthPenaltyMultiplier(double depthPenaltyMultiplier);
    //! Set the tree size penalty multiplier.
    CBoostedTreeFactory& treeSizePenaltyMultiplier(double treeSizePenaltyMultiplier);
    //! Set the sum of weights squared multiplier.
    CBoostedTreeFactory& leafWeightPenaltyMultiplier(double leafWeightPenaltyMultiplier);
    //! Set the penalty for changing the tree toppology when incrementally training.
    CBoostedTreeFactory& treeTopologyChangePenalty(double treeTopologyChangePenalty);
    //! Set the soft tree depth limit.
    CBoostedTreeFactory& softTreeDepthLimit(double softTreeDepthLimit);
    //! Set the soft tree depth tolerance. This controls how hard we'll try to
    //! respect the soft tree depth limit.
    CBoostedTreeFactory& softTreeDepthTolerance(double softTreeDepthTolerance);
    //! Set the fractional relative tolerance in the target maximum tree depth.
    CBoostedTreeFactory& maxTreeDepthTolerance(double maxTreeDepthTolerance);
    //! Set the amount we'll shrink the weights on each each iteration.
    CBoostedTreeFactory& eta(double eta);
    //! Set the amount we'll grow eta on each each iteration.
    CBoostedTreeFactory& etaGrowthRatePerTree(double etaGrowthRatePerTree);
    //! Set the maximum number of trees in the ensemble.
    CBoostedTreeFactory& maximumNumberTrees(std::size_t maximumNumberTrees);
    //! Set the fraction of features we'll use in the bag to build a tree.
    CBoostedTreeFactory& featureBagFraction(double featureBagFraction);
    //! Set the relative weight to assign changing old predictions in the loss
    //! function for incremental training.
    CBoostedTreeFactory& predictionChangeCost(double predictionChangeCost);
    //! Set the maximum number of optimisation rounds we'll use for hyperparameter
    //! optimisation per parameter for training.
    CBoostedTreeFactory& maximumOptimisationRoundsPerHyperparameterForTrain(std::size_t rounds);
    //! Set the number of restarts to use in global probing for Bayesian Optimisation.
    CBoostedTreeFactory& bayesianOptimisationRestarts(std::size_t restarts);
    //! Set the number of training examples we need per feature we'll include.
    CBoostedTreeFactory& rowsPerFeature(std::size_t rowsPerFeature);
    //! Set the number of training examples we need per feature we'll include.
    CBoostedTreeFactory& numberTopShapValues(std::size_t numberTopShapValues);
    //! Set the flag to enable or disable early stopping.
    CBoostedTreeFactory& earlyStoppingEnabled(bool enable);
    //! Set the row mask for new data with which we want to incrementally train.
    CBoostedTreeFactory& newTrainingRowMask(core::CPackedBitVector rowMask);
    //! Set the fraction of trees in the forest to retrain.
    CBoostedTreeFactory& retrainFraction(double fraction);
    //! Set the data summarization information.
    CBoostedTreeFactory& featureEncoder(TEncoderUPtr encoder);
    //! Set the best forest from the previous training.
    CBoostedTreeFactory& bestForest(TNodeVecVec forest);

    //! Set pointer to the analysis instrumentation.
    CBoostedTreeFactory&
    analysisInstrumentation(CDataFrameTrainBoostedTreeInstrumentationInterface& instrumentation);
    //! Set the callback function for training state recording.
    CBoostedTreeFactory& trainingStateCallback(TTrainingStateCallback callback);

    //! Estimate the maximum booking memory that training a boosted tree on a data
    //! frame with \p numberRows row and \p numberColumns columns will use.
    std::size_t estimateMemoryUsageTrain(std::size_t numberRows, std::size_t numberColumns) const;
    //! Estimate the maximum booking memory that incrementally training a boosted
    //! tree on a data frame with \p numberRows row and \p numberColumns columns
    //! will use.
    std::size_t estimateMemoryUsageTrainIncremental(std::size_t numberRows,
                                                    std::size_t numberColumns) const;
    //! Get the number of columns training the model will add to the data frame.
    //! \note This includes padding for alignment so should be prefered if possible.
    std::size_t numberExtraColumnsForTrain() const;
    //! Get the number of columns training the model will add to the data frame.
    static std::size_t numberExtraColumnsForTrain(std::size_t numberParameters);

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
    using TDoubleVec = std::vector<double>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TOptionalDouble = boost::optional<double>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TOptionalVector = boost::optional<TVector>;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TBoostedTreeImplUPtr = std::unique_ptr<CBoostedTreeImpl>;
    using TApplyParameter = std::function<bool(CBoostedTreeImpl&, double)>;
    using TAdjustTestLoss = std::function<double(double, double, double)>;

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

    //! Resize the data frame with the extra columns used by train.
    void prepareDataFrameForTrain(core::CDataFrame& frame) const;

    //! Resize the data frame with the extra columns used by incremental train.
    void prepareDataFrameForIncrementalTrain(core::CDataFrame& frame) const;

    //! Set up cross validation.
    void initializeCrossValidation(core::CDataFrame& frame) const;

    //! Encode categorical fields and at the same time select the features to use
    //! as regressors.
    void selectFeaturesAndEncodeCategories(const core::CDataFrame& frame) const;

    //! Determine the encoded feature types.
    void determineFeatureDataTypes(const core::CDataFrame& frame) const;

    //! Initialize the regressors sample distribution.
    bool initializeFeatureSampleDistribution() const;

    //! Set the initial values for the various hyperparameters.
    void initializeHyperparameters(core::CDataFrame& frame);

    //! Setup before initializing unset hyperparameters.
    void initializeHyperparametersSetup(core::CDataFrame& frame);

    //! Estimate a good search bounding box regularisation hyperparameters.
    void initializeUnsetRegularizationHyperparameters(core::CDataFrame& frame);

    //! Estimate a good range for the feature bag fraction search interval.
    void initializeUnsetFeatureBagFraction(core::CDataFrame& frame);

    //! Estimates a good range value for the downsample factor search interval.
    void initializeUnsetDownsampleFactor(core::CDataFrame& frame);

    //! Estimate a good range value for learn rate.
    void initializeUnsetEta(core::CDataFrame& frame);

    //! Estimate a good range value for tree topology penalty.
    void initializeUnsetTreeTopologyPenalty();

    //! Estimate the reduction in gain from a split and the total curvature of
    //! the loss function at a split.
    TDoubleDoublePrVec estimateTreeGainAndCurvature(core::CDataFrame& frame,
                                                    const TDoubleVec& percentiles) const;

    //! Perform a line search for the test loss w.r.t. a single regularization
    //! hyperparameter and apply Newton's method to find the minimum. The plan
    //! is to find a value near where the model starts to overfit.
    //!
    //! \return The interval to search during the main hyperparameter optimisation
    //! loop or null if this couldn't be found.
    TOptionalVector testLossLineSearch(core::CDataFrame& frame,
                                       const TApplyParameter& applyParameterStep,
                                       double intervalLeftEnd,
                                       double intervalRightEnd,
                                       double returnedIntervalLeftEndOffset,
                                       double returnedIntervalRightEndOffset,
                                       const TAdjustTestLoss& adjustTestLoss = noopAdjustTestLoss) const;

    //! Initialize the state for hyperparameter optimisation.
    void initializeHyperparameterOptimisation() const;

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
    TOptionalSize m_BayesianOptimisationRestarts;
    bool m_StratifyRegressionCrossValidation{true};
    double m_InitialDownsampleRowsPerFeature{200.0};
    double m_GainPerNode1stPercentile{0.0};
    double m_GainPerNode50thPercentile{0.0};
    double m_GainPerNode90thPercentile{0.0};
    double m_TotalCurvaturePerNode1stPercentile{0.0};
    double m_TotalCurvaturePerNode90thPercentile{0.0};
    std::size_t m_NumberThreads{1};
    TBoostedTreeImplUPtr m_TreeImpl;
    TVector m_LogDownsampleFactorSearchInterval{0.0};
    TVector m_LogFeatureBagFractionInterval{0.0};
    TVector m_LogDepthPenaltyMultiplierSearchInterval{0.0};
    TVector m_LogTreeSizePenaltyMultiplierSearchInterval{0.0};
    TVector m_LogLeafWeightPenaltyMultiplierSearchInterval{0.0};
    TVector m_SoftDepthLimitSearchInterval{0.0};
    TVector m_LogEtaSearchInterval{0.0};
    TVector m_LogTreeTopologyChangePenaltySearchInterval{0.0};
    TTrainingStateCallback m_RecordTrainingState{noopRecordTrainingState};
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeFactory_h
