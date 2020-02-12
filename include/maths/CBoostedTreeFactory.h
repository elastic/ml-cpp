/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeFactory_h
#define INCLUDED_ml_maths_CBoostedTreeFactory_h

#include <core/CDataFrame.h>

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
}
namespace maths {

class CNode;
class CBoostedTree;
class CBoostedTreeImpl;

//! Factory for CBoostedTree objects.
class MATHS_EXPORT CBoostedTreeFactory final {
public:
    using TVector = CVectorNx1<double, 3>;
    using TBoostedTreeUPtr = std::unique_ptr<CBoostedTree>;
    using TTrainingStateCallback = CBoostedTree::TTrainingStateCallback;
    using TLossFunctionUPtr = CBoostedTree::TLossFunctionUPtr;
    using TAnalysisInstrumentationPtr = CDataFrameAnalysisInstrumentationInterface*;

public:
    //! Construct a boosted tree object from parameters.
    static CBoostedTreeFactory constructFromParameters(std::size_t numberThreads,
                                                       TLossFunctionUPtr loss);

    //! Construct a boosted tree object from its serialized version.
    //!
    //! \warning Throws runtime error on fail to restore.
    static CBoostedTreeFactory constructFromString(std::istream& jsonStream);

    ~CBoostedTreeFactory();
    CBoostedTreeFactory(CBoostedTreeFactory&) = delete;
    CBoostedTreeFactory& operator=(CBoostedTreeFactory&) = delete;
    CBoostedTreeFactory(CBoostedTreeFactory&&);
    CBoostedTreeFactory& operator=(CBoostedTreeFactory&&);

    //! Set the objective to use when choosing the class assignments.
    CBoostedTreeFactory&
    classAssignmentObjective(CBoostedTree::EClassAssignmentObjective objective);
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
    //! Set the sum of leaf depth penalties multiplier.
    CBoostedTreeFactory& depthPenaltyMultiplier(double depthPenaltyMultiplier);
    //! Set the tree size penalty multiplier.
    CBoostedTreeFactory& treeSizePenaltyMultiplier(double treeSizePenaltyMultiplier);
    //! Set the sum of weights squared multiplier.
    CBoostedTreeFactory& leafWeightPenaltyMultiplier(double leafWeightPenaltyMultiplier);
    //! Set the soft tree depth limit.
    CBoostedTreeFactory& softTreeDepthLimit(double softTreeDepthLimit);
    //! Set the soft tree depth tolerance. This controls how hard we'll try to
    //! respect the soft tree depth limit.
    CBoostedTreeFactory& softTreeDepthTolerance(double softTreeDepthTolerance);
    //! Set the fractional relative tolerance in the target maximum tree depth.
    CBoostedTreeFactory& maxTreeDepthTolerance(double maxTreeDepthTolerance);
    //! Set the amount we'll shrink the weights on each each iteration.
    CBoostedTreeFactory& eta(double eta);
    //! Set the maximum number of trees in the ensemble.
    CBoostedTreeFactory& maximumNumberTrees(std::size_t maximumNumberTrees);
    //! Set the fraction of features we'll use in the bag to build a tree.
    CBoostedTreeFactory& featureBagFraction(double featureBagFraction);
    //! Set the maximum number of optimisation rounds we'll use for hyperparameter
    //! optimisation per parameter.
    CBoostedTreeFactory& maximumOptimisationRoundsPerHyperparameter(std::size_t rounds);
    //! Set the number of restarts to use in global probing for Bayesian Optimisation.
    CBoostedTreeFactory& bayesianOptimisationRestarts(std::size_t restarts);
    //! Set the number of training examples we need per feature we'll include.
    CBoostedTreeFactory& rowsPerFeature(std::size_t rowsPerFeature);
    //! Set the number of training examples we need per feature we'll include.
    CBoostedTreeFactory& topShapValues(std::size_t topShapValues);

    //! Set pointer to the analysis instrumentation.
    CBoostedTreeFactory& analysisInstrumentation(CDataFrameAnalysisInstrumentationInterface& instrumentation);
    //! Set the callback function for training state recording.
    CBoostedTreeFactory& trainingStateCallback(TTrainingStateCallback callback);

    //! Estimate the maximum booking memory that training the boosted tree on a
    //! data frame with \p numberRows row and \p numberColumns columns will use.
    std::size_t estimateMemoryUsage(std::size_t numberRows, std::size_t numberColumns) const;
    //! Get the number of columns training the model will add to the data frame.
    std::size_t numberExtraColumnsForTrain() const;
    //! Build a boosted tree object for a given data frame.
    TBoostedTreeUPtr buildFor(core::CDataFrame& frame, std::size_t dependentVariable);
    //! Restore a boosted tree object for a given data frame.
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
    using TApplyRegularizer = std::function<bool(CBoostedTreeImpl&, double)>;

private:
    CBoostedTreeFactory(std::size_t numberThreads, TLossFunctionUPtr loss);

    //! Compute the row masks for the missing values for each feature.
    void initializeMissingFeatureMasks(const core::CDataFrame& frame) const;

    //! Set up the number of folds we'll use for cross-validation.
    void initializeNumberFolds(core::CDataFrame& frame) const;

    //! Resize the data frame with the extra columns used by train.
    void resizeDataFrame(core::CDataFrame& frame) const;

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

    //! Estimate a good central value for the regularisation hyperparameters
    //! search bounding box.
    void initializeUnsetRegularizationHyperparameters(core::CDataFrame& frame);

    //! Estimates a good central value for the downsample factor search interval.
    void initializeUnsetDownsampleFactor(core::CDataFrame& frame);

    //! Estimate a good central value for learn rate.
    void initializeUnsetEta(core::CDataFrame& frame);

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
                                       const TApplyRegularizer& applyRegularizerStep,
                                       double intervalLeftEnd,
                                       double intervalRightEnd,
                                       double returnedIntervalLeftEndOffset,
                                       double returnedIntervalRightEndOffset) const;

    //! Initialize the state for hyperparameter optimisation.
    void initializeHyperparameterOptimisation() const;

    //! Get the number of hyperparameter tuning rounds to use.
    std::size_t numberHyperparameterTuningRounds() const;

    //! Setup monitoring for training progress.
    void initializeTrainingProgressMonitoring(const core::CDataFrame& frame);

    //! Refresh progress monitoring after restoring from saved training state.
    void resumeRestoredTrainingProgressMonitoring();

    //! The total number of progress steps used in the main loop.
    std::size_t mainLoopNumberSteps(double eta) const;

    //! The maximum number of trees to use in the hyperparameter optimisation loop.
    std::size_t mainLoopMaximumNumberTrees(double eta) const;

    //! Stubs out persistence.
    static void noopRecordTrainingState(CBoostedTree::TPersistFunc);

private:
    TOptionalDouble m_MinimumFrequencyToOneHotEncode;
    TOptionalSize m_BayesianOptimisationRestarts;
    bool m_StratifyRegressionCrossValidation = true;
    double m_InitialDownsampleRowsPerFeature = 200.0;
    std::size_t m_NumberThreads;
    TBoostedTreeImplUPtr m_TreeImpl;
    TVector m_LogDownsampleFactorSearchInterval;
    TVector m_LogDepthPenaltyMultiplierSearchInterval;
    TVector m_LogTreeSizePenaltyMultiplierSearchInterval;
    TVector m_LogLeafWeightPenaltyMultiplierSearchInterval;
    TVector m_SoftDepthLimitSearchInterval;
    TVector m_LogEtaSearchInterval;
    TTrainingStateCallback m_RecordTrainingState = noopRecordTrainingState;
    std::size_t m_TopShapValues = 0;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeFactory_h
