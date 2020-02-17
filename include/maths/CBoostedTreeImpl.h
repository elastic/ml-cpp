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
#include <maths/CBoostedTreeUtils.h>
#include <maths/CDataFrameAnalysisInstrumentationInterface.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CDataFrameUtils.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>

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
    using TTrainingStateCallback = CBoostedTree::TTrainingStateCallback;
    using TOptionalDouble = boost::optional<double>;
    using TRegularization = CBoostedTreeRegularization<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeRange = boost::integer_range<std::size_t>;
    using TAnalysisInstrumentationPtr = CDataFrameAnalysisInstrumentationInterface*;

public:
    static const double MINIMUM_RELATIVE_GAIN_PER_SPLIT;

public:
    CBoostedTreeImpl(std::size_t numberThreads,
                     TLossFunctionUPtr loss,
                     TAnalysisInstrumentationPtr instrumentation = nullptr);

    ~CBoostedTreeImpl();

    CBoostedTreeImpl& operator=(const CBoostedTreeImpl&) = delete;
    CBoostedTreeImpl& operator=(CBoostedTreeImpl&&);

    //! Train the model on the values in \p frame.
    void train(core::CDataFrame& frame, const TTrainingStateCallback& recordTrainStateCallback);

    //! Write the predictions of the best trained model to \p frame.
    //!
    //! \note Must be called only if a trained model is available.
    void predict(core::CDataFrame& frame) const;

    //! Compute SHAP values using the best trained model to \p frame.
    //!
    //! \note Must be called only if a trained model is available.
    void computeShapValues(core::CDataFrame& frame);

    //! Get the model produced by training if it has been run.
    const TNodeVecVec& trainedModel() const;

    //! Get the column containing the dependent variable.
    std::size_t columnHoldingDependentVariable() const;

    //! Get the number of columns training the model will add to the data frame.
    static std::size_t numberExtraColumnsForTrain(std::size_t numberLossParameters) {
        // We store as follows:
        //   1. The predicted values for the dependent variables
        //   2. The gradient of the loss function
        //   3. The upper triangle of the hessian of the loss function
        //   4. The example's weight
        return numberLossParameters * (numberLossParameters + 5) / 2 + 1;
    }

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Estimate the maximum booking memory that training the boosted tree on a data
    //! frame with \p numberRows row and \p numberColumns columns will use.
    std::size_t estimateMemoryUsage(std::size_t numberRows, std::size_t numberColumns) const;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Visit this tree trainer implementation.
    void accept(CBoostedTree::CVisitor& visitor);

    //! \return The best hyperparameters for validation error found so far.
    const CBoostedTreeHyperparameters& bestHyperparameters() const;

    //! Get the probability threshold at which to classify a row as class one.
    double probabilityAtWhichToAssignClassOne() const;

    //! Get the indices of the columns containing SHAP values.
    TSizeRange columnsHoldingShapValues() const;

    //! Get the number of largest SHAP values that will be returned for every row.
    std::size_t topShapValues() const;

    //! Get the number of columns in the original data frame.
    std::size_t numberInputColumns() const;

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
    using TVector = CDenseVector<double>;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TImmutableRadixSetVec = std::vector<core::CImmutableRadixSet<double>>;
    using TNodeVecVecDoublePr = std::pair<TNodeVecVec, double>;
    using TDataFrameCategoryEncoderUPtr = std::unique_ptr<CDataFrameCategoryEncoder>;
    using TDataTypeVec = CDataFrameUtils::TDataTypeVec;
    using TRegularizationOverride = CBoostedTreeRegularization<TOptionalDouble>;

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

    //! Compute the probability threshold at which to classify a row as class one.
    void computeProbabilityAtWhichToAssignClassOne(const core::CDataFrame& frame);

    //! Train the forest and compute loss moments on each fold.
    TMeanVarAccumulatorSizePr crossValidateForest(core::CDataFrame& frame);

    //! Initialize the predictions and loss function derivatives for the masked
    //! rows in \p frame.
    TNodeVec initializePredictionsAndLossDerivatives(core::CDataFrame& frame,
                                                     const core::CPackedBitVector& trainingRowMask,
                                                     const core::CPackedBitVector& testingRowMask) const;

    //! Train one forest on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVecVecDoublePr trainForest(core::CDataFrame& frame,
                                    const core::CPackedBitVector& trainingRowMask,
                                    const core::CPackedBitVector& testingRowMask,
                                    core::CLoopProgress& trainingProgress) const;

    //! Randomly downsamples the training row mask by the downsample factor.
    core::CPackedBitVector downsample(const core::CPackedBitVector& trainingRowMask) const;

    //! Get the candidate splits values for each feature.
    TImmutableRadixSetVec candidateSplits(const core::CDataFrame& frame,
                                          const core::CPackedBitVector& trainingRowMask) const;

    //! Train one tree on the rows of \p frame in the mask \p trainingRowMask.
    TNodeVec trainTree(core::CDataFrame& frame,
                       const core::CPackedBitVector& trainingRowMask,
                       const TImmutableRadixSetVec& candidateSplits,
                       const std::size_t maximumTreeSize) const;

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
    std::size_t m_NumberInputColumns = 0;
    TLossFunctionUPtr m_Loss;
    CBoostedTree::EClassAssignmentObjective m_ClassAssignmentObjective =
        CBoostedTree::E_MinimumRecall;
    bool m_StopCrossValidationEarly = true;
    TRegularizationOverride m_RegularizationOverride;
    TOptionalDouble m_DownsampleFactorOverride;
    TOptionalDouble m_EtaOverride;
    TOptionalSize m_NumberFoldsOverride;
    TOptionalSize m_MaximumNumberTreesOverride;
    TOptionalDouble m_FeatureBagFractionOverride;
    TRegularization m_Regularization;
    double m_ProbabilityAtWhichToAssignClassOne = 0.5;
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
    TDataFrameCategoryEncoderUPtr m_Encoder;
    TDataTypeVec m_FeatureDataTypes;
    TDoubleVec m_FeatureSampleProbabilities;
    TPackedBitVectorVec m_MissingFeatureRowMasks;
    TPackedBitVectorVec m_TrainingRowMasks;
    TPackedBitVectorVec m_TestingRowMasks;
    double m_BestForestTestLoss = boosted_tree_detail::INF;
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
    TAnalysisInstrumentationPtr m_Instrumentation; // no persist/restore

private:
    friend class CBoostedTreeFactory;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeImpl_h
