/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeFactory_h
#define INCLUDED_ml_maths_CBoostedTreeFactory_h

#include <core/CDataFrame.h>

#include <maths/CBoostedTree.h>
#include <maths/ImportExport.h>

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
    using TBoostedTreeUPtr = std::unique_ptr<CBoostedTree>;

public:
    //! Construct a boosted tree object from parameters.
    static CBoostedTreeFactory constructFromParameters(std::size_t numberThreads,
                                                       CBoostedTree::TLossFunctionUPtr loss);

    //! Construct a boosted tree object from its serialized version.
    static TBoostedTreeUPtr constructFromString(std::stringstream& jsonStringStream,
                                                core::CDataFrame& frame);

    ~CBoostedTreeFactory();
    CBoostedTreeFactory(CBoostedTreeFactory&) = delete;
    CBoostedTreeFactory& operator=(CBoostedTreeFactory&) = delete;
    CBoostedTreeFactory(CBoostedTreeFactory&&);
    CBoostedTreeFactory& operator=(CBoostedTreeFactory&&);

    //! Set the minimum fraction with a category value to one-hot encode.
    CBoostedTreeFactory& minimumFrequencyToOneHotEncode(double frequency);
    //! Set the number of folds to use for estimating the generalisation error.
    CBoostedTreeFactory& numberFolds(std::size_t numberFolds);
    //! Set the lambda regularisation parameter.
    CBoostedTreeFactory& lambda(double lambda);
    //! Set the gamma regularisation parameter.
    CBoostedTreeFactory& gamma(double gamma);
    //! Set the amount we'll shrink the weights on each each iteration.
    CBoostedTreeFactory& eta(double eta);
    //! Set the maximum number of trees in the ensemble.
    CBoostedTreeFactory& maximumNumberTrees(std::size_t maximumNumberTrees);
    //! Set the fraction of features we'll use in the bag to build a tree.
    CBoostedTreeFactory& featureBagFraction(double featureBagFraction);
    //! Set the maximum number of optimisation rounds we'll use for hyperparameter
    //! optimisation per parameter.
    CBoostedTreeFactory& maximumOptimisationRoundsPerHyperparameter(std::size_t rounds);
    //! Set the number of training examples we need per feature we'll include.
    CBoostedTreeFactory& rowsPerFeature(std::size_t rowsPerFeature);
    //! Set the callback function for progress monitoring.
    CBoostedTreeFactory& progressCallback(CBoostedTree::TProgressCallback callback);

    //! Estimate the maximum booking memory that training the boosted tree on a data
    //! frame with \p numberRows row and \p numberColumns columns will use.
    std::size_t estimateMemoryUsage(std::size_t numberRows, std::size_t numberColumns) const;
    //! Get the number of columns training the model will add to the data frame.
    std::size_t numberExtraColumnsForTrain() const;
    //! Build a boosted tree object for a given data frame.
    TBoostedTreeUPtr buildFor(core::CDataFrame& frame, std::size_t dependentVariable);

private:
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TBoostedTreeImplUPtr = std::unique_ptr<CBoostedTreeImpl>;

private:
    static const double MINIMUM_ETA;
    static const std::size_t MAXIMUM_NUMBER_TREES;

private:
    CBoostedTreeFactory(std::size_t numberThreads, CBoostedTree::TLossFunctionUPtr loss);

    //! Compute the row masks for the missing values for each feature.
    void initializeMissingFeatureMasks(const core::CDataFrame& frame) const;

    //! Get the (train, test) row masks for performing cross validation.
    std::pair<TPackedBitVectorVec, TPackedBitVectorVec> crossValidationRowMasks() const;

    //! Encode categorical fields and at the same time select the features to use
    //! as regressors.
    void selectFeaturesAndEncodeCategories(const core::CDataFrame& frame) const;

    //! Initialize the regressors sample distribution.
    bool initializeFeatureSampleDistribution() const;

    //! Read overrides for hyperparameters and if necessary estimate the initial
    //! values for \f$\lambda\f$ and \f$\gamma\f$ which match the gain from an
    //! overfit tree.
    void initializeHyperparameters(core::CDataFrame& frame) const;

    //! Initialize the state for hyperparameter optimisation.
    void initializeHyperparameterOptimisation() const;

    //! Get the number of hyperparameter tuning rounds to use.
    std::size_t numberHyperparameterTuningRounds() const;

private:
    double m_MinimumFrequencyToOneHotEncode;
    TBoostedTreeImplUPtr m_TreeImpl;
    CBoostedTree::TProgressCallback m_RecordProgress;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeFactory_h
