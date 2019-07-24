/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeFactory_h
#define INCLUDED_ml_maths_CBoostedTreeFactory_h

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CPackedBitVector.h>

#include <maths/CBayesianOptimisation.h>
#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeImpl.h>
#include <maths/ImportExport.h>

#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

class CNode;
class MATHS_EXPORT CBoostedTree;

//! Factory for CBoostedTree objects.
class CBoostedTreeFactory final {
public:
    using TBoostedTreeUPtr = std::unique_ptr<CBoostedTree>;

public:
    //! Construct a boosted tree object from parameters
    static CBoostedTreeFactory constructFromParameters(std::size_t numberThreads,
                                                       std::size_t dependentVariable,
                                                       CBoostedTree::TLossFunctionUPtr loss);

    //! Set the number of folds to use for estimating the generalisation error.
    CBoostedTreeFactory& numberFolds(std::size_t folds);
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
    //! Set the callback function for progress monitoring.
    CBoostedTreeFactory& progressCallback(CBoostedTree::TProgressCallback callback);
    //! Set the reference to the data frame.
    CBoostedTreeFactory& frame(core::CDataFrame& frame);
    //! Create and return a new boosted tree trainer.
    operator TBoostedTreeUPtr();
    //! Estimate the maximum booking memory that training the boosted tree on a data
    //! frame with \p numberRows row and \p numberColumns columns will use.
    std::size_t estimateMemoryUsage(std::size_t numberRows, std::size_t numberColumns) const;
    //! Get the number of columns training the model will add to the data frame.
    std::size_t numberExtraColumnsForTrain() const;

private:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TDoubleDoubleDoubleTr = std::tuple<double, double, double>;
    using TRowItr = core::CDataFrame::TRowItr;
    using TRowRef = core::CDataFrame::TRowRef;
    using TDataFramePtr = core::CDataFrame*;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TNodeVec = std::vector<CNode>;
    using TBoostedTreeImplUPtr = std::unique_ptr<CBoostedTreeImpl>;

private:
    CBoostedTreeFactory(std::size_t numberThreads,
                        std::size_t dependentVariable,
                        CBoostedTree::TLossFunctionUPtr loss);

    TBoostedTreeUPtr build();

    void initializeMissingFeatureMasks(const core::CDataFrame& frame);

    std::pair<TPackedBitVectorVec, TPackedBitVectorVec> crossValidationRowMasks() const;

    //! Initialize the regressors sample distribution.
    void initializeFeatureSampleDistribution(const core::CDataFrame& frame);

    //! Read overrides for hyperparameters and if necessary estimate the initial
    //! values for \f$\lambda\f$ and \f$\gamma\f$ which match the gain from an
    //! overfit tree.
    void initializeHyperparameters(core::CDataFrame& frame,
                                   CBoostedTree::TProgressCallback recordProgress);

    CBayesianOptimisation::TDoubleDoublePrVec hyperparameterBoundingBox() const;

    //! Get the number of hyperparameter tuning rounds to use.
    std::size_t numberHyperparameterTuningRounds() const;

private:
    TBoostedTreeUPtr m_Tree;
    TBoostedTreeImplUPtr m_Impl;
    TDataFramePtr m_Frame;
    CBoostedTree::TProgressCallback m_ProgressCallback;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeFactory_h
