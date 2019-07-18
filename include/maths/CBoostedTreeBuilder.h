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

#include <maths/CBoostedTree.h>
#include <maths/CBayesianOptimisation.h>

#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

class CNode;
class CBoostedTreeImpl;

//! SimpleFactory for CBoostedTree object
class CBoostedTreeBuilder final {
public:
//    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
//    using TNodeVec = std::vector<CNode>;
    using TCBoostedTreeUPtr = std::unique_ptr<CBoostedTree>;

public:
    CBoostedTreeBuilder(std::size_t numberThreads,
                                          std::size_t dependentVariable,
                                          CBoostedTree::TLossFunctionUPtr loss);

    //! Set the number of folds to use for estimating the generalisation error.
    CBoostedTreeBuilder& numberFolds(std::size_t folds);
    //! Set the lambda regularisation parameter.
    CBoostedTreeBuilder& lambda(double lambda);
    //! Set the gamma regularisation parameter.
    CBoostedTreeBuilder& gamma(double gamma);
    //! Set the amount we'll shrink the weights on each each iteration.
    CBoostedTreeBuilder& eta(double eta);
    //! Set the maximum number of trees in the ensemble.
    CBoostedTreeBuilder& maximumNumberTrees(std::size_t maximumNumberTrees);
    //! Set the fraction of features we'll use in the bag to build a tree.
    CBoostedTreeBuilder& featureBagFraction(double featureBagFraction);
    //! Set the maximum number of optimisation rounds we'll use for hyperparameter
    //! optimisation per parameter.
    CBoostedTreeBuilder& maximumOptimisationRoundsPerHyperparameter(std::size_t rounds);

    CBoostedTreeBuilder& progressCallback(CBoostedTree::TProgressCallback callback);

    CBoostedTreeBuilder& frame(core::CDataFrame& frame);

    TCBoostedTreeUPtr build();

    operator TCBoostedTreeUPtr();
    operator CBoostedTree&&();

private:

    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TDoubleDoubleDoubleTr = std::tuple<double, double, double>;
    using TRowItr = core::CDataFrame::TRowItr;
    using TRowRef = core::CDataFrame::TRowRef;

    // use raw pointer since CDataFrame has copy constructor deleted
    using TCDataFramePtr = core::CDataFrame*;
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TNodeVec = std::vector<CNode>;
    //    using TDoubleDoublePrVec = std::vector<std::pair<double, double>>;

private:
    void initializeMissingFeatureMasks(const core::CDataFrame& frame);

    std::pair<TPackedBitVectorVec, TPackedBitVectorVec>
    crossValidationRowMasks() const;

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
    TCBoostedTreeUPtr m_Tree;
    TCDataFramePtr m_Frame;
    CBoostedTree::TProgressCallback m_ProgressCallback;
};

}
}

#endif // INCLUDED_ml_maths_CBoostedTreeFactory_h
