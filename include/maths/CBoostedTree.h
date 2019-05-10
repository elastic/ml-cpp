/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTree_h
#define INCLUDED_ml_maths_CBoostedTree_h

#include <core/CDataFrame.h>

#include <maths/CBasicStatistics.h>
#include <maths/CDataFrameRegressionModel.h>
#include <maths/ImportExport.h>

#include <cstddef>
#include <memory>
#include <thread>

namespace ml {
namespace maths {
namespace boosted_tree {
//! \brief Computes the leaf value which minimizes the loss function.
class MATHS_EXPORT CArgMinLoss {
public:
    virtual ~CArgMinLoss() = default;

    //! Update with a point prediction and actual value.
    void add(double prediction, double actual);

    //! Returns the value at the node which minimises the loss for the
    //! at the predictions added.
    //!
    //! Formally, returns \f$x^* = arg\min_x\{\sum_i{L(p_i + x, a_i)}\}\f$
    //! for predictions and actuals \f$p_i\f$ and \f$a_i\f$, respectively.
    virtual double value() const = 0;

private:
    virtual void addImpl(double prediction, double actual) = 0;

private:
    std::mutex m_Mutex;
};

//! \brief Defines the loss function for the regression problem.
class MATHS_EXPORT CLoss {
public:
    using TArgMinLossUPtr = std::unique_ptr<CArgMinLoss>;

public:
    virtual ~CLoss() = default;
    //! The value of the loss function.
    virtual double value(double prediction, double actual) const = 0;
    //! The slope of the loss function.
    virtual double gradient(double prediction, double actual) const = 0;
    //! The curvature of the loss function.
    virtual double curvature(double prediction, double actual) const = 0;
    //! Get an object which computes the leaf value that minimises loss.
    virtual TArgMinLossUPtr minimizer() const = 0;
};

//! \brief Finds the leaf node value which minimises the MSE.
class MATHS_EXPORT CArgMinMse final : public CArgMinLoss {
public:
    double value() const override;

private:
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

private:
    void addImpl(double prediction, double actual) override;

private:
    TMeanAccumulator m_MeanError;
};

//! \brief The MSE loss function.
class MATHS_EXPORT CMse final : public CLoss {
public:
public:
    double value(double prediction, double actual) const override;
    double gradient(double prediction, double actual) const override;
    double curvature(double prediction, double actual) const override;
    TArgMinLossUPtr minimizer() const override;
};
}

//! \brief A boosted regression tree model.
//!
//! DESCRIPTION:\n
//! This is strongly based on xgboost. We deviate in two important respect: we have
//! hyperparameters which control the chance of selecting a feature in the feature
//! bag for a tree, we have different handling of categorical fields.
//!
//! The probability of selecting a feature behave like a feature weight, allowing us
//! to:
//!   1. Incorporate some estimate of strength of relationship between a feature and
//!      the target variable upfront,
//!   2. Use optimisation techniques suited for smooth cost functions to fine tune
//!      the features used during training.
//! All in all this gives us improved resilience to nuisance variables and allows
//! us to perform feature selection by imposing a hard cutoff on the minimum probability
//! of a feature we will accept in the final model.
//!
//! The original xgboost paper doesn't explicitly deal with categorical data, it assumes
//! there is a well ordering on each feature and looks for binary splits subject to this
//! ordering. This leaves two choices for categorical fields a) use some predefined order
//! knowing that only splits of the form \f$\{\{1,2,...,i\},\{i+1,i+2,...,m\}\}\f$ will
//! be considered or b) use hot-one-encoding knowing splits of the form \f$\{\{0\},\{1\}\}\f$
//! will then be considered for each category. The first choice will rule out good splits
//! because they aren't consistent with the ordering and the second choice will behave
//! poorly for fields with high cardinality because it will be impossible to accurately
//! estimate the change in loss corresponding to the splits.
// TODO
class MATHS_EXPORT CBoostedTree final : public CDataFrameRegressionModel {
public:
    using TProgressCallback = std::function<void(double)>;
    using TRowRef = core::CDataFrame::TRowRef;
    using TLossFunctionUPtr = std::unique_ptr<boosted_tree::CLoss>;

public:
    CBoostedTree(std::size_t numberThreads, std::size_t dependentVariable, TLossFunctionUPtr loss);
    ~CBoostedTree() override;

    //! \name Parameter Setters
    //@{
    //! Set the number of folds to use for estimating the generalisation error.
    CBoostedTree& numberFolds(std::size_t folds);
    //! Set the lambda regularisation parameter.
    CBoostedTree& lambda(double lambda);
    //! Set the gamma regularisation parameter.
    CBoostedTree& gamma(double gamma);
    //! Set the amount we'll shrink the weights on each each iteration.
    CBoostedTree& eta(double eta);
    //! Set the maximum number of trees in the ensemble.
    CBoostedTree& maximumNumberTrees(std::size_t maximumNumberTrees);
    //! Set the fraction of features we'll use in the bag to build a tree.
    CBoostedTree& featureBagFraction(double featureBagFraction);
    //! Set the maximum number of optimisation rounds we'll use for hyperparameter
    //! optimisation.
    CBoostedTree& maximumHyperparameterOptimisationRounds(std::size_t rounds);
    //@}

    //! Train the model on the values in \p frame.
    void train(core::CDataFrame& frame, TProgressCallback recordProgress = noop) override;

    //! Write the predictions of this model to \p frame.
    void predict(core::CDataFrame& frame, TProgressCallback recordProgress = noop) const override;

    //! Write this model to \p writer.
    void write(core::CRapidJsonConcurrentLineWriter& writer) const override;

    //! Get the feature weights the model has chosen.
    TDoubleVec featureWeights() const override;

    //! Get the number of columns training the model will add to the data frame.
    std::size_t numberExtraColumnsForTrain() const override;

    //! Get the column containing the model's prediction for the dependent variable.
    std::size_t columnHoldingPrediction(std::size_t numberColumns) const override;

private:
    class CImpl;
    using TImplUPtr = std::unique_ptr<CImpl>;

private:
    TImplUPtr m_Impl;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTree_h
