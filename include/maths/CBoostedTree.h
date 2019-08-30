/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTree_h
#define INCLUDED_ml_maths_CBoostedTree_h

#include <core/CDataFrame.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

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
    //! Returns true if the loss curvature is constant.
    virtual bool isCurvatureConstant() const = 0;
    //! Get an object which computes the leaf value that minimises loss.
    virtual TArgMinLossUPtr minimizer() const = 0;
    //! Get the name of the loss function
    virtual const std::string& name() const = 0;
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
    double value(double prediction, double actual) const override;
    double gradient(double prediction, double actual) const override;
    double curvature(double prediction, double actual) const override;
    bool isCurvatureConstant() const override;
    TArgMinLossUPtr minimizer() const override;
    const std::string& name() const override;

public:
    static const std::string NAME;
};
}

class CBoostedTreeImpl;

//! \brief A boosted regression tree model.
//!
//! DESCRIPTION:\n
//! This is strongly based on xgboost. We deviate in two important respect: we have
//! hyperparameters which control the chance of selecting a feature in the feature
//! bag for a tree, we have automatic handling of categorical fields, we roll in a
//! hyperparameter optimisation loop based on Bayesian Optimisation seeded with a
//! random search and we use an increasing learn rate training a single forest.
//!
//! The probability of selecting a feature behave like a feature weight, allowing us
//! to:
//!   1. Incorporate an estimate of strength of relationship between a regressor and
//!      the target variable upfront,
//!   2. Use optimisation techniques suited for smooth cost functions to fine tune
//!      the regressors the tree will focus on during training.
//! All in all this gives us improved resilience to nuisance regressors and allows
//! us to perform feature selection by imposing a hard cutoff on the minimum probability
//! of a feature we will accept in the final model.
//!
//! The original xgboost paper doesn't explicitly deal with categorical data, it assumes
//! there is a well ordering on each feature and looks for binary splits subject to this
//! ordering. We use a mixed strategy which considers one-hot, target mean and frequency
//! encoding. We choose the "best" strategy based on simultaneously maximising measures
//! of relevancy and redundancy in the feature set as a whole. We use the MICe statistic
//! proposed by Reshef for this purpose. See CDataFrameCategoryEncoder for more details.
class MATHS_EXPORT CBoostedTree final : public CDataFrameRegressionModel {
public:
    using TRowRef = core::CDataFrame::TRowRef;
    using TLossFunctionUPtr = std::unique_ptr<boosted_tree::CLoss>;
    using TDataFramePtr = core::CDataFrame*;

public:
    ~CBoostedTree() override;

    //! Train on the examples in the data frame supplied to the constructor.
    void train() override;

    //! Write the predictions to the data frame supplied to the constructor.
    //!
    //! \warning This can only be called after train.
    void predict() const override;

    //! Write the trained model to \p writer.
    //!
    //! \warning This can only be called after train.
    void write(core::CRapidJsonConcurrentLineWriter& writer) const override;

    //! Get the feature weights the model has chosen.
    const TDoubleVec& featureWeights() const override;

    //! Get the column containing the dependent variable.
    std::size_t columnHoldingDependentVariable() const override;

    //! Get the column containing the model's prediction for the dependent variable.
    std::size_t columnHoldingPrediction(std::size_t numberColumns) const override;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

private:
    using TImplUPtr = std::unique_ptr<CBoostedTreeImpl>;

private:
    CBoostedTree(core::CDataFrame& frame,
                 TProgressCallback recordProgress,
                 TMemoryUsageCallback recordMemoryUsage,
                 TTrainingStateCallback recordTrainingState,
                 TImplUPtr&& impl);

private:
    TImplUPtr m_Impl;

    friend class CBoostedTreeFactory;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTree_h
