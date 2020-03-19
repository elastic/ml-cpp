/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTreeLoss_h
#define INCLUDED_ml_maths_CBoostedTreeLoss_h

#include <maths/CBasicStatistics.h>
#include <maths/CKMeansOnline.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace maths {
namespace boosted_tree_detail {
class MATHS_EXPORT CArgMinLossImpl {
public:
    using TDoubleVector = CDenseVector<double>;
    using TMemoryMappedFloatVector = CMemoryMappedDenseVector<CFloatStorage>;

public:
    CArgMinLossImpl(double lambda);
    virtual ~CArgMinLossImpl() = default;

    virtual std::unique_ptr<CArgMinLossImpl> clone() const = 0;
    virtual bool nextPass() = 0;
    virtual void add(const TMemoryMappedFloatVector& prediction,
                     double actual,
                     double weight = 1.0) = 0;
    virtual void merge(const CArgMinLossImpl& other) = 0;
    virtual TDoubleVector value() const = 0;

protected:
    double lambda() const;

private:
    double m_Lambda;
};

//! \brief Finds the value to add to a set of predictions which minimises the
//! regularized MSE w.r.t. the actual values.
class MATHS_EXPORT CArgMinMseImpl final : public CArgMinLossImpl {
public:
    CArgMinMseImpl(double lambda);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(const TMemoryMappedFloatVector& prediction, double actual, double weight = 1.0) override;
    void merge(const CArgMinLossImpl& other) override;
    TDoubleVector value() const override;

private:
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

private:
    TMeanAccumulator m_MeanError;
};

//! \brief Finds the value to add to a set of predicted log-odds which minimises
//! regularised cross entropy loss w.r.t. the actual categories.
//!
//! DESCRIPTION:\n
//! We want to find the weight which minimizes the log-loss, i.e. which satisfies
//! <pre class="fragment">
//!   \f$\displaystyle arg\min_w{ \lambda w^2 -\sum_i{ a_i \log(S(p_i + w)) + (1 - a_i) \log(1 - S(p_i + w)) } }\f$
//! </pre>
//!
//! Rather than working with this function directly we we approximate it by computing
//! the predictions `p_i` and actual class counts in a uniform bucketing of the data,
//! i.e. we compute the weight which satisfies
//! <pre class="fragment">
//! \f$\displaystyle arg\min_w{ \lambda w^2 -\sum_{B}{ c_{1,B} \log(S(\bar{p}_B + w)) + c_{0,B} \log(1 - S(\bar{p}_B + w)) } }\f$
//! </pre>
//!
//! Here, \f$B\f$ ranges over the buckets, \f$\bar{p}_B\f$ denotes the B'th bucket
//! centre and \f$c_{0,B}\f$ and \f$c_{1,B}\f$ denote the counts of actual classes
//! 0 and 1, respectively, in the bucket \f$B\f$.
class MATHS_EXPORT CArgMinBinomialLogisticLossImpl final : public CArgMinLossImpl {
public:
    CArgMinBinomialLogisticLossImpl(double lambda);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(const TMemoryMappedFloatVector& prediction, double actual, double weight = 1.0) override;
    void merge(const CArgMinLossImpl& other) override;
    TDoubleVector value() const override;

private:
    using TMinMaxAccumulator = CBasicStatistics::CMinMax<double>;
    using TDoubleVector2x1 = CVectorNx1<double, 2>;
    using TDoubleVector2x1Vec = std::vector<TDoubleVector2x1>;

private:
    static constexpr std::size_t NUMBER_BUCKETS = 128;

private:
    std::size_t bucket(double prediction) const {
        double bucket{(prediction - m_PredictionMinMax.min()) / this->bucketWidth()};
        return std::min(static_cast<std::size_t>(bucket), m_BucketsClassCounts.size() - 1);
    }

    double bucketCentre(std::size_t bucket) const {
        return m_PredictionMinMax.min() +
               (static_cast<double>(bucket) + 0.5) * this->bucketWidth();
    }

    double bucketWidth() const {
        return m_PredictionMinMax.initialized()
                   ? m_PredictionMinMax.range() /
                         static_cast<double>(m_BucketsClassCounts.size())
                   : 0.0;
    }

private:
    std::size_t m_CurrentPass = 0;
    TMinMaxAccumulator m_PredictionMinMax;
    TDoubleVector2x1 m_ClassCounts;
    TDoubleVector2x1Vec m_BucketsClassCounts;
};

//! \brief Finds the value to add to a set of predicted multinomial logit which
//! minimises regularised cross entropy loss w.r.t. the actual classes.
//!
//! DESCRIPTION:\n
//! We want to find the weight which minimizes the log-loss, i.e. which satisfies
//! <pre class="fragment">
//!   \f$\displaystyle arg\min_w{ \lambda \|w\|^2 -\sum_i{ \log([softmax(p_i + w)]_{a_i}) } }\f$
//! </pre>
//!
//! Here, \f$a_i\f$ is the index of the i'th example's true class. Rather than
//! working with this function directly we approximate it by the means of the
//! predictions and counts of actual classes in a partition of the data, i.e.
//! we compute the weight which satisfies
//! <pre class="fragment">
//! \f$\displaystyle arg\min_w{ \lambda \|w\|^2 -\sum_P{ c_{a_i, P} \log([softmax(\bar{p}_P + w)]) } }\f$
//! </pre>
//!
//! Here, \f$P\f$ ranges over the subsets of the partition, \f$\bar{p}_P\f$ denotes
//! the mean of the predictions in the P'th subset and \f$c_{a_i, P}\f$ denote the
//! counts of each classes \f$\{a_i\}\f$ in the subset \f$P\f$. We compute this
//! partition by k-means.
class MATHS_EXPORT CArgMinMultinomialLogisticLossImpl final : public CArgMinLossImpl {
public:
    using TObjective = std::function<double(const TDoubleVector&)>;
    using TObjectiveGradient = std::function<TDoubleVector(const TDoubleVector&)>;

public:
    CArgMinMultinomialLogisticLossImpl(std::size_t numberClasses,
                                       double lambda,
                                       const CPRNG::CXorOShiro128Plus& rng);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(const TMemoryMappedFloatVector& prediction, double actual, double weight = 1.0) override;
    void merge(const CArgMinLossImpl& other) override;
    TDoubleVector value() const override;

    // Exposed for unit testing.
    TObjective objective() const;
    TObjectiveGradient objectiveGradient() const;

private:
    using TDoubleVectorVec = std::vector<TDoubleVector>;
    using TKMeans = CKMeansOnline<TDoubleVector>;

private:
    static constexpr std::size_t NUMBER_CENTRES = 128;
    static constexpr std::size_t NUMBER_RESTARTS = 3;

private:
    std::size_t m_NumberClasses = 0;
    std::size_t m_CurrentPass = 0;
    mutable CPRNG::CXorOShiro128Plus m_Rng;
    TDoubleVector m_ClassCounts;
    TDoubleVector m_DoublePrediction;
    TKMeans m_PredictionSketch;
    TDoubleVectorVec m_Centres;
    TDoubleVectorVec m_CentresClassCounts;
};
}

namespace boosted_tree {

//! \brief Computes the leaf value which minimizes the loss function.
class MATHS_EXPORT CArgMinLoss {
public:
    using TDoubleVector = CDenseVector<double>;
    using TMemoryMappedFloatVector = CMemoryMappedDenseVector<CFloatStorage>;

public:
    CArgMinLoss(const CArgMinLoss& other);
    CArgMinLoss(CArgMinLoss&& other) = default;

    CArgMinLoss& operator=(const CArgMinLoss& other);
    CArgMinLoss& operator=(CArgMinLoss&& other) = default;

    //! Start another pass over the predictions and actuals.
    //!
    //! \return True if we need to perform another pass to compute value().
    bool nextPass() const;

    //! Update with a point prediction and actual value.
    void add(const TMemoryMappedFloatVector& prediction, double actual, double weight = 1.0);

    //! Get the minimiser over the predictions and actual values added to both
    //! this and \p other.
    void merge(CArgMinLoss& other);

    //! Returns the value to add to the predictions which minimises the loss
    //! with respect to the actuals.
    //!
    //! Formally, returns \f$x^* = arg\min_x\{\sum_i{L(p_i + x, a_i)}\}\f$
    //! for predictions and actuals \f$p_i\f$ and \f$a_i\f$, respectively.
    TDoubleVector value() const;

private:
    using TArgMinLossImplUPtr = std::unique_ptr<boosted_tree_detail::CArgMinLossImpl>;

private:
    CArgMinLoss(const boosted_tree_detail::CArgMinLossImpl& impl);

private:
    TArgMinLossImplUPtr m_Impl;

    friend class CLoss;
};

//! \brief Defines the loss function for the regression problem.
class MATHS_EXPORT CLoss {
public:
    using TDoubleVector = CDenseVector<double>;
    using TMemoryMappedFloatVector = CMemoryMappedDenseVector<CFloatStorage>;
    using TWriter = std::function<void(std::size_t, double)>;

    enum EType {
        E_BinaryClassification,
        E_MulticlassClassification,
        E_Regression
    };

public:
    virtual ~CLoss() = default;
    //! Clone the loss.
    virtual std::unique_ptr<CLoss> clone() const = 0;
    //! Get the type of prediction problem to which this loss applies.
    virtual EType type() const = 0;
    //! The number of parameters to the loss function.
    virtual std::size_t numberParameters() const = 0;
    //! The value of the loss function.
    virtual double value(const TMemoryMappedFloatVector& prediction,
                         double actual,
                         double weight = 1.0) const = 0;
    //! The gradient of the loss function.
    virtual void gradient(const TMemoryMappedFloatVector& prediction,
                          double actual,
                          TWriter writer,
                          double weight = 1.0) const = 0;
    //! The Hessian of the loss function (flattened).
    virtual void curvature(const TMemoryMappedFloatVector& prediction,
                           double actual,
                           TWriter writer,
                           double weight = 1.0) const = 0;
    //! Returns true if the loss curvature is constant.
    virtual bool isCurvatureConstant() const = 0;
    //! Transforms a prediction from the forest to the target space.
    virtual TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const = 0;
    //! Get an object which computes the leaf value that minimises loss.
    virtual CArgMinLoss minimizer(double lambda,
                                  const CPRNG::CXorOShiro128Plus& rng) const = 0;
    //! Get the name of the loss function
    virtual const std::string& name() const = 0;

protected:
    CArgMinLoss makeMinimizer(const boosted_tree_detail::CArgMinLossImpl& impl) const;
};

//! \brief The MSE loss function.
class MATHS_EXPORT CMse final : public CLoss {
public:
    static const std::string NAME;

public:
    std::unique_ptr<CLoss> clone() const override;
    EType type() const override;
    std::size_t numberParameters() const override;
    double value(const TMemoryMappedFloatVector& prediction,
                 double actual,
                 double weight = 1.0) const override;
    void gradient(const TMemoryMappedFloatVector& prediction,
                  double actual,
                  TWriter writer,
                  double weight = 1.0) const override;
    void curvature(const TMemoryMappedFloatVector& prediction,
                   double actual,
                   TWriter writer,
                   double weight = 1.0) const override;
    bool isCurvatureConstant() const override;
    //! \return \p prediction.
    TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const override;
    CArgMinLoss minimizer(double lambda, const CPRNG::CXorOShiro128Plus& rng) const override;
    const std::string& name() const override;
};

//! \brief Implements loss for binomial logistic regression.
//!
//! DESCRIPTION:\n
//! This targets the cross entropy loss using the tree to predict class log-odds:
//! <pre class="fragment">
//!   \f$\displaystyle l_i(p) = -(1 - a_i) \log(1 - S(p)) - a_i \log(S(p))\f$
//! </pre>
//! where \f$a_i\f$ denotes the actual class of the i'th example, \f$p\f$ is the
//! prediction and \f$S(\cdot)\f$ denotes the logistic function.
class MATHS_EXPORT CBinomialLogisticLoss final : public CLoss {
public:
    static const std::string NAME;

public:
    std::unique_ptr<CLoss> clone() const override;
    EType type() const override;
    std::size_t numberParameters() const override;
    double value(const TMemoryMappedFloatVector& prediction,
                 double actual,
                 double weight = 1.0) const override;
    void gradient(const TMemoryMappedFloatVector& prediction,
                  double actual,
                  TWriter writer,
                  double weight = 1.0) const override;
    void curvature(const TMemoryMappedFloatVector& prediction,
                   double actual,
                   TWriter writer,
                   double weight = 1.0) const override;
    bool isCurvatureConstant() const override;
    //! \return (P(class 0), P(class 1)).
    TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const override;
    CArgMinLoss minimizer(double lambda, const CPRNG::CXorOShiro128Plus& rng) const override;
    const std::string& name() const override;
};

//!  \brief Implements loss for multinomial logistic regression.
//!
//! DESCRIPTION:\n
//! This targets the cross-entropy loss using the forest to predict the class
//! probabilities via the softmax function:
//! <pre class="fragment">
//!   \f$\displaystyle l_i(p) = -\sum_i a_{ij} \log(\sigma(p))\f$
//! </pre>
//! where \f$a_i\f$ denotes the actual class of the i'th example, \f$p\f$ denotes
//! the vector valued prediction and \f$\sigma(p)\$ is the softmax function, i.e.
//! \f$[\sigma(p)]_j = \frac{e^{p_i}}{\sum_k e^{p_k}}\f$.
class MATHS_EXPORT CMultinomialLogisticLoss final : public CLoss {
public:
    static const std::string NAME;

public:
    CMultinomialLogisticLoss(std::size_t numberClasses);
    EType type() const override;
    std::unique_ptr<CLoss> clone() const override;
    std::size_t numberParameters() const override;
    double value(const TMemoryMappedFloatVector& prediction,
                 double actual,
                 double weight = 1.0) const override;
    void gradient(const TMemoryMappedFloatVector& prediction,
                  double actual,
                  TWriter writer,
                  double weight = 1.0) const override;
    void curvature(const TMemoryMappedFloatVector& prediction,
                   double actual,
                   TWriter writer,
                   double weight = 1.0) const override;
    bool isCurvatureConstant() const override;
    //! \return (P(class 0), P(class 1), ..., P(class n)).
    TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const override;
    CArgMinLoss minimizer(double lambda, const CPRNG::CXorOShiro128Plus& rng) const override;
    const std::string& name() const override;

private:
    std::size_t m_NumberClasses;
};
}
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeLoss_h
