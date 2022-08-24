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

#ifndef INCLUDED_ml_maths_analytics_CBoostedTreeLoss_h
#define INCLUDED_ml_maths_analytics_CBoostedTreeLoss_h

#include <core/CStateRestoreTraverser.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/ImportExport.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CPRNG.h>
#include <maths/common/CSampling.h>
#include <maths/common/MathsTypes.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CDataFrame;
class CPackedBitVector;
}
namespace maths {
namespace analytics {
class CDataFrameCategoryEncoder;
namespace boosted_tree_detail {
class MATHS_ANALYTICS_EXPORT CArgMinLossImpl {
public:
    using TDoubleVector = common::CDenseVector<double>;
    using TMemoryMappedFloatVector = common::CMemoryMappedDenseVector<common::CFloatStorage>;
    using TNodeVec = analytics::CBoostedTree::TNodeVec;

public:
    explicit CArgMinLossImpl(double lambda);
    virtual ~CArgMinLossImpl() = default;

    virtual std::unique_ptr<CArgMinLossImpl> clone() const = 0;
    virtual bool nextPass() = 0;
    virtual void add(const CEncodedDataFrameRowRef& row,
                     bool newExample,
                     const TMemoryMappedFloatVector& prediction,
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
class MATHS_ANALYTICS_EXPORT CArgMinMseImpl : public CArgMinLossImpl {
public:
    explicit CArgMinMseImpl(double lambda);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(const CEncodedDataFrameRowRef& /*row*/,
             bool /*newExample*/,
             const TMemoryMappedFloatVector& prediction,
             double actual,
             double weight = 1.0) override {
        this->add(prediction, actual, weight);
    }
    void add(const TMemoryMappedFloatVector& prediction, double actual, double weight = 1.0);
    void merge(const CArgMinLossImpl& other) override;
    TDoubleVector value() const override;

protected:
    using TMeanAccumulator = common::CBasicStatistics::SSampleMean<double>::TAccumulator;

protected:
    const TMeanAccumulator& meanError() const { return m_MeanError; }

private:
    TMeanAccumulator m_MeanError;
};

//! \brief Finds the value to add to a set of predictions which minimises the
//! adjusted regularized MSE difference w.r.t. the actual values for incremental
//! training.
//!
//! DESCRIPTION:\n
//! This applies a correction to the loss based on the difference from the
//! predictions of a supplied tree (the one being retrained).
class MATHS_ANALYTICS_EXPORT CArgMinMseIncrementalImpl final : public CArgMinMseImpl {
public:
    CArgMinMseIncrementalImpl(double lambda, double eta, double mu, const TNodeVec& tree);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    void add(const CEncodedDataFrameRowRef& row,
             bool newExample,
             const TMemoryMappedFloatVector& prediction,
             double actual,
             double weight = 1.0) override;
    void merge(const CArgMinLossImpl& other) override;
    TDoubleVector value() const override;

private:
    double m_Eta{0.0};
    double m_Mu{0.0};
    const TNodeVec* m_Tree{nullptr};
    TMeanAccumulator m_MeanTreePredictions;
};

//! \brief Finds the value to add to a set of predictions which approximately
//! minimises the regularised mean squared logarithmic error (MSLE).
class MATHS_ANALYTICS_EXPORT CArgMinMsleImpl final : public CArgMinLossImpl {
public:
    using TObjective = std::function<double(double)>;

public:
    explicit CArgMinMsleImpl(double lambda, double offset = 1.0);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(const CEncodedDataFrameRowRef& /*row*/,
             bool /*newExample*/,
             const TMemoryMappedFloatVector& prediction,
             double actual,
             double weight = 1.0) override {
        this->add(prediction, actual, weight);
    }
    void add(const TMemoryMappedFloatVector& prediction, double actual, double weight = 1.0);
    void merge(const CArgMinLossImpl& other) override;
    TDoubleVector value() const override;

    // Exposed for unit testing.
    TObjective objective() const;

private:
    using TMinMaxAccumulator = common::CBasicStatistics::CMinMax<double>;
    using TMeanAccumulator = common::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TVector = common::CVectorNx1<double, 3>;
    using TVectorMeanAccumulator = common::CBasicStatistics::SSampleMean<TVector>::TAccumulator;
    using TVectorMeanAccumulatorVec = std::vector<TVectorMeanAccumulator>;
    using TVectorMeanAccumulatorVecVec = std::vector<TVectorMeanAccumulatorVec>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;

private:
    TSizeSizePr bucket(double prediction, double actual) const {
        auto bucketWidth{this->bucketWidth()};
        double bucketPrediction{(prediction - m_ExpPredictionMinMax.min()) /
                                bucketWidth.first};
        std::size_t predictionBucketIndex{std::min(
            static_cast<std::size_t>(bucketPrediction), m_Buckets.size() - 1)};

        double bucketActual{(actual - m_LogActualMinMax.min()) / bucketWidth.second};
        std::size_t actualBucketIndex{std::min(
            static_cast<std::size_t>(bucketActual), m_Buckets[0].size() - 1)};

        return std::make_pair(predictionBucketIndex, actualBucketIndex);
    }

    TDoubleDoublePr bucketWidth() const {
        double predictionBucketWidth{m_ExpPredictionMinMax.range() /
                                     static_cast<double>(m_Buckets.size())};
        double actualBucketWidth{m_LogActualMinMax.range() /
                                 static_cast<double>(m_Buckets[0].size())};
        return std::make_pair(predictionBucketWidth, actualBucketWidth);
    }

private:
    double m_Offset{1.0};
    std::size_t m_CurrentPass{0};
    TMinMaxAccumulator m_ExpPredictionMinMax;
    TMinMaxAccumulator m_LogActualMinMax;
    TVectorMeanAccumulatorVecVec m_Buckets;
    TMeanVarAccumulator m_MeanLogActual;
    TMeanAccumulator m_MeanError;
};

//! \brief Finds the value to add to a set of predictions which approximately
//! minimises the pseudo-Huber loss function.
class MATHS_ANALYTICS_EXPORT CArgMinPseudoHuberImpl final : public CArgMinLossImpl {
public:
    using TObjective = std::function<double(double)>;

public:
    CArgMinPseudoHuberImpl(double lambda, double delta);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(const CEncodedDataFrameRowRef& /*row*/,
             bool /*newExample*/,
             const TMemoryMappedFloatVector& prediction,
             double actual,
             double weight = 1.0) override {
        this->add(prediction, actual, weight);
    }
    void add(const TMemoryMappedFloatVector& prediction, double actual, double weight = 1.0);
    void merge(const CArgMinLossImpl& other) override;
    TDoubleVector value() const override;

    // Exposed for unit testing.
    TObjective objective() const;

private:
    using TMinMaxAccumulator = common::CBasicStatistics::CMinMax<double>;
    using TMeanAccumulator = common::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;

private:
    std::size_t bucket(double error) const {
        auto bucketWidth{this->bucketWidth()};
        double bucket{(error - m_ErrorMinMax.min()) / bucketWidth};
        std::size_t bucketIndex{
            std::min(static_cast<std::size_t>(bucket), m_Buckets.size() - 1)};
        return bucketIndex;
    }

    double bucketWidth() const {
        return m_ErrorMinMax.range() / static_cast<double>(m_Buckets.size());
    }

private:
    double m_DeltaSquared{1.0};
    std::size_t m_CurrentPass{0};
    TMeanAccumulatorVec m_Buckets;
    TMinMaxAccumulator m_ErrorMinMax;
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
class MATHS_ANALYTICS_EXPORT CArgMinBinomialLogisticLossImpl : public CArgMinLossImpl {
public:
    explicit CArgMinBinomialLogisticLossImpl(double lambda);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(const CEncodedDataFrameRowRef& /*row*/,
             bool /*newExample*/,
             const TMemoryMappedFloatVector& prediction,
             double actual,
             double weight = 1.0) override {
        this->add(prediction, actual, weight);
    }
    void add(const TMemoryMappedFloatVector& prediction, double actual, double weight = 1.0);
    void merge(const CArgMinLossImpl& other) override;
    TDoubleVector value() const override;

protected:
    using TMinMaxAccumulator = common::CBasicStatistics::CMinMax<double>;
    using TDoubleVector2x1 = common::CVectorNx1<double, 2>;
    using TDoubleVector2x1Vec = std::vector<TDoubleVector2x1>;
    using TObjective = std::function<double(double)>;

protected:
    static constexpr std::size_t NUMBER_BUCKETS{128};

protected:
    static std::size_t bucket(const TMinMaxAccumulator& minmax, double prediction) {
        double bucket{(prediction - minmax.min()) / bucketWidth(minmax)};
        return std::min(static_cast<std::size_t>(bucket), NUMBER_BUCKETS - 1);
    }

    static double bucketCentre(const TMinMaxAccumulator& minmax, std::size_t bucket) {
        return minmax.min() + (static_cast<double>(bucket) + 0.5) * bucketWidth(minmax);
    }

    static double bucketWidth(const TMinMaxAccumulator& minmax) {
        return minmax.initialized() ? minmax.range() / static_cast<double>(NUMBER_BUCKETS)
                                    : 0.0;
    }

    static double mid(const TMinMaxAccumulator& minmax) {
        return minmax.initialized() ? (minmax.min() + minmax.max()) / 2.0 : 0.0;
    }

    const TMinMaxAccumulator& predictionMinMax() const {
        return m_PredictionMinMax;
    }
    const TDoubleVector2x1& classCounts() const { return m_ClassCounts; }
    const TDoubleVector2x1Vec& bucketsClassCounts() const {
        return m_BucketsClassCounts;
    }

private:
    virtual TObjective objective() const;
    virtual double minWeight() const {
        return std::numeric_limits<double>::max();
    }
    virtual double maxWeight() const {
        return -std::numeric_limits<double>::max();
    }

private:
    std::size_t m_CurrentPass{0};
    TMinMaxAccumulator m_PredictionMinMax;
    TDoubleVector2x1 m_ClassCounts;
    TDoubleVector2x1Vec m_BucketsClassCounts;
};

//! \brief Finds the value to add to a set of predicted log-odds which minimises
//! adjusted regularised cross entropy loss w.r.t. the actual categories for
//! incremental training.
//!
//! DESCRIPTION:\n
//! This applies a correction to the loss based on the cross entropy between the
//! new predictions and the predictions of a supplied tree (the one being retrained).
class MATHS_ANALYTICS_EXPORT CArgMinBinomialLogisticLossIncrementalImpl final
    : public CArgMinBinomialLogisticLossImpl {
public:
    CArgMinBinomialLogisticLossIncrementalImpl(double lambda, double eta, double mu, const TNodeVec& tree);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(const CEncodedDataFrameRowRef& row,
             bool newExample,
             const TMemoryMappedFloatVector& prediction,
             double actual,
             double weight = 1.0) override;
    void merge(const CArgMinLossImpl& other) override;

private:
    using TDoubleVec = std::vector<double>;

private:
    TObjective objective() const override;
    double minWeight() const override { return m_TreePredictionMinMax.min(); }
    double maxWeight() const override { return m_TreePredictionMinMax.max(); }

private:
    double m_Eta{0.0};
    double m_Mu{0.0};
    const TNodeVec* m_Tree{nullptr};
    std::size_t m_CurrentPass{0};
    TMinMaxAccumulator m_TreePredictionMinMax;
    double m_Count{0.0};
    TDoubleVec m_BucketsCount;
};

//! \brief Finds the value to add to a set of predicted multinomial logit which
//! minimises regularised cross entropy loss w.r.t. the actual classes.
//!
//! DESCRIPTION:\n
//! We want to find the weight which minimizes the cross entropy, i.e. which
//! satisfies:
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
//! counts of each classes \f$\{a_i\}\f$ in the P'th subset. We compute this
//! partition via a weighted random sample where the weights are proportional to
//! the mean distance between each point and the rest of the sample set.
class MATHS_ANALYTICS_EXPORT CArgMinMultinomialLogisticLossImpl final : public CArgMinLossImpl {
public:
    using TObjective = std::function<double(const TDoubleVector&)>;
    using TObjectiveGradient = std::function<TDoubleVector(const TDoubleVector&)>;

public:
    CArgMinMultinomialLogisticLossImpl(std::size_t numberClasses,
                                       double lambda,
                                       const common::CPRNG::CXorOShiro128Plus& rng);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(const CEncodedDataFrameRowRef& /*row*/,
             bool /*newExample*/,
             const TMemoryMappedFloatVector& prediction,
             double actual,
             double weight = 1.0) override {
        this->add(prediction, actual, weight);
    }
    void add(const TMemoryMappedFloatVector& prediction, double actual, double weight = 1.0);
    void merge(const CArgMinLossImpl& other) override;
    TDoubleVector value() const override;

    // Exposed for unit testing.
    TObjective objective() const;
    TObjectiveGradient objectiveGradient() const;

private:
    using TDoubleVectorVec = std::vector<TDoubleVector>;
    using TSampler = common::CSampling::CVectorDissimilaritySampler<TDoubleVector>;

private:
    static constexpr std::size_t NUMBER_CENTRES{96};

private:
    std::size_t m_NumberClasses{0};
    std::size_t m_CurrentPass{0};
    mutable common::CPRNG::CXorOShiro128Plus m_Rng;
    TDoubleVector m_ClassCounts;
    TDoubleVector m_DoublePrediction;
    TSampler m_Sampler;
    TDoubleVectorVec m_Centres;
    TDoubleVectorVec m_CentresClassCounts;
};
}

namespace boosted_tree {

enum ELossType {
    E_MsleRegression,
    E_MseRegression,
    E_HuberRegression,
    E_BinaryClassification,
    E_MulticlassClassification
};

//! \brief Computes the leaf value which minimizes the loss function.
class MATHS_ANALYTICS_EXPORT CArgMinLoss {
public:
    using TDoubleVector = common::CDenseVector<double>;
    using TMemoryMappedFloatVector = common::CMemoryMappedDenseVector<common::CFloatStorage>;

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
    void add(const CEncodedDataFrameRowRef& row,
             bool newExample,
             const TMemoryMappedFloatVector& prediction,
             double actual,
             double weight = 1.0);

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
    explicit CArgMinLoss(const boosted_tree_detail::CArgMinLossImpl& impl);

private:
    TArgMinLossImplUPtr m_Impl;

    friend class CLoss;
};

//! \brief Defines the loss function for the regression or classification problem.
class MATHS_ANALYTICS_EXPORT CLoss {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TDoubleVector = common::CDenseVector<double>;
    using TMemoryMappedFloatVector = common::CMemoryMappedDenseVector<common::CFloatStorage>;
    using TWriter = std::function<void(std::size_t, double)>;
    using TLossUPtr = std::unique_ptr<CLoss>;
    using TNodeVec = CBoostedTree::TNodeVec;

public:
    virtual ~CLoss() = default;
    //! Clone the loss.
    virtual TLossUPtr clone() const = 0;
    //! Clone the loss for retraining \p tree.
    virtual TLossUPtr incremental(double eta, double mu, const TNodeVec& tree) const = 0;
    //! Clone the loss function if necessary pruning parameters.
    //!
    //! Computing the Hessian for loss functions with many parameters becomes
    //! cost prohibitive. This chooses the parameters which have the greatest
    //! impact on the loss based on the current predictions and "projects" on
    //! to these. The idea would be that these are computed afresh for each
    //! tree while boosting.
    virtual TLossUPtr project(std::size_t numberThreads,
                              core::CDataFrame& frame,
                              const core::CPackedBitVector& rowMask,
                              std::size_t targetColumn,
                              const TSizeVec& extraColumns,
                              common::CPRNG::CXorOShiro128Plus rng) const = 0;

    //! Get the type of prediction problem to which this loss applies.
    virtual ELossType type() const = 0;
    //! The number of predictions to compute.
    virtual std::size_t dimensionPrediction() const = 0;
    //! The number of gradients to compute.
    virtual std::size_t dimensionGradient() const = 0;

    //! The value of the loss function.
    virtual double value(const TMemoryMappedFloatVector& prediction,
                         double actual,
                         double weight = 1.0) const = 0;
    //! The gradient of the loss function.
    virtual void gradient(const CEncodedDataFrameRowRef& row,
                          bool newExample,
                          const TMemoryMappedFloatVector& prediction,
                          double actual,
                          const TWriter& writer,
                          double weight = 1.0) const = 0;
    //! The Hessian of the loss function (flattened).
    virtual void curvature(const CEncodedDataFrameRowRef& row,
                           bool newExample,
                           const TMemoryMappedFloatVector& prediction,
                           double actual,
                           const TWriter& writer,
                           double weight = 1.0) const = 0;
    //! Returns true if the loss curvature is constant.
    virtual bool isCurvatureConstant() const = 0;

    //! Compute a matched difference between predictions.
    virtual double difference(const TMemoryMappedFloatVector& prediction,
                              const TMemoryMappedFloatVector& previousPrediction,
                              double weight = 1.0) const = 0;

    //! Transforms a prediction from the forest to the target space.
    virtual TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const = 0;

    //! Get an object which computes the leaf value that minimises loss.
    virtual CArgMinLoss
    minimizer(double lambda, const common::CPRNG::CXorOShiro128Plus& rng) const = 0;

    //! Get the name of the loss function
    virtual const std::string& name() const = 0;

    //! Returns true if the loss function is used for regression.
    virtual bool isRegression() const = 0;

    //! Persist by passing information to \p inserter.
    void persistLoss(core::CStatePersistInserter& inserter) const;

    //! Initialize a loss object reading state from \p traverser.
    static TLossUPtr restoreLoss(core::CStateRestoreTraverser& traverser);

protected:
    static CArgMinLoss makeMinimizer(const boosted_tree_detail::CArgMinLossImpl& impl);

private:
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) = 0;
};

//! \brief The MSE loss function.
class MATHS_ANALYTICS_EXPORT CMse : public CLoss {
public:
    static const std::string NAME;

public:
    CMse() = default;
    explicit CMse(core::CStateRestoreTraverser& traverser);
    TLossUPtr clone() const override;
    TLossUPtr incremental(double eta, double mu, const TNodeVec& tree) const override;
    TLossUPtr project(std::size_t numberThreads,
                      core::CDataFrame& frame,
                      const core::CPackedBitVector& rowMask,
                      std::size_t targetColumn,
                      const TSizeVec& extraColumns,
                      common::CPRNG::CXorOShiro128Plus rng) const override;
    ELossType type() const override;
    std::size_t dimensionPrediction() const override;
    std::size_t dimensionGradient() const override;
    double value(const TMemoryMappedFloatVector& prediction,
                 double actual,
                 double weight = 1.0) const override;
    void gradient(const CEncodedDataFrameRowRef& /*row*/,
                  bool /*newExample*/,
                  const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const override {
        this->gradient(prediction, actual, writer, weight);
    }
    void gradient(const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const;
    void curvature(const CEncodedDataFrameRowRef& /*row*/,
                   bool /*newExample*/,
                   const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const override {
        this->curvature(prediction, actual, writer, weight);
    }
    void curvature(const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const;
    bool isCurvatureConstant() const override;
    double difference(const TMemoryMappedFloatVector& prediction,
                      const TMemoryMappedFloatVector& previousPrediction,
                      double weight = 1.0) const override;
    TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const override;
    CArgMinLoss minimizer(double lambda, const common::CPRNG::CXorOShiro128Plus& rng) const override;
    const std::string& name() const override;
    bool isRegression() const override;

private:
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
};

//! \brief The MSE loss function for incremental training.
class MATHS_ANALYTICS_EXPORT CMseIncremental final : public CMse {
public:
    static const std::string NAME;

public:
    CMseIncremental(double eta, double mu, const TNodeVec& tree);
    TLossUPtr clone() const override;
    TLossUPtr incremental(double eta, double mu, const TNodeVec& tree) const override;
    TLossUPtr project(std::size_t numberThreads,
                      core::CDataFrame& frame,
                      const core::CPackedBitVector& rowMask,
                      std::size_t targetColumn,
                      const TSizeVec& extraColumns,
                      common::CPRNG::CXorOShiro128Plus rng) const override;
    ELossType type() const override;
    std::size_t dimensionPrediction() const override;
    std::size_t dimensionGradient() const override;
    double value(const TMemoryMappedFloatVector& prediction,
                 double actual,
                 double weight = 1.0) const override;
    void gradient(const CEncodedDataFrameRowRef& row,
                  bool newExample,
                  const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const override;
    void curvature(const CEncodedDataFrameRowRef& /*row*/,
                   bool newExample,
                   const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const override {
        this->curvature(newExample, prediction, actual, writer, weight);
    }
    void curvature(bool newExample,
                   const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const;
    bool isCurvatureConstant() const override;
    double difference(const TMemoryMappedFloatVector& prediction,
                      const TMemoryMappedFloatVector& previousPrediction,
                      double weight = 1.0) const override;
    //! \return \p prediction.
    TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const override;
    CArgMinLoss minimizer(double lambda, const common::CPRNG::CXorOShiro128Plus& rng) const override;
    const std::string& name() const override;
    bool isRegression() const override;

private:
    void acceptPersistInserter(core::CStatePersistInserter&) const override {}
    bool acceptRestoreTraverser(core::CStateRestoreTraverser&) override;

private:
    double m_Eta{0.0};
    double m_Mu{0.0};
    const TNodeVec* m_Tree{nullptr};
};

//! \brief The MSLE loss function.
//!
//! DESCRIPTION:\n
//! Formally, the MSLE loss definition we use is \f$(\log(1+p) - \log(1+a))^2\f$.
//! However, we approximate this by a quadratic form which shares the position of
//! its minimum p = a and matches the value and derivative of MSLE loss function
//! at the current prediction. For example, if the current prediction for the i'th
//! training point is \f$p_i\f$, the loss is defined as
//! <pre class="fragment">
//!   \f$\displaystyle l_i(p) = c_i + w_i(p - a_i)^2\f$
//! </pre>
//! where \f$w_i = \frac{\log(1+p_i) - \log(1+a_i)}{(1+p_i)(p_i-a_i)}\f$ and \f$c_i\f$
//! is chosen so \f$l_i(p_i) = (\log(1+p_i) - \log(1+a_i))^2\f$.
class MATHS_ANALYTICS_EXPORT CMsle final : public CLoss {
public:
    static const std::string NAME;

public:
    explicit CMsle(double offset = 1.0);
    explicit CMsle(core::CStateRestoreTraverser& traverser);
    TLossUPtr clone() const override;
    TLossUPtr incremental(double eta, double mu, const TNodeVec& tree) const override;
    TLossUPtr project(std::size_t numberThreads,
                      core::CDataFrame& frame,
                      const core::CPackedBitVector& rowMask,
                      std::size_t targetColumn,
                      const TSizeVec& extraColumns,
                      common::CPRNG::CXorOShiro128Plus rng) const override;
    ELossType type() const override;
    std::size_t dimensionPrediction() const override;
    std::size_t dimensionGradient() const override;
    double value(const TMemoryMappedFloatVector& prediction,
                 double actual,
                 double weight = 1.0) const override;
    void gradient(const CEncodedDataFrameRowRef& /*row*/,
                  bool /*newExample*/,
                  const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const override {
        this->gradient(prediction, actual, writer, weight);
    }
    void gradient(const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const;
    void curvature(const CEncodedDataFrameRowRef& /*row*/,
                   bool /*newExample*/,
                   const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const override {
        this->curvature(prediction, actual, writer, weight);
    }
    void curvature(const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const;
    bool isCurvatureConstant() const override;
    double difference(const TMemoryMappedFloatVector& logPrediction,
                      const TMemoryMappedFloatVector& logPreviousPrediction,
                      double weight = 1.0) const override;
    //! \return exp(\p prediction).
    TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const override;
    CArgMinLoss minimizer(double lambda, const common::CPRNG::CXorOShiro128Plus& rng) const override;
    const std::string& name() const override;
    bool isRegression() const override;

private:
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

private:
    double m_Offset;
};

//! \brief The pseudo-Huber loss function.
//!
//! DESCRIPTION:\n
//! Formally, the pseudo-Huber loss definition we use is
//! \f$\delta^2 (\sqrt{1 + \frac{(a - p)^2}{\delta^2}} - 1)\f$.
//! However, we approximate this by a quadratic form which shares the position of
//! its minimum p = a and matches the value and derivative of the pseudo-Huber loss
//! function at the current prediction. For example, if the current prediction for
//! the i'th training point is \f$p_i\f$, the loss is defined as
//! <pre class="fragment">
//! \f[
//!     l_i(p) = \delta^2 \left(\sqrt{1 + \frac{(a_i - p_i)^{2}}{\delta^2}} - 1\right) +
//!              \frac{-a_i+p_i}{\sqrt{\frac{\delta^2 + (a_i-p_i)^{2}}{\delta^2}}} (p - p_i) +
//!              \frac{-a_i+p_i}{2\sqrt{\frac{\delta^2 + (a_i-p_i)^{2}}{\delta^2}}(a_i-p_i)} (p - p_i)^2
//! \f]
//! </pre>
//! For this approximation we compute first and second derivative (gradient and
//! curvature) with respect to p and then substitute \f$p = p_i\f$. As a result we
//! obtain the following formulas for the gradient:
//!   \f[\frac{-a_i + p_i}{\sqrt{\frac{\delta^2 + (a_i - p_i)^2}{\delta^2}}}\f]
//! and for the curvature:
//!   \f[\frac{1}{\sqrt{1 + \frac{(a_i - p_i)^2}{\delta^2}}}\f]
class MATHS_ANALYTICS_EXPORT CPseudoHuber final : public CLoss {
public:
    static const std::string NAME;

public:
    explicit CPseudoHuber(double delta);
    explicit CPseudoHuber(core::CStateRestoreTraverser& traverser);
    TLossUPtr clone() const override;
    TLossUPtr incremental(double eta, double mu, const TNodeVec& tree) const override;
    TLossUPtr project(std::size_t numberThreads,
                      core::CDataFrame& frame,
                      const core::CPackedBitVector& rowMask,
                      std::size_t targetColumn,
                      const TSizeVec& extraColumns,
                      common::CPRNG::CXorOShiro128Plus rng) const override;
    ELossType type() const override;
    std::size_t dimensionPrediction() const override;
    std::size_t dimensionGradient() const override;
    double value(const TMemoryMappedFloatVector& prediction,
                 double actual,
                 double weight = 1.0) const override;
    void gradient(const CEncodedDataFrameRowRef& /*row*/,
                  bool /*newExample*/,
                  const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const override {
        this->gradient(prediction, actual, writer, weight);
    }
    void gradient(const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const;
    void curvature(const CEncodedDataFrameRowRef& /*row*/,
                   bool /*newExample*/,
                   const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const override {
        this->curvature(prediction, actual, writer, weight);
    }
    void curvature(const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const;
    bool isCurvatureConstant() const override;
    double difference(const TMemoryMappedFloatVector& prediction,
                      const TMemoryMappedFloatVector& previousPrediction,
                      double weight = 1.0) const override;
    //! \return \p prediction.
    TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const override;
    CArgMinLoss minimizer(double lambda, const common::CPRNG::CXorOShiro128Plus& rng) const override;
    const std::string& name() const override;
    bool isRegression() const override;

private:
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

private:
    double m_Delta;
};

//! \brief The loss for binomial logistic regression.
//!
//! DESCRIPTION:\n
//! This targets the cross entropy loss using the tree to predict class log-odds:
//! <pre class="fragment">
//!   \f$\displaystyle l_i(p) = -(1 - a_i) \log(1 - S(p)) - a_i \log(S(p))\f$
//! </pre>
//! where \f$a_i\f$ denotes the actual class of the i'th example, \f$p\f$ is the
//! prediction and \f$S(\cdot)\f$ denotes the logistic function.
class MATHS_ANALYTICS_EXPORT CBinomialLogisticLoss : public CLoss {
public:
    static const std::string NAME;

public:
    CBinomialLogisticLoss() = default;
    explicit CBinomialLogisticLoss(core::CStateRestoreTraverser& traverser);
    TLossUPtr clone() const override;
    TLossUPtr incremental(double eta, double mu, const TNodeVec& tree) const override;
    TLossUPtr project(std::size_t numberThreads,
                      core::CDataFrame& frame,
                      const core::CPackedBitVector& rowMask,
                      std::size_t targetColumn,
                      const TSizeVec& extraColumns,
                      common::CPRNG::CXorOShiro128Plus rng) const override;
    ELossType type() const override;
    std::size_t dimensionPrediction() const override;
    std::size_t dimensionGradient() const override;
    double value(const TMemoryMappedFloatVector& prediction,
                 double actual,
                 double weight = 1.0) const override;
    void gradient(const CEncodedDataFrameRowRef& /*row*/,
                  bool /*newExample*/,
                  const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const override {
        this->gradient(prediction, actual, writer, weight);
    }
    void gradient(const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const;
    void curvature(const CEncodedDataFrameRowRef& /*row*/,
                   bool /*newExample*/,
                   const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const override {
        this->curvature(prediction, actual, writer, weight);
    }
    void curvature(const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const;
    bool isCurvatureConstant() const override;
    double difference(const TMemoryMappedFloatVector& prediction,
                      const TMemoryMappedFloatVector& previousPrediction,
                      double weight = 1.0) const override;
    //! \return (P(class 0), P(class 1)).
    TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const override;
    CArgMinLoss minimizer(double lambda, const common::CPRNG::CXorOShiro128Plus& rng) const override;
    const std::string& name() const override;
    bool isRegression() const override;

private:
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;
};

//! \brief The loss for incremental binomial logistic regression.
//!
//! DESCRIPTION:\n
//! This augments the standard loss function by adding the cross-entropy between
//! predictions and the supplied tree predictions.
class MATHS_ANALYTICS_EXPORT CBinomialLogisticLossIncremental final : public CBinomialLogisticLoss {
public:
    static const std::string NAME;

public:
    CBinomialLogisticLossIncremental(double eta, double mu, const TNodeVec& tree);
    TLossUPtr clone() const override;
    TLossUPtr incremental(double eta, double mu, const TNodeVec& tree) const override;
    TLossUPtr project(std::size_t numberThreads,
                      core::CDataFrame& frame,
                      const core::CPackedBitVector& rowMask,
                      std::size_t targetColumn,
                      const TSizeVec& extraColumns,
                      common::CPRNG::CXorOShiro128Plus rng) const override;
    ELossType type() const override;
    std::size_t dimensionPrediction() const override;
    std::size_t dimensionGradient() const override;
    double value(const TMemoryMappedFloatVector& prediction,
                 double actual,
                 double weight = 1.0) const override;
    void gradient(const CEncodedDataFrameRowRef& row,
                  bool newExample,
                  const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const override;
    void curvature(const CEncodedDataFrameRowRef& row,
                   bool newExample,
                   const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const override;
    bool isCurvatureConstant() const override;
    double difference(const TMemoryMappedFloatVector& prediction,
                      const TMemoryMappedFloatVector& previousPrediction,
                      double weight = 1.0) const override;
    //! \return (P(class 0), P(class 1)).
    TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const override;
    CArgMinLoss minimizer(double lambda, const common::CPRNG::CXorOShiro128Plus& rng) const override;
    const std::string& name() const override;
    bool isRegression() const override;

private:
    void acceptPersistInserter(core::CStatePersistInserter&) const override {}
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

private:
    double m_Eta{0.0};
    double m_Mu{0.0};
    const TNodeVec* m_Tree{nullptr};
};

//!  \brief The loss for multinomial logistic regression.
//!
//! DESCRIPTION:\n
//! This targets the cross-entropy loss using the forest to predict the class
//! probabilities via the softmax function:
//! <pre class="fragment">
//!   \f$\displaystyle l_i(p) = -\sum_i a_{ij} \log(\sigma(p))\f$
//! </pre>
//! where \f$a_i\f$ denotes the actual class of the i'th example, \f$p\f$ denotes
//! the vector valued prediction and \f$\sigma(p)\f$ is the softmax function, i.e.
//! \f$[\sigma(p)]_j = \frac{e^{p_i}}{\sum_k e^{p_k}}\f$.
class MATHS_ANALYTICS_EXPORT CMultinomialLogisticLoss : public CLoss {
public:
    static const std::string NAME;
    static constexpr std::size_t MAX_GRADIENT_DIMENSION{20};

public:
    explicit CMultinomialLogisticLoss(std::size_t numberClasses);
    explicit CMultinomialLogisticLoss(core::CStateRestoreTraverser& traverser);
    TLossUPtr clone() const override;
    TLossUPtr incremental(double eta, double mu, const TNodeVec& tree) const override;
    TLossUPtr project(std::size_t numberThreads,
                      core::CDataFrame& frame,
                      const core::CPackedBitVector& rowMask,
                      std::size_t targetColumn,
                      const TSizeVec& extraColumns,
                      common::CPRNG::CXorOShiro128Plus rng) const override;
    ELossType type() const override;
    std::size_t dimensionPrediction() const override;
    std::size_t dimensionGradient() const override;
    double value(const TMemoryMappedFloatVector& prediction,
                 double actual,
                 double weight = 1.0) const override;
    virtual void gradient(const TMemoryMappedFloatVector& prediction,
                          double actual,
                          const TWriter& writer,
                          double weight = 1.0) const;
    virtual void curvature(const TMemoryMappedFloatVector& prediction,
                           double actual,
                           const TWriter& writer,
                           double weight = 1.0) const;
    void gradient(const CEncodedDataFrameRowRef& /*row*/,
                  bool /*newExample*/,
                  const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const override {
        this->gradient(prediction, actual, writer, weight);
    }
    void curvature(const CEncodedDataFrameRowRef& /*row*/,
                   bool /*newExample*/,
                   const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const override {
        this->curvature(prediction, actual, writer, weight);
    }
    bool isCurvatureConstant() const override;
    double difference(const TMemoryMappedFloatVector& prediction,
                      const TMemoryMappedFloatVector& previousPrediction,
                      double weight = 1.0) const override;
    //! \return (P(class 0), P(class 1), ..., P(class n)).
    TDoubleVector transform(const TMemoryMappedFloatVector& prediction) const override;
    CArgMinLoss minimizer(double lambda, const common::CPRNG::CXorOShiro128Plus& rng) const override;
    const std::string& name() const override;
    bool isRegression() const override;

private:
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

private:
    std::size_t m_NumberClasses;
};

class MATHS_ANALYTICS_EXPORT CMultinomialLogisticSubsetLoss final
    : public CMultinomialLogisticLoss {
public:
    CMultinomialLogisticSubsetLoss(std::size_t numberClasses, const TSizeVec& classes);
    TLossUPtr clone() const override;
    TLossUPtr incremental(double eta, double mu, const TNodeVec& tree) const override;
    TLossUPtr project(std::size_t numberThreads,
                      core::CDataFrame& frame,
                      const core::CPackedBitVector& rowMask,
                      std::size_t targetColumn,
                      const TSizeVec& extraColumns,
                      common::CPRNG::CXorOShiro128Plus rng) const override;
    void gradient(const TMemoryMappedFloatVector& prediction,
                  double actual,
                  const TWriter& writer,
                  double weight = 1.0) const override;
    void curvature(const TMemoryMappedFloatVector& prediction,
                   double actual,
                   const TWriter& writer,
                   double weight = 1.0) const override;

private:
    using TIntVec = std::vector<int>;

private:
    TIntVec m_InClasses;
    TIntVec m_OutClasses;
};
}
}
}
}

#endif // INCLUDED_ml_maths_analytics_CBoostedTreeLoss_h
