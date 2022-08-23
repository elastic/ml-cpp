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

#include <maths/analytics/CBoostedTreeLoss.h>

#include <core/CDataFrame.h>
#include <core/CPersistUtils.h>
#include <core/Concurrency.h>

#include <maths/analytics/CBoostedTreeUtils.h>
#include <maths/analytics/CDataFrameCategoryEncoder.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CLbfgs.h>
#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CPRNG.h>
#include <maths/common/CSampling.h>
#include <maths/common/CSolvers.h>
#include <maths/common/CTools.h>
#include <maths/common/CToolsDetail.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>

namespace ml {
namespace maths {
namespace analytics {
using namespace boosted_tree_detail;
using TRowItr = core::CDataFrame::TRowItr;

namespace {
const double EPSILON{100.0 * std::numeric_limits<double>::epsilon()};
const double LOG_EPSILON{common::CTools::stableLog(EPSILON)};

// MSLE constants
const std::size_t MSLE_PREDICTION_INDEX{0};
const std::size_t MSLE_ACTUAL_INDEX{1};
const std::size_t MSLE_ERROR_INDEX{2};
const std::size_t MSLE_BUCKET_SIZE{32};
const std::size_t MSLE_OPTIMIZATION_ITERATIONS{15};

// Pseudo-Huber constants
const std::size_t HUBER_BUCKET_SIZE{128};
const std::size_t HUBER_OPTIMIZATION_ITERATIONS{15};

// Persistence and restoration
const std::string NUMBER_CLASSES_TAG{"number_classes"};
const std::string OFFSET_TAG{"offset"};
const std::string DELTA_TAG{"delta"};
const std::string NAME_TAG{"name"};

double logOneMinusLogistic(double logOdds) {
    // For large x logistic(x) = 1 - e^(-x) + O(e^(-2x))
    if (logOdds > -LOG_EPSILON) {
        return -logOdds;
    }
    return common::CTools::stableLog(1.0 - common::CTools::logisticFunction(logOdds));
}

double logLogistic(double logOdds) {
    // For small x logistic(x) = e^(x) + O(e^(2x))
    if (logOdds < LOG_EPSILON) {
        return logOdds;
    }
    return common::CTools::stableLog(common::CTools::logisticFunction(logOdds));
}
}

namespace boosted_tree_detail {

CArgMinLossImpl::CArgMinLossImpl(double lambda) : m_Lambda{lambda} {
}

double CArgMinLossImpl::lambda() const {
    return m_Lambda;
}

CArgMinMseImpl::CArgMinMseImpl(double lambda) : CArgMinLossImpl{lambda} {
}

std::unique_ptr<CArgMinLossImpl> CArgMinMseImpl::clone() const {
    return std::make_unique<CArgMinMseImpl>(*this);
}

bool CArgMinMseImpl::nextPass() {
    return false;
}

void CArgMinMseImpl::add(const TMemoryMappedFloatVector& prediction, double actual, double weight) {
    m_MeanError.add(actual - prediction(0), weight);
}

void CArgMinMseImpl::merge(const CArgMinLossImpl& other) {
    const auto* mse = dynamic_cast<const CArgMinMseImpl*>(&other);
    if (mse != nullptr) {
        m_MeanError += mse->m_MeanError;
    }
}

CArgMinMseImpl::TDoubleVector CArgMinMseImpl::value() const {

    // We searching for the value x^* which satisfies
    //
    //    x^* = argmin_x{ sum_i{(a_i - (p_i + x))^2} + lambda * x^2 }
    //
    // This is convex so there is one minimum where the derivative w.r.t. x is zero
    // and x^* = 1 / (n + lambda) sum_i{ a_i - p_i }. Denoting the mean prediction
    // error m = 1/n sum_i{ a_i - p_i } we have x^* = n / (n + lambda) m.

    double count{common::CBasicStatistics::count(m_MeanError)};
    double meanError{common::CBasicStatistics::mean(m_MeanError)};

    TDoubleVector result(1);
    result(0) = count == 0.0 ? 0.0 : count / (count + this->lambda()) * meanError;
    return result;
}

CArgMinMseIncrementalImpl::CArgMinMseIncrementalImpl(double lambda,
                                                     double eta,
                                                     double mu,
                                                     const TNodeVec& tree)
    : CArgMinMseImpl{lambda}, m_Eta{eta}, m_Mu{mu}, m_Tree{&tree} {
}

std::unique_ptr<CArgMinLossImpl> CArgMinMseIncrementalImpl::clone() const {
    return std::make_unique<CArgMinMseIncrementalImpl>(*this);
}

void CArgMinMseIncrementalImpl::add(const CEncodedDataFrameRowRef& row,
                                    bool newExample,
                                    const TMemoryMappedFloatVector& prediction,
                                    double actual,
                                    double weight) {
    this->CArgMinMseImpl::add(prediction, actual, weight);
    if (newExample == false) {
        m_MeanTreePredictions.add(root(*m_Tree).value(row, *m_Tree)(0));
    }
}

void CArgMinMseIncrementalImpl::merge(const CArgMinLossImpl& other) {
    const auto* mse = dynamic_cast<const CArgMinMseIncrementalImpl*>(&other);
    if (mse != nullptr) {
        this->CArgMinMseImpl::merge(*mse);
        m_MeanTreePredictions += mse->m_MeanTreePredictions;
    }
}

CArgMinMseIncrementalImpl::TDoubleVector CArgMinMseIncrementalImpl::value() const {

    // We searching for the value x^* which satisfies
    //
    //    x^* = argmin_x{ sum_i{(a_i - (p_i + x))^2 + 1{old} mu (p_i' / eta - x)^2} + lambda * x^2 }
    //
    // Here, a_i are the actuals, p_i the predictions and p_i' the predictions from
    // the tree being retrained. This is convex so there is one minimum where the
    // derivative w.r.t. x is zero and
    //
    //   x^* = 1 / (n (1 + n' / n mu) + lambda) sum_i{ a_i - p_i + mu / eta 1{old} p_i' }.
    //
    // where n' = sum_i 1{old}. Denoting the mean prediction error m = 1/n sum_i{ a_i - p_i }
    // and the mean tree predictions p' = 1/n' sum_i{p_i'} we have
    //
    //   x^* = n / (n (1 + n' / n mu) + lambda) (m + n' / n mu / eta p').
    //
    // In the following we absorb n' / n into the value of mu.

    double count{common::CBasicStatistics::count(this->meanError())};
    double meanError{common::CBasicStatistics::mean(this->meanError())};
    double oldCount{common::CBasicStatistics::count(m_MeanTreePredictions)};
    double meanTreePrediction{common::CBasicStatistics::mean(m_MeanTreePredictions)};
    double mu{(count == oldCount ? 1.0 : oldCount / count) * m_Mu};

    TDoubleVector result(1);
    result(0) = count == 0.0 ? 0.0
                             : count / (count * (1.0 + mu) + this->lambda()) *
                                   (meanError + mu / m_Eta * meanTreePrediction);
    return result;
}

CArgMinBinomialLogisticLossImpl::CArgMinBinomialLogisticLossImpl(double lambda)
    : CArgMinLossImpl{lambda}, m_ClassCounts{0},
      m_BucketsClassCounts(NUMBER_BUCKETS, TDoubleVector2x1{0.0}) {
}

std::unique_ptr<CArgMinLossImpl> CArgMinBinomialLogisticLossImpl::clone() const {
    return std::make_unique<CArgMinBinomialLogisticLossImpl>(*this);
}

bool CArgMinBinomialLogisticLossImpl::nextPass() {
    m_CurrentPass += bucketWidth(m_PredictionMinMax) > 0.0 ? 1 : 2;
    return m_CurrentPass < 2;
}

void CArgMinBinomialLogisticLossImpl::add(const TMemoryMappedFloatVector& prediction,
                                          double actual,
                                          double weight) {
    switch (m_CurrentPass) {
    case 0: {
        m_PredictionMinMax.add(prediction(0));
        m_ClassCounts(static_cast<std::size_t>(actual)) += weight;
        break;
    }
    case 1: {
        auto& count = m_BucketsClassCounts[bucket(m_PredictionMinMax, prediction(0))];
        count(static_cast<std::size_t>(actual)) += weight;
        break;
    }
    default:
        break;
    }
}

void CArgMinBinomialLogisticLossImpl::merge(const CArgMinLossImpl& other) {
    const auto* logistic = dynamic_cast<const CArgMinBinomialLogisticLossImpl*>(&other);
    if (logistic != nullptr) {
        switch (m_CurrentPass) {
        case 0:
            m_PredictionMinMax += logistic->m_PredictionMinMax;
            m_ClassCounts += logistic->m_ClassCounts;
            break;
        case 1:
            for (std::size_t i = 0; i < m_BucketsClassCounts.size(); ++i) {
                m_BucketsClassCounts[i] += logistic->m_BucketsClassCounts[i];
            }
            break;
        default:
            break;
        }
    }
}

CArgMinBinomialLogisticLossImpl::TDoubleVector CArgMinBinomialLogisticLossImpl::value() const {

    double minWeight;
    double maxWeight;

    // This is true if and only if all the predictions were identical. In this
    // case we only need one pass over the data and can compute the optimal
    // value from the counts of the two categories.
    if (bucketWidth(m_PredictionMinMax) == 0.0) {
        // This is the (unique) predicted value for the rows in leaf by the forest
        // so far (i.e. without the weight for the leaf we're about to add).
        double prediction{mid(m_PredictionMinMax)};

        // Weight shrinkage means the optimal weight will be somewhere between
        // the logit of the empirical probability and zero.
        double c0{m_ClassCounts(0) + 1.0};
        double c1{m_ClassCounts(1) + 1.0};
        double empiricalProbabilityC1{c1 / (c0 + c1)};
        double empiricalLogOddsC1{common::CTools::stableLog(
            empiricalProbabilityC1 / (1.0 - empiricalProbabilityC1))};
        minWeight = (empiricalProbabilityC1 < 0.5 ? empiricalLogOddsC1 : 0.0) - prediction;
        maxWeight = (empiricalProbabilityC1 < 0.5 ? 0.0 : empiricalLogOddsC1) - prediction;
    } else {
        // Choose a weight interval in which all probabilites vary from close to
        // zero to close to one. In particular, the idea is to minimize the leaf
        // weight on an interval [a, b] where if we add "a" the log-odds for all
        // rows <= -5, i.e. max prediction + a = -5, and if we add "b" the log-odds
        // for all rows >= 5, i.e. min prediction + a = 5.
        minWeight = -m_PredictionMinMax.max() - 5.0;
        maxWeight = -m_PredictionMinMax.min() + 5.0;
    }
    minWeight = std::min(minWeight, this->minWeight());
    maxWeight = std::max(maxWeight, this->maxWeight());

    TDoubleVector result(1);

    if (minWeight == maxWeight) {
        result(0) = minWeight;
        return result;
    }

    double minimum;
    double objectiveAtMinimum;
    auto objective = this->objective();
    std::size_t maxIterations{10};
    common::CSolvers::minimize(minWeight, maxWeight, objective(minWeight),
                               objective(maxWeight), objective, 1e-3,
                               maxIterations, minimum, objectiveAtMinimum);
    LOG_TRACE(<< "minimum = " << minimum << " objective(minimum) = " << objectiveAtMinimum);

    result(0) = minimum;
    return result;
}

CArgMinBinomialLogisticLossImpl::TObjective CArgMinBinomialLogisticLossImpl::objective() const {
    // This is true if all the predictions were identical. In this case we only
    // need one pass over the data and can compute the optimal value from the
    // counts of the two categories.
    if (bucketWidth(m_PredictionMinMax) == 0.0) {
        double prediction{mid(m_PredictionMinMax)};
        return [prediction, this](double weight) {
            double logOdds{prediction + weight};
            double c0{m_ClassCounts(0)};
            double c1{m_ClassCounts(1)};
            return this->lambda() * common::CTools::pow2(weight) -
                   c0 * logOneMinusLogistic(logOdds) - c1 * logLogistic(logOdds);
        };
    }
    return [this](double weight) {
        double loss{this->lambda() * common::CTools::pow2(weight)};
        for (std::size_t i = 0; i < m_BucketsClassCounts.size(); ++i) {
            double logOdds{bucketCentre(m_PredictionMinMax, i) + weight};
            double c0{m_BucketsClassCounts[i](0)};
            double c1{m_BucketsClassCounts[i](1)};
            loss -= c0 * logOneMinusLogistic(logOdds) + c1 * logLogistic(logOdds);
        }
        return loss;
    };
}

CArgMinBinomialLogisticLossIncrementalImpl::CArgMinBinomialLogisticLossIncrementalImpl(
    double lambda,
    double eta,
    double mu,
    const TNodeVec& tree)
    : CArgMinBinomialLogisticLossImpl{lambda}, m_Eta{eta}, m_Mu{mu}, m_Tree{&tree},
      m_BucketsCount(NUMBER_BUCKETS, 0.0) {
}

std::unique_ptr<CArgMinLossImpl> CArgMinBinomialLogisticLossIncrementalImpl::clone() const {
    return std::make_unique<CArgMinBinomialLogisticLossIncrementalImpl>(*this);
}

bool CArgMinBinomialLogisticLossIncrementalImpl::nextPass() {
    this->CArgMinBinomialLogisticLossImpl::nextPass();
    m_CurrentPass += bucketWidth(m_TreePredictionMinMax) > 0.0 ? 1 : 2;
    return m_CurrentPass < 2;
}

void CArgMinBinomialLogisticLossIncrementalImpl::add(const CEncodedDataFrameRowRef& row,
                                                     bool newExample,
                                                     const TMemoryMappedFloatVector& prediction,
                                                     double actual,
                                                     double weight) {
    this->CArgMinBinomialLogisticLossImpl::add(prediction, actual, weight);
    if (newExample == false) {
        switch (m_CurrentPass) {
        case 0: {
            double treePrediction{root(*m_Tree).value(row, *m_Tree)(0) / m_Eta};
            m_TreePredictionMinMax.add(treePrediction);
            m_Count += weight;
            break;
        }
        case 1: {
            double treePrediction{root(*m_Tree).value(row, *m_Tree)(0) / m_Eta};
            auto& count = m_BucketsCount[bucket(m_TreePredictionMinMax, treePrediction)];
            count += weight;
            break;
        }
        default:
            break;
        }
    }
}

void CArgMinBinomialLogisticLossIncrementalImpl::merge(const CArgMinLossImpl& other) {
    const auto* logistic =
        dynamic_cast<const CArgMinBinomialLogisticLossIncrementalImpl*>(&other);
    if (logistic != nullptr) {
        this->CArgMinBinomialLogisticLossImpl::merge(*logistic);
        switch (m_CurrentPass) {
        case 0:
            m_TreePredictionMinMax += logistic->m_TreePredictionMinMax;
            m_Count += logistic->m_Count;
            break;
        case 1:
            for (std::size_t i = 0; i < m_BucketsCount.size(); ++i) {
                m_BucketsCount[i] += logistic->m_BucketsCount[i];
            }
            break;
        default:
            break;
        }
    }
}

CArgMinBinomialLogisticLossIncrementalImpl::TObjective
CArgMinBinomialLogisticLossIncrementalImpl::objective() const {

    // This is true if all the forest and tree predictions were identical.
    if (bucketWidth(this->predictionMinMax()) == 0.0 &&
        bucketWidth(m_TreePredictionMinMax) == 0.0) {
        double prediction{mid(this->predictionMinMax())};
        double pOld{common::CTools::logisticFunction(mid(m_TreePredictionMinMax))};
        double mu{m_Mu * m_Count};
        return [prediction, pOld, mu, this](double weight) {
            double logOdds{prediction + weight};
            double c0{this->classCounts()(0)};
            double c1{this->classCounts()(1)};
            double logOneMinusPNew{logOneMinusLogistic(weight)};
            double logPNew{logLogistic(weight)};
            return this->lambda() * common::CTools::pow2(weight) -
                   c0 * logOneMinusLogistic(logOdds) - c1 * logLogistic(logOdds) -
                   mu * ((1.0 - pOld) * logOneMinusPNew + pOld * logPNew);
        };
    }

    // This is true if all the forest predictions were identical.
    if (bucketWidth(this->predictionMinMax()) == 0.0) {
        double prediction{mid(this->predictionMinMax())};
        return [prediction, this](double weight) {
            double logOdds{prediction + weight};
            double c0{this->classCounts()(0)};
            double c1{this->classCounts()(1)};
            double logOneMinusPNew{logOneMinusLogistic(weight)};
            double logPNew{logLogistic(weight)};
            double loss{this->lambda() * common::CTools::pow2(weight) -
                        c0 * logOneMinusLogistic(logOdds) - c1 * logLogistic(logOdds)};
            for (std::size_t i = 0; i < NUMBER_BUCKETS; ++i) {
                double pOld{common::CTools::logisticFunction(
                    bucketCentre(m_TreePredictionMinMax, i))};
                double mu{m_Mu * m_BucketsCount[i]};
                loss -= mu * ((1.0 - pOld) * logOneMinusPNew + pOld * logPNew);
            }
            return loss;
        };
    }

    // This is true if all the tree predictions were identical.
    if (bucketWidth(m_TreePredictionMinMax) == 0.0) {
        double pOld{common::CTools::logisticFunction(mid(m_TreePredictionMinMax))};
        double mu{m_Mu * m_Count};
        return [pOld, mu, this](double weight) {
            const auto& predictionMinMax = this->predictionMinMax();
            const auto& bucketsClassCounts = this->bucketsClassCounts();
            double logOneMinusPNew{logOneMinusLogistic(weight)};
            double logPNew{logLogistic(weight)};
            double loss{this->lambda() * common::CTools::pow2(weight) -
                        mu * ((1.0 - pOld) * logOneMinusPNew + pOld * logPNew)};
            for (std::size_t i = 0; i < NUMBER_BUCKETS; ++i) {
                double logOdds{bucketCentre(predictionMinMax, i) + weight};
                double c0{bucketsClassCounts[i](0)};
                double c1{bucketsClassCounts[i](1)};
                loss -= c0 * logOneMinusLogistic(logOdds) + c1 * logLogistic(logOdds);
            }
            return loss;
        };
    }

    return [this](double weight) {
        const auto& predictionMinMax = this->predictionMinMax();
        const auto& bucketsClassCounts = this->bucketsClassCounts();
        double logOneMinusPNew{logOneMinusLogistic(weight)};
        double logPNew{logLogistic(weight)};
        double loss{this->lambda() * common::CTools::pow2(weight)};
        for (std::size_t i = 0; i < NUMBER_BUCKETS; ++i) {
            double logOdds{bucketCentre(predictionMinMax, i) + weight};
            double c0{bucketsClassCounts[i](0)};
            double c1{bucketsClassCounts[i](1)};
            double mu{m_Mu * m_BucketsCount[i]};
            double pOld{common::CTools::logisticFunction(
                bucketCentre(m_TreePredictionMinMax, i))};
            loss -= c0 * logOneMinusLogistic(logOdds) + c1 * logLogistic(logOdds) +
                    mu * ((1.0 - pOld) * logOneMinusPNew + pOld * logPNew);
        }
        return loss;
    };
}

CArgMinMultinomialLogisticLossImpl::CArgMinMultinomialLogisticLossImpl(
    std::size_t numberClasses,
    double lambda,
    const common::CPRNG::CXorOShiro128Plus& rng)
    : CArgMinLossImpl{lambda}, m_NumberClasses{numberClasses}, m_Rng{rng},
      m_ClassCounts{TDoubleVector::Zero(numberClasses)}, m_Sampler{NUMBER_CENTRES} {
}

std::unique_ptr<CArgMinLossImpl> CArgMinMultinomialLogisticLossImpl::clone() const {
    return std::make_unique<CArgMinMultinomialLogisticLossImpl>(*this);
}

bool CArgMinMultinomialLogisticLossImpl::nextPass() {

    if (m_CurrentPass++ == 0) {
        m_Centres = std::move(m_Sampler.samples());
        std::sort(m_Centres.begin(), m_Centres.end());
        m_Centres.erase(std::unique(m_Centres.begin(), m_Centres.end()),
                        m_Centres.end());
        LOG_TRACE(<< "# centres = " << m_Centres.size());
        m_CurrentPass += m_Centres.size() == 1 ? 1 : 0;
        m_CentresClassCounts.resize(m_Centres.size(), TDoubleVector::Zero(m_NumberClasses));
    }

    LOG_TRACE(<< "current pass = " << m_CurrentPass);

    return m_CurrentPass < 2;
}

void CArgMinMultinomialLogisticLossImpl::add(const TMemoryMappedFloatVector& prediction,
                                             double actual,
                                             double weight) {

    using TMinAccumulator =
        common::CBasicStatistics::SMin<std::pair<double, std::size_t>>::TAccumulator;

    switch (m_CurrentPass) {
    case 0: {
        // We have a member variable to avoid allocating a tempory each time.
        m_DoublePrediction = prediction;
        m_Sampler.sample(m_DoublePrediction);
        m_ClassCounts(static_cast<std::size_t>(actual)) += weight;
        break;
    }
    case 1: {
        TMinAccumulator nearest;
        for (std::size_t i = 0; i < m_Centres.size(); ++i) {
            nearest.add({(m_Centres[i] - prediction).squaredNorm(), i});
        }
        auto& count = m_CentresClassCounts[nearest[0].second];
        count(static_cast<std::size_t>(actual)) += weight;
        break;
    }
    default:
        break;
    }
}

void CArgMinMultinomialLogisticLossImpl::merge(const CArgMinLossImpl& other) {
    const auto* logistic = dynamic_cast<const CArgMinMultinomialLogisticLossImpl*>(&other);
    if (logistic != nullptr) {
        switch (m_CurrentPass) {
        case 0:
            m_Sampler.merge(logistic->m_Sampler);
            m_ClassCounts += logistic->m_ClassCounts;
            break;
        case 1:
            for (std::size_t i = 0; i < m_CentresClassCounts.size(); ++i) {
                m_CentresClassCounts[i] += logistic->m_CentresClassCounts[i];
            }
            break;
        default:
            break;
        }
    }
}

CArgMinMultinomialLogisticLossImpl::TDoubleVector
CArgMinMultinomialLogisticLossImpl::value() const {

    // The optimisation objective is convex. To see this note that we can write
    // it as sum_i{ f_ij(w) } + ||w||^2 with f_ij(w) = -[log(softmax_j(z_i + w))].
    // Since the sum of convex functions is convex and ||.|| is clearly convex we
    // just require the f_ij to be convex. This is a standard result and follows from
    // the fact that their Hessian is of the form H = diag(p) - p p^t where 1-norm
    // of p is one. Convexity follows if this is positive definite. To verify note
    // that x^t H x = ||p^(1/2) x||^2 ||p^(1/2)||^2 - (p^t x)^2, which is greater
    // than 0 for all x via Cauchy-Schwarz. We optimize via L-BFGS. Note also that
    // we truncate lambda to be positive so the weights don't become too large for
    // leaves with only one class.

    TObjective objective{this->objective()};
    TObjectiveGradient objectiveGradient{this->objectiveGradient()};

    TDoubleVector wmin{TDoubleVector::Zero(m_NumberClasses)};

    double loss;
    common::CLbfgs<TDoubleVector> lgbfs{5};
    std::tie(wmin, loss) = lgbfs.minimize(objective, objectiveGradient, std::move(wmin));
    LOG_TRACE(<< "loss* = " << loss << " weight* = " << wmin.transpose());

    return wmin;
}

CArgMinMultinomialLogisticLossImpl::TObjective
CArgMinMultinomialLogisticLossImpl::objective() const {
    TDoubleVector logProbabilities{m_NumberClasses};
    double lambda{std::max(this->lambda(), 1e-6)};
    if (m_Centres.size() == 1) {
        return [logProbabilities, lambda, this](const TDoubleVector& weight) mutable -> double {
            logProbabilities = m_Centres[0] + weight;
            common::CTools::inplaceLogSoftmax(logProbabilities);
            return lambda * weight.squaredNorm() - m_ClassCounts.transpose() * logProbabilities;
        };
    }
    return [logProbabilities, lambda, this](const TDoubleVector& weight) mutable -> double {
        double loss{0.0};
        for (std::size_t i = 0; i < m_CentresClassCounts.size(); ++i) {
            if (m_CentresClassCounts[i].sum() > 0.0) {
                logProbabilities = m_Centres[i] + weight;
                common::CTools::inplaceLogSoftmax(logProbabilities);
                loss -= m_CentresClassCounts[i].transpose() * logProbabilities;
            }
        }
        return loss + lambda * weight.squaredNorm();
    };
}

CArgMinMultinomialLogisticLossImpl::TObjectiveGradient
CArgMinMultinomialLogisticLossImpl::objectiveGradient() const {
    TDoubleVector probabilities{m_NumberClasses};
    TDoubleVector lossGradient{m_NumberClasses};
    double lambda{std::max(this->lambda(), 1e-6)};
    if (m_Centres.size() == 1) {
        return [probabilities, lossGradient, lambda,
                this](const TDoubleVector& weight) mutable -> TDoubleVector {
            probabilities = m_Centres[0] + weight;
            common::CTools::inplaceSoftmax(probabilities);
            lossGradient = m_ClassCounts.array().sum() * probabilities - m_ClassCounts;
            return 2.0 * lambda * weight + lossGradient;
        };
    }
    return [probabilities, lossGradient, lambda,
            this](const TDoubleVector& weight) mutable -> TDoubleVector {
        lossGradient.array() = 0.0;
        for (std::size_t i = 0; i < m_CentresClassCounts.size(); ++i) {
            double n{m_CentresClassCounts[i].array().sum()};
            if (n > 0.0) {
                probabilities = m_Centres[i] + weight;
                common::CTools::inplaceSoftmax(probabilities);
                lossGradient -= m_CentresClassCounts[i] - n * probabilities;
            }
        }
        return 2.0 * lambda * weight + lossGradient;
    };
}

CArgMinMsleImpl::CArgMinMsleImpl(double lambda, double offset)
    : CArgMinLossImpl{lambda}, m_Offset{offset}, m_Buckets(MSLE_BUCKET_SIZE) {
    for (auto& bucket : m_Buckets) {
        bucket.resize(MSLE_BUCKET_SIZE);
    }
}

std::unique_ptr<CArgMinLossImpl> CArgMinMsleImpl::clone() const {
    return std::make_unique<CArgMinMsleImpl>(*this);
}

bool CArgMinMsleImpl::nextPass() {
    m_CurrentPass += this->bucketWidth().first > 0.0 ? 1 : 2;
    return m_CurrentPass < 2;
}

void CArgMinMsleImpl::add(const TMemoryMappedFloatVector& prediction, double actual, double weight) {
    double expPrediction{common::CTools::stableExp(prediction[0])};
    double logActual{common::CTools::fastLog(m_Offset + actual)};
    switch (m_CurrentPass) {
    case 0: {
        m_ExpPredictionMinMax.add(expPrediction);
        m_LogActualMinMax.add(logActual);
        m_MeanLogActual.add(logActual, weight);
        break;
    }
    case 1: {
        double logError{logActual - common::CTools::fastLog(m_Offset + expPrediction)};
        TVector example;
        example(MSLE_PREDICTION_INDEX) = expPrediction;
        example(MSLE_ACTUAL_INDEX) = logActual;
        example(MSLE_ERROR_INDEX) = logError;
        auto bucketIndex{this->bucket(expPrediction, logActual)};
        m_Buckets[bucketIndex.first][bucketIndex.second].add(example, weight);
        break;
    }
    default:
        break;
    }
}

void CArgMinMsleImpl::merge(const CArgMinLossImpl& other) {
    const auto* mlse = dynamic_cast<const CArgMinMsleImpl*>(&other);
    if (mlse != nullptr) {
        switch (m_CurrentPass) {
        case 0:
            m_ExpPredictionMinMax += mlse->m_ExpPredictionMinMax;
            m_LogActualMinMax += mlse->m_LogActualMinMax;
            m_MeanLogActual += mlse->m_MeanLogActual;
            break;
        case 1:
            for (std::size_t i = 0; i < m_Buckets.size(); ++i) {
                for (std::size_t j = 0; j < m_Buckets[i].size(); ++j) {
                    m_Buckets[i][j] += mlse->m_Buckets[i][j];
                }
            }
            break;
        default:
            break;
        }
    }
}

CArgMinMsleImpl::TDoubleVector CArgMinMsleImpl::value() const {
    TObjective objective;
    double minLogWeight;
    double maxLogWeight;

    objective = this->objective();
    if (this->bucketWidth().first == 0.0) {
        minLogWeight = -4.0;
        maxLogWeight = common::CBasicStatistics::mean(m_MeanLogActual);
    } else {
        // If the weight smaller than the minimum log error every prediction is low.
        // Conversely, if it's larger than the maximum log error every prediction is
        // high. In both cases, we can reduce the error by making the weight larger,
        // respectively smaller. Therefore, the optimal weight must lie between these
        // values.
        minLogWeight = std::numeric_limits<double>::max();
        maxLogWeight = -minLogWeight;
        for (const auto& bucketsPrediction : m_Buckets) {
            for (const auto& bucketActual : bucketsPrediction) {
                minLogWeight = std::min(
                    minLogWeight, common::CBasicStatistics::mean(bucketActual)(MSLE_ERROR_INDEX));
                maxLogWeight = std::max(
                    maxLogWeight, common::CBasicStatistics::mean(bucketActual)(MSLE_ERROR_INDEX));
            }
        }
    }

    double minimizer;
    double objectiveAtMinimum;
    std::size_t maxIterations{MSLE_OPTIMIZATION_ITERATIONS};
    common::CSolvers::minimize(minLogWeight, maxLogWeight, objective(minLogWeight),
                               objective(maxLogWeight), objective, 1e-5,
                               maxIterations, minimizer, objectiveAtMinimum);
    LOG_TRACE(<< "minimum = " << minimizer << " objective(minimum) = " << objectiveAtMinimum);

    TDoubleVector result(1);
    result(0) = minimizer;
    return result;
}

CArgMinMsleImpl::TObjective CArgMinMsleImpl::objective() const {
    return [this](double logWeight) {
        double weight{common::CTools::stableExp(logWeight)};
        if (this->bucketWidth().first == 0.0) {
            // prediction is constant
            double expPrediction{m_ExpPredictionMinMax.max()};
            double logPrediction{common::CTools::fastLog(m_Offset + expPrediction * weight)};
            double meanLogActual{common::CBasicStatistics::mean(m_MeanLogActual)};
            double meanLogActualSquared{common::CBasicStatistics::variance(m_MeanLogActual) +
                                        common::CTools::pow2(meanLogActual)};
            double loss{meanLogActualSquared - 2.0 * meanLogActual * logPrediction +
                        common::CTools::pow2(logPrediction)};
            return loss + this->lambda() * common::CTools::pow2(weight);
        }

        double loss{0.0};
        double totalCount{0.0};
        for (const auto& bucketPrediction : m_Buckets) {
            for (const auto& bucketActual : bucketPrediction) {
                double count{common::CBasicStatistics::count(bucketActual)};
                if (count > 0.0) {
                    const auto& bucketMean{common::CBasicStatistics::mean(bucketActual)};
                    double expPrediction{bucketMean(MSLE_PREDICTION_INDEX)};
                    double logActual{bucketMean(MSLE_ACTUAL_INDEX)};
                    double logPrediction{
                        common::CTools::fastLog(m_Offset + expPrediction * weight)};
                    loss += count * common::CTools::pow2(logActual - logPrediction);
                    totalCount += count;
                }
            }
        }
        return loss / totalCount + this->lambda() * common::CTools::pow2(weight);
    };
}

CArgMinPseudoHuberImpl::CArgMinPseudoHuberImpl(double lambda, double delta)
    : CArgMinLossImpl{lambda}, m_DeltaSquared{common::CTools::pow2(delta)},
      m_Buckets(HUBER_BUCKET_SIZE) {
}

std::unique_ptr<CArgMinLossImpl> CArgMinPseudoHuberImpl::clone() const {
    return std::make_unique<CArgMinPseudoHuberImpl>(*this);
}

bool CArgMinPseudoHuberImpl::nextPass() {
    m_CurrentPass += this->bucketWidth() > 0.0 ? 1 : 2;
    return m_CurrentPass < 2;
}

void CArgMinPseudoHuberImpl::add(const TMemoryMappedFloatVector& prediction,
                                 double actual,
                                 double weight) {
    switch (m_CurrentPass) {
    case 0: {
        m_ErrorMinMax.add(actual - prediction[0]);
        break;
    }
    case 1: {
        double error{actual - prediction[0]};
        auto bucketIndex{this->bucket(error)};
        m_Buckets[bucketIndex].add(error, weight);
        break;
    }
    default:
        break;
    }
}

void CArgMinPseudoHuberImpl::merge(const CArgMinLossImpl& other) {
    const auto* huber = dynamic_cast<const CArgMinPseudoHuberImpl*>(&other);
    if (huber != nullptr) {
        switch (m_CurrentPass) {
        case 0:
            m_ErrorMinMax += huber->m_ErrorMinMax;
            break;
        case 1:
            for (std::size_t i = 0; i < m_Buckets.size(); ++i) {
                m_Buckets[i] += huber->m_Buckets[i];
            }
            break;
        default:
            break;
        }
    }
}

CArgMinPseudoHuberImpl::TDoubleVector CArgMinPseudoHuberImpl::value() const {
    // Set the lower (upper) bounds for minimisation such that every example will
    // have the same sign error if the weight is smaller (larger) than this bound
    // and so would only increase the loss.
    double minWeight = m_ErrorMinMax.min();
    double maxWeight = m_ErrorMinMax.max();

    TObjective objective{this->objective()};
    double minimizer;
    double objectiveAtMinimum;
    std::size_t maxIterations{HUBER_OPTIMIZATION_ITERATIONS};
    common::CSolvers::minimize(minWeight, maxWeight, objective(minWeight),
                               objective(maxWeight), objective, 1e-5,
                               maxIterations, minimizer, objectiveAtMinimum);
    LOG_TRACE(<< "minimum = " << minimizer << " objective(minimum) = " << objectiveAtMinimum);

    TDoubleVector result(1);
    result(0) = minimizer;
    return result;
}

CArgMinPseudoHuberImpl::TObjective CArgMinPseudoHuberImpl::objective() const {
    return [this](double weight) {
        if (m_DeltaSquared > 0) {
            double loss{0.0};
            double totalCount{0.0};
            for (const auto& bucket : m_Buckets) {
                double count{common::CBasicStatistics::count(bucket)};
                if (count > 0.0) {
                    double error{common::CBasicStatistics::mean(bucket)};
                    loss += count * m_DeltaSquared *
                            (std::sqrt(1.0 + common::CTools::pow2(error - weight) / m_DeltaSquared) -
                             1.0);
                    totalCount += count;
                }
            }
            return loss / totalCount + this->lambda() * common::CTools::pow2(weight);
        }
        return 0.0;
    };
}
}

namespace boosted_tree {

void CLoss::persistLoss(core::CStatePersistInserter& inserter) const {
    auto persist = [this](core::CStatePersistInserter& inserter_) {
        this->acceptPersistInserter(inserter_);
    };
    inserter.insertLevel(this->name(), persist);
}

CLoss::TLossUPtr CLoss::restoreLoss(core::CStateRestoreTraverser& traverser) {
    const std::string& lossFunctionName{traverser.name()};
    try {
        if (lossFunctionName == CMse::NAME) {
            return std::make_unique<CMse>(traverser);
        }
        if (lossFunctionName == CMsle::NAME) {
            return std::make_unique<CMsle>(traverser);
        }
        if (lossFunctionName == CPseudoHuber::NAME) {
            return std::make_unique<CPseudoHuber>(traverser);
        }
        if (lossFunctionName == CBinomialLogisticLoss::NAME) {
            return std::make_unique<CBinomialLogisticLoss>(traverser);
        }
        if (lossFunctionName == CMultinomialLogisticLoss::NAME) {
            return std::make_unique<CMultinomialLogisticLoss>(traverser);
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Error restoring loss function " << lossFunctionName << " "
                  << e.what());
        return nullptr;
    }

    LOG_ERROR(<< "Error restoring loss function. Unknown loss function type '"
              << lossFunctionName << "'.");
    return nullptr;
}

CArgMinLoss::CArgMinLoss(const CArgMinLoss& other)
    : m_Impl{other.m_Impl->clone()} {
}

CArgMinLoss& CArgMinLoss::operator=(const CArgMinLoss& other) {
    if (this != &other) {
        m_Impl = other.m_Impl->clone();
    }
    return *this;
}

bool CArgMinLoss::nextPass() const {
    return m_Impl->nextPass();
}

void CArgMinLoss::add(const CEncodedDataFrameRowRef& row,
                      bool newExample,
                      const TMemoryMappedFloatVector& prediction,
                      double actual,
                      double weight) {
    return m_Impl->add(row, newExample, prediction, actual, weight);
}

void CArgMinLoss::merge(CArgMinLoss& other) {
    return m_Impl->merge(*other.m_Impl);
}

CArgMinLoss::TDoubleVector CArgMinLoss::value() const {
    return m_Impl->value();
}

CArgMinLoss::CArgMinLoss(const CArgMinLossImpl& impl) : m_Impl{impl.clone()} {
}

CArgMinLoss CLoss::makeMinimizer(const boosted_tree_detail::CArgMinLossImpl& impl) {
    return CArgMinLoss{impl};
}

CMse::CMse(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel([this](auto& traverser_) {
            return this->acceptRestoreTraverser(traverser_);
        }) == false) {
        throw std::runtime_error{"failed to restore CMse"};
    }
}

CMse::TLossUPtr CMse::clone() const {
    return std::make_unique<CMse>(*this);
}

CMse::TLossUPtr CMse::incremental(double eta, double mu, const TNodeVec& tree) const {
    return std::make_unique<CMseIncremental>(eta, mu, tree);
}

CMse::TLossUPtr CMse::project(std::size_t,
                              core::CDataFrame&,
                              const core::CPackedBitVector&,
                              std::size_t,
                              const TSizeVec&,
                              common::CPRNG::CXorOShiro128Plus) const {
    return this->clone();
}

ELossType CMse::type() const {
    return E_MseRegression;
}

std::size_t CMse::dimensionPrediction() const {
    return 1;
}

std::size_t CMse::dimensionGradient() const {
    return 1;
}

double CMse::value(const TMemoryMappedFloatVector& prediction, double actual, double weight) const {
    return weight * common::CTools::pow2(prediction(0) - actual);
}

void CMse::gradient(const TMemoryMappedFloatVector& prediction,
                    double actual,
                    const TWriter& writer,
                    double weight) const {
    writer(0, 2.0 * weight * (prediction(0) - actual));
}

void CMse::curvature(const TMemoryMappedFloatVector& /*prediction*/,
                     double /*actual*/,
                     const TWriter& writer,
                     double weight) const {
    writer(0, 2.0 * weight);
}

bool CMse::isCurvatureConstant() const {
    return true;
}

double CMse::difference(const TMemoryMappedFloatVector& prediction,
                        const TMemoryMappedFloatVector& previousPrediction,
                        double weight) const {
    return weight * common::CTools::pow2(prediction(0) - previousPrediction(0));
}

CMse::TDoubleVector CMse::transform(const TMemoryMappedFloatVector& prediction) const {
    return TDoubleVector{prediction};
}

CArgMinLoss CMse::minimizer(double lambda, const common::CPRNG::CXorOShiro128Plus&) const {
    return this->makeMinimizer(CArgMinMseImpl{lambda});
}

const std::string& CMse::name() const {
    return NAME;
}

bool CMse::isRegression() const {
    return true;
}

void CMse::acceptPersistInserter(core::CStatePersistInserter& /* inserter */) const {
}

bool CMse::acceptRestoreTraverser(core::CStateRestoreTraverser& /* traverser */) {
    return true;
}

const std::string CMse::NAME{"mse"};

CMseIncremental::CMseIncremental(double eta, double mu, const TNodeVec& tree)
    : m_Eta{eta}, m_Mu{mu}, m_Tree{&tree} {
}

CMseIncremental::TLossUPtr CMseIncremental::clone() const {
    return std::make_unique<CMseIncremental>(*this);
}

CMseIncremental::TLossUPtr
CMseIncremental::incremental(double eta, double mu, const TNodeVec& tree) const {
    return std::make_unique<CMseIncremental>(eta, mu, tree);
}

CMseIncremental::TLossUPtr CMseIncremental::project(std::size_t,
                                                    core::CDataFrame&,
                                                    const core::CPackedBitVector&,
                                                    std::size_t,
                                                    const TSizeVec&,
                                                    common::CPRNG::CXorOShiro128Plus) const {
    return this->clone();
}

ELossType CMseIncremental::type() const {
    return E_MseRegression;
}

std::size_t CMseIncremental::dimensionPrediction() const {
    return 1;
}

std::size_t CMseIncremental::dimensionGradient() const {
    return 1;
}

double CMseIncremental::value(const TMemoryMappedFloatVector& prediction,
                              double actual,
                              double weight) const {
    // This purposely doesn't include any loss term for changing the prediction.
    // This is used to estimate the quality of a retrained forest and select
    // hyperaparameters which penalise changing predictions such as mu. As such
    // we compute loss on a hold out from the old data to act as a proxy for how
    // much we might have damaged accuracy on the original training data.
    return this->CMse::value(prediction, actual, weight);
}

void CMseIncremental::gradient(const CEncodedDataFrameRowRef& row,
                               bool newExample,
                               const TMemoryMappedFloatVector& prediction,
                               double actual,
                               const TWriter& writer,
                               double weight) const {
    if (newExample) {
        this->CMse::gradient(prediction, actual, writer, weight);
    } else {
        double treePrediction{root(*m_Tree).value(row, *m_Tree)(0)};
        writer(0, 2.0 * weight * (prediction(0) - actual + m_Mu / m_Eta * treePrediction));
    }
}

void CMseIncremental::curvature(bool newExample,
                                const TMemoryMappedFloatVector& /*prediction*/,
                                double /*actual*/,
                                const TWriter& writer,
                                double weight) const {
    writer(0, 2.0 * weight * (1.0 + (newExample ? 0.0 : m_Mu)));
}

bool CMseIncremental::isCurvatureConstant() const {
    return true;
}

double CMseIncremental::difference(const TMemoryMappedFloatVector& prediction,
                                   const TMemoryMappedFloatVector& previousPrediction,
                                   double weight) const {
    return this->CMse::difference(prediction, previousPrediction, weight);
}

CMse::TDoubleVector CMseIncremental::transform(const TMemoryMappedFloatVector& prediction) const {
    return TDoubleVector{prediction};
}

CArgMinLoss CMseIncremental::minimizer(double lambda,
                                       const common::CPRNG::CXorOShiro128Plus&) const {
    return this->makeMinimizer(CArgMinMseIncrementalImpl{lambda, m_Eta, m_Mu, *m_Tree});
}

const std::string& CMseIncremental::name() const {
    return NAME;
}

bool CMseIncremental::isRegression() const {
    return true;
}

bool CMseIncremental::acceptRestoreTraverser(core::CStateRestoreTraverser&) {
    return false;
}

const std::string CMseIncremental::NAME{"mse_incremental"};

CMsle::CMsle(double offset) : m_Offset{offset} {
}

CMsle::CMsle(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel([this](auto& traverser_) {
            return this->acceptRestoreTraverser(traverser_);
        }) == false) {
        throw std::runtime_error{"failed to restore CMsle"};
    }
}

CMsle::TLossUPtr CMsle::clone() const {
    return std::make_unique<CMsle>(*this);
}

CMsle::TLossUPtr CMsle::incremental(double, double, const TNodeVec&) const {
    return nullptr;
}

CMsle::TLossUPtr CMsle::project(std::size_t,
                                core::CDataFrame&,
                                const core::CPackedBitVector&,
                                std::size_t,
                                const TSizeVec&,
                                common::CPRNG::CXorOShiro128Plus) const {
    return this->clone();
}

ELossType CMsle::type() const {
    return E_MsleRegression;
}

std::size_t CMsle::dimensionPrediction() const {
    return 1;
}

std::size_t CMsle::dimensionGradient() const {
    return 1;
}

double CMsle::value(const TMemoryMappedFloatVector& logPrediction, double actual, double weight) const {
    double prediction{common::CTools::stableExp(logPrediction(0))};
    double logOffsetPrediction{common::CTools::stableLog(m_Offset + prediction)};
    if (actual < 0.0) {
        HANDLE_FATAL(<< "Input error: target value needs to be non-negative to use "
                     << "with MSLE loss, received: " << actual);
    }
    double logOffsetActual{common::CTools::stableLog(m_Offset + actual)};
    return weight * common::CTools::pow2(logOffsetPrediction - logOffsetActual);
}

void CMsle::gradient(const TMemoryMappedFloatVector& logPrediction,
                     double actual,
                     const TWriter& writer,
                     double weight) const {
    double prediction{common::CTools::stableExp(logPrediction(0))};
    double log1PlusPrediction{common::CTools::stableLog(m_Offset + prediction)};
    double log1PlusActual{common::CTools::stableLog(m_Offset + actual)};
    writer(0, 2.0 * weight * (log1PlusPrediction - log1PlusActual) / (prediction + 1.0));
}

void CMsle::curvature(const TMemoryMappedFloatVector& logPrediction,
                      double actual,
                      const TWriter& writer,
                      double weight) const {
    double prediction{common::CTools::stableExp(logPrediction(0))};
    double logOffsetPrediction{common::CTools::stableLog(m_Offset + prediction)};
    double logOffsetActual{common::CTools::stableLog(m_Offset + actual)};
    // Apply L'Hopital's rule in the limit prediction -> actual.
    writer(0, prediction == actual
                  ? 0.0
                  : 2.0 * weight * (logOffsetPrediction - logOffsetActual) /
                        ((prediction + m_Offset) * (prediction - actual)));
}

bool CMsle::isCurvatureConstant() const {
    return false;
}

double CMsle::difference(const TMemoryMappedFloatVector& logPrediction,
                         const TMemoryMappedFloatVector& logPreviousPrediction,
                         double weight) const {
    double prediction{common::CTools::stableExp(logPrediction(0))};
    double previousPrediction{common::CTools::stableExp(logPreviousPrediction(0))};
    double logOffsetPrediction{common::CTools::stableLog(m_Offset + prediction)};
    double logOffsetPreviousPrediction{common::CTools::stableLog(m_Offset + previousPrediction)};
    return weight * common::CTools::pow2(logOffsetPrediction - logOffsetPreviousPrediction);
}

CMsle::TDoubleVector CMsle::transform(const TMemoryMappedFloatVector& prediction) const {
    TDoubleVector result{1};
    result(0) = std::exp(prediction(0));
    return result;
}

CArgMinLoss CMsle::minimizer(double lambda,
                             const common::CPRNG::CXorOShiro128Plus& /* rng */) const {
    return this->makeMinimizer(CArgMinMsleImpl{lambda});
}

const std::string& CMsle::name() const {
    return NAME;
}

bool CMsle::isRegression() const {
    return true;
}

void CMsle::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(OFFSET_TAG, m_Offset, inserter);
}

bool CMsle::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(OFFSET_TAG, core::CPersistUtils::restore(OFFSET_TAG, m_Offset, traverser))
    } while (traverser.next());
    return true;
}

const std::string CMsle::NAME{"msle"};

CPseudoHuber::CPseudoHuber(double delta) : m_Delta{delta} {};

CPseudoHuber::CPseudoHuber(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel([this](auto& traverser_) {
            return this->acceptRestoreTraverser(traverser_);
        }) == false) {
        throw std::runtime_error{"failed to restore CPseudoHuber"};
    }
}

CPseudoHuber::TLossUPtr CPseudoHuber::clone() const {
    return std::make_unique<CPseudoHuber>(*this);
}

CPseudoHuber::TLossUPtr CPseudoHuber::incremental(double, double, const TNodeVec&) const {
    return nullptr;
}

CPseudoHuber::TLossUPtr CPseudoHuber::project(std::size_t,
                                              core::CDataFrame&,
                                              const core::CPackedBitVector&,
                                              std::size_t,
                                              const TSizeVec&,
                                              common::CPRNG::CXorOShiro128Plus) const {
    return this->clone();
}

ELossType CPseudoHuber::type() const {
    return E_HuberRegression;
}

std::size_t CPseudoHuber::dimensionPrediction() const {
    return 1;
}

std::size_t CPseudoHuber::dimensionGradient() const {
    return 1;
}

double CPseudoHuber::value(const TMemoryMappedFloatVector& prediction,
                           double actual,
                           double weight) const {
    double delta2{common::CTools::pow2(m_Delta)};
    return weight * delta2 *
           (std::sqrt(1.0 + common::CTools::pow2(actual - prediction(0)) / delta2) - 1.0);
}

void CPseudoHuber::gradient(const TMemoryMappedFloatVector& prediction,
                            double actual,
                            const TWriter& writer,
                            double weight) const {
    writer(0, weight * (prediction(0) - actual) /
                  (std::sqrt(1.0 + common::CTools::pow2((actual - prediction(0)) / m_Delta))));
}

void CPseudoHuber::curvature(const TMemoryMappedFloatVector& prediction,
                             double actual,
                             const TWriter& writer,
                             double weight) const {
    double result{
        1.0 / (std::sqrt(1.0 + common::CTools::pow2((actual - prediction(0)) / m_Delta)))};
    writer(0, weight * result);
}

bool CPseudoHuber::isCurvatureConstant() const {
    return false;
}

double CPseudoHuber::difference(const TMemoryMappedFloatVector& prediction,
                                const TMemoryMappedFloatVector& previousPrediction,
                                double weight) const {
    double delta2{common::CTools::pow2(m_Delta)};
    return weight * delta2 *
           (std::sqrt(1.0 + common::CTools::pow2(prediction(0) - previousPrediction(0)) / delta2) -
            1.0);
}

CPseudoHuber::TDoubleVector
CPseudoHuber::transform(const TMemoryMappedFloatVector& prediction) const {
    TDoubleVector result{1};
    result(0) = prediction(0);
    return result;
}

CArgMinLoss CPseudoHuber::minimizer(double lambda,
                                    const common::CPRNG::CXorOShiro128Plus& /* rng */) const {
    return this->makeMinimizer(CArgMinPseudoHuberImpl{lambda, m_Delta});
}

const std::string& CPseudoHuber::name() const {
    return NAME;
}

bool CPseudoHuber::isRegression() const {
    return true;
}

void CPseudoHuber::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(DELTA_TAG, m_Delta, inserter);
}

bool CPseudoHuber::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(DELTA_TAG, core::CPersistUtils::restore(DELTA_TAG, m_Delta, traverser))
    } while (traverser.next());
    return true;
}

const std::string CPseudoHuber::NAME{"pseudo_huber"};

CBinomialLogisticLoss::CBinomialLogisticLoss(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel([this](auto& traverser_) {
            return this->acceptRestoreTraverser(traverser_);
        }) == false) {
        throw std::runtime_error{"failed to restore CBinomialLogisticLoss"};
    }
}

CBinomialLogisticLoss::TLossUPtr CBinomialLogisticLoss::clone() const {
    return std::make_unique<CBinomialLogisticLoss>(*this);
}

CBinomialLogisticLoss::TLossUPtr
CBinomialLogisticLoss::incremental(double eta, double mu, const TNodeVec& tree) const {
    return std::make_unique<CBinomialLogisticLossIncremental>(eta, mu, tree);
}

CBinomialLogisticLoss::TLossUPtr
CBinomialLogisticLoss::project(std::size_t,
                               core::CDataFrame&,
                               const core::CPackedBitVector&,
                               std::size_t,
                               const TSizeVec&,
                               common::CPRNG::CXorOShiro128Plus) const {
    return this->clone();
}

ELossType CBinomialLogisticLoss::type() const {
    return E_BinaryClassification;
}

std::size_t CBinomialLogisticLoss::dimensionPrediction() const {
    return 1;
}

std::size_t CBinomialLogisticLoss::dimensionGradient() const {
    return 1;
}

double CBinomialLogisticLoss::value(const TMemoryMappedFloatVector& prediction,
                                    double actual,
                                    double weight) const {
    return -weight * ((1.0 - actual) * logOneMinusLogistic(prediction(0)) +
                      actual * logLogistic(prediction(0)));
}

void CBinomialLogisticLoss::gradient(const TMemoryMappedFloatVector& prediction,
                                     double actual,
                                     const TWriter& writer,
                                     double weight) const {
    if (prediction(0) > -LOG_EPSILON && actual == 1.0) {
        writer(0, -weight * common::CTools::stableExp(-prediction(0)));
    } else {
        writer(0, weight * (common::CTools::logisticFunction(prediction(0)) - actual));
    }
}

void CBinomialLogisticLoss::curvature(const TMemoryMappedFloatVector& prediction,
                                      double /*actual*/,
                                      const TWriter& writer,
                                      double weight) const {
    if (prediction(0) > -LOG_EPSILON) {
        writer(0, weight * common::CTools::stableExp(-prediction(0)));
    } else {
        double probability{common::CTools::logisticFunction(prediction(0))};
        writer(0, weight * probability * (1.0 - probability));
    }
}

bool CBinomialLogisticLoss::isCurvatureConstant() const {
    return false;
}

double CBinomialLogisticLoss::difference(const TMemoryMappedFloatVector& prediction,
                                         const TMemoryMappedFloatVector& previousPrediction,
                                         double weight) const {
    // The cross entropy of the new predicted probabilities given the previous ones.
    double previousProbability{common::CTools::logisticFunction(previousPrediction(0))};
    return -weight * ((1.0 - previousProbability) * logOneMinusLogistic(prediction(0)) +
                      previousProbability * logLogistic(prediction(0)));
}

CBinomialLogisticLoss::TDoubleVector
CBinomialLogisticLoss::transform(const TMemoryMappedFloatVector& prediction) const {
    double p1{common::CTools::logisticFunction(prediction(0))};
    TDoubleVector result{2};
    result(0) = 1.0 - p1;
    result(1) = p1;
    return result;
}

CArgMinLoss CBinomialLogisticLoss::minimizer(double lambda,
                                             const common::CPRNG::CXorOShiro128Plus&) const {
    return this->makeMinimizer(CArgMinBinomialLogisticLossImpl{lambda});
}

const std::string& CBinomialLogisticLoss::name() const {
    return NAME;
}

bool CBinomialLogisticLoss::isRegression() const {
    return false;
}

void CBinomialLogisticLoss::acceptPersistInserter(core::CStatePersistInserter& /* inserter */) const {
}
bool CBinomialLogisticLoss::acceptRestoreTraverser(core::CStateRestoreTraverser& /* traverser */) {
    return true;
}

const std::string CBinomialLogisticLoss::NAME{"binomial_logistic"};

CBinomialLogisticLossIncremental::CBinomialLogisticLossIncremental(double eta,
                                                                   double mu,
                                                                   const TNodeVec& tree)
    : m_Eta{eta}, m_Mu{mu}, m_Tree{&tree} {
}

CBinomialLogisticLossIncremental::TLossUPtr CBinomialLogisticLossIncremental::clone() const {
    return std::make_unique<CBinomialLogisticLossIncremental>(*this);
}

CBinomialLogisticLossIncremental::TLossUPtr
CBinomialLogisticLossIncremental::incremental(double eta, double mu, const TNodeVec& tree) const {
    return std::make_unique<CBinomialLogisticLossIncremental>(eta, mu, tree);
}

CBinomialLogisticLossIncremental::TLossUPtr
CBinomialLogisticLossIncremental::project(std::size_t,
                                          core::CDataFrame&,
                                          const core::CPackedBitVector&,
                                          std::size_t,
                                          const TSizeVec&,
                                          common::CPRNG::CXorOShiro128Plus) const {
    return this->clone();
}

ELossType CBinomialLogisticLossIncremental::type() const {
    return E_BinaryClassification;
}

std::size_t CBinomialLogisticLossIncremental::dimensionPrediction() const {
    return 1;
}

std::size_t CBinomialLogisticLossIncremental::dimensionGradient() const {
    return 1;
}

double CBinomialLogisticLossIncremental::value(const TMemoryMappedFloatVector& prediction,
                                               double actual,
                                               double weight) const {
    // This purposely doesn't include any loss term for changing the prediction.
    // This is used to estimate the quality of a retrained forest and select
    // hyperaparameters which penalise changing predictions such as mu. As such
    // we compute loss on a hold out from the old data to act as a proxy for how
    // much we might have damaged accuracy on the original training data.
    return this->CBinomialLogisticLoss::value(prediction, actual, weight);
}

void CBinomialLogisticLossIncremental::gradient(const CEncodedDataFrameRowRef& row,
                                                bool newExample,
                                                const TMemoryMappedFloatVector& prediction,
                                                double actual,
                                                const TWriter& writer,
                                                double weight) const {
    if (newExample) {
        this->CBinomialLogisticLoss::gradient(prediction, actual, writer, weight);
    } else {
        double treePrediction{common::CTools::logisticFunction(
            root(*m_Tree).value(row, *m_Tree)(0) / m_Eta)};
        if (prediction(0) > -LOG_EPSILON && actual == 1.0) {
            writer(0, -weight * ((1.0 + m_Mu) * common::CTools::stableExp(-prediction(0)) +
                                 m_Mu * (treePrediction - 1.0)));
        } else {
            writer(0, weight * ((1.0 + m_Mu) * common::CTools::logisticFunction(prediction(0)) -
                                actual - m_Mu * treePrediction));
        }
    }
}

void CBinomialLogisticLossIncremental::curvature(const CEncodedDataFrameRowRef& /*row*/,
                                                 bool newExample,
                                                 const TMemoryMappedFloatVector& prediction,
                                                 double /*actual*/,
                                                 const TWriter& writer,
                                                 double weight) const {
    if (prediction(0) > -LOG_EPSILON) {
        writer(0, weight * (newExample ? 1.0 : 1.0 + m_Mu) *
                      common::CTools::stableExp(-prediction(0)));
    } else {
        double probability{common::CTools::logisticFunction(prediction(0))};
        writer(0, weight * (newExample ? 1.0 : 1.0 + m_Mu) * probability * (1.0 - probability));
    }
}

bool CBinomialLogisticLossIncremental::isCurvatureConstant() const {
    return false;
}

double CBinomialLogisticLossIncremental::difference(const TMemoryMappedFloatVector& prediction,
                                                    const TMemoryMappedFloatVector& previousPrediction,
                                                    double weight) const {
    return this->CBinomialLogisticLoss::difference(prediction, previousPrediction, weight);
}

CBinomialLogisticLossIncremental::TDoubleVector
CBinomialLogisticLossIncremental::transform(const TMemoryMappedFloatVector& prediction) const {
    return this->CBinomialLogisticLoss::transform(prediction);
}

CArgMinLoss CBinomialLogisticLossIncremental::minimizer(double lambda,
                                                        const common::CPRNG::CXorOShiro128Plus&) const {
    return this->makeMinimizer(
        CArgMinBinomialLogisticLossIncrementalImpl{lambda, m_Eta, m_Mu, *m_Tree});
}

const std::string& CBinomialLogisticLossIncremental::name() const {
    return NAME;
}

bool CBinomialLogisticLossIncremental::isRegression() const {
    return false;
}

bool CBinomialLogisticLossIncremental::acceptRestoreTraverser(core::CStateRestoreTraverser&) {
    return false;
}

const std::string CBinomialLogisticLossIncremental::NAME{"binomial_logistic_incremental"};

CMultinomialLogisticLoss::CMultinomialLogisticLoss(std::size_t numberClasses)
    : m_NumberClasses{numberClasses} {
}

CMultinomialLogisticLoss::CMultinomialLogisticLoss(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel([this](auto& traverser_) {
            return this->acceptRestoreTraverser(traverser_);
        }) == false) {
        throw std::runtime_error{"failed to restore CMultinomialLogisticLoss"};
    }
}

CMultinomialLogisticLoss::TLossUPtr CMultinomialLogisticLoss::clone() const {
    return std::make_unique<CMultinomialLogisticLoss>(m_NumberClasses);
}

CMultinomialLogisticLoss::TLossUPtr
CMultinomialLogisticLoss::incremental(double, double, const TNodeVec&) const {
    return nullptr;
}

CMultinomialLogisticLoss::TLossUPtr
CMultinomialLogisticLoss::project(std::size_t numberThreads,
                                  core::CDataFrame& frame,
                                  const core::CPackedBitVector& rowMask,
                                  std::size_t targetColumn,
                                  const TSizeVec& extraColumns,
                                  common::CPRNG::CXorOShiro128Plus rng) const {
    if (m_NumberClasses <= MAX_GRADIENT_DIMENSION) {
        return this->clone();
    }

    // Compute total loss over masked rows.
    auto result =
        frame
            .readRows(
                numberThreads, 0, frame.numberRows(),
                core::bindRetrievableState(
                    [&](TDoubleVec& losses, const TRowItr& beginRows, const TRowItr& endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            auto prediction = readPrediction(*row, extraColumns, m_NumberClasses);
                            double actual{readActual(*row, targetColumn)};
                            double weight{readExampleWeight(*row, extraColumns)};
                            losses[static_cast<int>(actual)] +=
                                this->value(prediction, actual, weight);
                        }
                    },
                    TDoubleVec(m_NumberClasses, 0.0)),
                &rowMask)
            .first;

    auto losses = std::move(result[0].s_FunctionState);
    for (std::size_t i = 1; i < result.size(); ++i) {
        for (std::size_t j = 1; j < losses.size(); ++j) {
            losses[j] += result[i].s_FunctionState[j];
        }
    }

    TSizeVec classes;
    common::CSampling::categoricalSampleWithoutReplacement(
        rng, losses, MAX_GRADIENT_DIMENSION - 1, classes);
    std::sort(classes.begin(), classes.end());

    return std::make_unique<CSubsetMultinomialLogisticLoss>(m_NumberClasses, classes);
}

ELossType CMultinomialLogisticLoss::type() const {
    return E_MulticlassClassification;
}

std::size_t CMultinomialLogisticLoss::dimensionPrediction() const {
    return m_NumberClasses;
}

std::size_t CMultinomialLogisticLoss::dimensionGradient() const {
    return std::min(m_NumberClasses, MAX_GRADIENT_DIMENSION);
}

double CMultinomialLogisticLoss::value(const TMemoryMappedFloatVector& prediction,
                                       double actual,
                                       double weight) const {
    double zmax{prediction.maxCoeff()};
    double logZ{0.0};
    for (int i = 0; i < prediction.size(); ++i) {
        logZ += std::exp(prediction(i) - zmax);
    }
    logZ = zmax + common::CTools::stableLog(logZ);

    // i.e. -log(z(actual))
    return weight * (logZ - prediction(static_cast<int>(actual)));
}

void CMultinomialLogisticLoss::gradient(const TMemoryMappedFloatVector& prediction,
                                        double actual,
                                        const TWriter& writer,
                                        double weight) const {

    // We prefer an implementation which avoids any memory allocations.

    double zmax{prediction.maxCoeff()};
    double pEps{0.0};
    double logZ{0.0};
    for (int i = 0; i < prediction.size(); ++i) {
        if (prediction(i) - zmax < LOG_EPSILON) {
            // Sum the contributions from classes whose predicted probability
            // is less than epsilon, for which we'd lose all nearly precision
            // when adding to the normalisation coefficient.
            pEps += std::exp(prediction(i) - zmax);
        } else {
            logZ += std::exp(prediction(i) - zmax);
        }
    }
    pEps = common::CTools::stable(pEps / logZ);
    logZ = zmax + common::CTools::stableLog(logZ);

    for (int i = 0; i < prediction.size(); ++i) {
        double pi{common::CTools::stableExp(prediction(i) - logZ)};
        if (i == static_cast<int>(actual)) {
            // We have that p = 1 / (1 + eps) and the gradient is p - 1.
            // Use a Taylor expansion and drop terms of O(eps^2) to get:
            writer(i, weight * (pi == 1.0 ? -pEps : pi - 1.0));
        } else {
            writer(i, weight * pi);
        }
    }
}

void CMultinomialLogisticLoss::curvature(const TMemoryMappedFloatVector& prediction,
                                         double /*actual*/,
                                         const TWriter& writer,
                                         double weight) const {

    // Return the lower triangle of the Hessian column major.

    // We prefer an implementation which avoids any memory allocations.

    double zmax{prediction.maxCoeff()};
    double pEps{0.0};
    double logZ{0.0};
    for (int i = 0; i < prediction.size(); ++i) {
        double pAdj{std::exp(prediction(i) - zmax)};
        if (prediction(i) - zmax < LOG_EPSILON) {
            // Sum the contributions from classes whose predicted probability
            // is less than epsilon, for which we'd lose all nearly precision
            // when adding to the normalisation coefficient.
            pEps += pAdj;
        } else {
            logZ += pAdj;
        }
    }
    pEps = common::CTools::stable(pEps / logZ);
    logZ = zmax + common::CTools::stableLog(logZ);

    std::size_t k{0};
    for (int i = 0; i < prediction.size(); ++i) {
        double pi{common::CTools::stableExp(prediction(i) - logZ)};
        // We have that p = 1 / (1 + eps) and the curvature is p (1 - p).
        // Use a Taylor expansion and drop terms of O(eps^2) to get:
        writer(k++, weight * (pi == 1.0 ? pEps : pi * (1.0 - pi)));
        for (int j = i + 1; j < prediction.size(); ++j) {
            double pij{common::CTools::stableExp(prediction(i) + prediction(j) - 2.0 * logZ)};
            writer(k++, -weight * pij);
        }
    }
    LOG_TRACE(<< "Wrote " << k << " curvatures");
}

bool CMultinomialLogisticLoss::isCurvatureConstant() const {
    return false;
}

double CMultinomialLogisticLoss::difference(const TMemoryMappedFloatVector& prediction,
                                            const TMemoryMappedFloatVector& previousPrediction,
                                            double weight) const {

    // The cross entropy of the new predicted probabilities given the previous ones.

    double zmax{prediction.maxCoeff()};
    double logZ{0.0};
    for (int i = 0; i < prediction.size(); ++i) {
        logZ += std::exp(prediction(i) - zmax);
    }
    logZ = zmax + common::CTools::stableLog(logZ);

    double result{0};
    auto previousProbabilities = this->transform(previousPrediction);
    for (int i = 0; i < prediction.size(); ++i) {
        result += previousProbabilities(i) * (logZ - prediction(i));
    }

    return weight * result;
}

CMultinomialLogisticLoss::TDoubleVector
CMultinomialLogisticLoss::transform(const TMemoryMappedFloatVector& prediction) const {
    TDoubleVector result{prediction};
    common::CTools::inplaceSoftmax(result);
    return result;
}

CArgMinLoss CMultinomialLogisticLoss::minimizer(double lambda,
                                                const common::CPRNG::CXorOShiro128Plus& rng) const {
    return this->makeMinimizer(
        CArgMinMultinomialLogisticLossImpl{m_NumberClasses, lambda, rng});
}

const std::string& CMultinomialLogisticLoss::name() const {
    return NAME;
}

bool CMultinomialLogisticLoss::isRegression() const {
    return false;
}

void CMultinomialLogisticLoss::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(NUMBER_CLASSES_TAG, m_NumberClasses, inserter);
}

bool CMultinomialLogisticLoss::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(NUMBER_CLASSES_TAG,
                core::CPersistUtils::restore(NUMBER_CLASSES_TAG, m_NumberClasses, traverser))
    } while (traverser.next());
    return true;
}

const std::string CMultinomialLogisticLoss::NAME{"multinomial_logistic"};

CSubsetMultinomialLogisticLoss::CSubsetMultinomialLogisticLoss(std::size_t numberClasses,
                                                               const TSizeVec& classes)
    : CMultinomialLogisticLoss{numberClasses} {
    m_InClasses.reserve(classes.size());
    for (auto i : classes) {
        m_InClasses.push_back(static_cast<int>(i));
    }
    m_OutClasses.resize(numberClasses - classes.size());
    std::iota(m_OutClasses.begin(), m_OutClasses.end(), 0);
    m_OutClasses.erase(std::set_difference(m_OutClasses.begin(), m_OutClasses.end(),
                                           m_InClasses.begin(), m_InClasses.end(),
                                           m_OutClasses.begin()),
                       m_OutClasses.end());
}

CSubsetMultinomialLogisticLoss::TLossUPtr CSubsetMultinomialLogisticLoss::clone() const {
    return std::make_unique<CSubsetMultinomialLogisticLoss>(*this);
}

CSubsetMultinomialLogisticLoss::TLossUPtr
CSubsetMultinomialLogisticLoss::incremental(double, double, const TNodeVec&) const {
    return nullptr;
}

CSubsetMultinomialLogisticLoss::TLossUPtr
CSubsetMultinomialLogisticLoss::project(std::size_t,
                                        core::CDataFrame&,
                                        const core::CPackedBitVector&,
                                        std::size_t,
                                        const TSizeVec&,
                                        common::CPRNG::CXorOShiro128Plus) const {
    return this->clone();
}

void CSubsetMultinomialLogisticLoss::gradient(const CEncodedDataFrameRowRef& /*row*/,
                                              bool /*newExample*/,
                                              const TMemoryMappedFloatVector& prediction,
                                              double actual,
                                              const TWriter& writer,
                                              double weight) const {

    // We prefer an implementation which avoids any memory allocations.

    int actual_{static_cast<int>(actual)};
    double zmax{prediction.maxCoeff()};
    double pEps{0.0};
    double logZ{0.0};
    double logPAgg{0.0};
    bool isActualIn{false};
    for (auto i : m_InClasses) {
        double pAdj{std::exp(prediction(i) - zmax)};
        // Sum the contributions from classes whose predicted probability
        // is less than epsilon, for which we'd lose all nearly precision
        // when adding to the normalisation coefficient.
        (prediction(i) - zmax < LOG_EPSILON ? pEps : logZ) += pAdj;
        isActualIn |= (actual_ == i);
    }
    for (auto i : m_OutClasses) {
        double pAdj{std::exp(prediction(i) - zmax)};
        logPAgg += pAdj;
    }
    (logPAgg < EPSILON * logZ ? pEps : logZ) += logPAgg;
    pEps = common::CTools::stable(pEps / logZ);
    logZ = zmax + std::log(logZ);
    logPAgg = zmax + std::log(logPAgg);

    for (std::size_t i = 0; i <= m_InClasses.size(); ++i) {
        double pi{common::CTools::stableExp(
            (i < m_InClasses.size() ? static_cast<double>(prediction(m_InClasses[i])) : logPAgg) -
            logZ)};
        bool isActual{i < m_InClasses.size() ? (m_InClasses[i] == actual_)
                                             : (isActualIn == false)};
        if (isActual) {
            // We have that p = 1 / (1 + eps) and the gradient is p - 1.
            // Use a Taylor expansion and drop terms of O(eps^2) to get:
            writer(i, weight * (pi == 1.0 ? -pEps : pi - 1.0));
        } else {
            writer(i, weight * pi);
        }
    }
}

void CSubsetMultinomialLogisticLoss::curvature(const CEncodedDataFrameRowRef& /*row*/,
                                               bool /*newExample*/,
                                               const TMemoryMappedFloatVector& prediction,
                                               double /*actual*/,
                                               const TWriter& writer,
                                               double weight) const {

    // Return the lower triangle of the Hessian column major.

    // We prefer an implementation which avoids any memory allocations.

    double zmax{prediction.maxCoeff()};
    double pEps{0.0};
    double logZ{0.0};
    double pAgg{0.0};
    for (auto i : m_InClasses) {
        double pAdj{std::exp(prediction(i) - zmax)};
        if (prediction(i) - zmax < LOG_EPSILON) {
            // Sum the contributions from classes whose predicted probability
            // is less than epsilon, for which we'd lose all nearly precision
            // when adding to the normalisation coefficient.
            pEps += pAdj;
        } else {
            logZ += pAdj;
        }
    }
    for (auto i : m_OutClasses) {
        double pAdj{std::exp(prediction(i) - zmax)};
        logZ += pAdj;
        pAgg += pAdj;
    }
    pEps = common::CTools::stable((pEps + (pAgg < EPSILON * logZ ? pAgg : 0.0)) / logZ);
    logZ = zmax + common::CTools::stableLog(logZ);
    pAgg = pAgg * common::CTools::stableExp(zmax - logZ);

    std::size_t k{0};
    for (std::size_t i = 0; i <= m_InClasses.size(); ++i) {
        double pi{i < m_InClasses.size()
                      ? common::CTools::stableExp(prediction(m_InClasses[i]) - logZ)
                      : pAgg};
        // We have that p = 1 / (1 + eps) and the curvature is p (1 - p).
        // Use a Taylor expansion and drop terms of O(eps^2) to get:
        writer(k++, weight * (pi == 1.0 ? pEps : pi * (1.0 - pi)));
        for (std::size_t j = i + 1; j <= m_InClasses.size(); ++j) {
            double pij{pi * (j < m_InClasses.size()
                                 ? common::CTools::stableExp(prediction(m_InClasses[j]) - logZ)
                                 : pAgg)};
            writer(k++, -weight * pij);
        }
    }
    LOG_TRACE(<< "Wrote " << k << " curvatures");
}
}
}
}
}
