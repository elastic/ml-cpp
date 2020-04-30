/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "core/CPersistUtils.h"
#include <maths/CBoostedTreeLoss.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLbfgs.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPRNG.h>
#include <maths/CSampling.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>

#include <limits>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;

namespace {
const double EPSILON{100.0 * std::numeric_limits<double>::epsilon()};
const double LOG_EPSILON{CTools::stableLog(EPSILON)};

// MSLE constants
const std::size_t MSLE_PREDICTION_INDEX{0};
const std::size_t MSLE_ACTUAL_INDEX{1};
const std::size_t MSLE_ERROR_INDEX{2};
const std::size_t MSLE_BUCKET_SIZE{32};
const std::size_t MSLE_OPTIMIZATION_ITERATIONS{15};

// Pseudo-Huber constants
const std::size_t HUBER_ERROR_INDEX{0};
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
    return CTools::stableLog(1.0 - CTools::logisticFunction(logOdds));
}

double logLogistic(double logOdds) {
    // For small x logistic(x) = e^(x) + O(e^(2x))
    if (logOdds < LOG_EPSILON) {
        return logOdds;
    }
    return CTools::stableLog(CTools::logisticFunction(logOdds));
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

    // We searching for the value x which minimises
    //
    //    x^* = argmin_x{ sum_i{(a_i - (p_i + x))^2} + lambda * x^2 }
    //
    // This is convex so there is one minimum where the derivative w.r.t. x is zero
    // and x^* = 1 / (n + lambda) sum_i{ a_i - p_i }. Denoting the mean prediction
    // error m = 1/n sum_i{ a_i - p_i } we have x^* = n / (n + lambda) m.

    double count{CBasicStatistics::count(m_MeanError)};
    double meanError{CBasicStatistics::mean(m_MeanError)};

    TDoubleVector result(1);
    result(0) = count == 0.0 ? 0.0 : count / (count + this->lambda()) * meanError;
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
    m_CurrentPass += this->bucketWidth() > 0.0 ? 1 : 2;
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
        auto& count = m_BucketsClassCounts[this->bucket(prediction(0))];
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

    std::function<double(double)> objective;
    double minWeight;
    double maxWeight;

    // This is true if and only if all the predictions were identical. In this
    // case we only need one pass over the data and can compute the optimal
    // value from the counts of the two categories.
    if (this->bucketWidth() == 0.0) {
        // This is the (unique) predicted value for the rows in leaf by the forest
        // so far (i.e. without the weight for the leaf we're about to add).
        double prediction{m_PredictionMinMax.initialized()
                              ? (m_PredictionMinMax.min() + m_PredictionMinMax.max()) / 2.0
                              : 0.0};
        objective = [prediction, this](double weight) {
            double logOdds{prediction + weight};
            double c0{m_ClassCounts(0)};
            double c1{m_ClassCounts(1)};
            return this->lambda() * CTools::pow2(weight) -
                   c0 * logOneMinusLogistic(logOdds) - c1 * logLogistic(logOdds);
        };

        // Weight shrinkage means the optimal weight will be somewhere between
        // the logit of the empirical probability and zero.
        double c0{m_ClassCounts(0) + 1.0};
        double c1{m_ClassCounts(1) + 1.0};
        double empiricalProbabilityC1{c1 / (c0 + c1)};
        double empiricalLogOddsC1{CTools::stableLog(
            empiricalProbabilityC1 / (1.0 - empiricalProbabilityC1))};
        minWeight = (empiricalProbabilityC1 < 0.5 ? empiricalLogOddsC1 : 0.0) - prediction;
        maxWeight = (empiricalProbabilityC1 < 0.5 ? 0.0 : empiricalLogOddsC1) - prediction;

    } else {
        objective = [this](double weight) {
            double loss{0.0};
            for (std::size_t i = 0; i < m_BucketsClassCounts.size(); ++i) {
                double logOdds{this->bucketCentre(i) + weight};
                double c0{m_BucketsClassCounts[i](0)};
                double c1{m_BucketsClassCounts[i](1)};
                loss -= c0 * logOneMinusLogistic(logOdds) + c1 * logLogistic(logOdds);
            }
            return loss + this->lambda() * CTools::pow2(weight);
        };

        // Choose a weight interval in which all probabilites vary from close to
        // zero to close to one. In particular, the idea is to minimize the leaf
        // weight on an interval [a, b] where if we add "a" the log-odds for all
        // rows <= -5, i.e. max prediction + a = -5, and if we add "b" the log-odds
        // for all rows >= 5, i.e. min prediction + a = 5.
        minWeight = -m_PredictionMinMax.max() - 5.0;
        maxWeight = -m_PredictionMinMax.min() + 5.0;
    }

    TDoubleVector result(1);

    if (minWeight == maxWeight) {
        result(0) = minWeight;
        return result;
    }

    double minimum;
    double objectiveAtMinimum;
    std::size_t maxIterations{10};
    CSolvers::minimize(minWeight, maxWeight, objective(minWeight), objective(maxWeight),
                       objective, 1e-3, maxIterations, minimum, objectiveAtMinimum);
    LOG_TRACE(<< "minimum = " << minimum << " objective(minimum) = " << objectiveAtMinimum);

    result(0) = minimum;
    return result;
}

CArgMinMultinomialLogisticLossImpl::CArgMinMultinomialLogisticLossImpl(std::size_t numberClasses,
                                                                       double lambda,
                                                                       const CPRNG::CXorOShiro128Plus& rng)
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

    using TMinAccumulator = CBasicStatistics::SMin<std::pair<double, std::size_t>>::TAccumulator;

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
    CLbfgs<TDoubleVector> lgbfs{5};
    std::tie(wmin, loss) = lgbfs.minimize(objective, objectiveGradient, std::move(wmin));
    LOG_TRACE(<< "loss* = " << loss << " weight* = " << wmin.transpose());

    return wmin;
}

CArgMinMultinomialLogisticLossImpl::TObjective
CArgMinMultinomialLogisticLossImpl::objective() const {
    TDoubleVector logProbabilities{m_NumberClasses};
    double lambda{std::max(this->lambda(), 1e-6)};
    if (m_Centres.size() == 1) {
        return [logProbabilities, lambda, this](const TDoubleVector& weight) mutable {
            logProbabilities = m_Centres[0] + weight;
            CTools::inplaceLogSoftmax(logProbabilities);
            return lambda * weight.squaredNorm() - m_ClassCounts.transpose() * logProbabilities;
        };
    }
    return [logProbabilities, lambda, this](const TDoubleVector& weight) mutable {
        double loss{0.0};
        for (std::size_t i = 0; i < m_CentresClassCounts.size(); ++i) {
            if (m_CentresClassCounts[i].sum() > 0.0) {
                logProbabilities = m_Centres[i] + weight;
                CTools::inplaceLogSoftmax(logProbabilities);
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
        return [probabilities, lossGradient, lambda, this](const TDoubleVector& weight) mutable {
            probabilities = m_Centres[0] + weight;
            CTools::inplaceSoftmax(probabilities);
            lossGradient = m_ClassCounts.array().sum() * probabilities - m_ClassCounts;
            return TDoubleVector{2.0 * lambda * weight + lossGradient};
        };
    }
    return [probabilities, lossGradient, lambda, this](const TDoubleVector& weight) mutable {
        lossGradient.array() = 0.0;
        for (std::size_t i = 0; i < m_CentresClassCounts.size(); ++i) {
            double n{m_CentresClassCounts[i].array().sum()};
            if (n > 0.0) {
                probabilities = m_Centres[i] + weight;
                CTools::inplaceSoftmax(probabilities);
                lossGradient -= m_CentresClassCounts[i] - n * probabilities;
            }
        }
        return TDoubleVector{2.0 * lambda * weight + lossGradient};
    };
}

CArgMinMsleImpl::CArgMinMsleImpl(double lambda)
    : CArgMinLossImpl{lambda}, m_Buckets(MSLE_BUCKET_SIZE) {
    for (auto& bucket : m_Buckets) {
        bucket.resize(MSLE_BUCKET_SIZE);
    }
}

std::unique_ptr<CArgMinLossImpl> CArgMinMsleImpl::clone() const {
    return std::make_unique<CArgMinMsleImpl>(*this);
}

bool CArgMinMsleImpl::nextPass() {
    ++m_CurrentPass;
    return this->bucketWidth().first > 0.0 && m_CurrentPass < 2;
}

void CArgMinMsleImpl::add(const TMemoryMappedFloatVector& prediction, double actual, double weight) {
    double expPrediction{std::exp(prediction[0])};
    double logActual{CTools::fastLog(1.0 + actual)};
    switch (m_CurrentPass) {
    case 0: {
        m_ExpPredictionMinMax.add(expPrediction);
        m_LogActualMinMax.add(logActual);
        m_MeanLogActual.add(logActual, weight);
        break;
    }
    case 1: {
        double logError{logActual - CTools::fastLog(1.0 + expPrediction)};

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
        maxLogWeight = CBasicStatistics::mean(m_MeanLogActual);
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
                    minLogWeight, CBasicStatistics::mean(bucketActual)(MSLE_ERROR_INDEX));
                maxLogWeight = std::max(
                    maxLogWeight, CBasicStatistics::mean(bucketActual)(MSLE_ERROR_INDEX));
            }
        }
    }

    double minimizer;
    double objectiveAtMinimum;
    std::size_t maxIterations{MSLE_OPTIMIZATION_ITERATIONS};
    CSolvers::minimize(minLogWeight, maxLogWeight, objective(minLogWeight),
                       objective(maxLogWeight), objective, 1e-5, maxIterations,
                       minimizer, objectiveAtMinimum);
    LOG_TRACE(<< "minimum = " << minimizer << " objective(minimum) = " << objectiveAtMinimum);

    TDoubleVector result(1);
    result(0) = minimizer;
    return result;
}

CArgMinMsleImpl::TObjective CArgMinMsleImpl::objective() const {
    return [this](double logWeight) {

        double weight{std::exp(logWeight)};
        if (this->bucketWidth().first == 0.0) {
            // prediction is constant
            double expPrediction{m_ExpPredictionMinMax.max()};
            double logPrediction{CTools::fastLog(1 + expPrediction * weight)};
            double meanLogActual{CBasicStatistics::mean(m_MeanLogActual)};
            double meanLogActualSquared{CBasicStatistics::variance(m_MeanLogActual) +
                                        CTools::pow2(meanLogActual)};
            double loss{meanLogActualSquared - 2 * meanLogActual * logPrediction +
                        CTools::pow2(logPrediction)};
            return loss + this->lambda() * CTools::pow2(weight);
        } else {
            double loss{0.0};
            double totalCount{0.0};
            for (const auto& bucketPrediction : m_Buckets) {
                for (const auto& bucketActual : bucketPrediction) {
                    double count{CBasicStatistics::count(bucketActual)};
                    if (count > 0.0) {
                        double expPrediction{CBasicStatistics::mean(
                            bucketActual)(MSLE_PREDICTION_INDEX)};
                        double logActual{CBasicStatistics::mean(bucketActual)(MSLE_ACTUAL_INDEX)};
                        double logPrediction{CTools::fastLog(1 + expPrediction * weight)};
                        loss += count * (CTools::pow2(logActual - logPrediction));
                        totalCount += count;
                    }
                }
            }
            return loss / totalCount + this->lambda() * CTools::pow2(weight);
        }
    };
}

CArgMinPseudoHuberImpl::CArgMinPseudoHuberImpl(double lambda, double delta)
    : CArgMinLossImpl{lambda}, m_Delta2{CTools::pow2(delta)},
      m_Buckets(HUBER_BUCKET_SIZE) {
}

std::unique_ptr<CArgMinLossImpl> CArgMinPseudoHuberImpl::clone() const {
    return std::make_unique<CArgMinPseudoHuberImpl>(*this);
}

bool CArgMinPseudoHuberImpl::nextPass() {
    ++m_CurrentPass;
    return this->bucketWidth() > 0.0 && m_CurrentPass < 2;
}

void CArgMinPseudoHuberImpl::add(const TMemoryMappedFloatVector& predictionVector,
                                 double actual,
                                 double weight) {
    double prediction{predictionVector[0]};
    switch (m_CurrentPass) {
    case 0: {
        m_ErrorMinMax.add(actual - prediction);
        break;
    }
    case 1: {
        double error{actual - prediction};

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
    TObjective objective;
    double minWeight = m_ErrorMinMax.min();
    double maxWeight = m_ErrorMinMax.max();

    objective = this->objective();
    double minimizer;
    double objectiveAtMinimum;
    std::size_t maxIterations{HUBER_OPTIMIZATION_ITERATIONS};
    CSolvers::minimize(minWeight, maxWeight, objective(minWeight), objective(maxWeight),
                       objective, 1e-5, maxIterations, minimizer, objectiveAtMinimum);
    LOG_TRACE(<< "minimum = " << minimizer << " objective(minimum) = " << objectiveAtMinimum);

    TDoubleVector result(1);
    result(0) = minimizer;
    return result;
}

CArgMinPseudoHuberImpl::TObjective CArgMinPseudoHuberImpl::objective() const {
    return [this](double weight) {
        if (m_Delta2 > 0) {

            double loss{0.0};
            double totalCount{0.0};
            for (const auto& bucket : m_Buckets) {
                double count{CBasicStatistics::count(bucket)};
                if (count > 0.0) {
                    double error{CBasicStatistics::mean(bucket)};
                    loss += count * m_Delta2 *
                            (std::sqrt(1.0 + CTools::pow2(error - weight) / m_Delta2) - 1.0);
                    totalCount += count;
                }
            }
            return loss / totalCount + this->lambda() * CTools::pow2(weight);
        } else {
            return 0.0;
        }
    };
}
}

namespace boosted_tree {

CLoss::TLossUPtr CLoss::restoreLoss(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& lossFunctionName = traverser.name();
        if (lossFunctionName == CMse::NAME) {
            return std::make_unique<CMse>(traverser);
        } else if (lossFunctionName == CMsle::NAME) {
            return std::make_unique<CMsle>(traverser);
        } else if (lossFunctionName == CPseudoHuber::NAME) {
            return std::make_unique<CPseudoHuber>(traverser);
        } else if (lossFunctionName == CBinomialLogisticLoss::NAME) {
            return std::make_unique<CBinomialLogisticLoss>(traverser);
        } else if (lossFunctionName == CMultinomialLogisticLoss::NAME) {
            return std::make_unique<CMultinomialLogisticLoss>(traverser);
        }

        LOG_ERROR(<< "Error restoring loss function. Unknown loss function type '"
                  << lossFunctionName << "'.");
        return nullptr;
    } while (traverser.next());
}

void CLoss::persistLoss(core::CStatePersistInserter& inserter) const {
    auto persist = [this](core::CStatePersistInserter& inserter_) {
        this->acceptPersistInserter(inserter_);
    };
    inserter.insertLevel(this->name(), persist);
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

void CArgMinLoss::add(const TMemoryMappedFloatVector& prediction, double actual, double weight) {
    return m_Impl->add(prediction, actual, weight);
}

void CArgMinLoss::merge(CArgMinLoss& other) {
    return m_Impl->merge(*other.m_Impl);
}

CArgMinLoss::TDoubleVector CArgMinLoss::value() const {
    return m_Impl->value();
}

CArgMinLoss::CArgMinLoss(const CArgMinLossImpl& impl) : m_Impl{impl.clone()} {
}

CArgMinLoss CLoss::makeMinimizer(const boosted_tree_detail::CArgMinLossImpl& impl) const {
    return {impl};
}

CMse::CMse(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel(std::bind(&CMse::acceptRestoreTraverser, this,
                                             std::placeholders::_1)) == false) {
        throw std::runtime_error{"failed to restore CMse"};
    }
}

std::unique_ptr<CLoss> CMse::clone() const {
    return std::make_unique<CMse>(*this);
}

CMse::EType CMse::type() const {
    return E_Regression;
}

std::size_t CMse::numberParameters() const {
    return 1;
}

double CMse::value(const TMemoryMappedFloatVector& prediction, double actual, double weight) const {
    return weight * CTools::pow2(prediction(0) - actual);
}

void CMse::gradient(const TMemoryMappedFloatVector& prediction,
                    double actual,
                    TWriter writer,
                    double weight) const {
    writer(0, 2.0 * weight * (prediction(0) - actual));
}

void CMse::curvature(const TMemoryMappedFloatVector& /*prediction*/,
                     double /*actual*/,
                     TWriter writer,
                     double weight) const {
    writer(0, 2.0 * weight);
}

bool CMse::isCurvatureConstant() const {
    return true;
}

CMse::TDoubleVector CMse::transform(const TMemoryMappedFloatVector& prediction) const {
    return TDoubleVector{prediction};
}

CArgMinLoss CMse::minimizer(double lambda, const CPRNG::CXorOShiro128Plus&) const {
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

CMsle::CMsle(double offset) : m_Offset{offset} {
}

CMsle::CMsle(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel(std::bind(&CMsle::acceptRestoreTraverser, this,
                                             std::placeholders::_1)) == false) {
        throw std::runtime_error{"failed to restore CMsle"};
    }
}

CLoss::EType CMsle::type() const {
    return E_Regression;
}

std::unique_ptr<CLoss> CMsle::clone() const {
    return std::make_unique<CMsle>(*this);
}

std::size_t CMsle::numberParameters() const {
    return 1;
}

double CMsle::value(const TMemoryMappedFloatVector& logPrediction, double actual, double weight) const {
    double prediction{std::exp(logPrediction(0))};
    double log1PlusPrediction{CTools::fastLog(1.0 + prediction)};
    if (actual < 0.0) {
        HANDLE_FATAL(<< "Input error: target value needs to be non-negative to use with MSLE loss, received: "
                     << actual);
    }
    double log1PlusActual{CTools::fastLog(1.0 + actual)};
    return weight * CTools::pow2(log1PlusPrediction - log1PlusActual);
}

void CMsle::gradient(const TMemoryMappedFloatVector& logPrediction,
                     double actual,
                     TWriter writer,
                     double weight) const {
    double prediction{std::exp(logPrediction(0))};
    double log1PlusPrediction{CTools::fastLog(1.0 + prediction)};
    double log1PlusActual{CTools::fastLog(1.0 + actual)};
    writer(0, 2 * weight * (log1PlusPrediction - log1PlusActual) / (prediction + 1));
}

void CMsle::curvature(const TMemoryMappedFloatVector& logPrediction,
                      double actual,
                      TWriter writer,
                      double weight) const {
    double prediction{std::exp(logPrediction(0))};
    double log1PlusPrediction{CTools::fastLog(1.0 + prediction)};
    double log1PlusActual{CTools::fastLog(1.0 + actual)};
    // Apply L'Hopital's rule in the limit prediction -> actual.
    writer(0, prediction == actual ? 0.0
                                   : 2.0 * weight * (log1PlusPrediction - log1PlusActual) /
                                         ((prediction + 1) * (prediction - actual)));
}

bool CMsle::isCurvatureConstant() const {
    return false;
}

CMsle::TDoubleVector CMsle::transform(const TMemoryMappedFloatVector& prediction) const {
    TDoubleVector result{1};
    result(0) = std::exp(prediction(0));
    return result;
}

CArgMinLoss CMsle::minimizer(double lambda, const CPRNG::CXorOShiro128Plus& /* rng */) const {
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
    if (traverser.traverseSubLevel(std::bind(&CPseudoHuber::acceptRestoreTraverser,
                                             this, std::placeholders::_1)) == false) {
        throw std::runtime_error{"failed to restore CPseudoHuber"};
    }
}

CLoss::EType CPseudoHuber::type() const {
    return E_Regression;
}

std::unique_ptr<CLoss> CPseudoHuber::clone() const {
    return std::make_unique<CPseudoHuber>(*this);
}

std::size_t CPseudoHuber::numberParameters() const {
    return 1;
}

double CPseudoHuber::value(const TMemoryMappedFloatVector& predictionVec,
                           double actual,
                           double weight) const {
    double delta2{CTools::pow2(m_Delta)};
    double prediction{predictionVec[0]};
    return weight * delta2 *
           (std::sqrt(1.0 + CTools::pow2(actual - prediction) / delta2) - 1.0);
}

void CPseudoHuber::gradient(const TMemoryMappedFloatVector& predictionVec,
                            double actual,
                            TWriter writer,
                            double weight) const {
    //\frac{- a_i + p_i}{\sqrt{1 + \frac{(a_i - p_i)^2}{\delta^2}}}
    double prediction{predictionVec(0)};
    writer(0, weight * (prediction - actual) /
                  (std::sqrt(1.0 + CTools::pow2((actual - prediction) / m_Delta))));
}

void CPseudoHuber::curvature(const TMemoryMappedFloatVector& predictionVec,
                             double actual,
                             TWriter writer,
                             double weight) const {
    // \frac{1}{\sqrt{1 + \frac{\left(a_{i} - p_{i}\right)^{2}}{\delta^{2}}}}
    double prediction{predictionVec(0)};
    // double delta2{CTools::pow2(m_Delta)};
    // double error2{CTools::pow2(actual - prediction)};
    // double tmp{1.0 + error2 / delta2};
    // double result{error2 / (delta2 * std::sqrt(tmp) * tmp)};
    double result{1.0/(std::sqrt(1.0+CTools::pow2((actual - prediction)/m_Delta)))};
    writer(0, weight * result);
}

bool CPseudoHuber::isCurvatureConstant() const {
    return false;
}

CMsle::TDoubleVector CPseudoHuber::transform(const TMemoryMappedFloatVector& prediction) const {
    TDoubleVector result{1};
    result(0) = prediction(0);
    return result;
}

CArgMinLoss CPseudoHuber::minimizer(double lambda,
                                    const CPRNG::CXorOShiro128Plus& /* rng */) const {
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
    if (traverser.traverseSubLevel(std::bind(&CBinomialLogisticLoss::acceptRestoreTraverser,
                                             this, std::placeholders::_1)) == false) {
        throw std::runtime_error{"failed to restore CBinomialLogisticLoss"};
    }
}

std::unique_ptr<CLoss> CBinomialLogisticLoss::clone() const {
    return std::make_unique<CBinomialLogisticLoss>(*this);
}

CBinomialLogisticLoss::EType CBinomialLogisticLoss::type() const {
    return E_BinaryClassification;
}

std::size_t CBinomialLogisticLoss::numberParameters() const {
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
                                     TWriter writer,
                                     double weight) const {
    if (prediction(0) > -LOG_EPSILON && actual == 1.0) {
        writer(0, -weight * CTools::stableExp(-prediction(0)));
    } else {
        writer(0, weight * (CTools::logisticFunction(prediction(0)) - actual));
    }
}

void CBinomialLogisticLoss::curvature(const TMemoryMappedFloatVector& prediction,
                                      double /*actual*/,
                                      TWriter writer,
                                      double weight) const {
    if (prediction(0) > -LOG_EPSILON) {
        writer(0, weight * CTools::stableExp(-prediction(0)));
    } else {
        double probability{CTools::logisticFunction(prediction(0))};
        writer(0, weight * probability * (1.0 - probability));
    }
}

bool CBinomialLogisticLoss::isCurvatureConstant() const {
    return false;
}

CBinomialLogisticLoss::TDoubleVector
CBinomialLogisticLoss::transform(const TMemoryMappedFloatVector& prediction) const {
    double p1{CTools::logisticFunction(prediction(0))};
    TDoubleVector result{2};
    result(0) = 1.0 - p1;
    result(1) = p1;
    return result;
}

CArgMinLoss CBinomialLogisticLoss::minimizer(double lambda,
                                             const CPRNG::CXorOShiro128Plus&) const {
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

CMultinomialLogisticLoss::CMultinomialLogisticLoss(std::size_t numberClasses)
    : m_NumberClasses{numberClasses} {
}

CMultinomialLogisticLoss::CMultinomialLogisticLoss(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel(std::bind(&CMultinomialLogisticLoss::acceptRestoreTraverser,
                                             this, std::placeholders::_1)) == false) {
        throw std::runtime_error{"failed to restore CMultinomialLogisticLoss"};
    }
}

std::unique_ptr<CLoss> CMultinomialLogisticLoss::clone() const {
    return std::make_unique<CMultinomialLogisticLoss>(m_NumberClasses);
}

CMultinomialLogisticLoss::EType CMultinomialLogisticLoss::type() const {
    return E_MulticlassClassification;
}

std::size_t CMultinomialLogisticLoss::numberParameters() const {
    return m_NumberClasses;
}

double CMultinomialLogisticLoss::value(const TMemoryMappedFloatVector& predictions,
                                       double actual,
                                       double weight) const {
    double zmax{predictions.maxCoeff()};
    double logZ{0.0};
    for (int i = 0; i < predictions.size(); ++i) {
        logZ += std::exp(predictions(i) - zmax);
    }
    logZ = zmax + CTools::stableLog(logZ);

    // i.e. -log(z(actual))
    return weight * (logZ - predictions(static_cast<std::size_t>(actual)));
}

void CMultinomialLogisticLoss::gradient(const TMemoryMappedFloatVector& predictions,
                                        double actual,
                                        TWriter writer,
                                        double weight) const {

    // We prefer an implementation which avoids any memory allocations.

    double zmax{predictions.maxCoeff()};
    double eps{0.0};
    double logZ{0.0};
    for (int i = 0; i < predictions.size(); ++i) {
        if (predictions(i) - zmax < LOG_EPSILON) {
            // Sum the contributions from classes whose predicted probability
            // is less than epsilon, for which we'd lose all nearly precision
            // when adding to the normalisation coefficient.
            eps += std::exp(predictions(i) - zmax);
        } else {
            logZ += std::exp(predictions(i) - zmax);
        }
    }
    eps = CTools::stable(eps);
    logZ = zmax + CTools::stableLog(logZ);

    for (int i = 0; i < predictions.size(); ++i) {
        if (i == static_cast<int>(actual)) {
            double probability{CTools::stableExp(predictions(i) - logZ)};
            if (probability == 1.0) {
                // We have that p = 1 / (1 + eps) and the gradient is p - 1.
                // Use a Taylor expansion and drop terms of O(eps^2) to get:
                writer(i, -weight * eps);
            } else {
                writer(i, weight * (probability - 1.0));
            }
        } else {
            writer(i, weight * CTools::stableExp(predictions(i) - logZ));
        }
    }
}

void CMultinomialLogisticLoss::curvature(const TMemoryMappedFloatVector& predictions,
                                         double /*actual*/,
                                         TWriter writer,
                                         double weight) const {

    // Return the lower triangle of the Hessian column major.

    // We prefer an implementation which avoids any memory allocations.

    double zmax{predictions.maxCoeff()};
    double eps{0.0};
    double logZ{0.0};
    for (int i = 0; i < predictions.size(); ++i) {
        if (predictions(i) - zmax < LOG_EPSILON) {
            // Sum the contributions from classes whose predicted probability
            // is less than epsilon, for which we'd lose all nearly precision
            // when adding to the normalisation coefficient.
            eps += std::exp(predictions(i) - zmax);
        } else {
            logZ += std::exp(predictions(i) - zmax);
        }
    }
    eps = CTools::stable(eps);
    logZ = zmax + CTools::stableLog(logZ);

    for (std::size_t i = 0, k = 0; i < m_NumberClasses; ++i) {
        double probability{CTools::stableExp(predictions(i) - logZ)};
        if (probability == 1.0) {
            // We have that p = 1 / (1 + eps) and the curvature is p (1 - p).
            // Use a Taylor expansion and drop terms of O(eps^2) to get:
            writer(k++, weight * eps);
        } else {
            writer(k++, weight * probability * (1.0 - probability));
        }
        for (std::size_t j = i + 1; j < m_NumberClasses; ++j) {
            double probabilities[]{CTools::stableExp(predictions(i) - logZ),
                                   CTools::stableExp(predictions(j) - logZ)};
            writer(k++, -weight * probabilities[0] * probabilities[1]);
        }
    }
}

bool CMultinomialLogisticLoss::isCurvatureConstant() const {
    return false;
}

CMultinomialLogisticLoss::TDoubleVector
CMultinomialLogisticLoss::transform(const TMemoryMappedFloatVector& prediction) const {
    TDoubleVector result{prediction};
    CTools::inplaceSoftmax(result);
    return result;
}

CArgMinLoss CMultinomialLogisticLoss::minimizer(double lambda,
                                                const CPRNG::CXorOShiro128Plus& rng) const {
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
}
}
}
