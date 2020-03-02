/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "maths/CLinearAlgebraEigen.h"
#include <maths/CBoostedTreeLoss.h>

#include <maths/CBasicStatistics.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>

namespace ml {
namespace maths {
using namespace boosted_tree_detail;

namespace {
double LOG_EPSILON{std::log(100.0 * std::numeric_limits<double>::epsilon())};

double logOneMinusLogistic(double logOdds) {
    // For large x logistic(x) = 1 - e^(-x) + O(e^(-2x))
    if (logOdds > -LOG_EPSILON) {
        return -logOdds;
    }
    return CTools::fastLog(1.0 - CTools::logisticFunction(logOdds));
}

double logLogistic(double logOdds) {
    // For small x logistic(x) = e^(x) + O(e^(2x))
    if (logOdds < LOG_EPSILON) {
        return logOdds;
    }
    return CTools::fastLog(CTools::logisticFunction(logOdds));
}

template<typename T>
CDenseVector<T> logSoftmax(CDenseVector<T> z) {
    // Version which handles overflow and underflow when taking exponentials.
    double zmax{z.maxCoeff()};
    z = z - zmax * CDenseVector<T>::Ones(z.size());
    double Z{z.array().exp().template lpNorm<1>()};
    z = z - std::log(Z) * CDenseVector<T>::Ones(z.size());
    return std::move(z);
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

CArgMinBinomialLogisticImpl::CArgMinBinomialLogisticImpl(double lambda)
    : CArgMinLossImpl{lambda}, m_ClassCounts{0},
      m_BucketsClassCounts(NUMBER_BUCKETS, TDoubleVector2x1{0.0}) {
}

std::unique_ptr<CArgMinLossImpl> CArgMinBinomialLogisticImpl::clone() const {
    return std::make_unique<CArgMinBinomialLogisticImpl>(*this);
}

bool CArgMinBinomialLogisticImpl::nextPass() {
    m_CurrentPass += this->bucketWidth() > 0.0 ? 1 : 2;
    return m_CurrentPass < 2;
}

void CArgMinBinomialLogisticImpl::add(const TMemoryMappedFloatVector& prediction,
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

void CArgMinBinomialLogisticImpl::merge(const CArgMinLossImpl& other) {
    const auto* logistic = dynamic_cast<const CArgMinBinomialLogisticImpl*>(&other);
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

CArgMinBinomialLogisticImpl::TDoubleVector CArgMinBinomialLogisticImpl::value() const {

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
        double empiricalLogOddsC1{
            std::log(empiricalProbabilityC1 / (1.0 - empiricalProbabilityC1))};
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

CArgMinMultinomialLogisticImpl::CArgMinMultinomialLogisticImpl(std::size_t numberClasses,
                                                               double lambda)
    : CArgMinLossImpl{lambda}, m_NumberClasses{numberClasses},
      m_ClassCounts{TDoubleVector::Zero(numberClasses)}, m_PredictionSketch{NUMBER_CENTRES},
      m_CentresClassCounts(NUMBER_CENTRES, TDoubleVector::Zero(numberClasses)) {
}

std::unique_ptr<CArgMinLossImpl> CArgMinMultinomialLogisticImpl::clone() const {
    return std::make_unique<CArgMinMultinomialLogisticImpl>(*this);
}

bool CArgMinMultinomialLogisticImpl::nextPass() {

    // Extract the k-centres.
    TKMeans::TSphericalClusterVecVec centres;
    if (m_PredictionSketch.kmeans(NUMBER_CENTRES, centres) == false) {
        m_Centres.push_back(TDoubleVector::Zero(m_NumberClasses));
        m_CurrentPass += 2;
    } else {
        m_Centres.reserve(centres.size());
        for (auto& centre : centres) {
            m_Centres.push_back(std::move(static_cast<TDoubleVector&>(centre[0])));
        }
        m_CurrentPass += m_Centres.size() == 1 ? 2 : 1;
    }

    // Clear memory used by k-means.
    m_PredictionSketch = TKMeans{0};

    return m_CurrentPass < 2;
}

void CArgMinMultinomialLogisticImpl::add(const TMemoryMappedFloatVector& prediction,
                                         double actual,
                                         double weight) {

    using TMinAccumulator = CBasicStatistics::SMin<std::pair<double, std::size_t>>::TAccumulator;

    switch (m_CurrentPass) {
    case 0: {
        // We have a member variable to avoid allocating a tempory each time.
        m_DoublePrediction = prediction;
        m_PredictionSketch.add(m_DoublePrediction, weight);
        m_ClassCounts(static_cast<std::size_t>(actual)) += weight;
        break;
    }
    case 1: {
        TMinAccumulator nearest;
        for (std::size_t i = 0; i < m_Centres.size(); ++i) {
            nearest.add({(m_Centres[i] - prediction).lpNorm<2>(), i});
        }
        auto& count = m_CentresClassCounts[nearest[0].second];
        count(static_cast<std::size_t>(actual)) += weight;
        break;
    }
    default:
        break;
    }
}

void CArgMinMultinomialLogisticImpl::merge(const CArgMinLossImpl& other) {
    const auto* logistic = dynamic_cast<const CArgMinMultinomialLogisticImpl*>(&other);
    if (logistic != nullptr) {
        switch (m_CurrentPass) {
        case 0:
            m_PredictionSketch.merge(logistic->m_PredictionSketch);
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

CArgMinMultinomialLogisticImpl::TDoubleVector CArgMinMultinomialLogisticImpl::value() const {

    std::function<double(const TDoubleVector&)> objective;

    TDoubleVector probabilities;

    // This is true if and only if all the predictions were identical. In this
    // case we only need one pass over the data and can compute the optimal
    // value from the counts of the categories.
    if (m_Centres.empty()) {

        objective = [&](const TDoubleVector& weight) {
            probabilities = m_Centres[0] + weight;
            probabilities = logSoftmax(std::move(probabilities));
            return this->lambda() * weight.lpNorm<2>() - m_ClassCounts.transpose() * probabilities;
        };

    } else {

        objective = [&](const TDoubleVector& weight) {
            double loss{0.0};
            for (std::size_t i = 0; i < m_CentresClassCounts.size(); ++i) {
                probabilities = m_Centres[i] + weight;
                probabilities = logSoftmax(std::move(probabilities));
                loss -= m_CentresClassCounts[i].transpose() * probabilities;
            }
            return loss + this->lambda() * weight.lpNorm<2>();
        };
    }

    // LBFGS with multiple restarts.
    // TODO
    return TDoubleVector{};
}
}

namespace boosted_tree {

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

std::unique_ptr<CLoss> CMse::clone() const {
    return std::make_unique<CMse>(*this);
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

CArgMinLoss CMse::minimizer(double lambda) const {
    return this->makeMinimizer(CArgMinMseImpl{lambda});
}

const std::string& CMse::name() const {
    return NAME;
}

const std::string CMse::NAME{"mse"};

std::unique_ptr<CLoss> CBinomialLogistic::clone() const {
    return std::make_unique<CBinomialLogistic>(*this);
}

std::size_t CBinomialLogistic::numberParameters() const {
    return 1;
}

double CBinomialLogistic::value(const TMemoryMappedFloatVector& prediction,
                                double actual,
                                double weight) const {
    return -weight * ((1.0 - actual) * logOneMinusLogistic(prediction(0)) +
                      actual * logLogistic(prediction(0)));
}

void CBinomialLogistic::gradient(const TMemoryMappedFloatVector& prediction,
                                 double actual,
                                 TWriter writer,
                                 double weight) const {
    if (prediction(0) > -LOG_EPSILON && actual == 1.0) {
        writer(0, -weight * std::exp(-prediction(0)));
    }
    writer(0, weight * (CTools::logisticFunction(prediction(0)) - actual));
}

void CBinomialLogistic::curvature(const TMemoryMappedFloatVector& prediction,
                                  double /*actual*/,
                                  TWriter writer,
                                  double weight) const {
    if (prediction(0) > -LOG_EPSILON) {
        writer(0, weight * std::exp(-prediction(0)));
    }
    double probability{CTools::logisticFunction(prediction(0))};
    writer(0, weight * probability * (1.0 - probability));
}

bool CBinomialLogistic::isCurvatureConstant() const {
    return false;
}

CBinomialLogistic::TDoubleVector
CBinomialLogistic::transform(const TMemoryMappedFloatVector& prediction) const {
    TDoubleVector result{prediction};
    result(0) = CTools::logisticFunction(result(0));
    return result;
}

CArgMinLoss CBinomialLogistic::minimizer(double lambda) const {
    return this->makeMinimizer(CArgMinBinomialLogisticImpl{lambda});
}

const std::string& CBinomialLogistic::name() const {
    return NAME;
}

const std::string CBinomialLogistic::NAME{"binomial_logistic"};

CMultinomialLogisticRegression::CMultinomialLogisticRegression(std::size_t numberClasses)
    : m_NumberClasses{numberClasses} {
}

std::unique_ptr<CLoss> CMultinomialLogisticRegression::clone() const {
    return std::make_unique<CMultinomialLogisticRegression>(m_NumberClasses);
}

std::size_t CMultinomialLogisticRegression::numberParameters() const {
    return m_NumberClasses;
}

double CMultinomialLogisticRegression::value(const TMemoryMappedFloatVector& predictions,
                                             double actual,
                                             double weight) const {
    double zmax{predictions.maxCoeff()};
    double logZ{0.0};
    for (int i = 0; i < predictions.size(); ++i) {
        logZ += std::exp(predictions(i) - zmax);
    }
    logZ = zmax + CTools::fastLog(logZ);

    // i.e. -log(z(actual))
    return weight * (logZ - predictions(static_cast<std::size_t>(actual)));
}

void CMultinomialLogisticRegression::gradient(const TMemoryMappedFloatVector& predictions,
                                              double actual,
                                              TWriter writer,
                                              double weight) const {

    // We prefer an implementation which avoids any memory allocations.

    double zmax{predictions.maxCoeff()};
    double logZ{0.0};
    for (int i = 0; i < predictions.size(); ++i) {
        logZ += std::exp(predictions(i) - zmax);
    }
    logZ = CTools::fastLog(logZ);

    for (int i = 0; i < predictions.size(); ++i) {
        if (i == static_cast<int>(actual)) {
            if (predictions(i) - zmax - logZ > -LOG_EPSILON) {
                writer(i, -weight * std::exp(-(predictions(i) - zmax - logZ)));
            }
            writer(i, weight * (std::exp(predictions(i) - zmax - logZ) - 1.0));
        }
        writer(i, weight * std::exp(predictions(i) - zmax - logZ));
    }
}

void CMultinomialLogisticRegression::curvature(const TMemoryMappedFloatVector& predictions,
                                               double /*actual*/,
                                               TWriter writer,
                                               double weight) const {

    // Return the lower triangle of the Hessian column major.

    // We prefer an implementation which avoids any memory allocations.

    double zmax{predictions.maxCoeff()};
    double logZ{0.0};
    for (int i = 0; i < predictions.size(); ++i) {
        logZ += std::exp(predictions(i) - zmax);
    }
    logZ = CTools::fastLog(logZ);

    for (std::size_t i = 0, k = 0; i < m_NumberClasses; ++i) {
        if (predictions(i) - zmax - logZ > -LOG_EPSILON) {
            writer(i, weight * std::exp(-(predictions(i) - zmax - logZ)));
        } else {
            double probability{std::exp(predictions(i) - zmax - logZ)};
            writer(i, weight * weight * probability * (1.0 - probability));
        }
        for (std::size_t j = i + 1; j < m_NumberClasses; ++j, ++k) {
            double probabilities[]{std::exp(predictions(i) - zmax - logZ),
                                   std::exp(predictions(j) - zmax - logZ)};
            writer(k, -weight * probabilities[0] * probabilities[1]);
        }
    }
}

bool CMultinomialLogisticRegression::isCurvatureConstant() const {
    return false;
}

CMultinomialLogisticRegression::TDoubleVector
CMultinomialLogisticRegression::transform(const TMemoryMappedFloatVector& prediction) const {
    TDoubleVector result{prediction};
    return CTools::softmax(std::move(result));
}

CArgMinLoss CMultinomialLogisticRegression::minimizer(double lambda) const {
    return this->makeMinimizer(CArgMinMultinomialLogisticImpl{m_NumberClasses, lambda});
}

const std::string& CMultinomialLogisticRegression::name() const {
    return NAME;
}

const std::string CMultinomialLogisticRegression::NAME{"multinomial_logistic"};
}
}
}
