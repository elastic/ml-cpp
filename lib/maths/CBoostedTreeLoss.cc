/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

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
    double Z{z.array().exp().matrix().template lpNorm<1>()};
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
                                                               double lambda,
                                                               const CPRNG::CXorOShiro128Plus& rng)
    : CArgMinLossImpl{lambda}, m_NumberClasses{numberClasses}, m_Rng{rng},
      m_ClassCounts{TDoubleVector::Zero(numberClasses)}, m_PredictionSketch{NUMBER_CENTRES} {
}

std::unique_ptr<CArgMinLossImpl> CArgMinMultinomialLogisticImpl::clone() const {
    return std::make_unique<CArgMinMultinomialLogisticImpl>(*this);
}

bool CArgMinMultinomialLogisticImpl::nextPass() {

    using TMeanAccumulator = CBasicStatistics::SSampleMean<TDoubleVector>::TAccumulator;

    if (m_CurrentPass == 0) {
        TKMeans::TSphericalClusterVecVec clusters;
        if (m_PredictionSketch.kmeans(NUMBER_CENTRES, clusters) == false) {
            m_Centres.push_back(TDoubleVector::Zero(m_NumberClasses));
            m_CurrentPass += 2;
        } else {
            // Extract the k-centres.
            m_Centres.reserve(clusters.size());
            for (const auto& cluster : clusters) {
                TMeanAccumulator centre{TDoubleVector::Zero(m_NumberClasses)};
                for (const auto& point : cluster) {
                    centre.add(point);
                }
                m_Centres.push_back(CBasicStatistics::mean(centre));
            }
            std::sort(m_Centres.begin(), m_Centres.end());
            m_Centres.erase(std::unique(m_Centres.begin(), m_Centres.end()),
                            m_Centres.end());
            LOG_TRACE(<< "# centres = " << m_Centres.size());
            m_CurrentPass += m_Centres.size() == 1 ? 2 : 1;
            m_CentresClassCounts.resize(m_Centres.size(),
                                        TDoubleVector::Zero(m_NumberClasses));
        }

        // Reclaim the memory used by k-means.
        m_PredictionSketch = TKMeans{0};
    } else {
        ++m_CurrentPass;
    }

    LOG_TRACE(<< "current pass = " << m_CurrentPass);

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

    using TMinAccumulator = CBasicStatistics::SMin<double>::TAccumulator;

    TDoubleVector weightBoundingBox[2];
    weightBoundingBox[0] = std::numeric_limits<double>::max() *
                           TDoubleVector::Ones(m_NumberClasses);
    weightBoundingBox[1] = -weightBoundingBox[0];
    if (m_Centres.size() == 1) {

        // Weight shrinkage means the optimal weight will be somewhere between
        // the logit of the empirical probability and zero.
        TDoubleVector empiricalProbabilities{m_ClassCounts.array() + 0.1};
        empiricalProbabilities = empiricalProbabilities /
                                 empiricalProbabilities.lpNorm<1>();
        TDoubleVector empiricalLogOdds{
            empiricalProbabilities.array().log().matrix() - m_Centres[0]};
        weightBoundingBox[0] = weightBoundingBox[0].array().min(0.0);
        weightBoundingBox[1] = weightBoundingBox[1].array().max(0.0);
        weightBoundingBox[0] = weightBoundingBox[0].array().min(empiricalLogOdds.array());
        weightBoundingBox[1] = weightBoundingBox[1].array().max(empiricalLogOdds.array());

    } else {

        for (const auto& centre : m_Centres) {
            weightBoundingBox[0] = weightBoundingBox[0].array().min(centre.array());
            weightBoundingBox[1] = weightBoundingBox[0].array().max(centre.array());
        }
    }
    LOG_TRACE(<< "bounding box blc = " << weightBoundingBox[0].transpose());
    LOG_TRACE(<< "bounding box trc = " << weightBoundingBox[1].transpose());

    // Optimize via LBFGS with multiple restarts.

    TMinAccumulator minLoss;
    TDoubleVector result;

    TDoubleVector x0(m_NumberClasses);
    TObjective objective{this->objective()};
    TObjectiveGradient objectiveGradient{this->objectiveGradient()};
    for (std::size_t i = 0; i < NUMBER_RESTARTS; ++i) {
        for (int j = 0; j < x0.size(); ++j) {
            double alpha{CSampling::uniformSample(m_Rng, 0.0, 1.0)};
            x0(j) = weightBoundingBox[0](j) +
                    alpha * (weightBoundingBox[1](j) - weightBoundingBox[0](j));
        }
        LOG_TRACE(<< "x0 = " << x0.transpose());

        double loss;
        CLbfgs<TDoubleVector> lgbfs{5};
        std::tie(x0, loss) = lgbfs.minimize(objective, objectiveGradient, std::move(x0));
        if (minLoss.add(loss)) {
            result = x0;
        }
        LOG_TRACE(<< "loss = " << loss << " weight for loss = " << x0.transpose());
    }
    LOG_TRACE(<< "minimum loss = " << minLoss << " weight* = " << result.transpose());

    return result;
}

CArgMinMultinomialLogisticImpl::TObjective CArgMinMultinomialLogisticImpl::objective() const {
    TDoubleVector logProbabilities;
    double lambda{this->lambda()};
    if (m_Centres.size() == 1) {
        return [logProbabilities, lambda, this](const TDoubleVector& weight) mutable {
            logProbabilities = m_Centres[0] + weight;
            logProbabilities = logSoftmax(std::move(logProbabilities));
            return lambda * weight.squaredNorm() - m_ClassCounts.transpose() * logProbabilities;
        };
    }
    return [logProbabilities, lambda, this](const TDoubleVector& weight) mutable {
        double loss{0.0};
        for (std::size_t i = 0; i < m_CentresClassCounts.size(); ++i) {
            logProbabilities = m_Centres[i] + weight;
            logProbabilities = logSoftmax(std::move(logProbabilities));
            loss -= m_CentresClassCounts[i].transpose() * logProbabilities;
        }
        return loss + lambda * weight.squaredNorm();
    };
}

CArgMinMultinomialLogisticImpl::TObjectiveGradient
CArgMinMultinomialLogisticImpl::objectiveGradient() const {
    TDoubleVector probabilities;
    double lambda{this->lambda()};
    if (m_Centres.size() == 1) {
        return [probabilities, lambda, this](const TDoubleVector& weight) mutable {
            probabilities = m_Centres[0] + weight;
            probabilities = CTools::softmax(std::move(probabilities));
            return TDoubleVector{2.0 * lambda * weight -
                                 (m_ClassCounts - m_ClassCounts.array().sum() * probabilities)};
        };
    }
    return [probabilities, lambda, this](const TDoubleVector& weight) mutable -> TDoubleVector {
        TDoubleVector lossGradient{TDoubleVector::Zero(m_NumberClasses)};
        for (std::size_t i = 0; i < m_CentresClassCounts.size(); ++i) {
            probabilities = m_Centres[i] + weight;
            probabilities = CTools::softmax(std::move(probabilities));
            lossGradient -= m_CentresClassCounts[i] -
                            m_CentresClassCounts[i].array().sum() * probabilities;
        }
        return TDoubleVector{2.0 * lambda * weight + lossGradient};
    };
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

CArgMinLoss CMse::minimizer(double lambda, const CPRNG::CXorOShiro128Plus&) const {
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

CArgMinLoss CBinomialLogistic::minimizer(double lambda, const CPRNG::CXorOShiro128Plus&) const {
    return this->makeMinimizer(CArgMinBinomialLogisticImpl{lambda});
}

const std::string& CBinomialLogistic::name() const {
    return NAME;
}

const std::string CBinomialLogistic::NAME{"binomial_logistic"};

CMultinomialLogistic::CMultinomialLogistic(std::size_t numberClasses)
    : m_NumberClasses{numberClasses} {
}

std::unique_ptr<CLoss> CMultinomialLogistic::clone() const {
    return std::make_unique<CMultinomialLogistic>(m_NumberClasses);
}

std::size_t CMultinomialLogistic::numberParameters() const {
    return m_NumberClasses;
}

double CMultinomialLogistic::value(const TMemoryMappedFloatVector& predictions,
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

void CMultinomialLogistic::gradient(const TMemoryMappedFloatVector& predictions,
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

void CMultinomialLogistic::curvature(const TMemoryMappedFloatVector& predictions,
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

bool CMultinomialLogistic::isCurvatureConstant() const {
    return false;
}

CMultinomialLogistic::TDoubleVector
CMultinomialLogistic::transform(const TMemoryMappedFloatVector& prediction) const {
    TDoubleVector result{prediction};
    return CTools::softmax(std::move(result));
}

CArgMinLoss CMultinomialLogistic::minimizer(double lambda,
                                            const CPRNG::CXorOShiro128Plus& rng) const {
    return this->makeMinimizer(CArgMinMultinomialLogisticImpl{m_NumberClasses, lambda, rng});
}

const std::string& CMultinomialLogistic::name() const {
    return NAME;
}

const std::string CMultinomialLogistic::NAME{"multinomial_logistic"};
}
}
}
