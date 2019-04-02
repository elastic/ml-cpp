/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBayesianOptimisation.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLbfgs.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>

#include <boost/math/distributions/normal.hpp>

namespace ml {
namespace maths {

CBayesianOptimisation::CBayesianOptimisation(TDoubleDoublePrVec parameterBounds)
    : m_ParameterBounds{std::move(parameterBounds)},
      m_DomainScales{SConstant<TVector>::get(m_ParameterBounds.size(), 1.0)},
      m_KernelParameters{SConstant<TVector>::get(m_ParameterBounds.size() + 1, 1.0)} {
}

void CBayesianOptimisation::add(TVector x, double fx, double vx) {
    m_Function.emplace_back(m_DomainScales.asDiagonal() * x,
                            m_RangeScale * (fx - m_RangeShift));
    m_ErrorVariances.push_back(m_RangeScale * vx);
}

CBayesianOptimisation::TVector CBayesianOptimisation::maximumExpectedImprovement() {
    this->precondition();
    this->maximumLikelihoodKernel();

    // We use an ADMM scheme.
    //
    // The upper and lower parameter box constraints are converted into one
    // inequality, i.e. x' = (x^t - a^t, b^t - x^t) >= 0. We use a standard
    // trick to augment the objective to capture the constraint. In particular,
    // we have 2 * dimension(x) slack variables y and solve
    //
    //  argmin_x -E[I(x)] + I(y)
    //      s.t. A * x - y = 0
    //
    // with I(y) = infinity y < 0 and 0 otherwise.

    auto EI = this->minusExpectedImprovement();

    return {};
}

CBayesianOptimisation::TLikelihoodFunc CBayesianOptimisation::minusLikelihood() const {

    TVector f_{this->function()};
    double v{this->meanErrorVariance()};

    return [ f = std::move(f_), v, this ](const TVector& a) {
        Eigen::LDLT<Eigen::MatrixXd> Kldl{this->kernel(a, v)};
        TVector Kinvf{Kldl.solve(f)};
        return 0.5 * f.transpose() * Kinvf + 0.5 * Kldl.vectorD().array().log().sum();
    };
}

CBayesianOptimisation::TLikelihoodGradientFunc
CBayesianOptimisation::minusLikelihoodGradient() const {

    TVector f_{this->function()};
    double v{this->meanErrorVariance()};

    return [ f = std::move(f_), v, this ](const TVector& a) {

        TMatrix K{this->kernel(a, v)};
        Eigen::LDLT<Eigen::MatrixXd> Kldl{K};

        TVector Kinvf{Kldl.solve(f)};

        TVector ones(f.size());
        ones.setOnes();
        TMatrix KInv{Kldl.solve(TMatrix{ones.asDiagonal()})};

        K.diagonal() -= v * ones;

        TVector gradient{a.size()};
        gradient.setZero();

        for (int i = 0; i < Kinvf.size(); ++i) {
            gradient(0) -=
                0.5 / a(0) *
                double{(Kinvf(i) * Kinvf.transpose() - KInv.row(i)) * K.col(i)};
        }
        for (int i = 0; i < a.size(); ++i) {
            TMatrix dist{this->distanceMatrix(i)};
            for (int j = 0; j < Kinvf.size(); ++j) {
                gradient(i + 1) +=
                    0.5 * double{(Kinvf(j) * Kinvf.transpose() - KInv.row(j)) *
                                 dist.col(j).cwiseProduct(K.col(j))};
            }
        }

        return gradient;
    };
}

CBayesianOptimisation::TExpectedImprovementFunc
CBayesianOptimisation::minusExpectedImprovement() const {

    TMatrix K{this->kernel(m_KernelParameters, this->meanErrorVariance())};
    Eigen::LDLT<Eigen::MatrixXd> Kldl_{K};

    TVector Kinvf_{Kldl_.solve(this->function())};

    double vx{this->meanErrorVariance()};

    double fmin{std::min_element(m_Function.begin(), m_Function.end(),
                                 [](const TVectorDoublePr& lhs, const TVectorDoublePr& rhs) {
                                     return lhs.second < rhs.second;
                                 })
                    ->second};

    return
        [ Kldl = std::move(Kldl_), Kinvf = std::move(Kinvf_), vx, fmin,
          this ](const TVector& x) {

        double Kxx;
        TVector Kxn;
        std::tie(Kxn, Kxx) = this->kernelCovariates(m_KernelParameters, x, vx);

        double sigma{std::max(Kxx - Kxn.transpose() * Kldl.solve(Kxn), 0.0)};

        if (sigma == 0.0) {
            return 0.0;
        }

        double mu{Kxn.transpose() * Kinvf};
        sigma = std::sqrt(sigma);

        boost::math::normal normal{mu, sigma};
        double z{(fmin - mu) / sigma};
        return -sigma * (z * CTools::safeCdf(normal, z) + CTools::safePdf(normal, z));
    };
}

void CBayesianOptimisation::maximumLikelihoodKernel() {

    // Use random restarts of L-BFGS to find maximum likelihood parameters.

    std::size_t n(m_KernelParameters.size());

    TDoubleVec scales;
    scales.reserve((m_Restarts - 1) * n);
    CSampling::uniformSample(m_Rng, std::log(0.2), std::log(5.0),
                             (m_Restarts - 1) * n, scales);

    auto l = this->minusLikelihood();
    auto g = this->minusLikelihoodGradient();

    CLbfgs<TVector> lbfgs{10};

    double lmax;
    TVector amax;
    std::tie(amax, lmax) = lbfgs.minimize(l, g, m_KernelParameters);

    TVector scale{n};
    for (std::size_t i = 0; i < m_Restarts; ++i) {

        TVector a{m_KernelParameters};
        for (std::size_t j = 0; j < n; ++j) {
            scale(j) = m_KernelParameters[i * n + j];
        }
        a = scale.cwiseProduct(a);

        double la;
        std::tie(a, la) = lbfgs.minimize(l, g, std::move(a));

        if (la < lmax) {
            lmax = la;
            amax = std::move(a);
        }
    }

    m_KernelParameters = std::move(amax);
}

void CBayesianOptimisation::precondition() {

    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TVectorMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<TVector>::TAccumulator;

    TVectorMeanVarAccumulator domainMoments(las::zero(m_DomainScales));
    TMeanVarAccumulator rangeMoments;

    for (const auto& value : m_Function) {
        domainMoments.add(value.first);
        rangeMoments.add(value.second);
    }

    m_DomainScales = las::ones(m_DomainScales).array() /
                     CBasicStatistics::variance(domainMoments).array().sqrt();
    m_RangeShift = CBasicStatistics::variance(rangeMoments);
    m_RangeScale = 1.0 / std::sqrt(CBasicStatistics::variance(rangeMoments));

    for (auto& value : m_Function) {
        value.first = m_DomainScales.asDiagonal() * value.first;
        value.second = m_RangeScale * (value.second - m_RangeShift);
    }
}

CBayesianOptimisation::TVector CBayesianOptimisation::function() const {
    TVector result(m_Function.size());
    for (std::size_t i = 0; i < m_Function.size(); ++i) {
        result(i) = m_Function[i].second;
    }
    return result;
}

double CBayesianOptimisation::meanErrorVariance() const {
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    TMeanAccumulator variance;
    variance.add(m_ErrorVariances);
    return CBasicStatistics::mean(variance);
}

CBayesianOptimisation::TMatrix CBayesianOptimisation::distanceMatrix(int coord) const {
    TMatrix result{m_Function.size(), m_Function.size()};
    for (std::size_t i = 0; i < m_Function.size(); ++i) {
        result(i, i) = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            result(i, j) = result(j, i) = CTools::pow2(
                (m_Function[i].first(coord) - m_Function[j].first(coord)));
        }
    }
    return result;
}

CBayesianOptimisation::TMatrix CBayesianOptimisation::kernel(const TVector& a, double v) const {
    TMatrix result{m_Function.size(), m_Function.size()};
    for (std::size_t i = 0; i < m_Function.size(); ++i) {
        result(i, i) = a(0) + v;
        for (std::size_t j = 0; j < i; ++j) {
            result(i, j) = result(j, i) =
                this->kernel(a, m_Function[i].first, m_Function[j].first);
        }
    }
    return result;
}

CBayesianOptimisation::TVectorDoublePr
CBayesianOptimisation::kernelCovariates(const TVector& a, const TVector& x, double vx) const {
    double Kxx{a(0) + vx};
    TVector Kxn(m_Function.size());
    for (std::size_t i = 0; i < m_Function.size(); ++i) {
        Kxn(i) = this->kernel(a, x, m_Function[i].first);
    }
    return {Kxn, Kxx};
}

double CBayesianOptimisation::kernel(const TVector& a, const TVector& x, const TVector& y) {
    return a(0) * std::exp(-(x - y).transpose() *
                           a.tail(a.size() - 1).asDiagonal() * (x - y));
}
}
}
