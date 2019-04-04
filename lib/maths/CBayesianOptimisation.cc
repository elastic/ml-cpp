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
    : m_DomainScales{SConstant<TVector>::get(parameterBounds.size(), 1.0)},
      m_A(parameterBounds.size()), m_B(parameterBounds.size()),
      m_KernelParameters{SConstant<TVector>::get(parameterBounds.size() + 1, 1.0)} {

    for (std::size_t i = 0; i < parameterBounds.size(); ++i) {
        m_A(i) = parameterBounds[i].first;
        m_B(i) = parameterBounds[i].second;
    }
}

void CBayesianOptimisation::add(TVector x, double fx, double vx) {
    m_Function.emplace_back(m_DomainScales.asDiagonal() * x,
                            m_RangeScale * (fx - m_RangeShift));
    m_ErrorVariances.push_back(CTools::pow2(m_RangeScale) * vx);
}

CBayesianOptimisation::TVector CBayesianOptimisation::maximumExpectedImprovement() {

    // Reapply conditioning and recompute the maximum likelihood kernel parameters.
    this->maximumLikelihoodKernel();

    TVector xmax;
    double fmax{0.0};

    TEIFunc minusEI;
    TEIGradientFunc minusEIGradient;
    std::tie(minusEI, minusEIGradient) = this->minusExpectedImprovementAndGradient();

    // Use random restarts inside the constraint bounding box.
    TVector interpolate(m_A.size());
    TDoubleVec interpolates;
    CSampling::uniformSample(m_Rng, 0.0, 1.0,
                             (m_Restarts - 1) * interpolate.size(), interpolates);

    CLbfgs<TVector> lbfgs{10};

    // We set rho to give the constraint and objective approximately equal priority
    // in the following constrained optimisation problem.
    double rho{std::sqrt(this->functionVariance())};

    for (std::size_t i = 0; i < m_Restarts; ++i) {

        for (int j = 0; j < interpolate.size(); ++i) {
            interpolate(j) = interpolates[i * m_Restarts + j];
        }
        TVector x0(m_A + interpolate.asDiagonal() * (m_B - m_A));

        TVector xcand;
        double fcand;
        std::tie(xcand, fcand) =
            lbfgs.constrainedMinimize(minusEI, minusEIGradient, m_A, m_B, x0, rho);

        if (-fcand > fmax) {
            xmax = std::move(xcand);
            fmax = -fcand;
        }
    }

    return xmax;
}

std::pair<CBayesianOptimisation::TLikelihoodFunc, CBayesianOptimisation::TLikelihoodGradientFunc>
CBayesianOptimisation::minusLikelihoodAndGradient() const {

    TVector f{this->function()};
    double v{this->meanErrorVariance()};

    auto likelihood = [f, v, this](const TVector& a) {
        Eigen::LDLT<Eigen::MatrixXd> Kldl{this->kernel(a, v)};
        TVector Kinvf{Kldl.solve(f)};
        return 0.5 * (f.transpose() * Kinvf + Kldl.vectorD().array().log().sum());
    };

    auto likelihoodGradient = [f, v, this](const TVector& a) {

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
                1.0 / a(0) *
                double{(Kinvf(i) * Kinvf.transpose() - KInv.row(i)) * K.col(i)};
        }
        for (int i = 1; i < a.size(); ++i) {
            TMatrix dKdai{this->dKerneld(a, i)};
            for (int j = 0; j < Kinvf.size(); ++j) {
                gradient(i) += 0.5 * double{(Kinvf(j) * Kinvf.transpose() - KInv.row(j)) *
                                            dKdai.col(j)};
            }
        }

        return gradient;
    };

    return {std::move(likelihood), std::move(likelihoodGradient)};
}

std::pair<CBayesianOptimisation::TEIFunc, CBayesianOptimisation::TEIGradientFunc>
CBayesianOptimisation::minusExpectedImprovementAndGradient() const {

    TMatrix K{this->kernel(m_KernelParameters, this->meanErrorVariance())};
    Eigen::LDLT<Eigen::MatrixXd> Kldl{K};

    TVector Kinvf{Kldl.solve(this->function())};

    double vx{this->meanErrorVariance()};

    double fmin{std::min_element(m_Function.begin(), m_Function.end(),
                                 [](const TVectorDoublePr& lhs, const TVectorDoublePr& rhs) {
                                     return lhs.second < rhs.second;
                                 })
                    ->second};

    auto EI = [Kldl, Kinvf, vx, fmin, this](const TVector& x) {

        double Kxx;
        TVector Kxn;
        std::tie(Kxn, Kxx) = this->kernelCovariates(m_KernelParameters, x, vx);

        double sigma{Kxx - Kxn.transpose() * Kldl.solve(Kxn)};

        if (sigma <= 0.0) {
            return 0.0;
        }

        double mu{Kxn.transpose() * Kinvf};
        sigma = std::sqrt(sigma);

        boost::math::normal normal{0.0, 1.0};
        double z{(fmin - mu) / sigma};
        return -sigma * (z * CTools::safeCdf(normal, z) + CTools::safePdf(normal, z));
    };

    auto EIGradient = [Kldl, Kinvf, vx, fmin, this](const TVector& x) {

        double Kxx;
        TVector Kxn;
        std::tie(Kxn, Kxx) = this->kernelCovariates(m_KernelParameters, x, vx);

        TVector KinvKxn{Kldl.solve(Kxn)};
        double sigma{Kxx - Kxn.transpose() * KinvKxn};

        if (sigma <= 0.0) {
            return las::zero(x);
        }

        double mu{Kxn.transpose() * Kinvf};
        sigma = std::sqrt(sigma);

        boost::math::normal normal{0.0, 1.0};
        double z{(fmin - mu) / sigma};
        double cdfz{CTools::safeCdf(normal, z)};
        double pdfz{CTools::safePdf(normal, z)};

        TVector muGradient{x.size()};
        TVector sigmaGradient{x.size()};
        for (int i = 0; i < x.size(); ++i) {
            TVector dKxndx{Kxn.size()};
            for (int j = 0; j < Kxn.size(); ++j) {
                const TVector& xj{m_Function[j].first};
                dKxndx(j) = 2.0 * m_KernelParameters(j + 1) * (xj(i) - x(i)) * Kxn(j);
            }
            muGradient(i) = Kinvf.transpose() * dKxndx;
            sigmaGradient(i) = -2.0 * KinvKxn.transpose() * dKxndx;
        }
        sigmaGradient /= 2.0 * sigma;

        TVector zGradient{((mu - fmin) / CTools::pow2(sigma)) * sigmaGradient -
                          muGradient / sigma};

        return TVector{-(z * cdfz + pdfz) * sigmaGradient - sigma * cdfz * zGradient};
    };

    return {std::move(EI), std::move(EIGradient)};
}

const CBayesianOptimisation::TVector& CBayesianOptimisation::maximumLikelihoodKernel() {

    // Use random restarts of L-BFGS to find maximum likelihood parameters.

    this->precondition();

    std::size_t n(m_KernelParameters.size());

    TDoubleVec scales;
    scales.reserve((m_Restarts - 1) * n);
    CSampling::uniformSample(m_Rng, std::log(0.1), std::log(4.0),
                             (m_Restarts - 1) * n, scales);

    TLikelihoodFunc l;
    TLikelihoodGradientFunc g;
    std::tie(l, g) = this->minusLikelihoodAndGradient();

    CLbfgs<TVector> lbfgs{10};

    double lmax;
    TVector amax;
    std::tie(amax, lmax) = lbfgs.minimize(l, g, m_KernelParameters, 1e-8, 75);

    TVector scale{n};
    for (std::size_t i = 1; i < m_Restarts; ++i) {

        TVector a{m_KernelParameters};
        for (std::size_t j = 0; j < n; ++j) {
            scale(j) = scales[(i - 1) * n + j];
        }
        a = scale.array().exp() * a.array();

        double la;
        std::tie(a, la) = lbfgs.minimize(l, g, std::move(a), 1e-8, 75);

        if (la < lmax) {
            lmax = la;
            amax = std::move(a);
        }
    }

    m_KernelParameters = std::move(amax);

    return m_KernelParameters;
}

void CBayesianOptimisation::precondition() {

    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TVectorMeanVarAccumulator =
        CBasicStatistics::SSampleMeanVar<CVector<double>>::TAccumulator;

    TVectorMeanVarAccumulator domainMoments(CVector<double>(m_DomainScales.size(), 0.0));
    TMeanVarAccumulator rangeMoments;

    for (const auto& value : m_Function) {
        domainMoments.add(fromDenseVector(value.first));
        rangeMoments.add(value.second);
    }

    TVector eps{las::ones(m_DomainScales) * std::numeric_limits<double>::epsilon()};

    m_DomainScales =
        las::ones(m_DomainScales)
            .cwiseQuotient((toDynamicDenseVector(CBasicStatistics::variance(domainMoments)) + eps)
                               .cwiseSqrt());
    m_RangeShift = CBasicStatistics::mean(rangeMoments);
    m_RangeScale = 1.0 / std::sqrt(CBasicStatistics::variance(rangeMoments) + eps(0));

    for (auto& value : m_Function) {
        value.first = m_DomainScales.asDiagonal() * value.first;
        value.second = m_RangeScale * (value.second - m_RangeShift);
    }
    for (auto& variance : m_ErrorVariances) {
        variance *= CTools::pow2(m_RangeScale);
    }
}

CBayesianOptimisation::TVector CBayesianOptimisation::function() const {
    TVector result(m_Function.size());
    for (std::size_t i = 0; i < m_Function.size(); ++i) {
        result(i) = m_Function[i].second;
    }
    return result;
}

double CBayesianOptimisation::functionVariance() const {

    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    TDoubleVec function(m_Function.size());
    for (std::size_t i = 0; i < m_Function.size(); ++i) {
        function[i] = m_Function[i].second;
    }
    std::sort(function.begin(), function.end());

    TMeanVarAccumulator moments;
    for (std::size_t i = 0, n = std::min(function.size(), std::size_t{10}); i < n; ++i) {
        moments.add(function[i]);
    }

    return CBasicStatistics::variance(moments);
}

double CBayesianOptimisation::meanErrorVariance() const {
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    TMeanAccumulator variance;
    variance.add(m_ErrorVariances);
    return CBasicStatistics::mean(variance);
}

CBayesianOptimisation::TMatrix CBayesianOptimisation::dKerneld(const TVector& a, int k) const {
    TMatrix result{m_Function.size(), m_Function.size()};
    for (std::size_t i = 0; i < m_Function.size(); ++i) {
        result(i, i) = 0.0;
        const TVector& xi{m_Function[i].first};
        for (std::size_t j = 0; j < i; ++j) {
            const TVector& xj{m_Function[j].first};
            result(i, j) = result(j, i) = 2.0 * a(k) *
                                          CTools::pow2(xi(k - 1) - xj(k - 1)) *
                                          this->kernel(a, xi, xj);
        }
    }
    return result;
}

CBayesianOptimisation::TMatrix CBayesianOptimisation::kernel(const TVector& a, double v) const {
    TMatrix result{m_Function.size(), m_Function.size()};
    for (std::size_t i = 0; i < m_Function.size(); ++i) {
        result(i, i) = CTools::pow2(a(0)) + v;
        const TVector& xi{m_Function[i].first};
        for (std::size_t j = 0; j < i; ++j) {
            const TVector& xj{m_Function[j].first};
            result(i, j) = result(j, i) = this->kernel(a, xi, xj);
        }
    }
    return result;
}

CBayesianOptimisation::TVectorDoublePr
CBayesianOptimisation::kernelCovariates(const TVector& a, const TVector& x, double vx) const {
    double Kxx{CTools::pow2(a(0)) + vx};
    TVector Kxn(m_Function.size());
    for (std::size_t i = 0; i < m_Function.size(); ++i) {
        Kxn(i) = this->kernel(a, x, m_Function[i].first);
    }
    return {Kxn, Kxx};
}

double CBayesianOptimisation::kernel(const TVector& a, const TVector& x, const TVector& y) {
    return CTools::pow2(a(0)) *
           std::exp(-(x - y).transpose() *
                    a.tail(a.size() - 1).cwiseAbs2().matrix().asDiagonal() * (x - y));
}
}
}
