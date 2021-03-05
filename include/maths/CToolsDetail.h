/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CToolsDetail_h
#define INCLUDED_ml_maths_CToolsDetail_h

#include <core/Constants.h>

#include <maths/CCompositeFunctions.h>
#include <maths/CIntegration.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CMixtureDistribution.h>
#include <maths/COrderings.h>
#include <maths/CTools.h>

#include <boost/math/constants/constants.hpp>

#include <algorithm>
#include <cmath>
#include <exception>

namespace ml {
namespace maths {

template<typename LOGF>
CTools::CMixtureProbabilityOfLessLikelySample::CSmoothedKernel<LOGF>::CSmoothedKernel(LOGF logf,
                                                                                      double logF0,
                                                                                      double k)
    : m_LogF(logf), m_LogF0(logF0), m_K(k),
      m_Scale(std::exp(m_LogF0) * (1.0 + std::exp(-k))) {
}

template<typename LOGF>
void CTools::CMixtureProbabilityOfLessLikelySample::CSmoothedKernel<LOGF>::k(double k) {
    double f0 = m_Scale / (1.0 + std::exp(-m_K));
    m_K = k;
    m_Scale = f0 * (1.0 + std::exp(-k));
}

template<typename LOGF>
bool CTools::CMixtureProbabilityOfLessLikelySample::CSmoothedKernel<LOGF>::
operator()(double x, double& result) const {
    // We use the fact that if:
    //   1 + exp(-k(f(x)/f0 - 1)) < (1 + eps) * exp(-k(f(x)/f0 - 1))
    //
    // then the kernel = scale to working precision. Canceling
    // O(1) terms in the exponential, taking logs and using the
    // fact that the log is monotonic increasing function this
    // reduces to
    //   0 < -k(f(x)/f0 - 1) + log(eps)
    //
    // which implies that we can simplify if
    //   f(x)/f0 < 1 + log(eps)/k

    result = 0.0;
    double logFx;
    if (!m_LogF(x, logFx)) {
        LOG_ERROR(<< "Failed to calculate likelihood at " << x);
        return false;
    }
    logFx -= m_LogF0;
    if (m_K * (logFx - 1.0) >= core::constants::LOG_MAX_DOUBLE) {
        return true;
    }
    double fx = std::exp(logFx);
    if (fx < 1.0 + core::constants::LOG_DOUBLE_EPSILON / m_K) {
        result = m_Scale * fx;
        return true;
    }
    result = m_Scale / (1.0 + std::exp(m_K * (fx - 1.0))) * fx;
    return true;
}

template<typename LOGF, typename EQUAL>
bool CTools::CMixtureProbabilityOfLessLikelySample::leftTail(const LOGF& logf,
                                                             std::size_t iterations,
                                                             const EQUAL& equal,
                                                             double& result) const {
    if (m_X <= m_A) {
        result = m_X;
        return true;
    }

    CCompositeFunctions::CMinusConstant<const LOGF&, double> f(logf, m_LogFx);

    try {
        double xr = m_A;
        double fr = f(xr);
        if (fr < 0.0) {
            result = m_A;
            return true;
        }
        double xl = xr;
        double fl = fr;
        if (m_MaxDeviation.count() > 0) {
            xl = xr - m_MaxDeviation[0];
            fl = f(xl);
        }

        iterations = std::max(iterations, std::size_t(4));
        std::size_t n = iterations - 2;
        if (!CSolvers::leftBracket(xl, xr, fl, fr, f, n)) {
            result = xl;
            return false;
        }
        n = iterations - n;
        CSolvers::solve(xl, xr, fl, fr, f, n, equal, result);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to find left root: " << e.what() << ", a = " << m_A
                  << ", logf(x) = " << m_LogFx << ", logf(a) = " << logf(m_A) << ", max deviation = "
                  << (m_MaxDeviation.count() > 0 ? m_MaxDeviation[0] : 0.0));
        return false;
    }
    return true;
}

template<typename LOGF, typename EQUAL>
bool CTools::CMixtureProbabilityOfLessLikelySample::rightTail(const LOGF& logf,
                                                              std::size_t iterations,
                                                              const EQUAL& equal,
                                                              double& result) const {
    if (m_X >= m_B) {
        result = m_X;
        return true;
    }

    CCompositeFunctions::CMinusConstant<const LOGF&, double> f(logf, m_LogFx);

    try {
        double xl = m_B;
        double fl = f(xl);
        if (fl < 0.0) {
            result = m_B;
            return true;
        }
        double xr = xl;
        double fr = fl;
        if (m_MaxDeviation.count() > 0) {
            xr = xl + m_MaxDeviation[0];
            fr = f(xr);
        }

        iterations = std::max(iterations, std::size_t(4));
        std::size_t n = iterations - 2;
        if (!CSolvers::rightBracket(xl, xr, fl, fr, f, n)) {
            result = xr;
            return false;
        }
        n = iterations - n;
        CSolvers::solve(xl, xr, fl, fr, f, n, equal, result);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to find right root: " << e.what() << ",b = " << m_B
                  << ", logf(x) = " << m_LogFx << ", logf(b) = " << logf(m_B));
        return false;
    }
    return true;
}

template<typename LOGF>
double CTools::CMixtureProbabilityOfLessLikelySample::calculate(const LOGF& logf,
                                                                double pTails) {
    TDoubleDoublePrVec intervals;
    this->intervals(intervals);

    double p = 0.0;
    TDoubleVec pIntervals(intervals.size(), 0.0);
    CSmoothedKernel<const LOGF&> kernel(logf, m_LogFx, 3.0);
    for (std::size_t i = 0; i < intervals.size(); ++i) {
        if (!CIntegration::gaussLegendre<CIntegration::OrderFour>(
                kernel, intervals[i].first, intervals[i].second, pIntervals[i])) {
            LOG_ERROR(<< "Couldn't integrate kernel over "
                      << core::CContainerPrinter::print(intervals[i]));
        }
    }

    p += pTails;
    kernel.k(15.0);
    CIntegration::adaptiveGaussLegendre<CIntegration::OrderTwo>(kernel, intervals, pIntervals,
                                                                2, // refinements
                                                                3,    // splits
                                                                1e-2, // tolerance
                                                                p);
    return truncate(p - pTails, 0.0, 1.0);
}

template<typename T>
double CTools::differentialEntropy(const CMixtureDistribution<T>& mixture) {
    using TModeVec = typename CMixtureDistribution<T>::TModeVec;

    static const double EPS = 1e-5;
    static const std::size_t INTERVALS = 8;

    const TDoubleVec& weights = mixture.weights();
    const TModeVec& modes = mixture.modes();

    if (weights.empty()) {
        return 0.0;
    }

    TDoubleDoublePrVec range;
    for (std::size_t i = 0; i < modes.size(); ++i) {
        range.push_back(TDoubleDoublePr(quantile(modes[i], EPS),
                                        quantile(modes[i], 1.0 - EPS)));
    }
    std::sort(range.begin(), range.end(), COrderings::SFirstLess());
    LOG_TRACE(<< "range = " << core::CContainerPrinter::print(range));
    std::size_t left = 0;
    for (std::size_t i = 1; i < range.size(); ++i) {
        if (range[left].second < range[i].first) {
            ++left;
            std::swap(range[left], range[i]);
        } else {
            range[left].second = std::max(range[left].second, range[i].second);
        }
    }
    range.erase(range.begin() + left + 1, range.end());
    LOG_TRACE(<< "range = " << core::CContainerPrinter::print(range));

    double result = 0.0;

    CDifferentialEntropyKernel<T> kernel(mixture);
    for (std::size_t i = 0; i < range.size(); ++i) {
        double a = range[i].first;
        double d = (range[i].second - range[i].first) / static_cast<double>(INTERVALS);

        for (std::size_t j = 0; j < INTERVALS; ++j, a += d) {
            double integral;
            if (CIntegration::gaussLegendre<CIntegration::OrderFive>(kernel, a, a + d, integral)) {
                result += integral;
            }
        }
    }

    LOG_TRACE(<< "result = " << result);
    return result;
}

template<typename T>
void CTools::inplaceSoftmax(CDenseVector<T>& z) {
    double zmax{z.maxCoeff()};
    z.array() -= zmax;
    z.array() = z.array().exp();
    z /= z.sum();
}

template<typename SCALAR>
void CTools::inplaceLogSoftmax(CDenseVector<SCALAR>& z) {
    // Handle under/overflow when taking exponentials by subtracting zmax.
    double zmax{z.maxCoeff()};
    z.array() -= zmax;
    double Z{z.array().exp().sum()};
    z.array() -= stableLog(Z);
}
}
}

#endif // INCLUDED_ml_maths_CToolsDetail_h
