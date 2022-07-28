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

#ifndef INCLUDED_ml_maths_common_CToolsDetail_h
#define INCLUDED_ml_maths_common_CToolsDetail_h

#include <core/Constants.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CCompositeFunctions.h>
#include <maths/common/CIntegration.h>
#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CMixtureDistribution.h>
#include <maths/common/COrderings.h>
#include <maths/common/CTools.h>

#include <boost/math/constants/constants.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <numeric>

namespace ml {
namespace maths {
namespace common {
class CTools::CMixtureProbabilityOfLessLikelySample {
public:
    //! Computes the value of the smooth kernel of an integral
    //! which approximates the probability of less likely samples.
    //!
    //! In particular, we write the integral as
    //! <pre class="fragment">
    //!   \f$P(\{s : f(s) < f(x)\}) = \int{I(f(s) < f(x)) f(s)}ds\f$
    //! </pre>
    //!
    //! and approximate the indicator function as
    //! <pre class="fragment">
    //!   \f$\displaystyle I(f(s) < f(x)) \approx (1+e^{-k}) \frac{e^{-k(f(s)/f(x)-1)}}{1+e^{-k(f(s)/f(x)-1)}}\f$
    //! </pre>
    //!
    //! Note that the larger the value of \f$k\f$ the better the
    //! approximation. Note also that this computes the scaled
    //! kernel, i.e. \f$k'(s) = k(s)/f(x)\f$ so the output must
    //! be scaled by \f$f(x)\f$ to recover the true probability.
    template<typename LOGF>
    class CSmoothedKernel : private core::CNonCopyable {
    public:
        CSmoothedKernel(LOGF logf, double logF0, double k)
            : m_LogF(logf), m_LogF0(logF0), m_K(k),
              m_Scale(std::exp(m_LogF0) * (1.0 + std::exp(-k))) {}

        void k(double k) {
            double f0 = m_Scale / (1.0 + std::exp(-m_K));
            m_K = k;
            m_Scale = f0 * (1.0 + std::exp(-k));
        }

        bool operator()(double x, double& result) const {
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

    private:
        LOGF m_LogF;
        double m_LogF0;
        double m_K;
        double m_Scale;
    };

public:
    //! \param[in] n The number of modes.
    //! \param[in] x The sample.
    //! \param[in] logFx The log of the p.d.f. at the sample.
    //! \param[in] a The left end of the interval to integrate.
    //! \param[in] b The left end of the interval to integrate.
    CMixtureProbabilityOfLessLikelySample(std::size_t n, double x, double logFx, double a, double b);

    //! Reinitialize the object for computing the the probability
    //! of \f$\{y : f(y) <= f(x)\}\f$.
    //!
    //! \param[in] x The sample.
    //! \param[in] logFx The log of the p.d.f. at the sample.
    void reinitialize(double x, double logFx);

    //! Add a mode of the distribution with mean \p mean and
    //! standard deviation \p sd with normalized weight \p weight.
    //!
    //! \param[in] weight The mode weight, i.e. the proportion of
    //! samples in the mode.
    //! \param[in] modeMean The mode mean.
    //! \param[in] modeSd The mode standard deviation.
    void addMode(double weight, double modeMean, double modeSd);

    //! Find the left tail argument with the same p.d.f. value as
    //! the sample.
    //!
    //! \param[in] logf The function which computes the log of the
    //! mixture p.d.f.
    //! \param[in] iterations The number of maximum number of
    //! evaluations of the logf function.
    //! \param[in] equal The function to test if two argument values
    //! are equal.
    //! \param[out] result Filled in with the argument with the same
    //! p.d.f. value as the sample in the left tail.
    //!
    //! \tparam LOGF The type of the function (object) which computes
    //! the log of the mixture p.d.f. It is expected to have a function
    //! like signature double (double).
    template<typename LOGF, typename EQUAL>
    bool leftTail(const LOGF& logf, std::size_t iterations, const EQUAL& equal, double& result) const {
        if (m_X <= m_A) {
            result = m_X;
            return true;
        }

        CCompositeFunctions::CMinusConstant<const LOGF&> f(logf, m_LogFx);

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

    //! Find the right tail argument with the same p.d.f. value
    //! as the sample.
    //!
    //! \param[in] logf The function which computes the log of the
    //! mixture p.d.f.
    //! \param[in] iterations The number of maximum number of
    //! evaluations of the logf function.
    //! \param[in] equal The function to test if two argument values
    //! are equal.
    //! \param[out] result Filled in with the argument with the same
    //! p.d.f. value as the sample in the right tail.
    //!
    //! \tparam LOGF The type of the function (object) which computes
    //! the log of the mixture p.d.f. It is expected to have a function
    //! like signature double (double).
    template<typename LOGF, typename EQUAL>
    bool rightTail(const LOGF& logf, std::size_t iterations, const EQUAL& equal, double& result) const {
        if (m_X >= m_B) {
            result = m_X;
            return true;
        }

        CCompositeFunctions::CMinusConstant<const LOGF&> f(logf, m_LogFx);

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

    //! Compute the probability of a less likely sample.
    //!
    //! \param[in] logf The function which computes the log of the
    //! mixture p.d.f.
    //! \param[in] pTails The probability in the distribution tails,
    //! which can be found from the c.d.f., and is not account for
    //! by the integration.
    //!
    //! \tparam LOGF The type of the function (object) which computes
    //! the log of the mixture p.d.f. It is expected to have a function
    //! like signature bool (double, double &) where the first argument
    //! is the p.d.f. argument and the second argument is filled in
    //! with the log p.d.f. at the first argument.
    template<typename LOGF>
    double calculate(const LOGF& logf, double pTails) {
        TDoubleDoublePrVec intervals;
        this->intervals(intervals);

        double p = 0.0;
        TDoubleVec pIntervals(intervals.size(), 0.0);
        CSmoothedKernel<const LOGF&> kernel(logf, m_LogFx, 3.0);
        for (std::size_t i = 0; i < intervals.size(); ++i) {
            if (!CIntegration::gaussLegendre<CIntegration::OrderFour>(
                    kernel, intervals[i].first, intervals[i].second, pIntervals[i])) {
                LOG_ERROR(<< "Couldn't integrate kernel over " << intervals[i]);
            }
        }

        p += pTails;
        kernel.k(15.0);
        CIntegration::adaptiveGaussLegendre<CIntegration::OrderTwo>(kernel, intervals, pIntervals,
                                                                    2, // refinements
                                                                    3, // splits
                                                                    1e-2, // tolerance
                                                                    p);
        return truncate(p - pTails, 0.0, 1.0);
    }

private:
    using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;

private:
    static const double LOG_ROOT_TWO_PI;

private:
    //! Compute the seed integration intervals.
    void intervals(TDoubleDoublePrVec& intervals);

private:
    //! The sample.
    double m_X;
    //! The log p.d.f. of the sample for which to compute the
    //! probability.
    double m_LogFx;
    //! The integration interval [a, b].
    double m_A, m_B;
    //! Filled in with the end points of the seed intervals for
    //! adaptive quadrature.
    TDoubleVec m_Endpoints;
    //! The maximum deviation of the sample from any mode.
    TMaxAccumulator m_MaxDeviation;
};

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
    LOG_TRACE(<< "range = " << range);
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
    LOG_TRACE(<< "range = " << range);

    double result = 0.0;

    CDifferentialEntropyKernel<T> kernel(mixture);
    for (auto[a, b] : range) {
        double d = (b - a) / static_cast<double>(INTERVALS);
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

template<typename COLLECTION>
void CTools::inplaceSoftmax(COLLECTION& z) {
    double Z{0.0};
    double zmax{*std::max_element(z.begin(), z.end())};
    for (auto& zi : z) {
        zi = stableExp(zi - zmax);
        Z += zi;
    }
    for (auto& zi : z) {
        zi /= Z;
    }
}

template<typename COLLECTION>
void CTools::inplaceLogSoftmax(COLLECTION& z) {
    double zmax{*std::max_element(z.begin(), z.end())};
    for (auto& zi : z) {
        zi -= zmax;
    }
    double logZ{std::log(std::accumulate(z.begin(), z.end(), 0.0, [](double sum, const auto& zi) {
        return sum + std::exp(zi);
    }))};
    for (auto& zi : z) {
        zi -= logZ;
    }
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
}

#endif // INCLUDED_ml_maths_common_CToolsDetail_h
