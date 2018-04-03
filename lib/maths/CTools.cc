/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <maths/CTools.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/Constants.h>

#include <maths/CChecksum.h>
#include <maths/CEqualWithTolerance.h>
#include <maths/CIntegration.h>
#include <maths/CLogTDistribution.h>
#include <maths/CMathsFuncs.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CPrior.h>
#include <maths/CSolvers.h>
#include <maths/CToolsDetail.h>

#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/optional.hpp>

#include <algorithm>
#include <cmath>
#include <ostream>

namespace boost {
namespace math {
namespace policies {

template<class T>
T user_overflow_error(const char* /*function*/, const char* /*message*/, const T& /*val*/) {
    return boost::numeric::bounds<T>::highest();
}
}
}
}

namespace ml {
namespace maths {

namespace {

using TDoubleBoolPr = std::pair<double, bool>;
using TDoubleDoublePr = std::pair<double, double>;
using TOptionalDoubleDoublePr = boost::optional<TDoubleDoublePr>;

namespace adapters {

template<typename DISTRIBUTION>
inline double pdf(const DISTRIBUTION& distribution, double x) {
    return CTools::safePdf(distribution, x);
}

inline double pdf(const CLogTDistribution& distribution, double x) {
    return ml::maths::pdf(distribution, x);
}

} // adapters::

inline TDoubleBoolPr stationaryPoint(const boost::math::beta_distribution<>& beta) {
    if (beta.alpha() < 1.0 && beta.beta() < 1.0) {
        // This is the unique minimum of the p.d.f.
        return {(beta.alpha() - 1.0) / (beta.alpha() + beta.beta() - 2.0), false};
    }
    return {boost::math::mode(beta), true};
}

//! Compute \f$x^2\f$.
inline double square(double x) {
    return x * x;
}

//! \brief p.d.f function adapter.
//!
//! DESCRIPTION:\n
//! Wrapper around a distribution object which evaluates the safe version
//! of the p.d.f. function. This is used to adapt the function for use
//! with the boost::math solvers.
template<typename DISTRIBUTION>
class CPdf {
public:
    CPdf(const DISTRIBUTION& distribution, double target) : m_Distribution(distribution), m_Target(target) {}

    double operator()(double x) const { return adapters::pdf(m_Distribution, x) - m_Target; }

private:
    DISTRIBUTION m_Distribution;
    double m_Target;
};

//! Convenience factory method for the CPdf object for \p distribution.
template<typename DISTRIBUTION>
inline CPdf<DISTRIBUTION> makePdf(const DISTRIBUTION& distribution, double target) {
    return CPdf<DISTRIBUTION>(distribution, target);
}

template<typename Distribution>
inline double continuousSafePdf(const Distribution& distribution, double x) {
    TDoubleDoublePr support = boost::math::support(distribution);
    if (x <= support.first || x >= support.second) {
        return 0.0;
    } else if (CMathsFuncs::isNan(x)) {
        LOG_ERROR("x = NaN, distribution = " << typeid(Distribution).name());
        return 0.0;
    }
    return boost::math::pdf(distribution, x);
}

template<typename Distribution>
inline double discreteSafePdf(const Distribution& distribution, double x) {
    // Note that the inequalities are strict this is needed because
    // the distribution is discrete and can have mass at the support
    // end points.

    TDoubleDoublePr support = boost::math::support(distribution);
    if (x < support.first || x > support.second) {
        return 0.0;
    } else if (CMathsFuncs::isNan(x)) {
        LOG_ERROR("x = NaN, distribution = " << typeid(Distribution).name());
        return 0.0;
    }
    return boost::math::pdf(distribution, x);
}

template<typename Distribution>
inline double continuousSafeCdf(const Distribution& distribution, double x) {
    TDoubleDoublePr support = boost::math::support(distribution);
    if (x <= support.first) {
        return 0.0;
    } else if (x >= support.second) {
        return 1.0;
    } else if (CMathsFuncs::isNan(x)) {
        LOG_ERROR("x = NaN, distribution = " << typeid(Distribution).name());
        return 0.0;
    }
    return boost::math::cdf(distribution, x);
}

template<typename Distribution>
inline double discreteSafeCdf(const Distribution& distribution, double x) {
    // Note that the inequalities are strict this is needed because
    // the distribution is discrete and can have mass at the support
    // end points.

    TDoubleDoublePr support = boost::math::support(distribution);
    if (x < support.first) {
        return 0.0;
    } else if (x > support.second) {
        return 1.0;
    } else if (CMathsFuncs::isNan(x)) {
        LOG_ERROR("x = NaN, distribution = " << typeid(Distribution).name());
        return 0.0;
    }
    return boost::math::cdf(distribution, x);
}

template<typename Distribution>
inline double continuousSafeCdfComplement(const Distribution& distribution, double x) {
    TDoubleDoublePr support = boost::math::support(distribution);
    if (x <= support.first) {
        return 1.0;
    } else if (x >= support.second) {
        return 0.0;
    } else if (CMathsFuncs::isNan(x)) {
        LOG_ERROR("x = NaN, distribution = " << typeid(Distribution).name());
        return 0.0;
    }
    return boost::math::cdf(boost::math::complement(distribution, x));
}

template<typename Distribution>
inline double discreteSafeCdfComplement(const Distribution& distribution, double x) {
    // Note that the inequalities are strict this is needed because
    // the distribution is discrete and can have mass at the support
    // end points.

    TDoubleDoublePr support = boost::math::support(distribution);
    if (x < support.first) {
        return 1.0;
    } else if (x > support.second) {
        return 0.0;
    } else if (CMathsFuncs::isNan(x)) {
        LOG_ERROR("x = NaN distribution = " << typeid(Distribution).name());
        return 0.0;
    }
    return boost::math::cdf(boost::math::complement(distribution, x));
}

const double EPSILON = std::numeric_limits<double>::epsilon();
const double MIN_DOUBLE = std::numeric_limits<double>::min();
const double NEG_INF = boost::numeric::bounds<double>::lowest();
const double POS_INF = boost::numeric::bounds<double>::highest();

} // unnamed::

//////// SMinusLogCdf Implementation ////////

namespace {

//! Computes -log(\p cdf) enforces limits and avoids underflow.
inline double safeMinusLogCdf(double cdf) {
    if (cdf == 0.0) {
        // log(0.0) == -HUGE_VALUE, which is too big for our purposes
        // and causes problems on Windows. In fact, we want to avoid
        // underflow since this will pollute the floating point
        // environment and *may* cause problems for some library
        // function implementations (see fe*exceptflags for more details).
        // The log of the minimum double should be small enough for
        // our purposes.
        return -core::constants::LOG_MIN_DOUBLE;
    }
    return std::max(-std::log(cdf), 0.0);
}
}

const double CTools::IMPROPER_CDF(0.5);

double CTools::SMinusLogCdf::operator()(const SImproperDistribution&, double) const {
    return -std::log(IMPROPER_CDF);
}

double CTools::SMinusLogCdf::operator()(const normal& normal_, double x) const {
    return safeMinusLogCdf(safeCdf(normal_, x));
}

double CTools::SMinusLogCdf::operator()(const students_t& students, double x) const {
    return safeMinusLogCdf(safeCdf(students, x));
}

double CTools::SMinusLogCdf::operator()(const negative_binomial& negativeBinomial, double x) const {
    return safeMinusLogCdf(safeCdf(negativeBinomial, x));
}

double CTools::SMinusLogCdf::operator()(const lognormal& logNormal, double x) const {
    return safeMinusLogCdf(safeCdf(logNormal, x));
}

double CTools::SMinusLogCdf::operator()(const CLogTDistribution& logt, double x) const {
    return safeMinusLogCdf(maths::cdf(logt, x));
}

double CTools::SMinusLogCdf::operator()(const gamma& gamma_, double x) const {
    return safeMinusLogCdf(safeCdf(gamma_, x));
}

double CTools::SMinusLogCdf::operator()(const beta& beta_, double x) const {
    return safeMinusLogCdf(safeCdf(beta_, x));
}

//////// SMinusLogCdfComplement Implementation ////////

double CTools::SMinusLogCdfComplement::operator()(const SImproperDistribution&, double) const {
    return -std::log(1.0 - IMPROPER_CDF);
}

double CTools::SMinusLogCdfComplement::operator()(const normal& normal_, double x) const {
    return safeMinusLogCdf(safeCdfComplement(normal_, x));
}

double CTools::SMinusLogCdfComplement::operator()(const students_t& students, double x) const {
    return safeMinusLogCdf(safeCdfComplement(students, x));
}

double CTools::SMinusLogCdfComplement::operator()(const negative_binomial& negativeBinomial, double x) const {
    return safeMinusLogCdf(safeCdfComplement(negativeBinomial, x));
}

double CTools::SMinusLogCdfComplement::operator()(const lognormal& logNormal, double x) const {
    return safeMinusLogCdf(safeCdfComplement(logNormal, x));
}

double CTools::SMinusLogCdfComplement::operator()(const CLogTDistribution& logt, double x) const {
    return safeMinusLogCdf(maths::cdfComplement(logt, x));
}

double CTools::SMinusLogCdfComplement::operator()(const gamma& gamma_, double x) const {
    return safeMinusLogCdf(safeCdfComplement(gamma_, x));
}

double CTools::SMinusLogCdfComplement::operator()(const beta& beta_, double x) const {
    return safeMinusLogCdf(safeCdfComplement(beta_, x));
}

//////// SProbabilityLessLikelySample Implementation ////////

CTools::CProbabilityOfLessLikelySample::CProbabilityOfLessLikelySample(maths_t::EProbabilityCalculation calculation)
    : m_Calculation(calculation) {
}

double CTools::CProbabilityOfLessLikelySample::operator()(const SImproperDistribution&, double, maths_t::ETail& tail) const {
    // For any finite sample this is one.
    tail = maths_t::E_MixedOrNeitherTail;
    return 1.0;
}

double CTools::CProbabilityOfLessLikelySample::operator()(const normal& normal_, double x, maths_t::ETail& tail) const {
    double px = 0.0;

    TDoubleDoublePr support = boost::math::support(normal_);
    if (!this->check(support, x, px, tail)) {
        return px;
    }

    switch (m_Calculation) {
    case maths_t::E_OneSidedBelow:
        px = truncate(2.0 * safeCdf(normal_, x), 0.0, 1.0);
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_LeftTail);
        break;

    case maths_t::E_TwoSided: {
        // The normal distribution is symmetric and single mode so the
        // probability of less likely events than x is:
        //   2 * std::min(cdf(x), 1 - cdf(x)).
        //
        // Note, we use the complement function to compute the 1 - cdf(x)
        // so that we aren't restricted to epsilon precision.

        double m = boost::math::mode(normal_);
        if (x < m) {
            px = truncate(2.0 * safeCdf(normal_, x), 0.0, 1.0);
        } else {
            px = truncate(2.0 * safeCdfComplement(normal_, x), 0.0, 1.0);
        }
        this->tail(x, m, tail);
        break;
    }
    case maths_t::E_OneSidedAbove:
        px = truncate(2.0 * safeCdfComplement(normal_, x), 0.0, 1.0);
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
        break;
    }

    return px;
}

double CTools::CProbabilityOfLessLikelySample::operator()(const students_t& students, double x, maths_t::ETail& tail) const {
    double px = 0.0;

    TDoubleDoublePr support = boost::math::support(students);
    if (!this->check(support, x, px, tail)) {
        return px;
    }

    switch (m_Calculation) {
    case maths_t::E_OneSidedBelow:
        px = truncate(2.0 * safeCdf(students, x), 0.0, 1.0);
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_LeftTail);
        break;

    case maths_t::E_TwoSided: {
        // Student's t distribution is symmetric and single mode so the
        // probability of less likely events than x is:
        //   2 * std::min(cdf(x), 1 - cdf(x)).
        //
        // Note, we use the complement function to compute the 1 - cdf(x)
        // so that we aren't restricted to epsilon precision.
        double m = boost::math::mode(students);
        if (x < m) {
            px = truncate(2.0 * safeCdf(students, x), 0.0, 1.0);
        } else {
            px = truncate(2.0 * safeCdfComplement(students, x), 0.0, 1.0);
        }
        this->tail(x, m, tail);
        break;
    }
    case maths_t::E_OneSidedAbove:
        px = truncate(2.0 * safeCdfComplement(students, x), 0.0, 1.0);
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
        break;
    }

    return px;
}

double CTools::CProbabilityOfLessLikelySample::operator()(const negative_binomial& negativeBinomial, double x, maths_t::ETail& tail) const {
    x = ::floor(x);

    double px = 0.0;

    TDoubleDoublePr support = boost::math::support(negativeBinomial);
    if (!this->check(support, x, px, tail)) {
        return px;
    }

    switch (m_Calculation) {
    case maths_t::E_OneSidedBelow:
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_LeftTail);
        return truncate(2.0 * safeCdf(negativeBinomial, x), 0.0, 1.0);

    case maths_t::E_TwoSided:
        // Fall through.
        break;

    case maths_t::E_OneSidedAbove:
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
        return truncate(2.0 * (safeCdfComplement(negativeBinomial, x) + safePdf(negativeBinomial, x)), 0.0, 1.0);
    }

    double fx = safePdf(negativeBinomial, x);
    double r = negativeBinomial.successes();
    double p = negativeBinomial.success_fraction();
    double m = boost::math::mode(negativeBinomial);
    LOG_TRACE("x = " << x << ", f(x) = " << fx);

    // If the number of successes <= 1 the distribution is single sided.
    if (r <= 1.0) {
        tail = maths_t::E_RightTail;
        return truncate(safeCdfComplement(negativeBinomial, x) + fx, 0.0, 1.0);
    }

    this->tail(x, m, tail);

    // If the f(x) <= f(0) and x is greater than the mode the probability
    // is just P(y > x).
    if (x > m) {
        double f0 = safePdf(negativeBinomial, 0.0);
        LOG_TRACE("f(0) = " << f0);
        if (fx <= f0) {
            return truncate(safeCdfComplement(negativeBinomial, x) + fx, 0.0, 1.0);
        }
    }

    // Trap the case f(x) = f(m).
    double fm = safePdf(negativeBinomial, m);

    LOG_TRACE("m = " << m << ", f(m) = " << fm);

    if (fx >= fm) {
        return 1.0;
    }

    const unsigned int MAX_ITERATIONS = 20u;
    std::size_t maxIterations = MAX_ITERATIONS;

    double b1, b2, f1, f2;
    if (x > m) {
        b1 = b2 = m;
        f1 = f2 = fm;

        double shrinkFactor = 1.5;
        double step = (1.0 / shrinkFactor - 1.0) * b1;
        for (;;) {
            if (maxIterations == 0 || f1 <= fx) {
                break;
            }

            b2 = b1;
            f2 = f1;
            b1 += step;
            f1 = safePdf(negativeBinomial, b1);
            --maxIterations;

            if (maxIterations <= 3 * MAX_ITERATIONS / 4) {
                shrinkFactor *= 2.0;
            }

            step = (maxIterations == MAX_ITERATIONS / 2 ? b1 : (1.0 / shrinkFactor - 1.0) * b1);
        }
    } else {
        // Noting that the binomial coefficient (k + r - 1)! / k! / (r - 1)!
        // is a monotonic increasing function of k, we have for any k':
        //   f(k') * (1 - p)^(k - k') < f(k)                  for k > k'
        //
        // Solving for k and using the fact that log(1 - p) is negative:
        //   k > k' + log(f(x)/f(k')) / log(1 - p)

        double logOneMinusP = std::log(1 - p);

        b1 = ::floor(m + std::log(std::max(fx, MIN_DOUBLE) / std::max(fm, MIN_DOUBLE)) / logOneMinusP);
        f1 = safePdf(negativeBinomial, b1);
        b2 = b1;
        f2 = f1;

        LOG_TRACE("b1 = " << b1 << ", f(b1) = " << f1);

        double growthFactor = 0.25;
        double step = growthFactor * b2;
        for (;;) {
            if (maxIterations == 0 || f2 <= fx) {
                break;
            }

            b1 = b2;
            f1 = f2;
            b2 += step;
            f2 = safePdf(negativeBinomial, b2);
            --maxIterations;

            // We compute successively tighter lower bounds on the
            // bracket point.
            double lowerBound = b2 + std::log(std::max(fx, MIN_DOUBLE) / std::max(f2, MIN_DOUBLE)) / logOneMinusP;
            LOG_TRACE("b2 = " << b2 << ", f2 = " << f2 << ", bound = " << lowerBound);

            if (maxIterations <= 3 * MAX_ITERATIONS / 4) {
                growthFactor *= 4.0;
            }
            step = std::max(growthFactor * b2, lowerBound - b2);
        }
    }

    LOG_TRACE("Initial bracket = (" << b1 << "," << b2 << ")"
                                    << ", f(bracket) = (" << f1 - fx << "," << f2 - fx << ")");

    px = x < m ? safeCdf(negativeBinomial, x) : safeCdfComplement(negativeBinomial, x);
    double y = POS_INF;

    try {
        // Note that this form of epsilon controls the maximum
        // relative error in the probability since p > px and
        // the error will be order eps * f(x) so we require that
        // eps * f(x) <= 0.05 * px.
        double eps = 0.05 * px / std::max(fx, MIN_DOUBLE);
        eps = std::max(eps, EPSILON * std::min(b1, b2));
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, eps);
        CSolvers::solve(b1, b2, f1 - fx, f2 - fx, makePdf(negativeBinomial, fx), maxIterations, equal, y);
        LOG_TRACE("bracket = (" << b1 << "," << b2 << ")"
                                << ", iterations = " << maxIterations << ", f(y) = " << safePdf(negativeBinomial, y) - fx
                                << ", eps = " << eps);
    } catch (const std::exception& e) {
        if (::fabs(f1 - fx) < 10.0 * EPSILON * fx) {
            y = b1;
        } else if (::fabs(f2 - fx) < 10.0 * EPSILON * fx) {
            y = b2;
        } else {
            LOG_ERROR("Failed in root finding: " << e.what() << ", x = " << x << ", bracket = (" << b1 << "," << b2 << ")"
                                                 << ", f(bracket) = (" << f1 - fx << "," << f2 - fx << ")");
            return truncate(px, 0.0, 1.0);
        }
    }

    if ((x < m && y < m) || (x > m && y > m) || !(x >= support.first && x <= support.second)) {
        LOG_ERROR("Bad root " << y << " (x = " << x << ")");
    }

    double py = x < m ? safeCdfComplement(negativeBinomial, y) : safeCdf(negativeBinomial, y);

    return truncate(px + py + fx, 0.0, 1.0);
}

double CTools::CProbabilityOfLessLikelySample::operator()(const lognormal& logNormal, double x, maths_t::ETail& tail) const {
    double px = 0.0;

    TDoubleDoublePr support = boost::math::support(logNormal);
    if (!this->check(support, x, px, tail)) {
        return px;
    }

    switch (m_Calculation) {
    case maths_t::E_OneSidedBelow:
        px = truncate(2.0 * safeCdf(logNormal, x), 0.0, 1.0);
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_LeftTail);
        break;

    case maths_t::E_TwoSided: {
        // Changing variables to x = exp(m) * exp(x') where m is the location
        // of the log normal distribution it is possible to show that the
        // equal point on the p.d.f. is at:
        //   exp(m) * exp(-s^2 + (s^4 + (log(x) - m)^2
        //                        + 2 * s^2 * (log(x) - m))^(1/2))  if x < mode
        //
        // and
        //   exp(m) * exp(-s^2 - (s^4 + (log(x) - m)^2
        //                        + 2 * s^2 * (log(x) - m))^(1/2))  if x > mode

        double logx = std::log(x);
        double squareScale = square(logNormal.scale());
        double discriminant =
            std::sqrt(square(squareScale) + (logx - logNormal.location() + 2.0 * squareScale) * (logx - logNormal.location()));
        double m = boost::math::mode(logNormal);
        this->tail(x, m, tail);
        double y = m * ::exp(x > m ? -discriminant : discriminant);
        if (x > y) {
            std::swap(x, y);
        }
        px = truncate(safeCdf(logNormal, x) + safeCdfComplement(logNormal, y), 0.0, 1.0);
        break;
    }
    case maths_t::E_OneSidedAbove:
        px = truncate(2.0 * safeCdfComplement(logNormal, x), 0.0, 1.0);
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
        break;
    }

    return px;
}

double CTools::CProbabilityOfLessLikelySample::operator()(const CLogTDistribution& logt, double x, maths_t::ETail& tail) const {
    double px = 0.0;

    TDoubleDoublePr support = maths::support(logt);
    if (!this->check(support, x, px, tail)) {
        return px;
    }

    switch (m_Calculation) {
    case maths_t::E_OneSidedBelow:
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_LeftTail);
        return truncate(2.0 * cdf(logt, x), 0.0, 1.0);

    case maths_t::E_TwoSided:
        // Fall through.
        break;

    case maths_t::E_OneSidedAbove:
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
        return truncate(2.0 * cdfComplement(logt, x), 0.0, 1.0);
    }

    const unsigned int MAX_ITERATIONS = 20u;

    double fx = pdf(logt, x);
    double m = mode(logt);
    LOG_TRACE("x = " << x << ", f(x) = " << fx);

    this->tail(x, m, tail);

    // Handle the case that x than the mode of the distribution.

    // Note that the p.d.f. can have a local minimum between zero
    // and the mode of the distribution.
    CLogTDistribution::TOptionalDouble localMinimum = maths::localMinimum(logt);
    if (!localMinimum) {
        // If there is no local minimum the distribution is single sided.
        return truncate(cdfComplement(logt, x), 0.0, 1.0);
    } else {
        double b1 = *localMinimum;
        double f1 = pdf(logt, b1);
        LOG_TRACE("b1 = " << b1 << ", f(b1) = " << f1);

        if (f1 > fx) {
            return truncate(cdfComplement(logt, x), 0.0, 1.0);
        } else if (x > m) {
            px = cdfComplement(logt, x);

            double b2 = m;
            double f2 = pdf(logt, m);
            LOG_TRACE("Initial bracket = (" << b1 << "," << b2 << ")"
                                            << ", f(bracket) = (" << f1 - fx << "," << f2 - fx << ")");

            double y = 0.0;
            try {
                // The gradient of the log normal p.d.f. can be very
                // large near the origin so we use the maximum of f1 and
                // f2 to be safe here rather that the value of f at the
                // root, i.e. f(x). Note that this form of epsilon controls
                // the maximum relative error in the probability since
                // p > px and the error will be order eps * f(x) so we
                // require that eps * f(x) <= 0.05 * px.
                double eps = 0.05 * px / std::max(std::max(f1, f2), MIN_DOUBLE);
                eps = std::max(eps, EPSILON * std::min(b1, b2));
                CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, eps);
                std::size_t maxIterations = MAX_ITERATIONS;
                CSolvers::solve(b1, b2, f1 - fx, f2 - fx, makePdf(logt, fx), maxIterations, equal, y);
                LOG_TRACE("bracket = (" << b1 << "," << b2 << ")"
                                        << ", iterations = " << maxIterations << ", f(y) = " << pdf(logt, y) - fx << ", eps = " << eps);
            } catch (const std::exception& e) {
                if (::fabs(f1 - fx) < 10.0 * EPSILON * fx) {
                    y = b1;
                } else if (::fabs(f2 - fx) < 10.0 * EPSILON * fx) {
                    y = b2;
                } else {
                    LOG_ERROR("Failed in root finding: " << e.what() << ", x = " << x << ", bracket = (" << b1 << "," << b2 << ")"
                                                         << ", f(bracket) = (" << f1 - fx << "," << f2 - fx << ")");
                    return truncate(px, 0.0, 1.0);
                }
            }

            return truncate(cdf(logt, y) + px, 0.0, 1.0);
        }
    }

    // Handle the case that x is less than the distribution mode.

    // For small x the density can be greater than the local mode.
    double fm = pdf(logt, m);
    LOG_TRACE("f(m) = " << fm);
    if (fx > fm) {
        return truncate(cdfComplement(logt, x), 0.0, 1.0);
    }

    // We use the fact that 1/3 log(x) < x^(1/8) for x > 0 to derive
    // the following lower bound for the density function if x > exp(l):
    //
    //   f(x|v,l,s) > f(exp(l)) * exp(l) * (s' / (3 * (exp(s'-l) * x)^(1/8)))^(v+1)
    //
    // where,
    //   v is the number of degrees freedom of the log-t distribution,
    //   l is the location of the log-t distribution,
    //   s is the scale of the log-t distribution
    //   s' = v^(1/2) * s.
    //
    // We can use this to solve for a lower bound for the value of x
    // denoted x* which satisfies:
    //   f(x*|v,l,s) = f
    //
    // The bound is relatively tight (for smallish v) since:
    //   g(x) = (1 - log(x) / 3) / x^(1/8)
    //
    // is a monotonic increasing function of x for large x and equals
    // 0.18 at x = 1000000.

    double v = logt.degreesFreedom();
    double l = logt.location();
    double s = logt.scale();

    // Initialize the bracket.
    double fl = pdf(logt, ::exp(l));
    double scale = std::sqrt(v) * s;
    double bound = 0.0;
    double fBound = POS_INF;
    if (fl < fx) {
        bound = ::exp(l);
        fBound = fl;
    } else {
        double t1 = l + std::log(fl / fx);
        double t2 = (l - scale) / 8.0 + std::log(scale / 3.0);
        double k0 = 8.0 * (t1 + (v + 1.0) * t2) / (v + 9.0);
        bound = std::max(::exp(l), ::exp(k0));
        fBound = pdf(logt, bound);
    }
    double b1 = fBound < fx ? m : bound;
    double f1 = fBound < fx ? fm : fBound;
    double b2 = bound;
    double f2 = fBound;
    LOG_TRACE("b1 = " << b1 << ", f(b1) = " << f1 << ", b2 = " << b2 << ", f(b2) = " << f2);

    std::size_t maxIterations = MAX_ITERATIONS;

    // Doubling search for the upper bracket with the possibility
    // of early exit based on an upper bound for the bracket. Note
    // that we accelerate the growth if we don't bracket the root
    // quickly and fallback to the bound if we haven't bracketed
    double step = std::max(b2, ::exp(l) - b2);
    double growthFactor = 1.0;
    for (;;) {
        if (maxIterations == 0 || f2 <= fx) {
            break;
        }

        b1 = b2;
        f1 = f2;
        b2 += step;
        f2 = pdf(logt, b2);
        --maxIterations;

        // We compute successively tighter upper bounds on the
        // bracket point.
        double upperBound = b2 * f2 / fx;
        LOG_TRACE("Bound = " << upperBound);

        if (maxIterations <= 3 * MAX_ITERATIONS / 4) {
            growthFactor *= 3.0;
        }
        if (maxIterations <= MAX_ITERATIONS / 2 || upperBound - b2 < 2.0 * growthFactor * b2) {
            step = upperBound - b2;
        } else {
            step = growthFactor * b2;
        }
    }

    LOG_TRACE("Initial bracket = (" << b1 << "," << b2 << ")"
                                    << ", f(bracket) = (" << f1 - fx << "," << f2 - fx << ")");

    px = cdf(logt, x);
    double y = POS_INF;

    try {
        // Note that this form of epsilon controls the maximum
        // relative error in the probability since p > px and
        // the error will be order eps * f(x) so we require that
        // eps * f(x) <= 0.05 * px.
        double eps = 0.05 * px / std::max(fx, MIN_DOUBLE);
        eps = std::max(eps, EPSILON * std::min(b1, b2));
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, eps);
        CSolvers::solve(b1, b2, f1 - fx, f2 - fx, makePdf(logt, fx), maxIterations, equal, y);
        LOG_TRACE("bracket = (" << b1 << "," << b2 << ")"
                                << ", iterations = " << maxIterations << ", f(y) = " << pdf(logt, y) - fx << ", eps = " << eps);
    } catch (const std::exception& e) {
        if (::fabs(f1 - fx) < 10.0 * EPSILON * fx) {
            y = b1;
        } else if (::fabs(f2 - fx) < 10.0 * EPSILON * fx) {
            y = b2;
        } else {
            LOG_ERROR("Failed in root finding: " << e.what() << ", x = " << x << ", bracket = (" << b1 << "," << b2 << ")"
                                                 << ", f(bracket) = (" << f1 - fx << "," << f2 - fx << ")");
            return truncate(px, 0.0, 1.0);
        }
    }

    return truncate(px + cdfComplement(logt, y), 0.0, 1.0);
}

double CTools::CProbabilityOfLessLikelySample::operator()(const gamma& gamma_, double x, maths_t::ETail& tail) const {
    double px = 0.0;

    TDoubleDoublePr support = boost::math::support(gamma_);
    if (!this->check(support, x, px, tail)) {
        return px;
    }

    switch (m_Calculation) {
    case maths_t::E_OneSidedBelow:
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_LeftTail);
        return truncate(2.0 * safeCdf(gamma_, x), 0.0, 1.0);

    case maths_t::E_TwoSided:
        // Fall through.
        break;

    case maths_t::E_OneSidedAbove:
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
        return truncate(2.0 * safeCdfComplement(gamma_, x), 0.0, 1.0);
    }

    // For alpha <= 1 the distribution is single sided.
    if (gamma_.shape() <= 1.0) {
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
        return truncate(safeCdfComplement(gamma_, x), 0.0, 1.0);
    }

    const double CONVERGENCE_TOLERANCE = 1e-4;
    const double PDF_TOLERANCE = 5e-2;
    const unsigned int MAX_ITERATIONS = 20u;

    double m = boost::math::mode(gamma_);
    LOG_TRACE("x = " << x << ", mode = " << m);

    this->tail(x, m, tail);

    double y[] = {2.0 * m - x, 0.0};
    unsigned int i = 0u;

    if (x == m) {
        return 1.0;
    } else if (x < m) {
        // For x < m we use the recurrence relation:
        //   y(n+1) = x + m * log(y(n) / x)
        //
        // Note it is relatively straightforward to show that:
        //   g(y) = x + m * log(y(n) / x)
        //
        // is a contractive mapping for y > m and x < m by showing
        // that the derivative is less than 1. This guarantees that
        // the following iteration (eventually) converges to the
        // unique fixed point (Banach fixed point theorem). Note that
        // this has poor convergence for x near the stationary point;
        // however, reflecting x in the stationary point gives a starting
        // point very near root in this case, i.e. it is equivalent to
        // initializing with a second order Taylor expansion about the
        // mode.

        for (;;) {
            y[(i + 1) % 2] = x + m * std::log(y[i % 2] / x);
            LOG_TRACE("y = " << y[(i + 1) % 2]);
            if (++i == MAX_ITERATIONS || ::fabs(y[1] - y[0]) < CONVERGENCE_TOLERANCE * std::max(y[0], y[1])) {
                break;
            }
        }
    } else {
        // For x > m we use the recurrence relation:
        //   y(n+1) = m - x * exp(-(x - y(n)) / m)
        //
        // Note it is relatively straightforward to show that:
        //   g(y) = x * exp(-(x - y) / m)
        //
        // is a contractive mapping for y > x and x < (a-1)/(a+b-1) by
        // showing that the derivative is less than 1. This guarantees
        // that the following iteration (eventually) converges to the
        // unique fixed point (Banach fixed point theorem). Note that
        // this has poor convergence for x near the stationary point;
        // however, reflecting x in the stationary point gives a starting
        // point very near root in this case, i.e. it is equivalent to
        // initializing with a second order Taylor expansion about the
        // mode.

        y[0] = std::max(y[0], m / 2.0);

        for (;;) {
            y[(i + 1) % 2] = x * ::exp(-(x - y[i % 2]) / m);
            LOG_TRACE("y = " << y[(i + 1) % 2]);
            if (++i == MAX_ITERATIONS || ::fabs(y[1] - y[0]) < CONVERGENCE_TOLERANCE * std::max(y[0], y[1])) {
                break;
            }
        }
    }

    double fx = safePdf(gamma_, x);
    double fy = safePdf(gamma_, y[i % 2]);
    LOG_TRACE("f(x) = " << fx << ", f(y) = " << fy);

    if (::fabs(fx - fy) <= PDF_TOLERANCE * std::max(fx, fy)) {
        if (x > y[i % 2]) {
            std::swap(x, y[i % 2]);
        }
        return truncate(safeCdf(gamma_, x) + safeCdfComplement(gamma_, y[i % 2]), 0.0, 1.0);
    }

    // The gamma density function can be rewritten as:
    //   f(y) = f(u) / u^(a - 1) / exp(-b * u) * exp(-b * (y - m * log(y)))
    //
    // for any u. Noting that log(u) is a concave function and expanding
    // about u:
    //   log(y) <= log(x) + (y - u) / u
    //
    // This implies that:
    //   f(y) < f(u) / u^(a - 1) / exp(-b * u) * exp(-b * (y - m * (log(u) - (y - u) / u)))
    //
    // From this it follows that if the initial f(y) > f(x) then a bracket
    // f(y') <= f(x) can be found using:
    //   y' = (1 + log(f(y) / f(x)) / b / (y - m)) * y

    double a = y[i % 2];
    double b = a;
    double fa = fy;
    double fb = fa;
    if (x > m && fy < fx) {
        b = m;
        fb = safePdf(gamma_, m);
    } else if (x > m && fy > fx) {
        b = (1.0 + gamma_.scale() / (a - m) * std::log(fa / fx)) * a;
        fb = safePdf(gamma_, b);
        std::swap(a, b);
        std::swap(fa, fb);
    } else if (fy < fx) {
        b = m;
        fb = safePdf(gamma_, m);
        std::swap(a, b);
        std::swap(fa, fb);
    } else {
        b = (1.0 + gamma_.scale() / (a - m) * std::log(fa / fx)) * a;
        fb = safePdf(gamma_, b);
    }

    LOG_TRACE("Initial bracket = (" << a << ", " << b << ")"
                                    << ", f(bracket) = (" << fa - fx << "," << fb - fx << ")");

    px = x > m ? safeCdfComplement(gamma_, x) : safeCdf(gamma_, x);

    try {
        // The gradient of the gamma p.d.f. can be very large
        // near the origin so we use the maximum of fa and
        // fb to be safe here rather that the value of f at the
        // root, i.e. f(x). Note that this form of epsilon controls
        // the maximum relative error in the probability since
        // p > px and the error will be order eps * f(x) so we
        // require that eps * f(x) <= 0.05 * px.
        double eps = 0.05 * px / std::max(std::max(fa, fb), MIN_DOUBLE);
        eps = std::max(eps, EPSILON * std::min(a, b));
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, eps);
        std::size_t maxIterations = MAX_ITERATIONS / 2;
        double candidate;
        CSolvers::solve(a, b, fa - fx, fb - fx, makePdf(gamma_, fx), maxIterations, equal, candidate);
        LOG_TRACE("bracket = (" << a << "," << b << ")"
                                << ", iterations = " << maxIterations << ", f(candidate) = " << safePdf(gamma_, candidate) - fx);

        if (::fabs(safePdf(gamma_, candidate) - fx) < ::fabs(fy - fx)) {
            y[i % 2] = candidate;
        }
    } catch (const std::exception& e) {
        if (::fabs(fa - fx) < 10.0 * EPSILON * fx) {
            y[i % 2] = a;
        } else if (::fabs(fb - fx) < 10.0 * EPSILON * fx) {
            y[i % 2] = b;
        } else {
            LOG_ERROR("Failed in bracketed solver: " << e.what() << ", x = " << x << ", bracket = (" << a << ", " << b << ")"
                                                     << ", f(bracket) = (" << fa - fx << "," << fb - fx << ")");
            return truncate(px, 0.0, 1.0);
        }
    }

    LOG_TRACE("f(x) = " << fx << ", f(y) = " << safePdf(gamma_, y[i % 2]));

    double py = x > y[i % 2] ? safeCdf(gamma_, y[i % 2]) : safeCdfComplement(gamma_, y[i % 2]);

    return truncate(px + py, 0.0, 1.0);
}

double CTools::CProbabilityOfLessLikelySample::operator()(const beta& beta_, double x, maths_t::ETail& tail) const {
    double px = 0.0;

    TDoubleDoublePr support(0.0, 1.0);
    if (!this->check(support, x, px, tail)) {
        return px;
    }

    switch (m_Calculation) {
    case maths_t::E_OneSidedBelow:
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_LeftTail);
        return truncate(2.0 * safeCdf(beta_, x), 0.0, 1.0);

    case maths_t::E_TwoSided:
        // Fall through.
        break;

    case maths_t::E_OneSidedAbove:
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
        return truncate(2.0 * safeCdfComplement(beta_, x), 0.0, 1.0);
    }

    if (beta_.alpha() < 1.0 && beta_.beta() < 1.0) {
        // The probability density function tends to infinity at x = 0
        // and x = 1 and has a unique minimum in the interval (0,1).
        //
        // We can't evaluate the function at 0 and 1 so can't use these
        // as a bracket values. The p.d.f. is monotonic increasing towards
        // 0 and 1 so we choose very nearby values.

        tail = maths_t::E_MixedOrNeitherTail;
        double eps = boost::math::tools::epsilon<double>();
        if (x <= eps || x >= 1.0 - eps) {
            return 1.0;
        }
        support = std::make_pair(eps, 1.0 - eps);
    } else if (beta_.alpha() == 1.0 && beta_.beta() == 1.0) {
        // The distribution is flat.
        tail = maths_t::E_MixedOrNeitherTail;
        return 1.0;
    } else if (beta_.alpha() <= 1.0 && beta_.beta() >= 1.0) {
        // The distribution is monotone decreasing.
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
        return truncate(safeCdfComplement(beta_, x), 0.0, 1.0);
    } else if (beta_.alpha() >= 1.0 && beta_.beta() <= 1.0) {
        // The distribution is monotone increasing.
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_LeftTail);
        return truncate(safeCdf(beta_, x), 0.0, 1.0);
    } else {
        // If alpha > 1 and beta > 1 the probability density function
        // tends to zero at x = 0 and x = 1 and has a unique maximum in
        // the interval (0,1).
    }

    // The following uses an iterative scheme to find the roots of:
    //   y^(a-1) * (1 - y)^(b-1) = x^(a-1) * (1 - x)^(b-1)
    //
    // This means that we can generally avoid evaluating the p.d.f. except
    // to check reasonable convergence of expansion and typically reduces
    // the runtime by an order of magnitude over root finding.

    const double CONVERGENCE_TOLERANCE = 1e-4;
    const double PDF_TOLERANCE = 5e-2;
    const unsigned int MAX_ITERATIONS = 6u;

    TDoubleBoolPr sp = stationaryPoint(beta_);
    double y[] = {2.0 * sp.first - x, 0.0};
    unsigned int i = 0u;

    this->tail(x, sp.first, tail);

    if (x < sp.first) {
        // For x < mode we use the recurrence relation:
        //   y(n+1) = 1 - (x / y(n))^((a-1)/(b-1)) * (1 - x)
        //
        // Note it is relatively straightforward to show that:
        //   g(y) = 1 - (x / y)^((a-1)/(b-1)) * (1 - x)
        //
        // is a contractive mapping for y > x and x < (a-1)/(a+b-1) by
        // showing that the derivative is less than 1. This guarantees
        // that the following iteration (eventually) converges to the
        // unique fixed point (Banach fixed point theorem). Note that
        // this has poor convergence for x near the stationary point;
        // however, reflecting x in the stationary point gives a starting
        // point very near root in this case, i.e. it is equivalent to
        // initializing with a second order Taylor expansion about the
        // mode.

        y[0] = std::min(y[0], (1.0 + sp.first) / 2.0);

        double k = (beta_.alpha() - 1.0) / (beta_.beta() - 1.0);
        for (;;) {
            y[(i + 1) % 2] = 1.0 - ::exp(k * std::log(x / y[i % 2])) * (1.0 - x);
            if (++i == MAX_ITERATIONS || ::fabs(y[1] - y[0]) < CONVERGENCE_TOLERANCE) {
                break;
            }
        }

        // Max sure y is supported by the p.d.f.
        if (y[i % 2] > support.second) {
            return truncate(sp.second ? safeCdf(beta_, x) : safeCdfComplement(beta_, x), 0.0, 1.0);
        }

        y[i % 2] = std::max(y[i % 2], sp.first);
    } else {
        // For x > mode we use the recurrence relation:
        //   y(n+1) = ((1 - x) / (1 - y(n)))^((b-1)/(a-1)) * x
        //
        // Note it is relatively straightforward to show that:
        //   g(y) = ((1 - x) / (1 - y)^((b-1)/(a-1)) * x
        //
        // is a contractive mapping for y > x and x < (a-1)/(a+b-1) by
        // showing that the derivative is less than 1. This guarantees
        // that the following iteration (eventually) converges to the
        // unique fixed point (Banach fixed point theorem). Note that
        // this has poor convergence for x near the stationary point;
        // however, reflecting x in the stationary point gives a starting
        // point very near the root in this case, i.e. it is equivalent to
        // initializing with a second order Taylor expansion about the
        // mode.

        y[0] = std::max(y[0], sp.first / 2.0);

        double k = (beta_.beta() - 1.0) / (beta_.alpha() - 1.0);
        for (;;) {
            y[(i + 1) % 2] = ::exp(k * std::log((1.0 - x) / (1.0 - y[i % 2]))) * x;
            if (++i == MAX_ITERATIONS || ::fabs(y[1] - y[0]) < CONVERGENCE_TOLERANCE) {
                break;
            }
        }

        // Max sure y is supported by the p.d.f.
        if (y[i % 2] < support.first) {
            return truncate(sp.second ? safeCdfComplement(beta_, x) : safeCdf(beta_, x), 0.0, 1.0);
        }

        y[i % 2] = std::min(y[i % 2], sp.first);
    }

    double fx = safePdf(beta_, x);
    double fy = safePdf(beta_, y[i % 2]);
    LOG_TRACE("f(x) = " << fx << ", f(y) = " << fy);

    TDoubleDoublePr bracket(support);
    TDoubleDoublePr fBracket(0.0, 0.0);

    try {
        double error = sp.second ? fy - fx : fx - fy;
        if (::fabs(error) <= PDF_TOLERANCE * std::max(fx, fy)) {
            if (x > y[i % 2]) {
                std::swap(x, y[i % 2]);
            }
            return truncate(sp.second ? safeCdf(beta_, x) + safeCdfComplement(beta_, y[i % 2])
                                      : safeCdf(beta_, y[i % 2]) - safeCdf(beta_, x),
                            0.0,
                            1.0);
        } else if (error > 0.0) {
            if (x < sp.first) {
                bracket = std::make_pair(y[i % 2], bracket.second);
                double fa = fy - fx;
                double fb = safePdf(beta_, bracket.second) - fx;
                fBracket = std::make_pair(fa, fb);
            } else {
                bracket = std::make_pair(bracket.first, y[i % 2]);
                double fa = safePdf(beta_, bracket.first) - fx;
                double fb = fy - fx;
                fBracket = std::make_pair(fa, fb);
            }
        } else {
            bracket = std::make_pair(sp.first, y[i % 2]);
            double fa = safePdf(beta_, sp.first) - fx;
            double fb = fy - fx;
            fBracket = std::make_pair(fa, fb);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to evaluate p.d.f.: " << e.what() << ", alpha = " << beta_.alpha() << ", beta = " << beta_.beta() << ", x = " << x
                                                << ", y = " << y[i % 2]);
        return 1.0;
    }

    if (bracket.first > bracket.second) {
        std::swap(bracket.first, bracket.second);
        std::swap(fBracket.first, fBracket.second);
    }

    LOG_TRACE("Initial bracket = " << core::CContainerPrinter::print(bracket)
                                   << ", f(bracket) = " << core::CContainerPrinter::print(fBracket));

    try {
        double eps = 0.05 / fx;
        eps = std::max(eps, EPSILON * std::min(bracket.first, bracket.second));
        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, eps);
        std::size_t maxIterations = MAX_ITERATIONS;
        double candidate;
        CSolvers::solve(
            bracket.first, bracket.second, fBracket.first, fBracket.second, makePdf(beta_, fx), maxIterations, equal, candidate);

        LOG_TRACE("bracket = " << core::CContainerPrinter::print(bracket) << ", iterations = " << maxIterations
                               << ", f(candidate) = " << safePdf(beta_, candidate) - fx << ", eps = " << eps);

        if (::fabs(safePdf(beta_, candidate) - fx) < ::fabs(fy - fx)) {
            y[i % 2] = candidate;
        }
    } catch (const std::exception& e) {
        if (::fabs(fBracket.first - fx) < 10.0 * EPSILON * fx) {
            y[i % 2] = bracket.first;
        } else if (::fabs(fBracket.second - fx) < 10.0 * EPSILON * fx) {
            y[i % 2] = bracket.second;
        } else {
            LOG_ERROR("Failed in bracketed solver: " << e.what() << ", x = " << x << ", bracket " << core::CContainerPrinter::print(bracket)
                                                     << ", f(bracket) = " << core::CContainerPrinter::print(fBracket));
            return 1.0;
        }
    }

    if (x > y[i % 2]) {
        std::swap(x, y[i % 2]);
    }

    return truncate(
        sp.second ? safeCdf(beta_, x) + safeCdfComplement(beta_, y[i % 2]) : safeCdf(beta_, y[i % 2]) - safeCdf(beta_, x), 0.0, 1.0);
}

bool CTools::CProbabilityOfLessLikelySample::check(const TDoubleDoublePr& support, double x, double& px, maths_t::ETail& tail) const {
    if (CMathsFuncs::isNan(x)) {
        LOG_ERROR("Bad argument x = " << x);
        tail = maths_t::E_MixedOrNeitherTail;
        return false;
    } else if (x < support.first) {
        switch (m_Calculation) {
        case maths_t::E_OneSidedBelow:
        case maths_t::E_TwoSided:
            px = 0.0;
            break;
        case maths_t::E_OneSidedAbove:
            px = 1.0;
            break;
        }
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_LeftTail);
        return false;
    } else if (x > support.second) {
        switch (m_Calculation) {
        case maths_t::E_OneSidedBelow:
            px = 1.0;
            break;
        case maths_t::E_TwoSided:
        case maths_t::E_OneSidedAbove:
            px = 0.0;
            break;
        }
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
        return false;
    }
    return true;
}

void CTools::CProbabilityOfLessLikelySample::tail(double x, double mode, maths_t::ETail& tail) const {
    if (x <= mode) {
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_LeftTail);
    }
    if (x >= mode) {
        tail = static_cast<maths_t::ETail>(tail | maths_t::E_RightTail);
    }
}

//////// CMixtureProbabilityOfLessLikelySample Implementation ////////

CTools::CMixtureProbabilityOfLessLikelySample::CMixtureProbabilityOfLessLikelySample(std::size_t n,
                                                                                     double x,
                                                                                     double logFx,
                                                                                     double a,
                                                                                     double b)
    : m_X(x), m_LogFx(logFx), m_A(a), m_B(b) {
    m_Endpoints.reserve(4 * n + 2);
    m_Endpoints.push_back(a);
    m_Endpoints.push_back(b);
}

void CTools::CMixtureProbabilityOfLessLikelySample::reinitialize(double x, double logFx) {
    m_X = x;
    m_LogFx = logFx;
    m_Endpoints.clear();
    m_Endpoints.push_back(m_A);
    m_Endpoints.push_back(m_B);
}

void CTools::CMixtureProbabilityOfLessLikelySample::addMode(double weight, double modeMean, double modeSd) {
    double deviation = m_LogFx - fastLog(weight) + LOG_ROOT_TWO_PI + fastLog(modeSd);
    if (deviation >= 0.0) {
        deviation = 0.0;
        m_Endpoints.push_back(truncate(modeMean - 2.0 * modeSd, m_A, m_B));
        m_Endpoints.push_back(truncate(modeMean + 2.0 * modeSd, m_A, m_B));
    } else if (deviation >= -0.5) {
        deviation = std::sqrt(-2.0 * deviation);
        m_Endpoints.push_back(truncate(modeMean - (deviation + 2.0) * modeSd, m_A, m_B));
        m_Endpoints.push_back(truncate(modeMean, m_A, m_B));
        m_Endpoints.push_back(truncate(modeMean + (deviation + 2.0) * modeSd, m_A, m_B));
    } else {
        deviation = std::sqrt(-2.0 * deviation);
        m_Endpoints.push_back(truncate(modeMean - (deviation + 2.0) * modeSd, m_A, m_B));
        m_Endpoints.push_back(truncate(modeMean - (deviation - 1.0) * modeSd, m_A, m_B));
        m_Endpoints.push_back(truncate(modeMean + (deviation - 1.0) * modeSd, m_A, m_B));
        m_Endpoints.push_back(truncate(modeMean + (deviation + 2.0) * modeSd, m_A, m_B));
    }
    m_MaxDeviation.add((2.0 + deviation) * modeSd);
}

void CTools::CMixtureProbabilityOfLessLikelySample::intervals(TDoubleDoublePrVec& intervals) {
    std::sort(m_Endpoints.begin(), m_Endpoints.end());
    m_Endpoints.erase(std::unique(m_Endpoints.begin(), m_Endpoints.end()), m_Endpoints.end());
    intervals.reserve(m_Endpoints.size() - 1);
    for (std::size_t i = 1u; i < m_Endpoints.size(); ++i) {
        intervals.emplace_back(m_Endpoints[i - 1], m_Endpoints[i]);
    }
    LOG_TRACE("intervals = " << core::CContainerPrinter::print(intervals));
}

const double CTools::CMixtureProbabilityOfLessLikelySample::LOG_ROOT_TWO_PI = 0.5 * std::log(boost::math::double_constants::two_pi);

//////// SIntervalExpectation Implementation ////////

double CTools::SIntervalExpectation::operator()(const normal& normal_, double a, double b) const {
    if (a > b) {
        std::swap(a, b);
    }
    if (a == POS_INF) {
        return POS_INF;
    }

    double mean = normal_.mean();
    double sd = normal_.standard_deviation();
    double s = std::sqrt(2.0) * sd;
    double a_ = a == NEG_INF ? a : (a - mean) / s;
    double b_ = b == POS_INF ? b : (b - mean) / s;
    double expa = a_ == NEG_INF ? 0.0 : ::exp(-a_ * a_);
    double expb = b_ == POS_INF ? 0.0 : ::exp(-b_ * b_);
    double erfa = a_ == NEG_INF ? -1.0 : boost::math::erf(a_);
    double erfb = b_ == POS_INF ? 1.0 : boost::math::erf(b_);

    if (erfb - erfa < std::sqrt(EPSILON)) {
        return expa == expb ? (a + b) / 2.0 : (a * expa + b * expb) / (expa + expb);
    }

    return mean + 2.0 * sd * (expa - expb) / boost::math::double_constants::root_two_pi / (erfb - erfa);
}

double CTools::SIntervalExpectation::operator()(const lognormal& logNormal, double a, double b) const {
    if (a > b) {
        std::swap(a, b);
    }
    if (a == POS_INF) {
        return POS_INF;
    }
    if (b <= 0.0) {
        return 0.0;
    }

    double location = logNormal.location();
    double scale = logNormal.scale();
    double mean = boost::math::mean(logNormal);
    double loga = a <= 0.0 ? NEG_INF : std::log(a);
    double logb = b == POS_INF ? POS_INF : std::log(b);
    double c = location + scale * scale;
    double s = std::sqrt(2.0) * scale;
    double a_ = loga == NEG_INF ? NEG_INF : (loga - location) / s;
    double b_ = logb == POS_INF ? POS_INF : (logb - location) / s;
    double erfa = loga == NEG_INF ? -1.0 : boost::math::erf((loga - c) / s);
    double erfb = logb == POS_INF ? 1.0 : boost::math::erf((logb - c) / s);

    if (erfb - erfa < std::sqrt(EPSILON)) {
        double expa = loga == NEG_INF ? 0.0 : ::exp(-a_ * a_);
        double expb = logb == POS_INF ? 0.0 : ::exp(-b_ * b_);
        return expa == expb ? (2.0 * a / (a + b)) * b : (expa + expb) / (expa / a + expb / b);
    }

    double erfa_ = a_ == NEG_INF ? -1.0 : boost::math::erf(a_);
    double erfb_ = b_ == POS_INF ? 1.0 : boost::math::erf(b_);
    return mean * (erfb - erfa) / (erfb_ - erfa_);
}

double CTools::SIntervalExpectation::operator()(const gamma& gamma_, double a, double b) const {
    if (a > b) {
        std::swap(a, b);
    }
    if (a == POS_INF) {
        return POS_INF;
    }
    if (b <= 0.0) {
        return 0.0;
    }

    double shape = gamma_.shape();
    double rate = 1.0 / gamma_.scale();
    double mean = boost::math::mean(gamma_);
    double gama = a <= 0.0 ? 0.0 : boost::math::gamma_p(shape + 1.0, rate * a);
    double gamb = b == POS_INF ? 1.0 : boost::math::gamma_p(shape + 1.0, rate * b);

    if (gamb - gama < std::sqrt(EPSILON)) {
        double expa = a <= 0.0 ? 0.0 : ::exp((shape - 1.0) * std::log(a) - rate * a);
        double expb = b == POS_INF ? 0.0 : ::exp((shape - 1.0) * std::log(b) - rate * b);
        return (a * expa + b * expb) / (expa + expb);
    }

    double gama_ = a <= 0.0 ? 0.0 : boost::math::gamma_p(shape, rate * a);
    double gamb_ = b == POS_INF ? 1.0 : boost::math::gamma_p(shape, rate * b);
    return mean * (gamb - gama) / (gamb_ - gama_);
}

//////// smallestProbability Implementation ////////

double CTools::smallestProbability(void) {
    return MIN_DOUBLE;
}

//////// safePdf Implementation ////////

namespace {

namespace math_policy {
using namespace boost::math::policies;
using AllowOverflow = policy<overflow_error<user_error>>;
}

inline boost::math::normal_distribution<double, math_policy::AllowOverflow>
allowOverflow(const boost::math::normal_distribution<>& normal) {
    return boost::math::normal_distribution<double, math_policy::AllowOverflow>(normal.mean(), normal.standard_deviation());
}

inline boost::math::students_t_distribution<double, math_policy::AllowOverflow>
allowOverflow(const boost::math::students_t_distribution<>& students) {
    return boost::math::students_t_distribution<double, math_policy::AllowOverflow>(students.degrees_of_freedom());
}

inline boost::math::poisson_distribution<double, math_policy::AllowOverflow>
allowOverflow(const boost::math::poisson_distribution<>& poisson) {
    return boost::math::poisson_distribution<double, math_policy::AllowOverflow>(poisson.mean());
}

inline boost::math::negative_binomial_distribution<double, math_policy::AllowOverflow>
allowOverflow(const boost::math::negative_binomial_distribution<>& negativeBinomial) {
    return boost::math::negative_binomial_distribution<double, math_policy::AllowOverflow>(negativeBinomial.successes(),
                                                                                           negativeBinomial.success_fraction());
}

inline boost::math::lognormal_distribution<double, math_policy::AllowOverflow>
allowOverflow(const boost::math::lognormal_distribution<>& logNormal) {
    return boost::math::lognormal_distribution<double, math_policy::AllowOverflow>(logNormal.location(), logNormal.scale());
}

inline boost::math::gamma_distribution<double, math_policy::AllowOverflow> allowOverflow(const boost::math::gamma_distribution<>& gamma) {
    return boost::math::gamma_distribution<double, math_policy::AllowOverflow>(gamma.shape(), gamma.scale());
}

inline boost::math::beta_distribution<double, math_policy::AllowOverflow> allowOverflow(const boost::math::beta_distribution<>& beta) {
    return boost::math::beta_distribution<double, math_policy::AllowOverflow>(beta.alpha(), beta.beta());
}

inline boost::math::binomial_distribution<double, math_policy::AllowOverflow>
allowOverflow(const boost::math::binomial_distribution<>& binomial) {
    return boost::math::binomial_distribution<double, math_policy::AllowOverflow>(binomial.trials(), binomial.success_fraction());
}

inline boost::math::chi_squared_distribution<double, math_policy::AllowOverflow>
allowOverflow(const boost::math::chi_squared_distribution<>& chi2) {
    return boost::math::chi_squared_distribution<double, math_policy::AllowOverflow>(chi2.degrees_of_freedom());
}
}

double CTools::safePdf(const normal& normal_, double x) {
    return continuousSafePdf(allowOverflow(normal_), x);
}

double CTools::safePdf(const students_t& students, double x) {
    return continuousSafePdf(allowOverflow(students), x);
}

double CTools::safePdf(const poisson& poisson_, double x) {
    return discreteSafePdf(allowOverflow(poisson_), x);
}

double CTools::safePdf(const negative_binomial& negativeBinomial, double x) {
    return discreteSafePdf(allowOverflow(negativeBinomial), x);
}

double CTools::safePdf(const lognormal& logNormal, double x) {
    return continuousSafePdf(allowOverflow(logNormal), x);
}

double CTools::safePdf(const gamma& gamma_, double x) {
    TDoubleDoublePr support = boost::math::support(gamma_);

    // The distribution at the 0 is either:
    //   0,
    //   b ^ a / (a - 1)!,
    //   infinite
    //
    // depending on the shape parameter.

    if (x == support.first) {
        if (gamma_.shape() < 1.0) {
            return POS_INF;
        } else if (gamma_.shape() == 1.0) {
            return 1.0 / gamma_.scale();
        }
        return 0.0;
    }

    return continuousSafePdf(allowOverflow(gamma_), x);
}

double CTools::safePdf(const beta& beta_, double x) {
    TDoubleDoublePr support = boost::math::support(beta_);

    // The distribution is either:
    //   0,
    //   1,
    //   1 / B(a, b) or
    //   infinity
    //
    // at the end points depending on the end point in question
    // and the values of the parameters. We explicitly handle all
    // cases because for some combinations of a and b the error
    // introduced by using a constant continuation of the function
    // from eps and 1 - eps can be very large.

    if (x == support.first) {
        if (beta_.alpha() < 1.0) {
            return POS_INF;
        } else if (beta_.alpha() == 1.0) {
            return 1.0 / boost::math::beta(beta_.alpha(), beta_.beta());
        } else {
            return 0.0;
        }
    } else if (x == support.second) {
        if (beta_.beta() < 1.0) {
            return POS_INF;
        } else if (beta_.beta() == 1.0) {
            return 1.0 / boost::math::beta(beta_.alpha(), beta_.beta());
        } else {
            return 0.0;
        }
    }

    return continuousSafePdf(allowOverflow(beta_), x);
}

double CTools::safePdf(const binomial& binomial_, double x) {
    return discreteSafePdf(allowOverflow(binomial_), x);
}

double CTools::safePdf(const chi_squared& chi2, double x) {
    TDoubleDoublePr support = boost::math::support(chi2);

    // Depending on the degrees of freedom the pdf at zero is either:
    //   0,
    //   1/2,
    //   infinity
    //
    // at zero.

    double df = chi2.degrees_of_freedom();

    if (x == support.first) {
        if (df < 2.0) {
            return POS_INF;
        } else if (df == 2.0) {
            return 0.5;
        } else {
            return 0.0;
        }
    }

    return continuousSafePdf(allowOverflow(chi2), x);
}

//////// safeCdf Implementation ////////

double CTools::safeCdf(const normal& normal_, double x) {
    return continuousSafeCdf(allowOverflow(normal_), x);
}

double CTools::safeCdf(const students_t& students, double x) {
    return continuousSafeCdf(allowOverflow(students), x);
}

double CTools::safeCdf(const poisson& poisson_, double x) {
    return discreteSafeCdf(allowOverflow(poisson_), x);
}

double CTools::safeCdf(const negative_binomial& negativeBinomial, double x) {
    return discreteSafeCdf(allowOverflow(negativeBinomial), x);
}

double CTools::safeCdf(const lognormal& logNormal, double x) {
    return continuousSafeCdf(allowOverflow(logNormal), x);
}

double CTools::safeCdf(const gamma& gamma_, double x) {
    return continuousSafeCdf(allowOverflow(gamma_), x);
}

double CTools::safeCdf(const beta& beta_, double x) {
    return continuousSafeCdf(allowOverflow(beta_), x);
}

double CTools::safeCdf(const binomial& binomial_, double x) {
    return discreteSafeCdf(allowOverflow(binomial_), x);
}

double CTools::safeCdf(const chi_squared& chi2, double x) {
    return continuousSafeCdf(allowOverflow(chi2), x);
}

//////// safeCdfComplement Implementation ////////

double CTools::safeCdfComplement(const normal& normal_, double x) {
    return continuousSafeCdfComplement(allowOverflow(normal_), x);
}

double CTools::safeCdfComplement(const students_t& students, double x) {
    return continuousSafeCdfComplement(allowOverflow(students), x);
}

double CTools::safeCdfComplement(const poisson& poisson_, double x) {
    return discreteSafeCdfComplement(allowOverflow(poisson_), x);
}

double CTools::safeCdfComplement(const negative_binomial& negativeBinomial, double x) {
    return discreteSafeCdfComplement(allowOverflow(negativeBinomial), x);
}

double CTools::safeCdfComplement(const lognormal& logNormal, double x) {
    return continuousSafeCdfComplement(allowOverflow(logNormal), x);
}

double CTools::safeCdfComplement(const gamma& gamma_, double x) {
    return continuousSafeCdfComplement(allowOverflow(gamma_), x);
}

double CTools::safeCdfComplement(const beta& beta_, double x) {
    return continuousSafeCdfComplement(allowOverflow(beta_), x);
}

double CTools::safeCdfComplement(const binomial& binomial_, double x) {
    return discreteSafeCdfComplement(allowOverflow(binomial_), x);
}

double CTools::safeCdfComplement(const chi_squared& chi2, double x) {
    return continuousSafeCdfComplement(allowOverflow(chi2), x);
}

//////// deviation Implementation ////////

namespace {
const double SMALL_PROBABILITY_DEVIATION = 1.0;
const double MINUSCULE_PROBABILITY_DEVIATION = 50.0;
const double MAX_DEVIATION = 100.0;
const double INV_LARGEST_SIGNIFICANT_PROBABILITY = 1.0 / LARGEST_SIGNIFICANT_PROBABILITY;
const double INV_SMALL_PROBABILITY = 1.0 / SMALL_PROBABILITY;
const double MINUS_LOG_SMALL_PROBABILITY = -std::log(SMALL_PROBABILITY);
const double MINUS_LOG_MINUSCULE_PROBABILITY = -std::log(MINUSCULE_PROBABILITY);
}

double CTools::deviation(double p) {
    const double MINUS_LOG_SMALLEST_PROBABILITY = -std::log(smallestProbability());

    double result = 0.0;

    double adjP = std::max(p, smallestProbability());
    if (adjP < LARGEST_SIGNIFICANT_PROBABILITY) {
        if (adjP >= SMALL_PROBABILITY) {
            // We use a linear scaling based on the inverse probability
            // into the range (0.0, 1.0].
            result = SMALL_PROBABILITY_DEVIATION * (1.0 / adjP - INV_LARGEST_SIGNIFICANT_PROBABILITY) /
                     (INV_SMALL_PROBABILITY - INV_LARGEST_SIGNIFICANT_PROBABILITY);
        } else if (adjP >= MINUSCULE_PROBABILITY) {
            // We use a linear scaling based on the log probability into
            // the range (1.0, 50.0].
            result = SMALL_PROBABILITY_DEVIATION + (MINUSCULE_PROBABILITY_DEVIATION - SMALL_PROBABILITY_DEVIATION) *
                                                       (-std::log(adjP) - MINUS_LOG_SMALL_PROBABILITY) /
                                                       (MINUS_LOG_MINUSCULE_PROBABILITY - MINUS_LOG_SMALL_PROBABILITY);
        } else {
            // We use a linear scaling based on the log probability into
            // the range (50.0, 100.0].
            result = MINUSCULE_PROBABILITY_DEVIATION + (MAX_DEVIATION - MINUSCULE_PROBABILITY_DEVIATION) *
                                                           (-std::log(adjP) - MINUS_LOG_MINUSCULE_PROBABILITY) /
                                                           (MINUS_LOG_SMALLEST_PROBABILITY - MINUS_LOG_MINUSCULE_PROBABILITY);
        }
    }

    if (!(result >= 0.0 && result <= MAX_DEVIATION)) {
        LOG_ERROR("Deviation " << result << " out of range, p =" << p);
    }

    return result;
}

double CTools::inverseDeviation(double deviation) {
    const double MINUS_LOG_SMALLEST_PROBABILITY = -std::log(smallestProbability());

    double result = 0.0;

    double adjDeviation = truncate(deviation, 0.0, MAX_DEVIATION);
    if (adjDeviation == 0.0) {
        result = (1.0 + LARGEST_SIGNIFICANT_PROBABILITY) / 2.0;
    } else if (adjDeviation <= SMALL_PROBABILITY_DEVIATION) {
        // We invert the linear scaling of the inverse probability
        // into the range (0.0, 1.0].
        result = 1.0 / (INV_LARGEST_SIGNIFICANT_PROBABILITY +
                        (INV_SMALL_PROBABILITY - INV_LARGEST_SIGNIFICANT_PROBABILITY) * deviation / SMALL_PROBABILITY_DEVIATION);
    } else if (adjDeviation <= MINUSCULE_PROBABILITY_DEVIATION) {
        // We invert the linear scaling of the log probability
        // into the range (1.0, 50.0].
        result = ::exp(-(MINUS_LOG_SMALL_PROBABILITY + (MINUS_LOG_MINUSCULE_PROBABILITY - MINUS_LOG_SMALL_PROBABILITY) *
                                                           (deviation - SMALL_PROBABILITY_DEVIATION) /
                                                           (MINUSCULE_PROBABILITY_DEVIATION - SMALL_PROBABILITY_DEVIATION)));
    } else {
        // We invert the linear scaling of the log probability
        // into the range (50.0, 100.0].
        result = ::exp(-(MINUS_LOG_MINUSCULE_PROBABILITY + (MINUS_LOG_SMALLEST_PROBABILITY - MINUS_LOG_MINUSCULE_PROBABILITY) *
                                                               (deviation - MINUSCULE_PROBABILITY_DEVIATION) /
                                                               (MAX_DEVIATION - MINUSCULE_PROBABILITY_DEVIATION)));
    }

    if (!(result >= 0.0 && result <= 1.0)) {
        LOG_ERROR("Probability " << result << " out of range, deviation =" << deviation);
    }

    return result;
}

//////// differentialEntropy Implementation ////////

double CTools::differentialEntropy(const poisson& poisson_) {
    // Approximate as sum over [mean - 5 * std, mean + 5 * std].

    double mean = boost::math::mean(poisson_);
    double deviation = boost::math::standard_deviation(poisson_);

    unsigned int a = static_cast<unsigned int>(std::max(mean - 5.0 * deviation, 0.0));
    unsigned int b = static_cast<unsigned int>(std::max(mean + 5.0 * deviation, 5.0));

    double result = 0.0;

    for (unsigned int x = a; x <= b; ++x) {
        double pdf = safePdf(poisson_, x);
        result -= log(pdf) * pdf;
    }

    return result;
}

double CTools::differentialEntropy(const normal& normal_) {
    // Equals log(2 * pi * e * v) / 2
    //
    // where,
    //   m is the mean and variance of the normal distribution.

    double variance = boost::math::variance(normal_);
    return 0.5 * std::log(boost::math::double_constants::two_pi * boost::math::double_constants::e * variance);
}

double CTools::differentialEntropy(const lognormal& logNormal) {
    // Equals log(2 * pi * e * v) / 2 + m.
    //
    // where,
    //   m and v are the mean and variance of the exponentiated normal
    //   distribution, respectively.

    double location = logNormal.location();
    double scale = logNormal.scale();
    return 0.5 * std::log(boost::math::double_constants::two_pi * boost::math::double_constants::e * square(scale)) + location;
}

double CTools::differentialEntropy(const gamma& gamma_) {
    // Equals k + log(t) + log(g(k)) + (1 - k) * f(k)
    //
    // where,
    //   k and t are the shape and scale of the gamma distribution, respectively.
    //   g(.) is the gamma function and
    //   f(.) is the digamma function, i.e. the derivative of log gamma.

    double shape = gamma_.shape();
    double scale = gamma_.scale();
    return shape + std::log(scale) + boost::math::lgamma(shape) + (1 - shape) * boost::math::digamma(shape);
}

//////// CGroup Implementation ////////

void CTools::CGroup::merge(const CGroup& other, double separation, double min, double max) {
    m_A = std::min(m_A, other.m_A);
    m_B = std::max(m_B, other.m_B);

    // Update the centre and truncate so that the group
    // centres are in range.
    m_Centre += other.m_Centre;
    double l{this->leftEndpoint(separation)};
    double r{this->rightEndpoint(separation)};
    m_Centre.s_Moments[0] += std::max(min - l, 0.0);
    m_Centre.s_Moments[0] += std::min(max - r, 0.0);
}

bool CTools::CGroup::overlap(const CGroup& other, double separation) const {
    const double TOL{1.0 + EPSILON};
    double ll{this->leftEndpoint(separation)};
    double lr{this->rightEndpoint(separation)};
    double rl{other.leftEndpoint(separation)};
    double rr{other.rightEndpoint(separation)};
    return !(TOL * (lr + separation) <= rl || ll >= TOL * (rr + separation) || TOL * (rr + separation) <= ll ||
             rl >= TOL * (lr + separation));
}

double CTools::CGroup::leftEndpoint(double separation) const {
    return CBasicStatistics::mean(m_Centre) - static_cast<double>(m_B - m_A) * separation / 2.0;
}

double CTools::CGroup::rightEndpoint(double separation) const {
    return CBasicStatistics::mean(m_Centre) + static_cast<double>(m_B - m_A) * separation / 2.0;
}

const CTools::CLookupTableForFastLog<CTools::FAST_LOG_PRECISION> CTools::FAST_LOG_TABLE;

double CTools::shiftLeft(double x, double eps) {
    if (x == NEG_INF) {
        return x;
    }
    return (x < 0.0 ? 1.0 + eps : 1.0 - eps) * x;
}

double CTools::shiftRight(double x, double eps) {
    if (x == POS_INF) {
        return x;
    }
    return (x < 0.0 ? 1.0 - eps : 1.0 + eps) * x;
}
}
}
