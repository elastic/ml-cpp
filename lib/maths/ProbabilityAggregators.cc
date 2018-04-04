/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/ProbabilityAggregators.h>

#include <core/Constants.h>
#include <core/CPersistUtils.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CIntegration.h>
#include <maths/CTools.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>

namespace ml
{
namespace maths
{

namespace
{
using TDoubleVec = std::vector<double>;
using TDoubleDoublePr = std::pair<double, double>;

//! Compute \f$x^2\f$.
inline double square(double x)
{
    return x * x;
}

//! Compute the deviation corresponding to a probability of less likely
//! samples \p p.
//!
//! This function takes the probability of less likely event for a sample
//! \f$x(i)\f$, \f$P(R(i))\f$, which it converts to a deviation \f$z(i)\f$
//! using the relation:
//! <pre class="fragment">
//!   \f$\displaystyle \frac{1 + erf(-z(i))}{2} = \frac{P(R(i))}{2}\f$
//! </pre>
//!
//! Note we work with this version since we square \f$z\f$ anyway and it
//! avoids loss of precision which we'd get when subtracting \f$P(R(i))\f$
//! from 1. See CJointProbabilityOfLessLikelySamples::calculate for details
//! of how the \f$z(i)\f$ are used to compute the joint probability.
bool deviation(double p, double &result)
{
    try
    {
        boost::math::normal_distribution<> normal(0.0, 1.0);
        result = square(boost::math::quantile(normal, p / 2.0));
        return true;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Unable to compute quantile: " << e.what()
                  << ", probability = " << p);
    }
    return false;
}

const double EPS = 0.1;

//! A custom, numerically robust, implementation of \f$(1 - x) ^ p\f$.
//!
//! \note It is assumed that p is integer.
double powOneMinusX(double x, double p)
{
    // For large p,
    //   (1 - x) ^ p ~= exp(-p * x).
    //
    // and this doesn't suffer from cancellation errors in the limit
    // p -> inf and x -> 0. For p * x << 1 we get much better precision
    // using the Taylor expansion:
    //   (1 - x) ^ p = 1 - p * x + p * (p - 1) * x^2 / 2! + ...
    //
    // and canceling the leading terms.

    if (x == 1.0)
    {
        return 0.0;
    }
    if (p == 1.0)
    {
        return 1.0 - x;
    }

    double y = p * x;
    if (std::fabs(y) < EPS)
    {
        static const double COEFFS[] =
            {
                -1.0,
                +1.0 / 2.0,
                -1.0 / 6.0,
                +1.0 / 24.0,
                -1.0 / 120.0,
                +1.0 / 720.0
            };
        static const std::size_t N = boost::size(COEFFS);

        double remainder = 0.0;
        double ti = 1.0;
        for (std::size_t i = 0u; i < N && p != 0.0; ++i, p -= 1.0)
        {
            ti *= p * x;
            remainder += COEFFS[i] * ti;
        }
        return 1.0 + remainder;
    }
    else if (p > 1000.0)
    {
        return std::exp(-y);
    }

    if (x > 1.0)
    {
        double sign = static_cast<int>(p) % 2 ? -1.0 : 1.0;
        return sign * std::exp(p * std::log(x - 1.0));
    }

    return std::exp(p * std::log(1.0 - x));
}

//! A custom, numerically robust, implementation of \f$1 - (1 - x) ^ p\f$.
//!
//! \note It is assumed that p is integer.
double oneMinusPowOneMinusX(double x, double p)
{
    // For large p,
    //   (1 - x) ^ p ~= exp(-p * x).
    //
    // and this doesn't suffer from cancellation errors in the limit
    // p -> inf and x -> 0. For p * x << 1 we get much better precision
    // using the Taylor expansion:
    //   (1 - x) ^ p = 1 - p * x + p * (p - 1) * x^2 / 2! + ...
    //
    // Note that this doesn't make use of powOneMinusX because we can
    // avoid the cancellation errors by using:
    //   1 - (1 - x) ^ p = p * x - p * (p - 1) * x^2 / 2 + ...
    //
    // when p * x is small.

    if (x == 1.0)
    {
        return 1.0;
    }
    if (p == 1.0)
    {
        return x;
    }

    double y = p * x;
    if (std::fabs(y) < EPS)
    {
        static const double COEFFS[] =
            {
                +1.0,
                -1.0 / 2.0,
                +1.0 / 6.0,
                -1.0 / 24.0,
                +1.0 / 120.0,
                -1.0 / 720.0
            };
        static const std::size_t N = boost::size(COEFFS);

        double result = 0.0;

        double ti = 1.0;
        for (std::size_t i = 0u; i < N && p != 0.0; ++i, p -= 1.0)
        {
            ti *= p * x;
            result += COEFFS[i] * ti;
        }

        return result;
    }
    else if (p > 1000.0)
    {
        return 1.0 - std::exp(-y);
    }

    if (x > 1.0)
    {
        double sign = static_cast<int>(p) % 2 ? -1.0 : 1.0;
        return 1.0 - sign * std::exp(p * std::log(x - 1.0));
    }

    return 1.0 - std::exp(p * std::log(1.0 - x));
}

//! A custom implementation of \f$\log(1 - x)\f$ which handles the
//! cancellation error for small x.
double logOneMinusX(double x)
{
    double result = 0.0;

    if (std::fabs(x) < EPS)
    {
        double xi = -x;
        for (std::size_t i = 0u; i < 6; ++i, xi *= -x)
        {
            result += xi / static_cast<double>(i + 1);
        }
    }
    else
    {
        result = std::log(1.0 - x);
    }

    return result;
}

//! \brief Calculates the probability of the m most extreme samples.
//!
//! DESCRIPTION:\n
//! This calculates the probability of the \f$m\f$ most extreme samples having
//! probabilities:
//! <pre class="fragment">
//!   \f$\displaystyle \{P_i\ :\ i = 1, ..., m \}\f$
//! </pre class="fragment">
//! where,\n
//!   \f$\displaystyle P_m < P_{m-1} < ... < P_1\f$\n
//!   \f$N\f$ samples have been observed in total.
//!
//! This is equal to the multi-dimensional integral:
//! <pre class="fragment">
//! \f$\displaystyle \frac{N!}{(N-m)!}\int_{0}^{P_m}\int_{t_m}^{P_{m-1}}...\int_{t_2}^{P_1}(1-t_1)^{N-m}dt_1...dt_m\f$
//! </pre class="fragment">
class CNumericalLogProbabilityOfMFromNExtremeSamples
{
    public:
        using TMinValueAccumulator = CBasicStatistics::COrderStatisticsHeap<double>;

        //! A recursive integrand for the multi-variable integration.
        class CLogIntegrand
        {
            public:
                //! \param limits The upper limits of integration.
                //! \param n The total number of samples.
                //! \param m The number of extreme samples.
                //! \param i The variable being integrated, i.e. \f$t_i\f$.
                CLogIntegrand(const TDoubleVec &limits,
                              const TDoubleVec &corrections,
                              std::size_t n,
                              std::size_t m,
                              std::size_t i) :
                        m_Limits(&limits),
                        m_Corrections(&corrections),
                        m_N(n), m_M(m), m_I(i)
                {
                }

                //! Wrapper around evaluate which adapts it for CIntegration::gaussLegendre.
                bool operator()(double x, double &result) const
                {
                    result = this->evaluate(x);
                    return true;
                }

            private:
                //! Evaluate the i'th integral at \p x.
                double evaluate(double x) const
                {
                    if (m_I == m_M)
                    {
                        return static_cast<double>(m_N - m_M) * logOneMinusX(x);
                    }
                    double result;
                    CLogIntegrand f(*m_Limits, *m_Corrections, m_N, m_M, m_I + 1u);
                    CIntegration::logGaussLegendre<CIntegration::OrderThree>(f, x, (*m_Limits)[m_I], result);
                    result += (*m_Corrections)[m_I];
                    return result;
                }

                const TDoubleVec *m_Limits;
                const TDoubleVec *m_Corrections;
                std::size_t m_N;
                std::size_t m_M;
                std::size_t m_I;
        };

    public:
        //! The maximum integral dimension.
        static const std::size_t MAX_DIMENSION;

    public:
        //! \param p The probabilities (in sorted order).
        //! \param n The total number of samples.
        CNumericalLogProbabilityOfMFromNExtremeSamples(const TMinValueAccumulator &p,
                                                       std::size_t n) :
                m_N(n)
        {
            if (p.count() > 0)
            {
                // For large n the integral is dominated from the contributions
                // near the lowest probability.
                m_P.push_back(p[0]);
                m_Corrections.push_back(0.0);
                for (std::size_t i = 1u; i < std::min(p.count(), MAX_DIMENSION); ++i)
                {
                    m_P.push_back(truncate(p[i], m_P[i-1]));
                    m_Corrections.push_back(p[i] == p[i-1] ? 0.0 : std::log(p[i] - p[i-1]) - std::log(m_P[i] - m_P[i-1]));
                }
            }
        }

        //! Calculate the probability (by numerical integration).
        double calculate()
        {
            double result;
            CLogIntegrand f(m_P, m_Corrections, m_N, m_P.size(), 1u);
            CIntegration::logGaussLegendre<CIntegration::OrderThree>(f, 0, m_P[0], result);
            result +=  boost::math::lgamma(static_cast<double>(m_N) + 1.0)
                     - boost::math::lgamma(static_cast<double>(m_N - m_P.size()) + 1.0);
            return result;
        }

    private:
        double truncate(double p, double pMinus1) const
        {
            static const double CUTOFF[] =
                {
                    1.0e32, 1.0e16, 1.0e8, 1.0e4, 100.0
                };
            return std::min(p, (m_N >= boost::size(CUTOFF) ? 100.0 : CUTOFF[m_N]) * pMinus1);
        }

    private:
        TDoubleVec m_P;
        TDoubleVec m_Corrections;
        std::size_t m_N;
};

const std::size_t CNumericalLogProbabilityOfMFromNExtremeSamples::MAX_DIMENSION(10u);

const char DELIMITER(':');

} // unnamed::


//////// CJointProbabilityOfLessLikelySample Implementation ////////

CJointProbabilityOfLessLikelySamples::CJointProbabilityOfLessLikelySamples() :
        m_Distance(0.0), m_NumberSamples(0.0)
{
}

bool CJointProbabilityOfLessLikelySamples::fromDelimited(const std::string &value)
{
    core::CPersistUtils::CBuiltinFromString converter(DELIMITER);

    TDoubleDoublePr distanceAndNumberSamples;
    if (converter(value, distanceAndNumberSamples))
    {
        m_Distance = distanceAndNumberSamples.first;
        m_NumberSamples = distanceAndNumberSamples.second;
        return true;
    }

    double onlySample;
    if (converter(value, onlySample))
    {
        m_OnlyProbability.reset(onlySample);
        return true;
    }

    LOG_ERROR("Failed to initialize joint probability from " << value);

    return false;
}

std::string CJointProbabilityOfLessLikelySamples::toDelimited() const
{
    core::CPersistUtils::CBuiltinToString converter(DELIMITER);
    if (m_OnlyProbability)
    {
        return converter(*m_OnlyProbability);
    }
    TDoubleDoublePr distanceAndNumberSamples(m_Distance, m_NumberSamples);
    return converter(distanceAndNumberSamples);
}

const CJointProbabilityOfLessLikelySamples &
    CJointProbabilityOfLessLikelySamples::operator+=(const CJointProbabilityOfLessLikelySamples &other)
{
    if (m_NumberSamples == 0.0)
    {
        m_OnlyProbability = other.m_OnlyProbability;
    }
    else if (other.m_NumberSamples == 0.0)
    {
        // Nothing to do.
    }
    else if (m_OnlyProbability && other.m_OnlyProbability)
    {
        double d;
        if (deviation(*m_OnlyProbability, d))
        {
            m_Distance += d;
        }
        if (deviation(*other.m_OnlyProbability, d))
        {
            m_Distance += d;
        }
        m_OnlyProbability.reset();
    }
    else if (m_OnlyProbability)
    {
        double d;
        if (deviation(*m_OnlyProbability, d))
        {
            m_Distance += d;
        }
        m_Distance += other.m_Distance;
        m_OnlyProbability.reset();
    }
    else if (other.m_OnlyProbability)
    {
        double d;
        if (deviation(*other.m_OnlyProbability, d))
        {
            m_Distance += d;
        }
    }
    else
    {
        m_Distance += other.m_Distance;
    }
    m_NumberSamples += other.m_NumberSamples;
    return *this;
}

void CJointProbabilityOfLessLikelySamples::add(double probability, double weight)
{
    // Round up to epsilon to stop z overflowing in the case the probability
    // is very small.
    if (probability < CTools::smallestProbability())
    {
        probability = CTools::smallestProbability();
    }

    if (m_NumberSamples == 0.0 && weight == 1.0)
    {
        m_OnlyProbability = probability;
        m_NumberSamples = weight;
        return;
    }

    double d;
    if (m_OnlyProbability && deviation(*m_OnlyProbability, d))
    {
        m_Distance += d;
        m_OnlyProbability.reset();
    }
    if (deviation(probability, d))
    {
        m_Distance += d * weight;
        m_NumberSamples += weight;
    }
}

bool CJointProbabilityOfLessLikelySamples::calculate(double &result) const
{
    result = 1.0;

    // This is defined as one for the case there are no samples.
    if (m_OnlyProbability)
    {
        result = CTools::truncate(*m_OnlyProbability, 0.0, 1.0);
        return true;
    }

    // We use a small positive threshold on the distance because of overflow
    // in the method boost uses to compute the incomplete gamma function. The
    // result will be very close to one in this case anyway.
    if (m_NumberSamples == 0.0 || m_Distance / m_NumberSamples < 1e-8)
    {
        return true;
    }

    // We want to find the probability of seeing a more extreme collection
    // of independent samples {y(1), y(2), ... , y(n)} than a specified
    // collection {x(1), x(2), ... , x(n)}, where this is defined as the
    // probability of the set R:
    //   { {y(i)} : Product_j{ L(y(j)) } <= Product_j{ L(y(j)) } }.
    //
    // We will assume that y(i) ~ N(m(i), v(i)). In this case, it is possible
    // to show that:
    //   P(R) = gi(n/2, Sum_i{ z(i)^2 } / 2) / g(n/2)
    //
    // where,
    //   z(i) = (x(i) - m(i)) / v(i) ^ (1/2).
    //   gi(., .) is the upper incomplete gamma function.
    //   g(.) is the gamma function.

    try
    {
        result = boost::math::gamma_q(m_NumberSamples / 2.0, m_Distance / 2.0);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Unable to compute probability: " << e.what()
                  << ", m_NumberSamples = " << m_NumberSamples
                  << ", m_Distance = " << m_Distance);
        return false;
    }

    if (!(result >= 0.0 && result <= 1.0))
    {
        LOG_ERROR("Invalid joint probability = " << result
                  << ", m_NumberSamples = " << m_NumberSamples
                  << ", m_Distance = " << m_Distance);
    }

    result = CTools::truncate(result, 0.0, 1.0);

    return true;
}

bool CJointProbabilityOfLessLikelySamples::averageProbability(double &result) const
{
    result = 1.0;

    // This is defined as one for the case there are no samples.
    if (m_OnlyProbability)
    {
        result = CTools::truncate(*m_OnlyProbability, 0.0, 1.0);
        return true;
    }
    if (m_NumberSamples == 0.0 || m_Distance == 0.0)
    {
        return true;
    }

    // This is the constant probability p s.t. if we added n lots of p we'd
    // get the same joint probability and is a measurement of the typical
    // probability in a set of independent samples.

    try
    {
        boost::math::normal_distribution<> normal(0.0, 1.0);
        result = 2.0 * boost::math::cdf(normal, -std::sqrt(m_Distance / m_NumberSamples));
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Unable to compute probability: " << e.what()
                  << ", m_NumberSamples = " << m_NumberSamples
                  << ", m_Distance = " << m_Distance);
        return false;
    }

    if (!(result >= 0.0 && result <= 1.0))
    {
        LOG_ERROR("Invalid average probability = " << result
                  << ", m_NumberSamples = " << m_NumberSamples
                  << ", m_Distance = " << m_Distance);
    }

    result = CTools::truncate(result, 0.0, 1.0);

    return true;
}

CJointProbabilityOfLessLikelySamples::TOptionalDouble
    CJointProbabilityOfLessLikelySamples::onlyProbability() const
{
    return m_OnlyProbability;
}

double CJointProbabilityOfLessLikelySamples::distance() const
{
    return m_Distance;
}

double CJointProbabilityOfLessLikelySamples::numberSamples() const
{
    return m_NumberSamples;
}

uint64_t CJointProbabilityOfLessLikelySamples::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_OnlyProbability);
    seed = CChecksum::calculate(seed, m_Distance);
    return CChecksum::calculate(seed, m_NumberSamples);
}

std::ostream &CJointProbabilityOfLessLikelySamples::print(std::ostream &o) const
{
    return o << '(' << m_NumberSamples << ", " << m_Distance << ')';
}

std::ostream &operator<<(std::ostream &o,
                         const CJointProbabilityOfLessLikelySamples &probability)
{
    return probability.print(o);
}

CJointProbabilityOfLessLikelySamples &
    CJointProbabilityOfLessLikelySamples::SAddProbability::operator()(
            CJointProbabilityOfLessLikelySamples &jointProbability,
            const double probability,
            const double weight) const
{
    jointProbability.add(probability, weight);
    return jointProbability;
}


//////// CLogJointProbabilityOfLessLikelySample Implementation ////////

CLogJointProbabilityOfLessLikelySamples::CLogJointProbabilityOfLessLikelySamples()
{
}

const CLogJointProbabilityOfLessLikelySamples &
    CLogJointProbabilityOfLessLikelySamples::operator+=(const CLogJointProbabilityOfLessLikelySamples &other)
{
    this->CJointProbabilityOfLessLikelySamples::operator+=(other);
    return *this;
}

void CLogJointProbabilityOfLessLikelySamples::add(double probability, double weight)
{
    this->CJointProbabilityOfLessLikelySamples::add(probability, weight);
}

bool CLogJointProbabilityOfLessLikelySamples::calculateLowerBound(double &result) const
{
    result = 0.0;

    // This is defined as log(1) = 0 for the case there are no samples.
    if (this->onlyProbability())
    {
        result = std::min(std::log(*this->onlyProbability()), 0.0);
        return true;
    }
    if (this->numberSamples() == 0.0 || this->distance() == 0.0)
    {
        return true;
    }

    // We want to evaluate:
    //   log(P) = log(gi(n/2, Sum_i{ z(i)^2 } / 2)) - log(g(n/2))
    //
    // where,
    //   z(i) = (x(i) - m(i)) / v(i) ^ (1/2).
    //   gi(.,.) is the upper incomplete gamma function.
    //   g(.) is the gamma function.
    //
    // As Sum_i{ z(i)^2 / 2 } gets large this gets very small and we end up evaluating
    // a lower bound for the upper incomplete gamma function and hence a lower bound
    // for log(P). The upper incomplete gamma function is defined as:
    //   gi(s, x) = Int_{x}^{inf} t^(s-1) * exp(-t) dt
    //
    // Noting that:
    //   gi(s, x) = x^(s-1) * exp(-x) * Int_{x}^{inf}{ (t/x)^(s-1) * exp(-(t-x)) }dt
    //
    // and changing variables to t' = t - x, then:
    //   gi(s, x) >= x^(s-1) * exp(-x) * Int_{0}^{inf}{ (1 + t')^floor(s-1) * exp(-t') }dt'
    //
    // We can expand the sum and move it outside the integral to finally obtain:
    //   gi(s, x) >= x^(s-1) * exp(-x) * Sum_j{ p!/(p - j)! * 1/x^j }
    //
    // where p = floor(s-1) and the sum is from 0 to p.
    //
    // There are two bounds for Sum_j{ p!/(p - j)! * 1/x^j } available. One rewrites
    // the sum as a strictly smaller sum of a geometric progression using Stirling's
    // approximation and the other uses Lagrange's form of the remainder. First, note
    // that:
    //   Sum_j{ p!/(p - j)! * 1/x^j } = p! / x^p * Sum_j{ x^(p-j) / (p - j)! }
    //                                = p! / x^p * Sum_j{ x^j / j! }
    //
    // Now,
    //   Sum_j{ x^j / j! } = (Sum_j=0,m{ x^j / j! } + Sum_j=m+1,n{ x^j / j! })
    //                                                                      (1)
    //
    // Stirling's approximation gives that:
    //   j! <= e * j^(j+1/2) * exp(-j)
    //
    // So,
    //   Sum_j=0,r{ x^j / j! } >= 1/e Sum_j{ x^j / (j^(j+1/2) * exp(-j)) }
    //                         >= 1/e/r^(1/2) * Sum_j{ (e*x/r)^j }
    //                          = 1/e/r^(1/2) * (1 - (e*x/r)^(r+1)) / (1 - (e*x/r))
    //
    // Using this in (1) gives,
    //   Sum_j{ x^j / j! } >= 1/e/m^(1/2) * (1 - (e*x/m)^(m+1)) / (1 - e*x/m)
    //                        + exp(m)*x^(m+1)/p^(m+3/2) * (1 - (e*x/p)^(p-m)) / (1 - e*x/p)
    //                                                                      (2)
    //
    // We are free to maximize this w.r.t. m and we want this bound to be as
    // tight as possible in the limit x >> n since our other bound is good if
    // x < n. So simplifying (2) in this limit, setting differential of the
    // log to zero we get:
    //   -1/2m - log(m) + log(x) = 0
    //
    // which is nearly solved by m = x. So we use m = min(x, p) in (2), since
    // it must be less than or equal to p, for our first bound.
    //
    // For the second bound note that the l.h.s. of (1) is the Taylor expansion
    // of e^x at zero and Lagrange's form for the remainder is:
    //   R(x, p) = exp(z) * x^(p+1) / (p+1)!                    for z in [0, x]
    //
    // This implies that:
    //   Sum_j{ x^j / j! } >= (1 - x^(p+1)/(p+1)!) * exp(x)
    //
    // We take maximum of the two bounds.

    // If upper incomplete gamma function doesn't underflow use the "exact" value
    // (we want 1 d.p. of precision).
    double probability;
    if (this->calculate(probability) && probability > 10.0 * boost::numeric::bounds<double>::smallest())
    {
        LOG_TRACE("probability = " << probability);
        result = std::log(probability);
        return true;
    }

    static const double E = boost::math::double_constants::e;
    static const double LOG_DOUBLE_MAX = std::log(0.1 * boost::numeric::bounds<double>::highest());

    double s = this->numberSamples() / 2.0;
    double x = this->distance() / 2.0;

    double bound = boost::numeric::bounds<double>::lowest();

    try
    {
        double logx = std::log(x);
        double p = std::floor(s - 1.0);
        double logPFactorial = boost::math::lgamma(p + 1.0);
        double m = std::floor(std::min(x, p) + 0.5);
        double logm = std::log(m);

        double b1 = 0.0;

        if ((m + 1.0) * (1.0 + logx - logm) >= LOG_DOUBLE_MAX)
        {
            // Handle the case that (e*x/m)^(m+1) overflows.
            b1 = -1.0 - 0.5 * logm + m * (1.0 + logx - logm);
        }
        else if (E * x / m != 1.0)
        {
            double r = 1.0 - E * x / m;
            b1 = -1.0 - 0.5 * logm + std::log(oneMinusPowOneMinusX(r, m + 1.0) / r);
        }
        else
        {
            // Use L'Hopital's rule to show that:
            //   lim   { (1 - r^(m+1)) / (1 - r) } = m + 1
            //  r -> 1
            b1 = -1.0 - 0.5 * logm + std::log(m + 1.0);
        }

        if (p > m)
        {
            double t = 0.0;

            double logp = std::log(p);
            if ((p - m) * (1.0 + logx - logp) >= LOG_DOUBLE_MAX)
            {
                // Handle the case that (e*x/p)^(p-m) overflows.
                t = m + (m + 1.0) * logx - (m + 1.5) * logp
                    + (p - m - 1.0) * (1.0 + logx - logp);
            }
            else if (E * x / p != 1.0)
            {
                double r = 1.0 - E * x / p;
                t = m + (m + 1.0) * logx - (m + 1.5) * logp
                    + std::log(oneMinusPowOneMinusX(r, p - m) / r);
            }
            else
            {
                // Use L'Hopital's rule to show that:
                //   lim   { (1 - r^(p - m)) / (1 - r) } = p - m
                //  r -> 1
                t = m + (m + 1.0) * logx - (m + 1.5) * logp + std::log(p - m);
            }

            double normalizer = std::max(b1, t);
            b1 = normalizer + std::log(std::exp(b1 - normalizer) + std::exp(t - normalizer));
        }

        double b2 = 0.0;
        if ((p + 1.0) * std::log(x) < logPFactorial + std::log(p + 1.0))
        {
            b2 = std::log(1.0 - std::exp((p + 1.0) * logx - logPFactorial) / (p + 1.0)) + x;
        }

        double logSum = logPFactorial - p * logx + std::max(b1, b2);

        bound = (s - 1.0) * logx - x + logSum - boost::math::lgamma(s);

        LOG_TRACE("s = " << s << ", x = " << x
                  << ", p = " << p << ", m = " << m
                  << ", b1 = " << b1 << ", b2 = " << b2
                  << ", log(sum) = " << logSum
                  << ", bound = " << bound);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Failed computing bound: " << e.what()
                   << ", s = " << s << ", x = " << x);
    }

    result = std::min(bound, 0.0);
    LOG_TRACE("result = " << result);

    return true;
}

bool CLogJointProbabilityOfLessLikelySamples::calculateUpperBound(double &result) const
{
    result = 0.0;

    // This is defined as log(1) = 0 for the case there are no samples.
    if (this->onlyProbability())
    {
        result = std::min(std::log(*this->onlyProbability()), 0.0);
        return true;
    }
    if (this->numberSamples() == 0.0 || this->distance() == 0.0)
    {
        return true;
    }

    // We want to evaluate:
    //   log(P) = log(gi(n/2, Sum_i{ z(i)^2 } / 2)) - log(g(n/2))
    //
    // where,
    //   z(i) = (x(i) - m(i)) / v(i) ^ (1/2).
    //   gi(.,.) is the upper incomplete gamma function.
    //   g(.) is the gamma function.
    //
    // As Sum_i{ z(i)^2 / 2 } gets large this gets very small and we end up evaluating
    // an upper bound for the upper incomplete gamma function and hence an upper bound
    // for log(P). The upper incomplete gamma function is defined as:
    //   gi(s, x) = Int_{x}^{inf} t^(s-1) * exp(-t) dt
    //
    // Noting that:
    //   gi(s, x) = x^(s-1) * exp(-x) * Int_{x}^{inf}{ (t/x)^(s-1) * exp(-(t-x)) }dt
    //
    // and changing variables to t' = t - x, then:
    //   gi(s, x) <= x^(s-1) * exp(-x) * Int_{0}^{inf}{ (1 + t')^ceil(s-1) * exp(-t') }dt'
    //
    // We can expand the sum and move it outside the integral to finally obtain:
    //   gi(s, x) <= x^(s-1) * exp(-x) * Sum_j{ p!/(p - j)! * 1/x^j }
    //
    // where p = ceil(s-1) and the sum is from 0 to p.
    //
    // There are two bounds for Sum_j{ p!/(p - j)! * 1/x^j } available. One
    // rewrites the sum as a strictly greater sum of a geometric progression
    // and the other uses Lagrange's form of the remainder. In particular,
    //   Sum_j{ p!/(p - j)! * 1/x^j } <= Sum_j{ (p/x)^j }
    //                                 = (1 - (p/x)^(p+1)) / (1 - p/x)      (1)
    //
    // For the other bound note that we can rewrite the sum as follows:
    //   Sum_j{ p!(p - j)! * 1/x^j } = p! / x^p * Sum_j{ x^(p-j) / (p - j)! }
    //                               = p! / x^p * Sum_j{ x^j / j! }
    //
    // This is the Taylor expansion of exp(x) at zero and Lagrange's form for the
    // remainder is:
    //   R(x, p) = exp(z) * x^(p+1) / (p+1)!                    for z in [0, x]
    //           > 0
    //
    // It follows that:
    //   Sum_j{ p!/(p - j)! * 1/x^j } <= p! / x^p * exp(x).                 (2)
    //
    // Note that (1) is tight for p/x << 1 and (2) is tight for p/x >> 1. We take
    // the minimum. In general, the upper gamma function should only underflow
    // for p/x << 1.

    // If upper incomplete gamma function likely isn't going to underflow
    // use the "exact" value. Note that we want 1 d.p. of precision.
    double probability;
    if (this->calculate(probability) && probability > 10.0 * boost::numeric::bounds<double>::smallest())
    {
        LOG_TRACE("probability = " << probability);
        result = std::log(probability);
        return true;
    }

    static const double LOG_DOUBLE_MAX = std::log(0.10 * boost::numeric::bounds<double>::highest());

    double s = this->numberSamples() / 2.0;
    double x = this->distance() / 2.0;

    double bound = 0.0;

    try
    {
        double p = std::ceil(s - 1.0);

        double b1 = 0.0;
        if ((p + 1.0) * std::log(p / x) >= LOG_DOUBLE_MAX)
        {
            // Handle the case that (p/x)^(p+1) is going to overflow. In this case
            // (1 - (p/x)^(p+1)) / (1 - p/x) < (p/x)^(p+1) / (p/x - 1) but they are
            // essentially equal.
            b1 = (p + 1.0) * std::log(p / x) - std::log(p / x - 1.0);
        }
        else if (p != x)
        {
            double r = 1.0 - p / x;
            b1 = std::log(oneMinusPowOneMinusX(r, p + 1.0) / r);
        }
        else
        {
            // Use L'Hopital's rule to show that:
            //   lim   { (1 - r^(p+1)) / (1 - r) } = p + 1
            //  r -> 1
            b1 = std::log(p + 1);
        }

        double b2 = boost::math::lgamma(p + 1.0) - p * std::log(x) + x;

        double logSum = std::min(b1, b2);

        bound = (s - 1.0) * std::log(x) - x + logSum - boost::math::lgamma(s);

        LOG_TRACE("s = " << s << ", x = " << x
                  << ", b1 = " << b1 << ", b2 = " << b2
                  << ", log(sum) = " << logSum
                  << ", bound = " << bound);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Failed computing bound: " << e.what()
                   << ", s = " << s << ", x = " << x);
    }

    result = std::min(bound, 0.0);
    LOG_TRACE("result = " << result);

    return true;
}


//////// CProbabilityOfExtremeSample Implementation ////////

CProbabilityOfExtremeSample::CProbabilityOfExtremeSample() :
        m_NumberSamples(0.0)
{
}

bool CProbabilityOfExtremeSample::fromDelimited(const std::string &value)
{
    std::size_t i = value.find_first_of(DELIMITER);
    if (!core::CStringUtils::stringToType(value.substr(0, i), m_NumberSamples))
    {
        LOG_ERROR("Failed to extract number samples from " << value);
        return false;
    }
    return m_MinValue.fromDelimited(value.substr(i+1));
}

std::string CProbabilityOfExtremeSample::toDelimited() const
{
    return   core::CStringUtils::typeToString(m_NumberSamples)
           + DELIMITER
           + m_MinValue.toDelimited();
}

const CProbabilityOfExtremeSample &
    CProbabilityOfExtremeSample::operator+=(const CProbabilityOfExtremeSample &other)
{
    m_MinValue += other.m_MinValue;
    m_NumberSamples += other.m_NumberSamples;
    return *this;
}

bool CProbabilityOfExtremeSample::add(double probability, double weight)
{
    bool result = m_MinValue.add(probability);
    m_NumberSamples += weight;
    return result;
}

bool CProbabilityOfExtremeSample::calculate(double &result) const
{
    result = 1.0;
    if (m_NumberSamples > 0)
    {
        result = CTools::truncate(oneMinusPowOneMinusX(m_MinValue[0], m_NumberSamples), 0.0, 1.0);
    }
    return true;
}

uint64_t CProbabilityOfExtremeSample::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_MinValue);
    return CChecksum::calculate(seed, m_NumberSamples);
}

std::ostream &CProbabilityOfExtremeSample::print(std::ostream &o) const
{
    return o << "(" << m_NumberSamples << ", " << m_MinValue.print() << ")";
}

std::ostream &operator<<(std::ostream &o,
                         const CProbabilityOfExtremeSample &probability)
{
    return probability.print(o);
}


//////// CProbabilityOfMFromNMostExtremeSamples Implementation ////////

CLogProbabilityOfMFromNExtremeSamples::CLogProbabilityOfMFromNExtremeSamples(std::size_t m) :
        m_MinValues(m),
        m_NumberSamples(0u)
{
}

bool CLogProbabilityOfMFromNExtremeSamples::fromDelimited(const std::string &value)
{
    std::size_t i = value.find_first_of(DELIMITER);
    if (!core::CStringUtils::stringToType(value.substr(0, i), m_NumberSamples))
    {
        LOG_ERROR("Failed to extract number samples from " << value);
        return false;
    }
    return m_MinValues.fromDelimited(value.substr(i+1));
}

std::string CLogProbabilityOfMFromNExtremeSamples::toDelimited() const
{
    return   core::CStringUtils::typeToString(m_NumberSamples)
           + DELIMITER
           + m_MinValues.toDelimited();
}

const CLogProbabilityOfMFromNExtremeSamples &
    CLogProbabilityOfMFromNExtremeSamples::operator+=(const CLogProbabilityOfMFromNExtremeSamples &other)
{
    m_MinValues += other.m_MinValues;
    m_NumberSamples += other.m_NumberSamples;
    return *this;
}

void CLogProbabilityOfMFromNExtremeSamples::add(const double probability)
{
    m_MinValues.add(probability);
    ++m_NumberSamples;
}

bool CLogProbabilityOfMFromNExtremeSamples::calculate(double &result)
{
    result = 0.0;

    if (m_NumberSamples == 0)
    {
        return true;
    }

    // We can express the probability as the sum of two terms:
    //   1 - (1 - pM)^N +
    //   Sum_i( 2^M * N! * c(i, M) / (i+1) / (N-M)! * (1 - (1 - pM/2)^(i+1)) )
    //
    // It is possible to derive recurrence relations for the coefficients
    // c(i, M) as follows:
    //
    //   c(m+1)    = c(m) / 2 / (N - M + m)
    //   c(0, m+1) = -1 * ( c(M+1) * (1 - pm) ^ (N-M+m)
    //                     + Sum_i( c(i, m+1) * (1 - pm/2)^(i+1) ) )
    //   c(i, m+1) = c(i-1, m) / i
    //
    // where i > 0 in the last relation and c(0) = 1 and c(0, 0) = 0.
    //
    // The strategy in computing this quantity is to maintain the coefficients
    // {c(i, .)} in normalized form so that the largest is equal to 1 and to
    // keep track of a normalization factor, which is the log of the largest
    // coefficient, to avoid underflow. Note that since:
    //   c(m+1) = (N - M)!/(N - M + m + 1)!/2^m
    //
    // it can become very small! This means we always work to full precision.

    std::size_t M = m_MinValues.count();
    std::size_t N = m_NumberSamples;

    LOG_TRACE("M = " << M << ", N = " << N);

    double logc = 0.0;

    double logLargestCoeff = 0.0;
    TDoubleVec coeffs;
    if (M > 1)
    {
        coeffs.reserve(M - 1);
    }

    m_MinValues.sort();
    for (std::size_t i = 0u; i < M; ++i)
    {
        m_MinValues[i] = CTools::truncate(m_MinValues[i], CTools::smallestProbability(), 1.0);
    }

    for (std::size_t m = 1u; m < M; ++m)
    {
        double p = m_MinValues[M - m];
        LOG_TRACE("p(" << m << ") = " << p);

        logc -= std::log(2.0 * static_cast<double>(N - M + m));

        // Update the coefficients (they are stored in reverse order).
        double sum = 0.0;
        for (std::size_t i = 0u; i < coeffs.size(); ++i)
        {
            double index = static_cast<double>(coeffs.size() - i);
            coeffs[i] /= index;
            sum += coeffs[i] * powOneMinusX(p / 2.0, index);
        }
        LOG_TRACE("sum = " << sum);

        // Re-normalize the "c" term before adding. Note that somewhat
        // surprisingly 1 / logLargestCoeff can be infinity, because
        // small numbers lose precision before underflowing. That means
        // that the following calculation can't use the re-normalized
        // "c" directly because it might be infinite. Instead, we make
        // use the fact that c * (1 - p)^(N - M + m) won't overflow.
        double q = CTools::truncate(powOneMinusX(p, static_cast<double>(N - M + m)), 0.0, 1.0);
        coeffs.push_back(-sum - q * std::exp(logc - logLargestCoeff));
        LOG_TRACE("c(0) = " << coeffs.back());

        // Re-normalize the coefficients if they aren't all identically zero.
        double cmax = 0.0;
        for (std::size_t i = 0u; i < coeffs.size(); ++i)
        {
            if (std::fabs(coeffs[i]) > 1.0 / boost::numeric::bounds<double>::highest())
            {
                cmax = std::max(cmax, std::fabs(coeffs[i]));
            }
        }
        if (cmax > 0.0)
        {
            LOG_TRACE("cmax = " << cmax);
            for (std::size_t i = 0u; i < coeffs.size(); ++i)
            {
                coeffs[i] /= cmax;
            }
            logLargestCoeff += std::log(cmax);
            LOG_TRACE("logLargestCoeff = " << logLargestCoeff);
        }
    }

    // Re-normalize in the case that we haven't been able to in the loop
    // because of overflow.
    double cmax = 0.0;
    for (std::size_t i = 0u; i < coeffs.size(); ++i)
    {
        cmax = std::max(cmax, std::fabs(coeffs[i]));
    }
    if (cmax > 0.0 && cmax < 1.0 / boost::numeric::bounds<double>::highest())
    {
        logLargestCoeff = std::log(cmax);
        for (std::size_t i = 0u; i < coeffs.size(); ++i)
        {
            coeffs[i] /= cmax;
        }
    }
    LOG_TRACE("coeffs = " << core::CContainerPrinter::print(coeffs));

    double pM = m_MinValues[0];
    LOG_TRACE("p(" << M << ") = " << pM);

    double pMin = oneMinusPowOneMinusX(pM, static_cast<double>(N));
    LOG_TRACE("1 - (1 - p(" << M << "))^" << N << " = " << pMin);

    if (M > 1)
    {
        double logScale = static_cast<double>(M) * std::log(2.0)
                          + boost::math::lgamma(static_cast<double>(N + 1))
                          - boost::math::lgamma(static_cast<double>(N - M + 1))
                          + logLargestCoeff;
        LOG_TRACE("log(scale) = " << logScale);

        double sum = 0.0;
        double positive = 0.0;
        double negative = 0.0;
        TDoubleVec terms;
        terms.reserve(coeffs.size());
        for (std::size_t i = 0u; i < coeffs.size(); ++i)
        {
            double index = static_cast<double>(coeffs.size() - i);
            double c = coeffs[i] / index;
            double p = oneMinusPowOneMinusX(pM / 2.0, index);
            LOG_TRACE("term(" << index << ") = " << (c * p)
                      << " (c(" << index << ") = " << c
                      << ", 1 - (1 - p(M)/2)^" << index << " = " << p << ")");
            terms.push_back(c * p);
            sum += std::fabs(c * p);
            (c * p < 0.0 ? negative : positive) += std::fabs(c * p);
        }
        LOG_TRACE("negative = " << negative << ", positive = " << positive);

        if (sum == 0.0)
        {
            result = std::log(pMin);
        }
        else
        {
            // To minimize cancellation errors we add pMin inside the loop
            // and compute weights s.t. Sum_i( w(i) ) = 1.0 and w(i) * pMin
            // is roughly the same size as the i'th coefficient.

            static const double PRECISION = 1e6;

            result = 0.0;
            double condition = 0.0;
            double logPMin = std::log(pMin);
            if (logPMin - logScale > core::constants::LOG_MAX_DOUBLE)
            {
                for (std::size_t i = 0u; i < terms.size(); ++i)
                {
                    LOG_TRACE("remainder(" << i << ") = " << std::fabs(terms[i]));
                    result += std::fabs(terms[i]);
                }
                result = std::log(result * pMin / sum);
            }
            else
            {
                if (logPMin - logScale < core::constants::LOG_MIN_DOUBLE)
                {
                    pMin = 0.0;
                    for (std::size_t i = 0u; i < terms.size(); ++i)
                    {
                        result += terms[i];
                        condition = std::max(condition, std::fabs(terms[i]));
                    }
                }
                else
                {
                    pMin /= std::exp(logScale);
                    LOG_TRACE("pMin = " << pMin);
                    for (std::size_t i = 0u; i < terms.size(); ++i)
                    {
                        double remainder = std::fabs(terms[i]) * pMin / sum + terms[i];
                        result += remainder;
                        double absTerms[] = { std::fabs(terms[i]), std::fabs(terms[i] * pMin / sum), std::fabs(remainder) };
                        condition = std::max(condition, *std::max_element(absTerms, absTerms + 3));
                    }
                }

                LOG_TRACE("result = " << result << ", condition = " << condition);

                if (result <= 0.0 || condition > PRECISION * result)
                {
                    // Whoops we've lost all our precision. Fall back to numerical
                    // integration (note this caps M <= 10 so the runtime doesn't
                    // blow up).
                    LOG_TRACE("Falling back to numerical integration");
                    CNumericalLogProbabilityOfMFromNExtremeSamples numerical(m_MinValues, N);
                    result = numerical.calculate();
                }
                else
                {
                    result = logScale + std::log(result);
                }
            }
        }
    }
    else
    {
        result = std::log(pMin);
    }

    // Numerical error means the probability can be slightly greater
    // than one on occasion we use a tolerance which should be much
    // larger than necessary, but we are only interested in values
    // well outside the range as indicative of a genuine problem.
    for (std::size_t i = 0u; i < 2; ++i)
    {
        if (!(result < 0.001))
        {
            std::ostringstream minValues;
            minValues << std::setprecision(16) << "[" << m_MinValues[0];
            for (std::size_t j = 1u; j < m_MinValues.count(); ++j)
            {
                minValues << " " << m_MinValues[j];
            }
            minValues << "]";
            LOG_ERROR("Invalid log(extreme probability) = " << result
                      << ", m_NumberSamples = " << m_NumberSamples
                      << ", m_MinValues = " << minValues.str()
                      << ", coeffs = " << core::CContainerPrinter::print(coeffs)
                      << ", log(max{coeffs}) = " << logLargestCoeff
                      << ", pM = " << pM
                      << ", pMin = " << pMin);
            result = 0.0;
        }
        else
        {
            break;
        }
        LOG_TRACE("Falling back to numerical integration");
        CNumericalLogProbabilityOfMFromNExtremeSamples numerical(m_MinValues, N);
        result = numerical.calculate();
    }

    result = std::min(result, 0.0);
    LOG_TRACE("result = " << result);

    return true;
}

bool CLogProbabilityOfMFromNExtremeSamples::calibrated(double &result)
{
    // This probability systematically decreases for increasing min(M, N).
    // Ideally, we would like the probability to be calibrated, such that,
    // with probability P it is less than or equal to P for individual
    // probabilities computed by randomly sampling a matched distribution.
    // This is approximately achieved by scaling the log probability based
    // on Monte-Carlo analysis of the value of the function calculated as
    // a function of min(M, N). The following is a fit to the empirical
    // function.

    if (this->calculate(result))
    {
        std::size_t n = std::min(m_MinValues.count(), m_NumberSamples);
        if (n == 0)
        {
            return true;
        }
        result /= 1.0 + std::log(static_cast<double>(n)) / 2.1;
        return true;
    }

    return false;
}

uint64_t CLogProbabilityOfMFromNExtremeSamples::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_MinValues);
    return CChecksum::calculate(seed, m_NumberSamples);
}

}
}
