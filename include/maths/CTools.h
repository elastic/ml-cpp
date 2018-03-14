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

#ifndef INCLUDED_ml_maths_CTools_h
#define INCLUDED_ml_maths_CTools_h

#include <core/CoreTypes.h>
#include <core/CIEEE754.h>
#include <core/CNonInstantiatable.h>

#include <maths/CBasicStatistics.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/array.hpp>
#include <boost/math/distributions/fwd.hpp>
#include <boost/math/policies/policy.hpp>

#include <cmath>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <vector>

namespace ml {
namespace maths {
class CLogTDistribution;
template<typename T> class CMixtureDistribution;

//! \brief A collection of utility functionality.
//!
//! DESCRIPTION:\n
//! A collection of utility functions primarily intended for use within the
//! maths library.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This class is really just a proxy for a namespace, but a object has
//! been intentionally used to force a single point for the declaration
//! and definition of utility functions within the maths library. As such
//! all member functions should be static and it should be state-less.
//! If your functionality doesn't fit this pattern just make it a nested
//! class.
class MATHS_EXPORT CTools : private core::CNonInstantiatable {
    public:
        BOOST_MATH_DECLARE_DISTRIBUTIONS(double, boost::math::policies::policy<>)
        using TDoubleDoublePr = std::pair<double, double>;
        using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
        using TDoubleVec = std::vector<double>;

        //! The c.d.f. value for all x for an improper distribution.
        static const double IMPROPER_CDF;

        //! \brief A tag for an improper distribution, which is 0 everywhere.
        struct SImproperDistribution {};

        //! \brief Computes minus the log of the c.d.f. of a specified sample
        //! of an R.V. for various distributions.
        struct MATHS_EXPORT SMinusLogCdf {
            double operator()(const SImproperDistribution &, double x) const;
            double operator()(const normal &normal_, double x) const;
            double operator()(const students_t &students, double x) const;
            double operator()(const negative_binomial &negativeBinomial, double x) const;
            double operator()(const lognormal &logNormal, double x) const;
            double operator()(const CLogTDistribution &logt, double x) const;
            double operator()(const gamma &gamma_, double x) const;
            double operator()(const beta &beta_, double x) const;
        };

        //! \brief Computes minus the log of the 1 - c.d.f. of a specified
        //! sample of an R.V. for various distributions using full double
        //! precision, i.e. these do not lose precision when the result is
        //! close to 1 and the smallest value is the minimum double rather
        //! than epsilon.
        struct MATHS_EXPORT SMinusLogCdfComplement {
            double operator()(const SImproperDistribution &, double) const;
            double operator()(const normal &normal_, double x) const;
            double operator()(const students_t &students, double x) const;
            double operator()(const negative_binomial &negativeBinomial, double x) const;
            double operator()(const lognormal &logNormal, double x) const;
            double operator()(const CLogTDistribution &logt, double x) const;
            double operator()(const gamma &gamma_, double x) const;
            double operator()(const beta &beta_, double x) const;
        };

        //! \brief Computes the probability of seeing a more extreme sample
        //! of an R.V. for various distributions.
        //!
        //! The one sided below calculation computes the probability of the set:
        //! <pre class="fragment">
        //!   \f$\{y\ |\ y \leq x\}\f$
        //! </pre>
        //!
        //! and normalizes the result so that it equals one at the distribution
        //! median.
        //!
        //! The two sided calculation computes the probability of the set:
        //! <pre class="fragment">
        //!   \f$\{y\ |\ f(y) \leq f(x)\}\f$
        //! </pre>
        //!
        //! where,\n
        //!   \f$f(.)\f$ is the p.d.f. of the random variable.
        //!
        //! The one sided above calculation computes the probability of the set:
        //! <pre class="fragment">
        //!   \f$\{y\ |\ y \geq x\}\f$
        //! </pre>
        //!
        //! and normalizes the result so that it equals one at the distribution
        //! median.
        class MATHS_EXPORT CProbabilityOfLessLikelySample {
            public:
                CProbabilityOfLessLikelySample(maths_t::EProbabilityCalculation calculation);

                double operator()(const SImproperDistribution &, double, maths_t::ETail &tail) const;
                double operator()(const normal &normal_, double x, maths_t::ETail &tail) const;
                double operator()(const students_t &students, double x, maths_t::ETail &tail) const;
                double operator()(const negative_binomial &negativeBinomial, double x, maths_t::ETail &tail) const;
                double operator()(const lognormal &logNormal, double x, maths_t::ETail &tail) const;
                double operator()(const CLogTDistribution &logt, double x, maths_t::ETail &tail) const;
                double operator()(const gamma &gamma_, double x, maths_t::ETail &tail) const;
                double operator()(const beta &beta_, double x, maths_t::ETail &tail) const;

            private:
                //! Check the value is supported.
                bool check(const TDoubleDoublePr &support,
                           double x,
                           double &px,
                           maths_t::ETail &tail) const;

                //! Update the tail.
                void tail(double x,
                          double mode,
                          maths_t::ETail &tail) const;

                //! The style of calculation which, i.e. one or two tail.
                maths_t::EProbabilityCalculation m_Calculation;
        };

        //! \brief Computes the probability of seeing a more extreme sample
        //! from a mixture model.
        //!
        //! \sa CProbabilityOfLessLikelySample
        class MATHS_EXPORT CMixtureProbabilityOfLessLikelySample {
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
                        CSmoothedKernel(LOGF logf, double logF0, double k);

                        void k(double k);
                        bool operator()(double x, double &result) const;

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
                CMixtureProbabilityOfLessLikelySample(std::size_t n,
                                                      double x,
                                                      double logFx,
                                                      double a,
                                                      double b);

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
                bool leftTail(const LOGF &logf,
                              std::size_t iterations,
                              const EQUAL &equal,
                              double &result) const;

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
                bool rightTail(const LOGF &logf,
                               std::size_t iterations,
                               const EQUAL &equal,
                               double &result) const;

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
                double calculate(const LOGF &logf, double pTails);

            private:
                using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;

            private:
                static const double LOG_ROOT_TWO_PI;

            private:
                //! Compute the seed integration intervals.
                void intervals(TDoubleDoublePrVec &intervals);

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

        //! \brief Computes the expectation conditioned on a particular interval.
        //!
        //! DESCRIPTION:\n
        //! Computes the expectation of various R.V.s on the condition that the
        //! variable is in a specified interval. In particular, this is the
        //! quantity:
        //! <pre class="fragment">
        //!   \f$E[ X 1{[a,b]} ] / E[ 1{a,b]} ]\f$
        //! </pre>
        struct MATHS_EXPORT SIntervalExpectation {
            double operator()(const normal &normal_, double a, double b) const;
            double operator()(const lognormal &logNormal, double a, double b) const;
            double operator()(const gamma &gamma_, double a, double b) const;
        };

        //! The smallest value of probability we permit.
        //!
        //! This is used to stop calculations under/overflowing if we
        //! allow the probability to be zero (for example).
        static double smallestProbability(void);

        //! \name Safe Probability Density Function
        //! Unfortunately, boost::math::pdf and boost::math::cdf don't
        //! handle values outside of the distribution support very well.
        //! By default they throw and if you suppress this behaviour
        //! they return 0.0 for the cdf! This wraps up the pdf and cdf
        //! calls and does the appropriate checking. The functions are
        //! extended to the whole real line in the usual way by treating
        //! them as continuous.
        //@{
        static double safePdf(const normal &normal_, double x);
        static double safePdf(const students_t &students, double x);
        static double safePdf(const poisson &poisson_, double x);
        static double safePdf(const negative_binomial &negativeBinomial, double x);
        static double safePdf(const lognormal &logNormal, double x);
        static double safePdf(const gamma &gamma_, double x);
        static double safePdf(const beta &beta_, double x);
        static double safePdf(const binomial &binomial_, double x);
        static double safePdf(const chi_squared &chi2, double x);
        //@}

        //! \name Safe Cumulative Density Function
        //! Wrappers around the boost::math::cdf functions which extend
        //! them to the whole real line.
        //! \see safePdf for details.
        //@{
        static double safeCdf(const normal &normal_, double x);
        static double safeCdf(const students_t &students, double x);
        static double safeCdf(const poisson &poisson_, double x);
        static double safeCdf(const negative_binomial &negativeBinomial, double x);
        static double safeCdf(const lognormal &logNormal, double x);
        static double safeCdf(const gamma &gamma_, double x);
        static double safeCdf(const beta &beta_, double x);
        static double safeCdf(const binomial &binomial_, double x);
        static double safeCdf(const chi_squared &chi2, double x);
        //@}

        //! \name Safe Cumulative Density Function Complement
        //! Wrappers around the boost::math::cdf functions for complement
        //! distributions which extend them to the whole real line.
        //! \see safePdf for details.
        //@{
        static double safeCdfComplement(const normal &normal_, double x);
        static double safeCdfComplement(const students_t &students, double x);
        static double safeCdfComplement(const poisson &poisson_, double x);
        static double safeCdfComplement(const negative_binomial &negativeBinomial, double x);
        static double safeCdfComplement(const lognormal &logNormal, double x);
        static double safeCdfComplement(const gamma &gamma_, double x);
        static double safeCdfComplement(const beta &beta_, double x);
        static double safeCdfComplement(const binomial &binomial_, double x);
        static double safeCdfComplement(const chi_squared &chi2, double x);
        //@}

        //! Compute the deviation from the probability of seeing a more
        //! extreme event for a distribution, i.e. for a sample \f$x\f$
        //! from a R.V. the probability \f$P(R)\f$ of the set:
        //! <pre class="fragment">
        //!   \f$ R = \{y\ |\ f(y) \leq f(x)\} \f$
        //! </pre>
        //! where,\n
        //!   \f$f(.)\f$ is the p.d.f. of the random variable.\n\n
        //! This is a monotonically decreasing function of \f$P(R)\f$ and
        //! is chosen so that for \f$P(R)\f$ near one it is zero and as
        //! \f$P(R) \rightarrow 0\f$ it saturates at 100.
        static double deviation(double p);

        //! The inverse of the deviation function.
        static double inverseDeviation(double deviation);

        //! \name Differential Entropy
        //! Compute the differential entropy of the specified distribution.\n\n
        //! The differential entropy of an R.V. is defined as:
        //! <pre class="fragment">
        //!   \f$ -E[\log(f(x))] \f$
        //! </pre>
        //! where,\n
        //!   \f$f(x)\f$ is the probability density function.\n\n
        //! This computes the differential entropy in units of "nats",
        //! i.e. the logarithm is the natural logarithm.
        //@{
        static double differentialEntropy(const poisson &poisson_);
        static double differentialEntropy(const normal &normal_);
        static double differentialEntropy(const lognormal &logNormal);
        static double differentialEntropy(const gamma &gamma_);
        template<typename T>
        class CDifferentialEntropyKernel {
            public:
                CDifferentialEntropyKernel(const CMixtureDistribution<T> &mixture) :
                    m_Mixture(&mixture) {
                }

                inline bool operator()(double x, double &result) const {
                    double                           fx = pdf(*m_Mixture, x);
                    result = fx == 0.0 ? 0.0 : -fx * std::log(fx);
                    return true;
                }

            private:
                const CMixtureDistribution<T> *m_Mixture;
        };
        template<typename T>
        static double differentialEntropy(const CMixtureDistribution<T> &mixture);
        //@}

        //! Check if \p log will underflow the smallest positive value of T.
        //!
        //! \tparam T must be a floating point type.
        template<typename T>
        static bool logWillUnderflow(T log) {
            static const T LOG_DENORM_MIN = std::log(std::numeric_limits<T>::min());
            return log < LOG_DENORM_MIN;
        }

    //! \name Fast Log
    private:
        //! The precision to use for fastLog, which gives good runtime
        //! accuracy tradeoff.
        static const int FAST_LOG_PRECISION = 14;

        //! Shift used to index the lookup table in fastLog.
        static const std::size_t FAST_LOG_SHIFT = 52 - FAST_LOG_PRECISION;

        //! \brief Creates a lookup table for log2(x) with specified
        //! accuracy.
        //!
        //! DESCRIPTION:\n
        //! This implements a singleton lookup table for all values
        //! of log base 2 of x for the mantissa of x in the range
        //! [0, 2^52-1]. The specified accuracy, \p N, determines the
        //! size of the lookup table, and values are equally spaced,
        //! i.e. the separation is 2^52 / 2^N. This is used by fastLog
        //! to read off the log base 2 to the specified precision.
        //!
        //! This is taken from the approach given in
        //! http://www.icsi.berkeley.edu/pubs/techreports/TR-07-002.pdf
        template<int BITS>
        class CLookupTableForFastLog {
            public:
                static const std::size_t BINS = 1 << BITS;

            public:
                using TArray = boost::array<double, BINS>;

            public:
                //! Builds the table.
                CLookupTableForFastLog(void) {
                    // Notes:
                    //   1) The shift is the maximum mantissa / BINS.
                    //   2) The sign bit is set to 0 which is positive.
                    //   3) The exponent is set to 1022, which is 0 in two's
                    //      complement.
                    //   4) This implementation is endian neutral because it
                    //      is constructing a look up from the mantissa value
                    //      (interpreted as an integer) to the corresponding
                    //      double value and fastLog uses the same approach
                    //      to extract the mantissa.
                    uint64_t                   dx = 0x10000000000000ull / BINS;
                    core::CIEEE754::SDoubleRep x;
                    x.s_Sign = 0;
                    x.s_Mantissa = (dx / 2) & core::CIEEE754::IEEE754_MANTISSA_MASK;
                    x.s_Exponent = 1022;
                    for (std::size_t i = 0u; i < BINS; ++i,
                         x.s_Mantissa = (x.s_Mantissa + dx) & core::CIEEE754::IEEE754_MANTISSA_MASK) {
                        double value;
                        static_assert(sizeof(double) == sizeof(core::CIEEE754::SDoubleRep),
                                      "SDoubleRep definition unsuitable for memcpy to double");
                        // Use memcpy() rather than union to adhere to strict
                        // aliasing rules
                        std::memcpy(&value, &x, sizeof(double));
                        m_Table[i] = std::log2(value);
                    }
                }

                //! Lookup log2 for a given mantissa.
                const double &operator[](uint64_t mantissa) const {
                    return m_Table[mantissa >> FAST_LOG_SHIFT];
                }

            private:
                //! The quantized log base 2 for the mantissa range.
                TArray m_Table;
        };

        //! The table used for computing fast log.
        static const CLookupTableForFastLog<FAST_LOG_PRECISION> FAST_LOG_TABLE;

    public:
        //! Approximate implementation of log(\p x), which is accurate
        //! to FAST_LOG_PRECISION bits of precision.
        //!
        //! \param[in] x The value for which to compute the natural log.
        //! \note This is taken from the approach given in
        //! http://www.icsi.berkeley.edu/pubs/techreports/TR-07-002.pdf
        static double fastLog(double x) {
            uint64_t mantissa;
            int      log2;
            core::CIEEE754::decompose(x, mantissa, log2);
            return 0.693147180559945 * (FAST_LOG_TABLE[mantissa] + log2);
        }
    //@}

    private:
        //! Get the location of the point \p x.
        template<typename T>
        static double location(T x) {
            return x;
        }
        //! Set \p x to \p y.
        template<typename T>
        static void setLocation(T &x, double y) {
            x = static_cast<T>(y);
        }
        //! Get a writable location of the point \p x.
        template<typename T>
        static double location(const typename CBasicStatistics::SSampleMean<T>::TAccumulator &x) {
            return CBasicStatistics::mean(x);
        }
        //! Set the mean of \p x to \p y.
        template<typename T>
        static void setLocation(typename CBasicStatistics::SSampleMean<T>::TAccumulator &x, double y) {
            x.s_Moments[0] = static_cast<T>(y);
        }

        //! \brief Utility class to represent points which are adjacent
        //! in the spreading algorithm.
        class MATHS_EXPORT CGroup {
            public:
                using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

            public:
                //! Create a new points group.
                template<typename T>
                CGroup(std::size_t index, const T &points) :
                    m_A(index),
                    m_B(index),
                    m_Centre() {
                    m_Centre.add(location(points[index]));
                }

                //! Merge this group and \p other group.
                void merge(const CGroup &other,
                           double separation,
                           double min,
                           double max);

                //! Check if this group and \p other group overlap.
                bool overlap(const CGroup &other,
                             double separation) const;

                //! Update the locations of the points in this group based
                //! on its centre position.
                template<typename T>
                bool spread(double separation, T &points) const {
                    if (m_A == m_B) {
                        return false;
                    }
                    bool   result = false;
                    double x = this->leftEndpoint(separation);
                    for (std::size_t i = m_A; i <= m_B; ++i, x += separation) {
                        if (location(points[i]) != x) {
                            setLocation(points[i], x);
                            result = true;
                        }
                    }
                    return result;
                }

            private:
                //! Get the position of the left end point of this group.
                double leftEndpoint(double separation) const;

                //! Get the position of the right end point of this group.
                double rightEndpoint(double separation) const;

                std::size_t m_A;
                std::size_t m_B;
                TMeanAccumulator m_Centre;
        };

        //! \brief Orders two points by their position.
        class CPointLess {
            public:
                template<typename T>
                bool operator()(const T &lhs, const T &rhs) const {
                    return location(lhs) < location(rhs);
                }
        };

    public:
        //! \brief Ensure the points are at least \p separation apart.\n\n
        //! This solves the problem of finding the new positions for the
        //! points \f$\{x_i\}\f$ such that there is no pair of points for
        //! which \f$\left \|x_j - x_i \right \| < s\f$ where \f$s\f$
        //! denotes the minimum separation \p separation and the total
        //! square distance the points move, i.e.
        //! <pre class="fragment">
        //!   \f$ \sum_i{(x_i' - x_i)^2} \f$
        //! </pre>
        //! is minimized.
        //!
        //! \param[in] a The left end point of the interval containing
        //! the shifted points.
        //! \param[in] b The right end point of the interval containing
        //! the shifted points.
        //! \param[in] separation The minimum permitted separation between
        //! points.
        //! \param[in,out] points The points to spread.
        template<typename T>
        static void spread(double a, double b, double separation, T &points);

        //! Compute the sign of \p x and return T(-1) if it is negative and T(1)
        //! otherwise.
        //!
        //! \param[in] x The value for which to check the sign.
        //! \note Conversion of 0 and -1 to T should be well defined.
        //! \note Zero maps to 1.
        template<typename T>
        static T sign(const T &x) {
            return x < T(0) ? T(-1) : T(1);
        }

        //! Truncate \p x to the range [\p a, \p b].
        //!
        //! \tparam T Must support operator<.
        template<typename T>
        static const T &truncate(const T &x, const T &a, const T &b) {
            return x < a ? a : (b < x ? b : x);
        }

        //! Component-wise truncation of stack vectors.
        template<typename T, std::size_t N>
        static CVectorNx1<T, N> truncate(const CVectorNx1<T, N> &x,
                                         const CVectorNx1<T, N> &a,
                                         const CVectorNx1<T, N> &b) {
            CVectorNx1<T, N> result(x);
            for (std::size_t i = 0u; i < N; ++i) {
                result(i) = truncate(result(i), a(i), b(i));
            }
            return result;
        }

        //! Component-wise truncation of heap vectors.
        template<typename T>
        static CVector<T> truncate(const CVector<T> &x,
                                   const CVector<T> &a,
                                   const CVector<T> &b) {
            CVector<T> result(x);
            for (std::size_t i = 0u; i < result.dimension(); ++i) {
                result(i) = truncate(result(i), a(i), b(i));
            }
            return result;
        }

        //! Component-wise truncation of small vector.
        template<typename T, std::size_t N>
        static core::CSmallVector<T, N> truncate(const core::CSmallVector<T, N> &x,
                                                 const core::CSmallVector<T, N> &a,
                                                 const core::CSmallVector<T, N> &b) {
            core::CSmallVector<T, N> result(x);
            for (std::size_t i = 0u; i < result.size(); ++i) {
                result[i] = truncate(result[i], a[i], b[i]);
            }
            return result;
        }

        //! Shift \p x to the left by \p eps times \p x.
        static double shiftLeft(double x, double eps = std::numeric_limits<double>::epsilon());

        //! Shift \p x to the right by \p eps times \p x.
        static double shiftRight(double x, double eps = std::numeric_limits<double>::epsilon());

        //! Sigmoid function of \p p.
        static double sigmoid(double p) {
            return 1.0 / (1.0 + 1.0 / p);
        }

        //! A smooth Heaviside function centred at one.
        //!
        //! This is a smooth version of the Heaviside function implemented
        //! as \f$sigmoid\left(\frac{sign (x - 1)}{wb}\right)\f$ normalized
        //! to the range [0, 1], where \f$b\f$ is \p boundary and \f$w\f$
        //! is \p width. Note, if \p sign is one this is a step up and if
        //! it is -1 it is a step down.
        //!
        //! \param[in] x The argument.
        //! \param[in] width The step width.
        //! \param[in] sign Determines whether it's a step up or down.
        static double smoothHeaviside(double x, double width, double sign = 1.0) {
            return sigmoid(std::exp(sign * (x - 1.0) / width))
                   / sigmoid(std::exp(1.0 / width));
        }
};

}
}

#endif // INCLUDED_ml_maths_CTools_h
