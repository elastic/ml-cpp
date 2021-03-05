/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTools_h
#define INCLUDED_ml_maths_CTools_h

#include <core/CIEEE754.h>
#include <core/CNonInstantiatable.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebraFwd.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/math/distributions/fwd.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/optional.hpp>
#include <boost/static_assert.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <numeric>
#include <vector>

namespace ml {
namespace maths {
class CLogTDistribution;
template<typename T>
class CMixtureDistribution;

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
        double operator()(const SImproperDistribution&, double x) const;
        double operator()(const normal& normal_, double x) const;
        double operator()(const students_t& students, double x) const;
        double operator()(const negative_binomial& negativeBinomial, double x) const;
        double operator()(const lognormal& logNormal, double x) const;
        double operator()(const CLogTDistribution& logt, double x) const;
        double operator()(const gamma& gamma_, double x) const;
        double operator()(const beta& beta_, double x) const;
    };

    //! \brief Computes minus the log of the 1 - c.d.f. of a specified
    //! sample of an R.V. for various distributions using full double
    //! precision, i.e. these do not lose precision when the result is
    //! close to 1 and the smallest value is the minimum double rather
    //! than epsilon.
    struct MATHS_EXPORT SMinusLogCdfComplement {
        double operator()(const SImproperDistribution&, double) const;
        double operator()(const normal& normal_, double x) const;
        double operator()(const students_t& students, double x) const;
        double operator()(const negative_binomial& negativeBinomial, double x) const;
        double operator()(const lognormal& logNormal, double x) const;
        double operator()(const CLogTDistribution& logt, double x) const;
        double operator()(const gamma& gamma_, double x) const;
        double operator()(const beta& beta_, double x) const;
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

        double operator()(const SImproperDistribution&, double, maths_t::ETail& tail) const;
        double operator()(const normal& normal_, double x, maths_t::ETail& tail) const;
        double operator()(const students_t& students, double x, maths_t::ETail& tail) const;
        double operator()(const negative_binomial& negativeBinomial,
                          double x,
                          maths_t::ETail& tail) const;
        double operator()(const lognormal& logNormal, double x, maths_t::ETail& tail) const;
        double operator()(const CLogTDistribution& logt, double x, maths_t::ETail& tail) const;
        double operator()(const gamma& gamma_, double x, maths_t::ETail& tail) const;
        double operator()(const beta& beta_, double x, maths_t::ETail& tail) const;

    private:
        //! Check the value is supported.
        bool check(const TDoubleDoublePr& support, double x, double& px, maths_t::ETail& tail) const;

        //! Update the tail.
        void tail(double x, double mode, maths_t::ETail& tail) const;

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
            bool operator()(double x, double& result) const;

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
        bool leftTail(const LOGF& logf, std::size_t iterations, const EQUAL& equal, double& result) const;

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
        bool rightTail(const LOGF& logf, std::size_t iterations, const EQUAL& equal, double& result) const;

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
        double calculate(const LOGF& logf, double pTails);

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
        double operator()(const normal& normal_, double a, double b) const;
        double operator()(const lognormal& logNormal, double a, double b) const;
        double operator()(const gamma& gamma_, double a, double b) const;
    };

    //! The smallest value of probability we permit.
    //!
    //! This is used to stop calculations under/overflowing if we
    //! allow the probability to be zero (for example).
    static double smallestProbability();

    //! \name Safe Probability Density Function
    //! Unfortunately, boost::math::pdf and boost::math::cdf don't
    //! handle values outside of the distribution support very well.
    //! By default they throw and if you suppress this behaviour
    //! they return 0.0 for the cdf! This wraps up the pdf and cdf
    //! calls and does the appropriate checking. The functions are
    //! extended to the whole real line in the usual way by treating
    //! them as continuous.
    //@{
    static double safePdf(const normal& normal_, double x);
    static double safePdf(const students_t& students, double x);
    static double safePdf(const poisson& poisson_, double x);
    static double safePdf(const negative_binomial& negativeBinomial, double x);
    static double safePdf(const lognormal& logNormal, double x);
    static double safePdf(const gamma& gamma_, double x);
    static double safePdf(const beta& beta_, double x);
    static double safePdf(const binomial& binomial_, double x);
    static double safePdf(const chi_squared& chi2, double x);
    //@}

    //! \name Safe Cumulative Density Function
    //! Wrappers around the boost::math::cdf functions which extend
    //! them to the whole real line.
    //! \see safePdf for details.
    //@{
    static double safeCdf(const normal& normal_, double x);
    static double safeCdf(const students_t& students, double x);
    static double safeCdf(const poisson& poisson_, double x);
    static double safeCdf(const negative_binomial& negativeBinomial, double x);
    static double safeCdf(const lognormal& logNormal, double x);
    static double safeCdf(const gamma& gamma_, double x);
    static double safeCdf(const beta& beta_, double x);
    static double safeCdf(const binomial& binomial_, double x);
    static double safeCdf(const chi_squared& chi2, double x);
    //@}

    //! \name Safe Cumulative Density Function Complement
    //! Wrappers around the boost::math::cdf functions for complement
    //! distributions which extend them to the whole real line.
    //! \see safePdf for details.
    //@{
    static double safeCdfComplement(const normal& normal_, double x);
    static double safeCdfComplement(const students_t& students, double x);
    static double safeCdfComplement(const poisson& poisson_, double x);
    static double safeCdfComplement(const negative_binomial& negativeBinomial, double x);
    static double safeCdfComplement(const lognormal& logNormal, double x);
    static double safeCdfComplement(const gamma& gamma_, double x);
    static double safeCdfComplement(const beta& beta_, double x);
    static double safeCdfComplement(const binomial& binomial_, double x);
    static double safeCdfComplement(const chi_squared& chi2, double x);
    //@}

    //! Compute the anomalousness from the probability of seeing a
    //! more extreme event for a distribution, i.e. for a sample
    //! \f$x\f$ from a R.V. the probability \f$P(R)\f$ of the set:
    //! <pre class="fragment">
    //!   \f$ R = \{y\ |\ f(y) \leq f(x)\} \f$
    //! </pre>
    //! where,\n
    //!   \f$f(.)\f$ is the p.d.f. of the random variable.\n\n
    //! This is a monotonically decreasing function of \f$P(R)\f$ and
    //! is chosen so that for \f$P(R)\f$ near one it is zero and as
    //! \f$P(R) \rightarrow 0\f$ it saturates at 100.
    static double anomalyScore(double p);

    //! The inverse of the anomalyScore function.
    static double inverseAnomalyScore(double deviation);

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
    static double differentialEntropy(const poisson& poisson_);
    static double differentialEntropy(const normal& normal_);
    static double differentialEntropy(const lognormal& logNormal);
    static double differentialEntropy(const gamma& gamma_);
    template<typename T>
    class CDifferentialEntropyKernel {
    public:
        CDifferentialEntropyKernel(const CMixtureDistribution<T>& mixture)
            : m_Mixture(&mixture) {}

        inline bool operator()(double x, double& result) const {
            double fx = pdf(*m_Mixture, x);
            result = fx == 0.0 ? 0.0 : -fx * std::log(fx);
            return true;
        }

    private:
        const CMixtureDistribution<T>* m_Mixture;
    };
    template<typename T>
    static double differentialEntropy(const CMixtureDistribution<T>& mixture);
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
        using TArray = std::array<double, BINS>;

    public:
        //! Builds the table.
        CLookupTableForFastLog() {
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
            std::uint64_t dx = 0x10000000000000ull / BINS;
            core::CIEEE754::SDoubleRep x;
            x.s_Sign = 0;
            x.s_Mantissa = (dx / 2) & core::CIEEE754::IEEE754_MANTISSA_MASK;
            x.s_Exponent = 1022;
            for (std::size_t i = 0; i < BINS;
                 ++i, x.s_Mantissa = (x.s_Mantissa + dx) & core::CIEEE754::IEEE754_MANTISSA_MASK) {
                double value;
                static_assert(sizeof(double) == sizeof(core::CIEEE754::SDoubleRep),
                              "SDoubleRep definition unsuitable for memcpy to double");
                // Use memcpy() rather than union to adhere to strict
                // aliasing rules
                std::memcpy(&value, &x, sizeof(double));
                m_Table[i] = stable(std::log2(value));
            }
        }

        //! Lookup log2 for a given mantissa.
        const double& operator[](std::uint64_t mantissa) const {
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
        std::uint64_t mantissa;
        int log2;
        core::CIEEE754::decompose(x, mantissa, log2);
        return 0.693147180559945 * (FAST_LOG_TABLE[mantissa] + log2);
    }
    //@}

    //! Compute the sign of \p x and return T(-1) if it is negative and T(1)
    //! otherwise.
    //!
    //! \param[in] x The value for which to check the sign.
    //! \note Conversion of 0 and -1 to T should be well defined.
    //! \note Zero maps to 1.
    template<typename T>
    static T sign(const T& x) {
        return x < T(0) ? T(-1) : T(1);
    }

    //! Truncate \p x to the range [\p a, \p b].
    //!
    //! \tparam T Must support operator<.
    template<typename T>
    static const T& truncate(const T& x, const T& a, const T& b) {
        return x < a ? a : (b < x ? b : x);
    }

    //! Component-wise truncation of stack vectors.
    template<typename T, std::size_t N>
    static CVectorNx1<T, N> truncate(const CVectorNx1<T, N>& x,
                                     const CVectorNx1<T, N>& a,
                                     const CVectorNx1<T, N>& b) {
        CVectorNx1<T, N> result(x);
        for (std::size_t i = 0; i < N; ++i) {
            result(i) = truncate(result(i), a(i), b(i));
        }
        return result;
    }

    //! Component-wise truncation of heap vectors.
    template<typename T>
    static CVector<T>
    truncate(const CVector<T>& x, const CVector<T>& a, const CVector<T>& b) {
        CVector<T> result(x);
        for (std::size_t i = 0; i < result.dimension(); ++i) {
            result(i) = truncate(result(i), a(i), b(i));
        }
        return result;
    }

    //! Component-wise truncation of small vector.
    template<typename T, std::size_t N>
    static core::CSmallVector<T, N> truncate(const core::CSmallVector<T, N>& x,
                                             const core::CSmallVector<T, N>& a,
                                             const core::CSmallVector<T, N>& b) {
        core::CSmallVector<T, N> result(x);
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] = truncate(result[i], a[i], b[i]);
        }
        return result;
    }

    //! Shift \p x to the left by \p eps times \p x.
    static double shiftLeft(double x, double eps = std::numeric_limits<double>::epsilon());

    //! Shift \p x to the right by \p eps times \p x.
    static double shiftRight(double x, double eps = std::numeric_limits<double>::epsilon());

    //! Compute \f$x^2\f$.
    static double pow2(double x) { return x * x; }

    //! Compute a value from \p x which will be stable across platforms.
    static double stable(double x) { return core::CIEEE754::dropbits(x, 1); }

    //! A version of std::log which is stable across platforms.
    static double stableLog(double x) { return stable(std::log(x)); }

    //! A version of std::log which is stable across platforms.
    static double stableExp(double x) { return stable(std::exp(x)); }

    //! Sigmoid function of \p p.
    static double sigmoid(double p) { return 1.0 / (1.0 + 1.0 / p); }

    //! The logistic function.
    //!
    //! i.e. \f$sigmoid\left(\frac{sign (x - x0)}{width}\right)\f$.
    //!
    //! \param[in] x The argument.
    //! \param[in] width The step width.
    //! \param[in] x0 The centre of the step.
    //! \param[in] sign Determines whether it's a step up or down.
    static double
    logisticFunction(double x, double width = 1.0, double x0 = 0.0, double sign = 1.0) {
        return sigmoid(stableExp(std::copysign(1.0, sign) * (x - x0) / width));
    }

    //! Compute the softmax for the multinomial logit values \p logit.
    //!
    //! i.e. \f$[\sigma(z)]_i = \frac{exp(z_i)}{\sum_j exp(z_j)}\f$.
    //!
    //! \tparam COLLECTION Is assumed to be a collection type, i.e. it
    //! must support iterator based access.
    template<typename COLLECTION>
    static void inplaceSoftmax(COLLECTION& z) {
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

    //! Compute the log of the softmax for the multinomial logit values \p logit.
    template<typename COLLECTION>
    static void inplaceLogSoftmax(COLLECTION& z) {
        double zmax{*std::max_element(z.begin(), z.end())};
        for (auto& zi : z) {
            zi -= zmax;
        }
        double logZ{std::log(std::accumulate(
            z.begin(), z.end(), 0.0,
            [](double sum, const auto& zi) { return sum + std::exp(zi); }))};
        for (auto& zi : z) {
            zi -= logZ;
        }
    }

    //! Specialize the softmax for CDenseVector.
    template<typename T>
    static void inplaceSoftmax(CDenseVector<T>& z);

    //! Specialize the log(softmax) for CDenseVector.
    template<typename SCALAR>
    static void inplaceLogSoftmax(CDenseVector<SCALAR>& z);

    //! Linearly interpolate a function on the interval [\p a, \p b].
    static double linearlyInterpolate(double a, double b, double fa, double fb, double x);

    //! Log-linearly interpolate a function on the interval [\p a, \p b].
    //!
    //! \warning The caller must ensure that \p a and \p b are positive.
    static double logLinearlyInterpolate(double a, double b, double fa, double fb, double x);

    //! A custom, numerically robust, implementation of \f$(1 - x) ^ p\f$.
    //!
    //! \note It is assumed that p is integer.
    static double powOneMinusX(double x, double p);

    //! A custom, numerically robust, implementation of \f$1 - (1 - x) ^ p\f$.
    //!
    //! \note It is assumed that p is integer.
    static double oneMinusPowOneMinusX(double x, double p);

    //! A custom implementation of \f$\log(1 - x)\f$ which handles the
    //! cancellation error for small x.
    static double logOneMinusX(double x);

    //! A wrapper around lgamma which handles corner cases if requested
    static bool lgamma(double value, double& result, bool checkForFinite = true);
};
}
}

#endif // INCLUDED_ml_maths_CTools_h
