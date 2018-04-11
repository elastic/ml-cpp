/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CLogNormalMeanPrecConjugate.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CNonCopyable.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CIntegration.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CLogTDistribution.h>
#include <maths/CMathsFuncs.h>
#include <maths/COrderings.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/numeric/conversion/bounds.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>

namespace ml {
namespace maths {

namespace {

using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble4Vec = core::CSmallVector<double, 4>;
using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
using TSizeVec = std::vector<std::size_t>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TWeightStyleVec = maths_t::TWeightStyleVec;

//! Compute x * x.
inline double pow2(double x) {
    return x * x;
}

const double MINIMUM_LOGNORMAL_SHAPE = 100.0;

namespace detail {

using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;

//! \brief Adds "weight" x "right operand" to the "left operand".
struct SPlusWeight {
    double operator()(double lhs, double rhs, double weight = 1.0) const { return lhs + weight * rhs; }
};

//! Get the effective location and scale of the sample.
//!
//! \param[in] vs The count variance scale.
//! \param[in] mean The normal prior mean.
//! \param[in] precision The normal prior precision.
//! \param[in] rate The gamma prior rate.
//! \param[in] shape The gamma prior shape.
//! \param[out] location The effective location of sample distribution.
//! \param[out] scale The effective scale of sample distribution.
inline void
locationAndScale(double vs, double r, double s, double mean, double precision, double rate, double shape, double& location, double& scale) {
    double t = vs == 1.0 ? r : r + std::log(s + vs * (1.0 - s));
    double scaledPrecision = t == r ? precision : t / r * precision;
    double scaledRate = t == r ? rate : t / r * rate;
    location = mean + (r - t) / 2.0;
    scale = std::sqrt((scaledPrecision + 1.0) / scaledPrecision * scaledRate / shape);
}

//! Evaluate \p func on the joint predictive distribution for \p samples
//! (integrating over the prior for the exponentiated normal mean and
//! precision) and aggregate the results using \p aggregate.
//!
//! \param weightStyles Controls the interpretation of weights that are
//! associated with each sample. See maths_t::ESampleWeightStyle for more
//! details.
//! \param samples The weighted samples.
//! \param weights The weights of each sample in \p samples.
//! \param func The function to evaluate.
//! \param aggregate The function to aggregate the results of \p func.
//! \param isNonInformative True if the prior is non-informative.
//! \param offset The constant offset of the data, in particular it is
//! assumed that \p samples are distributed as exp(Y) - "offset", where
//! Y is a normally distributed R.V.
//! \param shape The shape of the marginal precision prior.
//! \param rate The rate of the marginal precision prior.
//! \param mean The mean of the conditional mean prior.
//! \param precision The precision of the conditional mean prior.
//! \param result Filled in with the aggregation of results of \p func.
template<typename FUNC, typename AGGREGATOR, typename RESULT>
bool evaluateFunctionOnJointDistribution(const TWeightStyleVec& weightStyles,
                                         const TDouble1Vec& samples,
                                         const TDouble4Vec1Vec& weights,
                                         FUNC func,
                                         AGGREGATOR aggregate,
                                         bool isNonInformative,
                                         double offset,
                                         double shape,
                                         double rate,
                                         double mean,
                                         double precision,
                                         RESULT& result) {
    result = RESULT();

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute distribution for empty sample set");
        return false;
    }

    // Computing the true joint marginal distribution of all the samples
    // by integrating the joint likelihood over the prior distribution
    // for the exponentiated Gaussian mean and precision is not tractable.
    // We will approximate the joint p.d.f. as follows:
    //   Integral{ Product_i{ L(x(i) | m,p) } * f(m,p) }dm*dp
    //      ~= Product_i{ Integral{ L(x(i) | m,p) * f(m,p) }dm*dp }.
    //
    // where,
    //   L(. | m,p) is the likelihood function and
    //   f(m,p) is the prior for the exponentiated Gaussian mean and precision.
    //
    // This becomes increasingly accurate as the prior distribution narrows.

    try {
        if (isNonInformative) {
            // The non-informative prior is improper and effectively 0 everywhere.
            // (It is acceptable to approximate all finite samples as at the median
            // of this distribution.)
            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = maths_t::count(weightStyles, weights[i]);
                result = aggregate(result, func(CTools::SImproperDistribution(), samples[i] + offset), n);
            }
        } else if (shape > MINIMUM_LOGNORMAL_SHAPE) {
            // For large shape the marginal likelihood is very well approximated
            // by a log-normal distribution. In particular, the true distribution
            // is log t with  2 * a degrees of freedom, location m and scale
            // s = ((p+1)/p * b/a) ^ (1/2). This implies that the p.d.f is
            // proportional to:
            //   f(x) ~ 1 / x * (1 + 1/(2*a) * (log(x) - m)^2 / s^2) ^ -((2*a + 1) / 2).
            //
            // To compute the log-normal distribution we use the identity:
            //   lim n->inf { (1 + 1 / (2*n) * p * x^2)^-n } = exp(-p * x^2 / 2).
            //
            // This gives that in the limit of large a the p.d.f. tends to:
            //   f(x) ~ 1 / x * exp(-(log(x) - m) ^ 2 / s ^ 2 / 2).
            //
            // This is log-normal with:
            //    mean = m and
            //   scale = ((p+1)/p * b/a) ^ (1/2).

            double r = rate / shape;
            double s = std::exp(-r);

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = maths_t::count(weightStyles, weights[i]);
                double varianceScale =
                    maths_t::seasonalVarianceScale(weightStyles, weights[i]) * maths_t::countVarianceScale(weightStyles, weights[i]);
                double location;
                double scale;
                locationAndScale(varianceScale, r, s, mean, precision, rate, shape, location, scale);
                boost::math::lognormal_distribution<> lognormal(location, scale);
                result = aggregate(result, func(lognormal, samples[i] + offset), n);
            }
        } else {
            // The marginal likelihood is log t with 2 * a degrees of freedom,
            // location m and scale s = (a * p / (p + 1) / b) ^ (1/2).

            double r = rate / shape;
            double s = std::exp(-r);

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = maths_t::count(weightStyles, weights[i]);
                double varianceScale =
                    maths_t::seasonalVarianceScale(weightStyles, weights[i]) * maths_t::countVarianceScale(weightStyles, weights[i]);
                double location;
                double scale;
                locationAndScale(varianceScale, r, s, mean, precision, rate, shape, location, scale);
                CLogTDistribution logt(2.0 * shape, location, scale);
                result = aggregate(result, func(logt, samples[i] + offset), n);
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Error calculating joint c.d.f.: " << e.what());
        return false;
    }

    LOG_TRACE(<< "result = " << result);

    return true;
}

//! \brief Evaluates a specified function object, which must be default constructible,
//! on the joint distribution of a set of the samples at a specified offset.
//!
//! This thin wrapper around the evaluateFunctionOnJointDistribution function
//! so that it can be integrated over the hidden variable representing the
//! actual value of a discrete datum which we assume is in the interval [n, n+1].
template<typename F>
class CEvaluateOnSamples : core::CNonCopyable {
public:
    CEvaluateOnSamples(const TWeightStyleVec& weightStyles,
                       const TDouble1Vec& samples,
                       const TDouble4Vec1Vec& weights,
                       bool isNonInformative,
                       double offset,
                       double mean,
                       double precision,
                       double shape,
                       double rate)
        : m_WeightStyles(weightStyles),
          m_Samples(samples),
          m_Weights(weights),
          m_IsNonInformative(isNonInformative),
          m_Offset(offset),
          m_Mean(mean),
          m_Precision(precision),
          m_Shape(shape),
          m_Rate(rate) {}

    bool operator()(double x, double& result) const {
        return evaluateFunctionOnJointDistribution(m_WeightStyles,
                                                   m_Samples,
                                                   m_Weights,
                                                   F(),
                                                   SPlusWeight(),
                                                   m_IsNonInformative,
                                                   m_Offset + x,
                                                   m_Shape,
                                                   m_Rate,
                                                   m_Mean,
                                                   m_Precision,
                                                   result);
    }

private:
    const TWeightStyleVec& m_WeightStyles;
    const TDouble1Vec& m_Samples;
    const TDouble4Vec1Vec& m_Weights;
    bool m_IsNonInformative;
    double m_Offset;
    double m_Mean;
    double m_Precision;
    double m_Shape;
    double m_Rate;
};

//! \brief Kernel for computing the marginal likelihood's mean.
//!
//! This is used to evaluate the integral of the likelihood mean w.r.t. the
//! prior on the likelihood precision. Note that the integral over the prior
//! on the mean can be performed analytically so the kernel is:
//! <pre class="fragment">
//!   \f$\(\displaystyle e^{m+\frac{1}{p}\left(\frac{1}{t} + 1\right)} f(p)\)\f$
//! </pre>
//! Here, \(m\) is the expected mean, and the prior on the precision\(p\) is
//! gamma distributed.
class CMeanKernel {
public:
    using TValue = CVectorNx1<double, 2>;

public:
    CMeanKernel(double m, double p, double a, double b) : m_M(m), m_P(p), m_A(a), m_B(b) {}

    bool operator()(double x, TValue& result) const {
        try {
            boost::math::gamma_distribution<> gamma(m_A, 1.0 / m_B);
            double fx = boost::math::pdf(gamma, x);
            result(0) = std::exp(m_M + 0.5 / x * (1.0 / m_P + 1.0)) * fx;
            result(1) = fx;
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to calculate mean kernel: " << e.what() << ", m = " << m_M << ", p = " << m_P << ", a = " << m_A
                      << ", b = " << m_B << ", x = " << x);
            return false;
        }
        return true;
    }

private:
    double m_M, m_P, m_A, m_B;
};

//! \brief Kernel for computing the marginal likelihood's variance.
//!
//! This is used to evaluate the integral of the likelihood variance w.r.t.
//! the prior on the likelihood precision. Note that the integral over the
//! prior on the mean can be performed analytically.
class CVarianceKernel {
public:
    using TValue = CVectorNx1<double, 2>;

public:
    CVarianceKernel(double mean, double m, double p, double a, double b) : m_Mean(mean), m_M(m), m_P(p), m_A(a), m_B(b) {}

    bool operator()(const TValue& x, TValue& result) const {
        try {
            boost::math::gamma_distribution<> gamma(m_A, 1.0 / m_B);
            boost::math::normal_distribution<> normal(m_M, std::sqrt(1.0 / x(0) / m_P));
            double fx = boost::math::pdf(normal, x(1)) * boost::math::pdf(gamma, x(0));
            double m = std::exp(x(1) + 0.5 / x(0));
            result(0) = (m * m * (std::exp(1.0 / x(0)) - 1.0) + pow2(m - m_Mean)) * fx;
            result(1) = fx;
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to calculate mean kernel: " << e.what() << ", m = " << m_M << ", p = " << m_P << ", a = " << m_A
                      << ", b = " << m_B << ", x = " << x);
            return false;
        }
        return true;
    }

private:
    double m_Mean, m_M, m_P, m_A, m_B;
};

//! \brief Computes the probability of seeing less likely samples at a specified
//! offset.
//!
//! This thin wrapper around the evaluateFunctionOnJointDistribution function
//! so that it can be integrated over the hidden variable representing the
//! actual value of a discrete datum which we assume is in the interval [n, n+1].
class CProbabilityOfLessLikelySamples : core::CNonCopyable {
public:
    CProbabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                    const TWeightStyleVec& weightStyles,
                                    const TDouble1Vec& samples,
                                    const TDouble4Vec1Vec& weights,
                                    bool isNonInformative,
                                    double offset,
                                    double mean,
                                    double precision,
                                    double shape,
                                    double rate)
        : m_Calculation(calculation),
          m_WeightStyles(weightStyles),
          m_Samples(samples),
          m_Weights(weights),
          m_IsNonInformative(isNonInformative),
          m_Offset(offset),
          m_Mean(mean),
          m_Precision(precision),
          m_Shape(shape),
          m_Rate(rate),
          m_Tail(0) {}

    bool operator()(double x, double& result) const {
        CJointProbabilityOfLessLikelySamples probability;
        maths_t::ETail tail = maths_t::E_UndeterminedTail;

        if (!evaluateFunctionOnJointDistribution(
                m_WeightStyles,
                m_Samples,
                m_Weights,
                boost::bind<double>(CTools::CProbabilityOfLessLikelySample(m_Calculation), _1, _2, boost::ref(tail)),
                CJointProbabilityOfLessLikelySamples::SAddProbability(),
                m_IsNonInformative,
                m_Offset + x,
                m_Shape,
                m_Rate,
                m_Mean,
                m_Precision,
                probability) ||
            !probability.calculate(result)) {
            LOG_ERROR(<< "Failed to compute probability of less likely samples"
                      << ", samples = " << core::CContainerPrinter::print(m_Samples) << ", offset = " << m_Offset + x);
            return false;
        }

        m_Tail = m_Tail | tail;

        return true;
    }

    maths_t::ETail tail() const { return static_cast<maths_t::ETail>(m_Tail); }

private:
    maths_t::EProbabilityCalculation m_Calculation;
    const TWeightStyleVec& m_WeightStyles;
    const TDouble1Vec& m_Samples;
    const TDouble4Vec1Vec& m_Weights;
    bool m_IsNonInformative;
    double m_Offset;
    double m_Mean;
    double m_Precision;
    double m_Shape;
    double m_Rate;
    mutable int m_Tail;
};

//! \brief Wraps up log marginal likelihood function so that it can be integrated
//! over the hidden variable representing the actual value of a discrete datum
//! which we assume is in the interval [n, n+1].
//!
//! We assume the data are described by X = exp(Y) - u where, Y is normally
//! distributed and u is a constant offset. The log marginal likelihood
//! function of the samples is the likelihood function for the data integrated
//! over the prior distribution for the mean and scale. It can be shown that:
//!   log( L(x | m, p, a, b) ) =
//!     log( 1 / (2 * pi) ^ (n/2)
//!          * (p / (p + n)) ^ (1/2)
//!          * b ^ a
//!          * Gamma(a + n/2)
//!          / Gamma(a)
//!          / (b + 1/2 * (n * var(log(x))
//!                        + p * n * (mean(log(x)) - m)^2 / (p + n))) ^ (a + n/2)
//!          / Product_i{ x(i) } ).
//!
//! Here,
//!   x = {y(i) + u} and {y(i)} is the sample vector and u the constant offset.
//!   n = |x| the number of elements in the sample vector.
//!   mean(.) is the sample mean function.
//!   var(.) is the sample variance function.
//!   m and p are the prior Gaussian mean and precision, respectively.
//!   a and b are the prior Gamma shape and rate, respectively.
class CLogMarginalLikelihood : core::CNonCopyable {
public:
    CLogMarginalLikelihood(const TWeightStyleVec& weightStyles,
                           const TDouble1Vec& samples,
                           const TDouble4Vec1Vec& weights,
                           double offset,
                           double mean,
                           double precision,
                           double shape,
                           double rate)
        : m_WeightStyles(weightStyles),
          m_Samples(samples),
          m_Weights(weights),
          m_Offset(offset),
          m_Mean(mean),
          m_Precision(precision),
          m_Shape(shape),
          m_Rate(rate),
          m_NumberSamples(0.0),
          m_Scales(),
          m_Constant(0.0),
          m_ErrorStatus(maths_t::E_FpNoErrors) {
        this->precompute();
    }

    //! Evaluate the log marginal likelihood at the offset \p x.
    bool operator()(double x, double& result) const {
        if (m_ErrorStatus & maths_t::E_FpFailed) {
            return false;
        }

        double logSamplesSum = 0.0;
        TMeanVarAccumulator logSampleMoments;

        try {
            for (std::size_t i = 0u; i < m_Samples.size(); ++i) {
                double n = maths_t::countForUpdate(m_WeightStyles, m_Weights[i]);
                double sample = m_Samples[i] + m_Offset + x;
                if (sample <= 0.0) {
                    // Technically, the marginal likelihood is zero here
                    // so the log would be infinite. We use minus max
                    // double because log(0) = HUGE_VALUE, which causes
                    // problems for Windows. Calling code is notified
                    // when the calculation overflows and should avoid
                    // taking the exponential since this will underflow
                    // and pollute the floating point environment. This
                    // may cause issues for some library function
                    // implementations (see fe*exceptflag for more details).
                    result = boost::numeric::bounds<double>::lowest();
                    this->addErrorStatus(maths_t::E_FpOverflowed);
                    return false;
                }

                double logSample = std::log(sample);
                double w = m_Scales.empty() ? 1.0 : 1.0 / m_Scales[i].first;
                double shift = m_Scales.empty() ? 0.0 : m_Scales[i].second;

                logSamplesSum += n * logSample;
                logSampleMoments.add(logSample - shift, n * w);
            }
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to calculate likelihood: " << e.what());
            this->addErrorStatus(maths_t::E_FpFailed);
            return false;
        }

        double weightedNumberSamples = CBasicStatistics::count(logSampleMoments);
        double logSamplesMean = CBasicStatistics::mean(logSampleMoments);
        double logSamplesSquareDeviation = (weightedNumberSamples - 1.0) * CBasicStatistics::variance(logSampleMoments);

        double impliedShape = m_Shape + 0.5 * m_NumberSamples;
        double impliedRate =
            m_Rate + 0.5 * (logSamplesSquareDeviation +
                            m_Precision * weightedNumberSamples * pow2(logSamplesMean - m_Mean) / (m_Precision + weightedNumberSamples));

        result = m_Constant - impliedShape * std::log(impliedRate) - logSamplesSum;

        return true;
    }

    //! Retrieve the error status for the integration.
    maths_t::EFloatingPointErrorStatus errorStatus() const { return m_ErrorStatus; }

private:
    static const double LOG_2_PI;

private:
    //! Compute all the constants in the integrand.
    void precompute() {
        try {
            double logVarianceScaleSum = 0.0;

            if (maths_t::hasSeasonalVarianceScale(m_WeightStyles, m_Weights) || maths_t::hasCountVarianceScale(m_WeightStyles, m_Weights)) {
                m_Scales.reserve(m_Weights.size());
                double r = m_Rate / m_Shape;
                double s = std::exp(-r);
                for (std::size_t i = 0u; i < m_Weights.size(); ++i) {
                    double varianceScale = maths_t::seasonalVarianceScale(m_WeightStyles, m_Weights[i]) *
                                           maths_t::countVarianceScale(m_WeightStyles, m_Weights[i]);

                    // Get the scale and shift of the exponentiated Gaussian.
                    if (varianceScale == 1.0) {
                        m_Scales.emplace_back(1.0, 0.0);
                    } else {
                        double t = r + std::log(s + varianceScale * (1.0 - s));
                        m_Scales.emplace_back(t / r, 0.5 * (r - t));
                        logVarianceScaleSum += std::log(t / r);
                    }
                }
            }

            m_NumberSamples = 0.0;
            double weightedNumberSamples = 0.0;

            for (std::size_t i = 0u; i < m_Weights.size(); ++i) {
                double n = maths_t::countForUpdate(m_WeightStyles, m_Weights[i]);
                m_NumberSamples += n;
                weightedNumberSamples += n / (m_Scales.empty() ? 1.0 : m_Scales[i].first);
            }

            double impliedShape = m_Shape + 0.5 * m_NumberSamples;
            double impliedPrecision = m_Precision + weightedNumberSamples;

            m_Constant = 0.5 * (std::log(m_Precision) - std::log(impliedPrecision)) - 0.5 * m_NumberSamples * LOG_2_PI -
                         0.5 * logVarianceScaleSum + boost::math::lgamma(impliedShape) - boost::math::lgamma(m_Shape) +
                         m_Shape * std::log(m_Rate);
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Error calculating marginal likelihood: " << e.what());
            this->addErrorStatus(maths_t::E_FpFailed);
        }
    }

    //! Update the error status.
    void addErrorStatus(maths_t::EFloatingPointErrorStatus status) const {
        m_ErrorStatus = static_cast<maths_t::EFloatingPointErrorStatus>(m_ErrorStatus | status);
    }

private:
    const TWeightStyleVec& m_WeightStyles;
    const TDouble1Vec& m_Samples;
    const TDouble4Vec1Vec& m_Weights;
    double m_Offset;
    double m_Mean;
    double m_Precision;
    double m_Shape;
    double m_Rate;
    double m_NumberSamples;
    TDoubleDoublePrVec m_Scales;
    double m_Constant;
    mutable maths_t::EFloatingPointErrorStatus m_ErrorStatus;
};

const double CLogMarginalLikelihood::LOG_2_PI = std::log(boost::math::double_constants::two_pi);

//! \brief Wraps up the sample total square deviation of the logs of a
//! collection of samples, i.e.
//! <pre class="fragment">
//!   \f$sum_i{(\log(x_i) - \frac{1}{n}sum_j{\log(x_j)})^2}\f$
//! </pre>
//!
//! so that it can be integrated over the hidden variable representing the
//! actual value of a discrete datum which we assume is in the interval
//! [n, n+1].
class CLogSampleSquareDeviation : core::CNonCopyable {
public:
    CLogSampleSquareDeviation(const TWeightStyleVec& weightStyles, const TDouble1Vec& samples, const TDouble4Vec1Vec& weights, double mean)
        : m_WeightStyles(weightStyles), m_Samples(samples), m_Weights(weights), m_Mean(mean) {}

    bool operator()(double x, double& result) const {
        result = 0.0;
        for (std::size_t i = 0u; i < m_Samples.size(); ++i) {
            double residual = m_Samples[i];
            if (residual <= 0.0) {
                continue;
            }
            double n = maths_t::countForUpdate(m_WeightStyles, m_Weights[i]);
            residual = std::log(residual + x) - m_Mean;
            result += n * pow2(residual);
        }
        return true;
    }

private:
    const TWeightStyleVec& m_WeightStyles;
    const TDouble1Vec& m_Samples;
    const TDouble4Vec1Vec& m_Weights;
    double m_Mean;
};

} // detail::

// We use short field names to reduce the state size
const std::string OFFSET_TAG("a");
const std::string GAUSSIAN_MEAN_TAG("b");
const std::string GAUSSIAN_PRECISION_TAG("c");
const std::string GAMMA_SHAPE_TAG("d");
const std::string GAMMA_RATE_TAG("e");
const std::string NUMBER_SAMPLES_TAG("f");
//const std::string MINIMUM_TAG("g"); No longer used
//const std::string MAXIMUM_TAG("h"); No longer used
const std::string DECAY_RATE_TAG("i");
const std::string EMPTY_STRING;
}

CLogNormalMeanPrecConjugate::CLogNormalMeanPrecConjugate(maths_t::EDataType dataType,
                                                         double offset,
                                                         double gaussianMean,
                                                         double gaussianPrecision,
                                                         double gammaShape,
                                                         double gammaRate,
                                                         double decayRate,
                                                         double offsetMargin)
    : CPrior(dataType, decayRate),
      m_Offset(offset),
      m_OffsetMargin(offsetMargin),
      m_GaussianMean(gaussianMean),
      m_GaussianPrecision(gaussianPrecision),
      m_GammaShape(gammaShape),
      m_GammaRate(gammaRate) {
}

CLogNormalMeanPrecConjugate::CLogNormalMeanPrecConjugate(const SDistributionRestoreParams& params,
                                                         core::CStateRestoreTraverser& traverser,
                                                         double offsetMargin)
    : CPrior(params.s_DataType, params.s_DecayRate),
      m_Offset(0.0),
      m_OffsetMargin(offsetMargin),
      m_GaussianMean(0.0),
      m_GaussianPrecision(0.0),
      m_GammaShape(0.0),
      m_GammaRate(0.0) {
    traverser.traverseSubLevel(boost::bind(&CLogNormalMeanPrecConjugate::acceptRestoreTraverser, this, _1));
}

bool CLogNormalMeanPrecConjugate::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(
            DECAY_RATE_TAG, double decayRate, core::CStringUtils::stringToType(traverser.value(), decayRate), this->decayRate(decayRate))
        RESTORE_BUILT_IN(OFFSET_TAG, m_Offset)
        RESTORE_BUILT_IN(GAUSSIAN_MEAN_TAG, m_GaussianMean)
        RESTORE_BUILT_IN(GAUSSIAN_PRECISION_TAG, m_GaussianPrecision)
        RESTORE_BUILT_IN(GAMMA_SHAPE_TAG, m_GammaShape)
        RESTORE_BUILT_IN(GAMMA_RATE_TAG, m_GammaRate)
        RESTORE_SETUP_TEARDOWN(NUMBER_SAMPLES_TAG,
                               double numberSamples,
                               core::CStringUtils::stringToType(traverser.value(), numberSamples),
                               this->numberSamples(numberSamples))
    } while (traverser.next());

    return true;
}

CLogNormalMeanPrecConjugate
CLogNormalMeanPrecConjugate::nonInformativePrior(maths_t::EDataType dataType, double offset, double decayRate, double offsetMargin) {
    return CLogNormalMeanPrecConjugate(dataType,
                                       offset + offsetMargin,
                                       NON_INFORMATIVE_MEAN,
                                       NON_INFORMATIVE_PRECISION,
                                       NON_INFORMATIVE_SHAPE,
                                       NON_INFORMATIVE_RATE,
                                       decayRate,
                                       offsetMargin);
}

CLogNormalMeanPrecConjugate::EPrior CLogNormalMeanPrecConjugate::type() const {
    return E_LogNormal;
}

CLogNormalMeanPrecConjugate* CLogNormalMeanPrecConjugate::clone() const {
    return new CLogNormalMeanPrecConjugate(*this);
}

void CLogNormalMeanPrecConjugate::setToNonInformative(double offset, double decayRate) {
    *this = nonInformativePrior(this->dataType(), offset + this->offsetMargin(), decayRate, this->offsetMargin());
}

double CLogNormalMeanPrecConjugate::offsetMargin() const {
    return m_OffsetMargin;
}

bool CLogNormalMeanPrecConjugate::needsOffset() const {
    return true;
}

double
CLogNormalMeanPrecConjugate::adjustOffset(const TWeightStyleVec& weightStyles, const TDouble1Vec& samples, const TDouble4Vec1Vec& weights) {
    COffsetCost cost(*this);
    CApplyOffset apply(*this);
    return this->adjustOffsetWithCost(weightStyles, samples, weights, cost, apply);
}

double CLogNormalMeanPrecConjugate::offset() const {
    return m_Offset;
}

void CLogNormalMeanPrecConjugate::addSamples(const TWeightStyleVec& weightStyles,
                                             const TDouble1Vec& samples,
                                             const TDouble4Vec1Vec& weights) {
    if (samples.empty()) {
        return;
    }

    if (samples.size() != weights.size()) {
        LOG_ERROR(<< "Mismatch in samples '" << core::CContainerPrinter::print(samples) << "' and weights '"
                  << core::CContainerPrinter::print(weights) << "'");
        return;
    }

    this->adjustOffset(weightStyles, samples, weights);
    this->CPrior::addSamples(weightStyles, samples, weights);

    // We assume the data are described by X = exp(Y) - u where, Y is normally
    // distributed and u is a constant offset.
    //
    // If y = {y(i)} denotes the sample vector, then x = {y(i) + u} are
    // log-normally distributed with mean m' and inverse scale s', and the
    // likelihood function is:
    //   likelihood(x | m', s') ~
    //       Product_i{ 1 / x(i) * exp(-s' * (log(x(i)) - m')^2 / 2) }.
    //
    // The conjugate joint prior for m' and s' is gamma-normal and the update
    // of the posterior with n independent samples comes from:
    //   likelihood(x | m', s') * prior(m', s' | m, p, a, b)            (1)
    //
    // where,
    //   prior(m', s' | m, p, a, b) ~
    //       (s' * p)^(1/2) * exp(-s' * p * (m' - m)^2 / 2)
    //       * s'^(a - 1) * exp(-b * s')
    //
    // i.e. the conditional distribution of the mean is Gaussian with mean m
    // and precision s' * p and the marginal distribution of s' is gamma with
    // shape a and rate b. Equation (1) implies that the parameters of the prior
    // distribution update as follows:
    //   m -> (p * m + n * mean(log(x))) / (p + n)
    //   p -> p + n
    //   a -> a + n/2
    //   b -> b + 1/2 * (n * var(log(x)) + p * n * (mean(log(x)) - m)^2 / (p + n))
    //
    // where,
    //   mean(.) is the sample mean function.
    //   var(.) is the sample variance function.
    //
    // Note that the weight of the sample x(i) is interpreted as its count,
    // i.e. n(i), so for example updating with {(x, 2)} is equivalent to
    // updating with {x, x}.
    //
    // If the data are discrete then we approximate the discrete distribution
    // by saying it is uniform on the intervals [n,n+1] for each integral n.
    // This is like saying that the data are samples from:
    //   X' = X + Z
    //
    // where,
    //   Z is uniform in the interval [0,1].
    //
    // We care about the limiting behaviour of the filter, i.e. as the number
    // of samples n->inf. In this case, the uniform law of large numbers gives
    // that:
    //   Sum_i( f(x(i) + z(i)) ) -> E[ Sum_i( f(x(i) + Z) ) ]
    //
    // We use this to evaluate:
    //   mean(log(x(i) + z(i)))
    //       -> 1/n * Sum_i( (x(i) + 1) * log(x(i) + 1) - x(i) * log(x(i)) - 1 )
    //
    // To evaluate var(log(x(i) + z(i))) we use numerical integration for
    // simplicity because naive calculation of the exact integral suffers from
    // cancellation errors.

    double numberSamples = 0.0;
    double scaledNumberSamples = 0.0;
    double logSamplesMean = 0.0;
    double logSamplesSquareDeviation = 0.0;

    double r = m_GammaRate / m_GammaShape;
    double s = std::exp(-r);
    try {
        if (this->isInteger()) {
            // Filled in with samples rescaled to have approximately unit
            // variance scale.
            TDouble1Vec scaledSamples;
            scaledSamples.resize(samples.size(), 1.0);

            TMeanAccumulator logSamplesMean_;
            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = maths_t::countForUpdate(weightStyles, weights[i]);
                double varianceScale =
                    maths_t::seasonalVarianceScale(weightStyles, weights[i]) * maths_t::countVarianceScale(weightStyles, weights[i]);
                double x = samples[i] + m_Offset;
                numberSamples += n;
                double t = varianceScale == 1.0 ? r : r + std::log(s + varianceScale * (1.0 - s));
                double shift = (r - t) / 2.0;
                double scale = r == t ? 1.0 : t / r;
                scaledSamples[i] = scale;
                double logxInvPlus1 = std::log(1.0 / x + 1.0);
                double logxPlus1 = std::log(x + 1.0);
                logSamplesMean_.add(x * logxInvPlus1 + logxPlus1 - 1.0 - shift, n / scale);
            }
            scaledNumberSamples = CBasicStatistics::count(logSamplesMean_);
            logSamplesMean = CBasicStatistics::mean(logSamplesMean_);

            double mean =
                (m_GaussianPrecision * m_GaussianMean + scaledNumberSamples * logSamplesMean) / (m_GaussianPrecision + scaledNumberSamples);
            for (std::size_t i = 0u; i < scaledSamples.size(); ++i) {
                double scale = scaledSamples[i];
                scaledSamples[i] =
                    scale == 1.0 ? samples[i] + m_Offset : std::exp(mean + (std::log(samples[i] + m_Offset) - mean) / std::sqrt(scale));
            }

            detail::CLogSampleSquareDeviation deviationFunction(weightStyles, scaledSamples, weights, logSamplesMean);
            CIntegration::gaussLegendre<CIntegration::OrderFive>(deviationFunction, 0.0, 1.0, logSamplesSquareDeviation);
        } else {
            TMeanVarAccumulator logSamplesMoments;
            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = maths_t::countForUpdate(weightStyles, weights[i]);
                double varianceScale =
                    maths_t::seasonalVarianceScale(weightStyles, weights[i]) * maths_t::countVarianceScale(weightStyles, weights[i]);
                double x = samples[i] + m_Offset;
                if (x <= 0.0) {
                    LOG_ERROR(<< "Discarding " << x << " it's not log-normal");
                    continue;
                }
                numberSamples += n;
                double t = varianceScale == 1.0 ? r : r + std::log(s + varianceScale * (1.0 - s));
                double scale = r == t ? 1.0 : t / r;
                double shift = (r - t) / 2.0;
                logSamplesMoments.add(std::log(x) - shift, n / scale);
            }
            scaledNumberSamples = CBasicStatistics::count(logSamplesMoments);
            logSamplesMean = CBasicStatistics::mean(logSamplesMoments);
            logSamplesSquareDeviation = (scaledNumberSamples - 1.0) * CBasicStatistics::variance(logSamplesMoments);
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to update likelihood: " << e.what());
        return;
    }

    m_GammaShape += 0.5 * numberSamples;
    m_GammaRate += 0.5 * (logSamplesSquareDeviation + m_GaussianPrecision * scaledNumberSamples * pow2(logSamplesMean - m_GaussianMean) /
                                                          (m_GaussianPrecision + scaledNumberSamples));

    m_GaussianMean =
        (m_GaussianPrecision * m_GaussianMean + scaledNumberSamples * logSamplesMean) / (m_GaussianPrecision + scaledNumberSamples);
    m_GaussianPrecision += scaledNumberSamples;

    // If the coefficient of variation of the data is too small we run
    // in to numerical problems. We truncate the variation by modeling
    // the impact of an actual variation (standard deviation divided by
    // mean) in the data of size MINIMUM_COEFFICIENT_OF_VARATION on the
    // prior parameters.

    if (m_GaussianPrecision > 1.5) {
        // The idea is to model the impact of a coefficient of variation
        // equal to MINIMUM_COEFFICIENT_OF_VARIATION on the parameters
        // of the prior this will affect. In particular, this enters in
        // to the gamma rate and shape by its impact on the mean of the
        // log of the samples. Note that:
        //   mean(log(x'(i)) = 1/n * Sum_i{ log(m + (x'(i) - m)) }
        //                   = 1/n * Sum_i{ log(m) + log(1 + d(i) / m) }
        //                   ~ 1/n * Sum_i{ log(m) + d(i) / m - (d(i)/m)^2 / 2 }
        //
        // where x(i) are our true samples, which are all very nearly m
        // (since their variation is tiny by assumption) and x'(i) are
        // our samples assuming minimum coefficient of variation. Finally,
        // we note that:
        //   E[d(i)] = 0
        //   E[(d(i)/m)^2] = "coefficient of variation"^2
        //
        // From which we derive the results below.

        double minimumRate = (2.0 * m_GammaShape - 1.0) * pow2(MINIMUM_COEFFICIENT_OF_VARIATION);

        if (m_GammaRate < minimumRate) {
            double extraVariation = (minimumRate - m_GammaRate) / (m_GaussianPrecision - 1.0);
            m_GammaRate = minimumRate;
            m_GaussianMean -= 0.5 * extraVariation;
        }
    }

    LOG_TRACE(<< "logSamplesMean = " << logSamplesMean << ", logSamplesSquareDeviation = " << logSamplesSquareDeviation
              << ", numberSamples = " << numberSamples << ", scaledNumberSamples = " << scaledNumberSamples);
    LOG_TRACE(<< "m_GammaShape = " << m_GammaShape << ", m_GammaRate = " << m_GammaRate << ", m_GaussianMean = " << m_GaussianMean
              << ", m_GaussianPrecision = " << m_GaussianPrecision << ", m_Offset = " << m_Offset);

    if (this->isBad()) {
        LOG_ERROR(<< "Update failed (" << this->debug() << ")");
        LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR(<< "weights = " << core::CContainerPrinter::print(weights));
        this->setToNonInformative(this->offsetMargin(), this->decayRate());
    }
}

void CLogNormalMeanPrecConjugate::propagateForwardsByTime(double time) {
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
        LOG_ERROR(<< "Bad propagation time " << time);
        return;
    }

    if (this->isNonInformative()) {
        // Nothing to be done.
        return;
    }

    double alpha = std::exp(-this->decayRate() * time);
    double beta = 1.0 - alpha;

    m_GaussianPrecision = alpha * m_GaussianPrecision + beta * NON_INFORMATIVE_PRECISION;

    // We want to increase the variance of the gamma distribution while
    // holding its mean constant s.t. in the limit t -> inf var -> inf.
    // The mean and variance are a / b and a / b^2, respectively, for
    // shape a and rate b so choose a factor f in the range [0, 1] and
    // update as follows:
    //   a -> f * a
    //   b -> f * b
    //
    // Thus the mean is unchanged and variance is increased by 1 / f.

    double factor = std::min((alpha * m_GammaShape + beta * NON_INFORMATIVE_SHAPE) / m_GammaShape, 1.0);

    m_GammaShape *= factor;
    m_GammaRate *= factor;

    this->numberSamples(this->numberSamples() * alpha);

    LOG_TRACE(<< "time = " << time << ", alpha = " << alpha << ", m_GaussianPrecision = " << m_GaussianPrecision
              << ", m_GammaShape = " << m_GammaShape << ", m_GammaRate = " << m_GammaRate << ", numberSamples = " << this->numberSamples());
}

CLogNormalMeanPrecConjugate::TDoubleDoublePr CLogNormalMeanPrecConjugate::marginalLikelihoodSupport() const {
    return std::make_pair(-m_Offset, boost::numeric::bounds<double>::highest());
}

double CLogNormalMeanPrecConjugate::marginalLikelihoodMean() const {
    return this->isInteger() ? this->mean() - 0.5 : this->mean();
}

double CLogNormalMeanPrecConjugate::marginalLikelihoodMode(const TWeightStyleVec& weightStyles, const TDouble4Vec& weights) const {
    if (this->isNonInformative()) {
        return std::exp(m_GaussianMean) - m_Offset;
    }

    // We use the fact that for large precision the marginal likelihood
    // is log-normally distributed and for small precision it is log-t.
    // See evaluateFunctionOnJointDistribution for more discussion.

    double varianceScale = 1.0;
    try {
        varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights) * maths_t::countVarianceScale(weightStyles, weights);
    } catch (const std::exception& e) { LOG_ERROR(<< "Failed to get variance scale: " << e.what()); }
    try {
        double r = m_GammaRate / m_GammaShape;
        double s = std::exp(-r);
        double location;
        double scale;
        detail::locationAndScale(varianceScale, r, s, m_GaussianMean, m_GaussianPrecision, m_GammaRate, m_GammaShape, location, scale);
        LOG_TRACE(<< "location = " << location << ", scale = " << scale);
        if (m_GammaShape > MINIMUM_LOGNORMAL_SHAPE) {
            boost::math::lognormal_distribution<> logNormal(location, scale);
            return boost::math::mode(logNormal) - m_Offset;
        }
        CLogTDistribution logt(2.0 * m_GammaShape, location, scale);
        double result = mode(logt) - m_Offset - (this->isInteger() ? 0.5 : 0.0);
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute marginal likelihood mode: " << e.what() << ", gaussian mean = " << m_GaussianMean
                  << ", gaussian precision = " << m_GaussianPrecision << ", gamma rate = " << m_GammaRate
                  << ", gamma shape = " << m_GammaShape);
    }

    // Fall back to using the exponentiated Gaussian's mean and precision.
    double normalMean = this->normalMean();
    double normalPrecision = this->normalPrecision() / varianceScale;
    return (normalPrecision == 0.0 ? 0.0 : std::exp(normalMean - 1.0 / normalPrecision)) - m_Offset;
}

double CLogNormalMeanPrecConjugate::marginalLikelihoodVariance(const TWeightStyleVec& weightStyles, const TDouble4Vec& weights) const {
    if (this->isNonInformative()) {
        return boost::numeric::bounds<double>::highest();
    }

    // This is just
    //   E_{M, P}[Var(X | M, P) + (E[X | M,P] - E[X])^2]        (1)
    //
    // where M, P likelihood mean and precision, respectively. Note that X
    // conditioned on M and P is log normal and
    //   Var(X | M, P) = exp(2*M + 1/P) * (exp(1/P) - 1)          (2)
    //
    // Pulling these together we have
    //   E_{M, P}[ exp(2*M + 1/P) * (exp(1/P) - 1) + (exp(M + 1/P) - E[X])^2 ]
    //
    // Note that although the integrand factorizes and we can perform the
    // integration over M in closed form, this version suffers from numerical
    // stability issues which can lead the variance to be negative. When the
    // prior is narrow we use the approximation:
    //   E[exp(1/P * (2/t + 1)) * ((exp(1/P) - 1) - E[X]^2)]
    //       ~= exp(1/E[P] * (2/p + 1)) * (exp(1/E[P]) - 1)
    //       ~= exp(b/a * (2/p + 1)) * (exp(b/a) - 1)
    //
    // Note that b / a > 0 so this is necessarily non-negative.

    double varianceScale = 1.0;
    try {
        varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights) * maths_t::countVarianceScale(weightStyles, weights);
    } catch (const std::exception& e) { LOG_ERROR(<< "Failed to get variance scale: " << e.what()); }
    double vh = std::exp(2.0 * m_GaussianMean + m_GammaRate / m_GammaShape * (2.0 / m_GaussianPrecision + 1.0)) *
                (std::exp(m_GammaRate / m_GammaShape) - 1.0);

    if (m_GammaShape < MINIMUM_LOGNORMAL_SHAPE) {
        try {
            detail::CVarianceKernel f(this->marginalLikelihoodMean(), m_GaussianMean, m_GaussianPrecision, m_GammaShape, m_GammaRate);
            boost::math::gamma_distribution<> gamma(m_GammaShape, 1.0 / m_GammaRate);
            TDoubleVec a(2);
            TDoubleVec b(2);
            a[0] = boost::math::quantile(gamma, 0.03);
            b[0] = boost::math::quantile(gamma, 0.97);

            boost::math::normal_distribution<> normal(m_GaussianMean, 1.0 / a[0] / m_GaussianPrecision);
            a[1] = boost::math::quantile(normal, 0.03);
            b[1] = boost::math::quantile(normal, 0.97);

            detail::CVarianceKernel::TValue variance;
            if (CIntegration::sparseGaussLegendre<CIntegration::OrderThree, CIntegration::TwoDimensions>(f, a, b, variance)) {
                double vl = variance(0) / variance(1);
                double alpha = std::min(2.0 * (1.0 - m_GammaShape / MINIMUM_LOGNORMAL_SHAPE), 1.0);
                return varianceScale * alpha * vl + (1.0 - alpha) * vh;
            }
        } catch (const std::exception& e) { LOG_ERROR(<< "Failed to calculate variance: " << e.what()); }
    }
    return varianceScale * vh;
}

CLogNormalMeanPrecConjugate::TDoubleDoublePr
CLogNormalMeanPrecConjugate::marginalLikelihoodConfidenceInterval(double percentage,
                                                                  const TWeightStyleVec& weightStyles,
                                                                  const TDouble4Vec& weights) const {
    if (this->isNonInformative()) {
        return this->marginalLikelihoodSupport();
    }

    percentage /= 100.0;
    percentage = CTools::truncate(percentage, 0.0, 1.0);

    // We use the fact that the marginal likelihood is a log-t distribution.

    try {
        double varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights) * maths_t::countVarianceScale(weightStyles, weights);

        double r = m_GammaRate / m_GammaShape;
        double s = std::exp(-r);
        double location;
        double scale;
        detail::locationAndScale(varianceScale, r, s, m_GaussianMean, m_GaussianPrecision, m_GammaRate, m_GammaShape, location, scale);
        LOG_TRACE(<< "location = " << location << ", scale = " << scale);

        if (m_GammaShape > MINIMUM_LOGNORMAL_SHAPE) {
            boost::math::lognormal_distribution<> logNormal(location, scale);
            double x1 = boost::math::quantile(logNormal, (1.0 - percentage) / 2.0) - m_Offset - (this->isInteger() ? 0.5 : 0.0);
            double x2 = percentage > 0.0
                            ? boost::math::quantile(logNormal, (1.0 + percentage) / 2.0) - m_Offset - (this->isInteger() ? 0.5 : 0.0)
                            : x1;
            LOG_TRACE(<< "x1 = " << x1 << ", x2 = " << x2);
            return std::make_pair(x1, x2);
        }
        CLogTDistribution logt(2.0 * m_GammaShape, location, scale);
        double x1 = quantile(logt, (1.0 - percentage) / 2.0) - m_Offset - (this->isInteger() ? 0.5 : 0.0);
        double x2 = percentage > 0.0 ? quantile(logt, (1.0 + percentage) / 2.0) - m_Offset - (this->isInteger() ? 0.5 : 0.0) : x1;
        LOG_TRACE(<< "x1 = " << x1 << ", x2 = " << x2);
        return std::make_pair(x1, x2);
    } catch (const std::exception& e) { LOG_ERROR(<< "Failed to compute confidence interval: " << e.what()); }

    return this->marginalLikelihoodSupport();
}

maths_t::EFloatingPointErrorStatus CLogNormalMeanPrecConjugate::jointLogMarginalLikelihood(const TWeightStyleVec& weightStyles,
                                                                                           const TDouble1Vec& samples,
                                                                                           const TDouble4Vec1Vec& weights,
                                                                                           double& result) const {
    result = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute likelihood for empty sample set");
        return maths_t::E_FpFailed;
    }

    if (samples.size() != weights.size()) {
        LOG_ERROR(<< "Mismatch in samples '" << core::CContainerPrinter::print(samples) << "' and weights '"
                  << core::CContainerPrinter::print(weights) << "'");
        return maths_t::E_FpFailed;
    }

    if (this->isNonInformative()) {
        // The non-informative likelihood is improper and effectively
        // zero everywhere. We use minus max double because
        // log(0) = HUGE_VALUE, which causes problems for Windows.
        // Calling code is notified when the calculation overflows
        // and should avoid taking the exponential since this will
        // underflow and pollute the floating point environment. This
        // may cause issues for some library function implementations
        // (see fe*exceptflag for more details).
        result = boost::numeric::bounds<double>::lowest();
        return maths_t::E_FpOverflowed;
    }

    detail::CLogMarginalLikelihood logMarginalLikelihood(
        weightStyles, samples, weights, m_Offset, m_GaussianMean, m_GaussianPrecision, m_GammaShape, m_GammaRate);
    if (this->isInteger()) {
        CIntegration::logGaussLegendre<CIntegration::OrderThree>(logMarginalLikelihood, 0.0, 1.0, result);
    } else {
        logMarginalLikelihood(0.0, result);
    }

    maths_t::EFloatingPointErrorStatus status =
        static_cast<maths_t::EFloatingPointErrorStatus>(logMarginalLikelihood.errorStatus() | CMathsFuncs::fpStatus(result));
    if (status & maths_t::E_FpFailed) {
        LOG_ERROR(<< "Failed to compute log likelihood (" << this->debug() << ")");
        LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR(<< "weights = " << core::CContainerPrinter::print(weights));
    } else if (status & maths_t::E_FpOverflowed) {
        LOG_TRACE(<< "Log likelihood overflowed for (" << this->debug() << ")");
        LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));
        LOG_TRACE(<< "weights = " << core::CContainerPrinter::print(weights));
    }
    return status;
}

void CLogNormalMeanPrecConjugate::sampleMarginalLikelihood(std::size_t numberSamples, TDouble1Vec& samples) const {
    samples.clear();

    if (numberSamples == 0 || this->numberSamples() == 0.0) {
        return;
    }

    if (this->isNonInformative()) {
        // We can't sample the marginal likelihood directly. This should
        // only happen if we've had one sample so just return that sample.
        samples.push_back(std::exp(m_GaussianMean) - m_Offset);
        return;
    }

    // The sampling strategy is to split the marginal likelihood up into
    // equal quantiles and then compute the expectation on each quantile
    // and sample that point, i.e. effectively sample the points:
    //   { n * E[ X * I{[q_n(i), q_n((i+1))]} ] }
    //
    // where,
    //   X is a R.V. whose distribution is the marginal likelihood.
    //   I{.} is the indicator function.
    //   q_n(.) is the n'th quantile function.
    //   i ranges over 0 to n-1.
    //   n is the number of samples.
    //
    // This sampling strategy has various nice properties:
    //   1) The sample quantiles match as many quantiles of the distribution
    //      as possible.
    //   2) The sample mean always matches the distribution mean:
    //        Sum_i( E[X * I{[q_n(i), q_n((i+1))]}] ) = E[X * 1] = E[X]
    //      by linearity of the expectation operator and since the indicators
    //      range over the entire support for X.
    //   3) As the number of samples increase each sample moment tends
    //      asymptotically to each corresponding distribution moment.
    //
    // X is log t-distributed. However, the expectation diverges for the
    // first and last quantile: it is equivalent to the moment generating
    // function of a t-distribution. So, instead we sample the approximate
    // log-normal distribution and use the relationship:
    //   E[ X * I{[x1,x2]} ] =
    //     [ 1/2
    //       * exp(m + (p+1)/p/2*b/a)                                       x2
    //       * erf((p/(p+1)*a/b * (x - m) - 1) / (2 * p/(p+1)*a/b) ^ (1/2) ]
    //                                                                      x1
    //   m and p are the prior Gaussian mean and precision, respectively.
    //   a and b are the prior Gamma shape and rate, respectively.
    //   erf(.) is the error function.

    samples.reserve(numberSamples);

    double scale = std::sqrt((m_GaussianPrecision + 1.0) / m_GaussianPrecision * m_GammaRate / m_GammaShape);
    try {
        boost::math::lognormal_distribution<> lognormal(m_GaussianMean, scale);

        double mean = boost::math::mean(lognormal);

        LOG_TRACE(<< "mean = " << mean << ", scale = " << scale << ", numberSamples = " << numberSamples);

        TDoubleDoublePr support = this->marginalLikelihoodSupport();

        double lastPartialExpectation = 0.0;

        for (std::size_t i = 1u; i < numberSamples; ++i) {
            double q = static_cast<double>(i) / static_cast<double>(numberSamples);
            double xq = std::log(boost::math::quantile(lognormal, q));

            double z = (xq - m_GaussianMean - scale * scale) / scale / boost::math::double_constants::root_two;

            double partialExpectation = mean * (1.0 + boost::math::erf(z)) / 2.0;

            double sample = static_cast<double>(numberSamples) * (partialExpectation - lastPartialExpectation) - m_Offset;

            LOG_TRACE(<< "sample = " << sample);

            // Sanity check the sample: should be in the distribution support.
            if (sample >= support.first && sample <= support.second) {
                samples.push_back(sample);
            } else {
                LOG_ERROR(<< "Sample out of bounds: sample = " << sample - m_Offset << ", gaussianMean = " << m_GaussianMean
                          << ", scale = " << scale << ", q = " << q << ", x(q) = " << xq << ", mean = " << mean);
            }

            lastPartialExpectation = partialExpectation;
        }

        double sample = static_cast<double>(numberSamples) * (mean - lastPartialExpectation) - m_Offset;

        LOG_TRACE(<< "sample = " << sample);

        if (sample >= support.first && sample <= support.second) {
            samples.push_back(sample);
        } else {
            LOG_ERROR(<< "Sample out of bounds: sample = " << sample << ", gaussianMean = " << m_GaussianMean << ", scale = " << scale
                      << ", mean = " << mean);
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to sample: " << e.what() << ", gaussianMean " << m_GaussianMean << ", scale = " << scale);
    }
}

bool CLogNormalMeanPrecConjugate::minusLogJointCdf(const TWeightStyleVec& weightStyles,
                                                   const TDouble1Vec& samples,
                                                   const TDouble4Vec1Vec& weights,
                                                   double& lowerBound,
                                                   double& upperBound) const {
    using TMinusLogCdf = detail::CEvaluateOnSamples<CTools::SMinusLogCdf>;

    lowerBound = upperBound = 0.0;

    TMinusLogCdf minusLogCdf(
        weightStyles, samples, weights, this->isNonInformative(), m_Offset, m_GaussianMean, m_GaussianPrecision, m_GammaShape, m_GammaRate);

    if (this->isInteger()) {
        // If the data are discrete we compute the approximate expectation
        // w.r.t. to the hidden offset of the samples Z, which is uniform
        // on the interval [0,1].
        double value;
        if (!CIntegration::logGaussLegendre<CIntegration::OrderThree>(minusLogCdf, 0.0, 1.0, value)) {
            LOG_ERROR(<< "Failed computing c.d.f. for " << core::CContainerPrinter::print(samples));
            return false;
        }

        lowerBound = upperBound = value;
        return true;
    }

    double value;
    if (!minusLogCdf(0.0, value)) {
        LOG_ERROR(<< "Failed computing c.d.f for " << core::CContainerPrinter::print(samples));
        return false;
    }

    lowerBound = upperBound = value;
    return true;
}

bool CLogNormalMeanPrecConjugate::minusLogJointCdfComplement(const TWeightStyleVec& weightStyles,
                                                             const TDouble1Vec& samples,
                                                             const TDouble4Vec1Vec& weights,
                                                             double& lowerBound,
                                                             double& upperBound) const {
    using TMinusLogCdfComplement = detail::CEvaluateOnSamples<CTools::SMinusLogCdfComplement>;

    lowerBound = upperBound = 0.0;

    TMinusLogCdfComplement minusLogCdfComplement(
        weightStyles, samples, weights, this->isNonInformative(), m_Offset, m_GaussianMean, m_GaussianPrecision, m_GammaShape, m_GammaRate);

    if (this->isInteger()) {
        // If the data are discrete we compute the approximate expectation
        // w.r.t. to the hidden offset of the samples Z, which is uniform
        // on the interval [0,1].
        double value;
        if (!CIntegration::logGaussLegendre<CIntegration::OrderThree>(minusLogCdfComplement, 0.0, 1.0, value)) {
            LOG_ERROR(<< "Failed computing c.d.f. complement for " << core::CContainerPrinter::print(samples));
            return false;
        }

        lowerBound = upperBound = value;
        return true;
    }

    double value;
    if (!minusLogCdfComplement(0.0, value)) {
        LOG_ERROR(<< "Failed computing c.d.f complement for " << core::CContainerPrinter::print(samples));
        return false;
    }

    lowerBound = upperBound = value;
    return true;
}

bool CLogNormalMeanPrecConjugate::probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                                 const TWeightStyleVec& weightStyles,
                                                                 const TDouble1Vec& samples,
                                                                 const TDouble4Vec1Vec& weights,
                                                                 double& lowerBound,
                                                                 double& upperBound,
                                                                 maths_t::ETail& tail) const {
    lowerBound = upperBound = 0.0;
    tail = maths_t::E_UndeterminedTail;

    detail::CProbabilityOfLessLikelySamples probability(calculation,
                                                        weightStyles,
                                                        samples,
                                                        weights,
                                                        this->isNonInformative(),
                                                        m_Offset,
                                                        m_GaussianMean,
                                                        m_GaussianPrecision,
                                                        m_GammaShape,
                                                        m_GammaRate);

    if (this->isInteger()) {
        // If the data are discrete we compute the approximate expectation
        // w.r.t. to the hidden offset of the samples Z, which is uniform
        // on the interval [0,1].
        double value;
        if (!CIntegration::gaussLegendre<CIntegration::OrderThree>(probability, 0.0, 1.0, value)) {
            LOG_ERROR(<< "Failed computing probability for " << core::CContainerPrinter::print(samples));
            return false;
        }

        lowerBound = upperBound = value;
        tail = probability.tail();

        return true;
    }

    double value;
    if (!probability(0.0, value)) {
        LOG_ERROR(<< "Failed computing probability for " << core::CContainerPrinter::print(samples));
        return false;
    }

    lowerBound = upperBound = value;
    tail = probability.tail();

    return true;
}

bool CLogNormalMeanPrecConjugate::isNonInformative() const {
    return m_GammaRate == NON_INFORMATIVE_RATE || m_GaussianPrecision == NON_INFORMATIVE_PRECISION;
}

void CLogNormalMeanPrecConjugate::print(const std::string& indent, std::string& result) const {
    result += core_t::LINE_ENDING + indent + "log-normal ";
    if (this->isNonInformative()) {
        result += "non-informative";
        return;
    }

    double scale = std::sqrt((m_GaussianPrecision + 1.0) / m_GaussianPrecision * m_GammaRate / m_GammaShape);
    try {
        boost::math::lognormal_distribution<> lognormal(m_GaussianMean, scale);
        double mean = boost::math::mean(lognormal);
        double deviation = boost::math::standard_deviation(lognormal);
        result += "mean = " + core::CStringUtils::typeToStringPretty(mean - m_Offset) +
                  " sd = " + core::CStringUtils::typeToStringPretty(deviation);
        return;
    } catch (const std::exception&) {}
    result += "mean = <unknown> variance = <unknown>";
}

std::string CLogNormalMeanPrecConjugate::printJointDensityFunction() const {
    if (this->isNonInformative()) {
        // The non-informative prior is improper and effectively 0 everywhere.
        return std::string();
    }

    // We'll plot the prior over a range where most of the mass is.

    static const double RANGE = 0.99;
    static const unsigned int POINTS = 51;

    boost::math::gamma_distribution<> gamma(m_GammaShape, 1.0 / m_GammaRate);

    double precision = m_GaussianPrecision * this->normalPrecision();
    boost::math::normal_distribution<> gaussian(m_GaussianMean, 1.0 / std::sqrt(precision));

    double xStart = boost::math::quantile(gamma, (1.0 - RANGE) / 2.0);
    double xEnd = boost::math::quantile(gamma, (1.0 + RANGE) / 2.0);
    double xIncrement = (xEnd - xStart) / (POINTS - 1.0);
    double x = xStart;

    double yStart = boost::math::quantile(gaussian, (1.0 - RANGE) / 2.0);
    double yEnd = boost::math::quantile(gaussian, (1.0 + RANGE) / 2.0);
    double yIncrement = (yEnd - yStart) / (POINTS - 1.0);
    double y = yStart;

    std::ostringstream xCoordinates;
    std::ostringstream yCoordinates;
    xCoordinates << "x = [";
    yCoordinates << "y = [";
    for (unsigned int i = 0u; i < POINTS; ++i, x += xIncrement, y += yIncrement) {
        xCoordinates << x << " ";
        yCoordinates << y << " ";
    }
    xCoordinates << "];" << core_t::LINE_ENDING;
    yCoordinates << "];" << core_t::LINE_ENDING;

    std::ostringstream pdf;
    pdf << "pdf = [";
    x = xStart;
    for (unsigned int i = 0u; i < POINTS; ++i, x += xIncrement) {
        y = yStart;
        for (unsigned int j = 0u; j < POINTS; ++j, y += yIncrement) {
            double conditionalPrecision = m_GaussianPrecision * x;
            boost::math::normal_distribution<> conditionalGaussian(m_GaussianMean, 1.0 / std::sqrt(conditionalPrecision));

            pdf << (CTools::safePdf(gamma, x) * CTools::safePdf(conditionalGaussian, y)) << " ";
        }
        pdf << core_t::LINE_ENDING;
    }
    pdf << "];" << core_t::LINE_ENDING << "mesh(x, y, pdf);";

    return xCoordinates.str() + yCoordinates.str() + pdf.str();
}

uint64_t CLogNormalMeanPrecConjugate::checksum(uint64_t seed) const {
    seed = this->CPrior::checksum(seed);
    seed = CChecksum::calculate(seed, m_Offset);
    seed = CChecksum::calculate(seed, m_GaussianMean);
    seed = CChecksum::calculate(seed, m_GaussianPrecision);
    seed = CChecksum::calculate(seed, m_GammaShape);
    return CChecksum::calculate(seed, m_GammaRate);
}

void CLogNormalMeanPrecConjugate::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CLogNormalMeanPrecConjugate");
}

std::size_t CLogNormalMeanPrecConjugate::memoryUsage() const {
    return 0;
}

std::size_t CLogNormalMeanPrecConjugate::staticSize() const {
    return sizeof(*this);
}

void CLogNormalMeanPrecConjugate::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(DECAY_RATE_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(OFFSET_TAG, m_Offset, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(GAUSSIAN_MEAN_TAG, m_GaussianMean, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(GAUSSIAN_PRECISION_TAG, m_GaussianPrecision, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(GAMMA_SHAPE_TAG, m_GammaShape, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(GAMMA_RATE_TAG, m_GammaRate, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(NUMBER_SAMPLES_TAG, this->numberSamples(), core::CIEEE754::E_SinglePrecision);
}

double CLogNormalMeanPrecConjugate::normalMean() const {
    return m_GaussianMean;
}

double CLogNormalMeanPrecConjugate::normalPrecision() const {
    if (this->isNonInformative()) {
        return 0.0;
    }

    try {
        boost::math::gamma_distribution<> gamma(m_GammaShape, 1.0 / m_GammaRate);
        return boost::math::mean(gamma);
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to create prior: " << e.what() << " shape = " << m_GammaShape << ", rate = " << m_GammaRate);
    }

    return 0.0;
}

CLogNormalMeanPrecConjugate::TDoubleDoublePr CLogNormalMeanPrecConjugate::confidenceIntervalNormalMean(double percentage) const {
    if (this->isNonInformative()) {
        return std::make_pair(boost::numeric::bounds<double>::lowest(), boost::numeric::bounds<double>::highest());
    }

    // Compute the symmetric confidence interval around the median of the
    // distribution from the percentiles, i.e. find the percentiles q(1)
    // and q(2) which satisfy:
    //   q(1) + q(2) = 1.0
    //   q(2) - q(1) = percentage
    //
    // We assume the data are described by X = exp(Y) - u where, Y is normally
    // distributed and u is a constant offset.
    //
    // The marginal prior distribution for the mean, M, of Y is a t distribution
    // with 2 * a degrees of freedom, location m and precision a * p / b.
    // We can compute the percentiles for this distribution from the student's
    // t distribution by noting that:
    //   (p * a / b) ^ (1/2) * (M - m) ~ student's t
    //
    // So the percentiles of the student's t map to the percentiles of M as follows:
    //   x_m_q = m + x_q_students / (a * p / b) ^ (1/2).

    percentage /= 100.0;
    double lowerPercentile = 0.5 * (1.0 - percentage);
    double upperPercentile = 0.5 * (1.0 + percentage);

    boost::math::students_t_distribution<> students(2.0 * m_GammaShape);

    double xLower = boost::math::quantile(students, lowerPercentile);
    double xUpper = boost::math::quantile(students, upperPercentile);

    boost::math::gamma_distribution<> gamma(m_GammaShape, 1.0 / m_GammaRate);
    double precision = m_GaussianPrecision * this->normalPrecision();
    xLower = m_GaussianMean + xLower / std::sqrt(precision);
    xUpper = m_GaussianMean + xUpper / std::sqrt(precision);

    return std::make_pair(xLower, xUpper);
}

CLogNormalMeanPrecConjugate::TDoubleDoublePr CLogNormalMeanPrecConjugate::confidenceIntervalNormalPrecision(double percentage) const {
    if (this->isNonInformative()) {
        return std::make_pair(boost::numeric::bounds<double>::lowest(), boost::numeric::bounds<double>::highest());
    }

    percentage /= 100.0;
    double lowerPercentile = 0.5 * (1.0 - percentage);
    double upperPercentile = 0.5 * (1.0 + percentage);

    // The marginal prior distribution for the precision is gamma.
    boost::math::gamma_distribution<> gamma(m_GammaShape, 1.0 / m_GammaRate);

    return std::make_pair(boost::math::quantile(gamma, lowerPercentile), boost::math::quantile(gamma, upperPercentile));
}

bool CLogNormalMeanPrecConjugate::equalTolerance(const CLogNormalMeanPrecConjugate& rhs, const TEqualWithTolerance& equal) const {
    LOG_DEBUG(<< m_GaussianMean << " " << rhs.m_GaussianMean << ", " << m_GaussianPrecision << " " << rhs.m_GaussianPrecision << ", "
              << m_GammaShape << " " << rhs.m_GammaShape << ", " << m_GammaRate << " " << rhs.m_GammaRate);
    return equal(m_GaussianMean, rhs.m_GaussianMean) && equal(m_GaussianPrecision, rhs.m_GaussianPrecision) &&
           equal(m_GammaShape, rhs.m_GammaShape) && equal(m_GammaRate, rhs.m_GammaRate);
}

double CLogNormalMeanPrecConjugate::mean() const {
    if (this->isNonInformative()) {
        return std::exp(m_GaussianMean) - m_Offset;
    }

    // This is just
    //   E_{M, P}[E[X | M, P]] - u                                (1)
    //
    // where M, P likelihood mean and precision, respectively, and u is
    // the offset. Note that X conditioned on M and P is log normal and
    //   E[X | M, P] = exp(M + 1/P/2)                             (2)
    //
    // Since the prior and the rhs of (2) factorizes so does the expectation
    // and we can evaluate the expectation over m' in (1) to give
    //   E_{M, P}[E[X | M, P]] = exp(m) * E_{P}[exp(1/2/P * (1/t + 1))]
    //
    // where m and t are the conditional prior mean and precision scale,
    // respectively. This last integral has no closed form solution, we
    // evaluate by numerical integration when the prior is wide and use:
    //   E[exp(1/2/P * (1/t + 1))] ~= exp(1/2/E[P] * (1/t + 1))
    //                             ~= exp(b/a/2 * (1/t + 1))
    //
    // when it is narrow.

    if (m_GammaShape < MINIMUM_LOGNORMAL_SHAPE) {
        try {
            detail::CMeanKernel f(m_GaussianMean, m_GaussianPrecision, m_GammaShape, m_GammaRate);
            boost::math::gamma_distribution<> gamma(m_GammaShape, 1.0 / m_GammaRate);
            double a = boost::math::quantile(gamma, 0.1);
            double b = boost::math::quantile(gamma, 0.9);
            detail::CMeanKernel::TValue result;
            if (CIntegration::gaussLegendre<CIntegration::OrderSeven>(f, a, b, result)) {
                return result(0) / result(1) - m_Offset;
            }
        } catch (const std::exception& e) { LOG_ERROR(<< "Failed to calculate mean: " << e.what()); }
    }
    return std::exp(m_GaussianMean + 0.5 * m_GammaRate / m_GammaShape * (1.0 / m_GaussianPrecision + 1.0)) - m_Offset;
}

bool CLogNormalMeanPrecConjugate::isBad() const {
    return !CMathsFuncs::isFinite(m_Offset) || !CMathsFuncs::isFinite(m_GaussianMean) || !CMathsFuncs::isFinite(m_GaussianPrecision) ||
           !CMathsFuncs::isFinite(m_GammaShape) || !CMathsFuncs::isFinite(m_GammaRate);
}

std::string CLogNormalMeanPrecConjugate::debug() const {
    std::ostringstream result;
    result << std::scientific << std::setprecision(15) << m_Offset << " " << m_GaussianMean << " " << m_GaussianMean << " " << m_GammaShape
           << " " << m_GammaRate;
    return result.str();
}

const double CLogNormalMeanPrecConjugate::NON_INFORMATIVE_MEAN = 0.0;
const double CLogNormalMeanPrecConjugate::NON_INFORMATIVE_PRECISION = 0.0;
const double CLogNormalMeanPrecConjugate::NON_INFORMATIVE_SHAPE = 1.0;
const double CLogNormalMeanPrecConjugate::NON_INFORMATIVE_RATE = 0.0;
}
}
