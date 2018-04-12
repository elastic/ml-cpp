/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CPoissonMeanConjugate.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CMathsFuncs.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <boost/bind.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/numeric/conversion/bounds.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

namespace ml {
namespace maths {

namespace {

const double MINIMUM_GAUSSIAN_MEAN = 100.0;

// Wrapper for static cast which can be used with STL algorithms.
template<typename TARGET_TYPE>
struct SStaticCast {
    template<typename SOURCE_TYPE>
    inline TARGET_TYPE operator()(const SOURCE_TYPE& source) const {
        return static_cast<TARGET_TYPE>(source);
    }
};

namespace detail {

using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble4Vec = core::CSmallVector<double, 4>;
using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
using TWeightStyleVec = maths_t::TWeightStyleVec;

//! Adds "weight" x "right operand" to the "left operand".
struct SPlusWeight {
    double operator()(double lhs, double rhs, double weight = 1.0) const {
        return lhs + weight * rhs;
    }
};

//! Evaluate \p func on the joint predictive distribution for \p samples
//! (integrating over the prior for the Poisson rate) and aggregate the
//! results using \p aggregate.
//!
//! \param[in] weightStyles Controls the interpretation of the weight(s) that
//! are associated with each sample. See maths_t::ESampleWeightStyle for more
//! details.
//! \param[in] samples The weighted samples.
//! \param[in] func The function to evaluate.
//! \param[in] aggregate The function to aggregate the results of \p func.
//! \param[in] offset The offset to apply to the data.
//! \param[in] isNonInformative True if the prior is non-informative.
//! \param[in] shape The shape of the rate prior.
//! \param[in] rate The rate of the rate prior.
//! \param[out] result Filled in with the aggregation of results of \p func.
template<typename FUNC, typename AGGREGATOR, typename RESULT>
bool evaluateFunctionOnJointDistribution(const TWeightStyleVec& weightStyles,
                                         const TDouble1Vec& samples,
                                         const TDouble4Vec1Vec& weights,
                                         FUNC func,
                                         AGGREGATOR aggregate,
                                         double offset,
                                         bool isNonInformative,
                                         double shape,
                                         double rate,
                                         RESULT& result) {
    result = RESULT();

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute distribution for empty sample set");
        return false;
    }

    // In order to compute the true joint marginal likelihood of the samples
    // integrating over the prior distribution for the mean of the Poisson
    // process we end up having to evaluate partial sums over the binomial
    // coefficients which are not closed form. The brute force evaluation
    // doesn't scale. There are various approximation schemes we could consider,
    // but in the first instance I think it is sufficient to approximate:
    //   Integral{ Product_i{ L(x(i) | u) } * f(u) }du
    //      ~= Product_i{ Integral{ L(x(i) | u) * f(u) }du }.
    //
    // where,
    //   L(. | u) is the likelihood function and
    //   f(u) is the prior for the mean.
    //
    // This becomes increasingly accurate as the prior distribution narrows.

    try {
        if (isNonInformative) {
            // The non-informative prior is improper and effectively 0 everywhere.
            // (It is acceptable to approximate all finite samples as at the median
            // of this distribution.)
            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double x = samples[i] + offset;
                double n = maths_t::count(weightStyles, weights[i]);
                result = aggregate(result, func(CTools::SImproperDistribution(), x), n);
            }
        } else {
            // The marginal likelihood for a single sample is the negative
            // binomial distribution:
            //   f(x | p, r) = Gamma(r + x) * p^r * (1 - p)^x / x! / Gamma(r)
            //
            // where,
            //   p = 1 - 1 / (b+1)
            //   r = a
            //
            // For large prior mean the marginal likelihood is well approximated
            // by a moment matched Gaussian, i.e. N(a/b, a * (b+1)/b^2) where
            // "a" is the shape and "b" is the rate of the gamma distribution,
            // and the error function is significantly cheaper to compute.

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = maths_t::count(weightStyles, weights[i]);
                double x = samples[i] + offset;

                double mean = shape / rate;
                if (mean > MINIMUM_GAUSSIAN_MEAN) {
                    double deviation = std::sqrt((rate + 1.0) / rate * mean);
                    boost::math::normal_distribution<> normal(mean, deviation);
                    result = aggregate(result, func(normal, x), n);
                } else {
                    double r = shape;
                    double p = rate / (rate + 1.0);
                    boost::math::negative_binomial_distribution<> negativeBinomial(r, p);
                    result = aggregate(result, func(negativeBinomial, x), n);
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Error calculating joint c.d.f."
                  << " offset = " << offset << ", shape = " << shape
                  << ", rate = " << rate << ": " << e.what());
        return false;
    }

    LOG_TRACE(<< "result = " << result);

    return true;
}

} // detail::

// We use short field names to reduce the state size
const std::string SHAPE_TAG("a");
const std::string RATE_TAG("b");
const std::string NUMBER_SAMPLES_TAG("c");
const std::string OFFSET_TAG("d");
//const std::string MINIMUM_TAG("e"); No longer used
//const std::string MAXIMUM_TAG("f"); No longer used
const std::string DECAY_RATE_TAG("g");
const std::string EMPTY_STRING;
}

CPoissonMeanConjugate::CPoissonMeanConjugate(double offset, double shape, double rate, double decayRate /*= 0.0*/)
    : CPrior(maths_t::E_IntegerData, decayRate), m_Offset(offset),
      m_Shape(shape), m_Rate(rate) {
}

CPoissonMeanConjugate::CPoissonMeanConjugate(const SDistributionRestoreParams& params,
                                             core::CStateRestoreTraverser& traverser)
    : CPrior(maths_t::E_IntegerData, params.s_DecayRate), m_Offset(0.0),
      m_Shape(0.0), m_Rate(0.0) {
    traverser.traverseSubLevel(
        boost::bind(&CPoissonMeanConjugate::acceptRestoreTraverser, this, _1));
}

bool CPoissonMeanConjugate::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(DECAY_RATE_TAG, double decayRate,
                               core::CStringUtils::stringToType(traverser.value(), decayRate),
                               this->decayRate(decayRate))
        RESTORE_BUILT_IN(OFFSET_TAG, m_Offset)
        RESTORE_BUILT_IN(SHAPE_TAG, m_Shape)
        RESTORE_BUILT_IN(RATE_TAG, m_Rate)
        RESTORE_SETUP_TEARDOWN(NUMBER_SAMPLES_TAG, double numberSamples,
                               core::CStringUtils::stringToType(traverser.value(), numberSamples),
                               this->numberSamples(numberSamples))
    } while (traverser.next());

    return true;
}

CPoissonMeanConjugate CPoissonMeanConjugate::nonInformativePrior(double offset, double decayRate) {
    // We'll use the improper distribution:
    //   lim "a -> 1+, k -> inf" { Gamma(a, k) }
    //
    // Since we have defined the gamma distribution in terms of the inverse
    // scale "k -> inf" is equivalent to "b = 1 / k -> 0.0".

    return CPoissonMeanConjugate(offset, NON_INFORMATIVE_SHAPE,
                                 NON_INFORMATIVE_RATE, decayRate);
}

CPoissonMeanConjugate::EPrior CPoissonMeanConjugate::type() const {
    return E_Poisson;
}

CPoissonMeanConjugate* CPoissonMeanConjugate::clone() const {
    return new CPoissonMeanConjugate(*this);
}

void CPoissonMeanConjugate::setToNonInformative(double offset, double decayRate) {
    *this = nonInformativePrior(offset, decayRate);
}

bool CPoissonMeanConjugate::needsOffset() const {
    return true;
}

double CPoissonMeanConjugate::adjustOffset(const TWeightStyleVec& /*weightStyles*/,
                                           const TDouble1Vec& samples,
                                           const TDouble4Vec1Vec& /*weights*/) {
    if (samples.empty() ||
        CMathsFuncs::beginFinite(samples) == CMathsFuncs::endFinite(samples)) {
        return 0.0;
    }

    // Ideally we'd like to minimize a suitable measure of the difference
    // between the two marginal likelihoods w.r.t. to the new prior parameters.
    // However, even the Kullback-Leibler divergence can't be computed in
    // closed form, so there is no easy way of doing this. We would have
    // to use a non-linear maximization scheme, which computes the divergence
    // numerically to evaluate the objective function, but this will be
    // slow and it would be difficult to analyse its numerical stability.
    // Instead we simply sample marginal likelihood and update the shifted
    // prior with these samples.

    static const double EPS = 0.01;
    static const double OFFSET_MARGIN = 0.0;

    double minimumSample = *std::min_element(CMathsFuncs::beginFinite(samples),
                                             CMathsFuncs::endFinite(samples));
    if (minimumSample + m_Offset >= OFFSET_MARGIN) {
        return 0.0;
    }

    TWeightStyleVec weightStyle(1, maths_t::E_SampleCountWeight);
    double offset = OFFSET_MARGIN - minimumSample;
    TDouble1Vec resamples;
    this->sampleMarginalLikelihood(ADJUST_OFFSET_SAMPLE_SIZE, resamples);
    double weight = this->numberSamples() / static_cast<double>(resamples.size());
    TDouble4Vec1Vec weights(resamples.size(), TDouble4Vec(1, weight));

    double before = 0.0;
    if (!resamples.empty()) {
        this->jointLogMarginalLikelihood(CConstantWeights::COUNT, resamples, weights, before);
    }

    // Reset the parameters.
    m_Offset = (offset < 0.0 ? (1.0 - EPS) : (1.0 + EPS)) * offset;
    m_Shape = NON_INFORMATIVE_SHAPE;
    m_Rate = NON_INFORMATIVE_RATE;
    this->numberSamples(0.0);

    if (resamples.empty()) {
        return 0.0;
    }

    for (auto& sample : resamples) {
        sample = std::max(sample, OFFSET_MARGIN - offset);
    }

    LOG_TRACE(<< "resamples = " << core::CContainerPrinter::print(resamples)
              << ", weight = " << weight << ", offset = " << m_Offset);

    this->addSamples(weightStyle, resamples, weights);

    double after;
    this->jointLogMarginalLikelihood(CConstantWeights::COUNT, resamples, weights, after);

    return std::min(after - before, 0.0);
}

double CPoissonMeanConjugate::offset() const {
    return m_Offset;
}

void CPoissonMeanConjugate::addSamples(const TWeightStyleVec& weightStyles,
                                       const TDouble1Vec& samples,
                                       const TDouble4Vec1Vec& weights) {
    if (samples.empty()) {
        return;
    }

    if (samples.size() != weights.size()) {
        LOG_ERROR(<< "Mismatch in samples '"
                  << core::CContainerPrinter::print(samples) << "' and weights '"
                  << core::CContainerPrinter::print(weights) << "'");
        return;
    }

    this->adjustOffset(weightStyles, samples, weights);
    this->CPrior::addSamples(weightStyles, samples, weights);

    // The update of the posterior with n independent samples of the
    // Poisson distribution comes from:
    //   ( exp(-u) * u^x ) * ( u^(a - 1) * exp(-b * u) )
    //     "likelihood"               "prior"
    //
    // This implies that the parameters of the gamma distribution,
    // Gamma(u | a, b) where a is the shape parameter and b is the
    // rate parameter, update as follows:
    //   a -> a + n(i) * Sum_i( x(i) )
    //   b -> b + Sum_i( n(i) * x(i) )
    //
    // where,
    //   x = {x(1), x(2), ... , x(n)} the sample vector.
    //
    // Note that the weight of the sample x(i) is interpreted as its
    // count, i.e. n(i), so for example updating with {(x, 2)} is
    // equivalent to updating with {x, x}.

    double numberSamples = 0.0;
    double sampleSum = 0.0;
    try {
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double n = maths_t::countForUpdate(weightStyles, weights[i]);
            double x = samples[i] + m_Offset;
            if (!CMathsFuncs::isFinite(x) || x < 0.0) {
                LOG_ERROR(<< "Discarding " << x << " it's not Poisson");
                continue;
            }
            numberSamples += n;
            sampleSum += n * x;
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to update likelihood: " << e.what());
        return;
    }

    m_Shape += sampleSum;
    m_Rate += numberSamples;

    LOG_TRACE(<< "# samples = " << numberSamples
              << ", sampleSum = " << sampleSum << ", m_Shape = " << m_Shape
              << ", m_Rate = " << m_Rate << ", m_Offset = " << m_Offset);
}

void CPoissonMeanConjugate::propagateForwardsByTime(double time) {
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
        LOG_ERROR(<< "Bad propagation time " << time);
        return;
    }

    if (this->isNonInformative()) {
        // There is nothing to be done.
        return;
    }

    double alpha = std::exp(-this->decayRate() * time);

    // We want to increase the variance of the gamma distribution
    // while holding its mean constant s.t. in the limit t -> inf
    // var -> inf. The mean and variance are a / b and a / b^2,
    // respectively, for shape "a" and rate "b" so choose a factor
    // f in the range [0, 1] and update as follows:
    //   a' -> f * a
    //   b' -> f * b
    //
    // Thus the mean is unchanged and variance is increased by 1 / f.

    double factor = std::min(
        (alpha * m_Shape + (1.0 - alpha) * NON_INFORMATIVE_SHAPE) / m_Shape, 1.0);

    m_Shape *= factor;
    m_Rate *= factor;

    this->numberSamples(this->numberSamples() * alpha);

    LOG_TRACE(<< "time = " << time << ", alpha = " << alpha
              << ", m_Shape = " << m_Shape << ", m_Rate = " << m_Rate
              << ", numberSamples = " << this->numberSamples());
}

CPoissonMeanConjugate::TDoubleDoublePr CPoissonMeanConjugate::marginalLikelihoodSupport() const {
    return std::make_pair(-m_Offset, boost::numeric::bounds<double>::highest());
}

double CPoissonMeanConjugate::marginalLikelihoodMean() const {
    if (this->isNonInformative()) {
        return -m_Offset;
    }

    // We use the fact that E[X} = E_{a,b}[E[X | a,b]]
    //                           = E_{a,b}[ m ]
    //                           = "prior mean"

    return this->priorMean() - m_Offset;
}

double CPoissonMeanConjugate::marginalLikelihoodMode(const TWeightStyleVec& /*weightStyles*/,
                                                     const TDouble4Vec& /*weights*/) const {
    if (this->isNonInformative()) {
        return -m_Offset;
    }

    // boost::math::negative_binomial_distribution is broken for
    // successes <= 1.0.

    if (m_Shape <= 1.0) {
        return -m_Offset;
    }

    // We use the fact that the marginal likelihood is negative
    // binomial.

    try {
        double r = m_Shape;
        double p = m_Rate / (m_Rate + 1.0);
        boost::math::negative_binomial_distribution<> negativeBinomial(r, p);
        return boost::math::mode(negativeBinomial) - m_Offset;
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute marginal likelihood mode: " << e.what()
                  << ", prior shape = " << m_Shape << ", prior rate = " << m_Rate);
    }

    return -m_Offset;
}

double CPoissonMeanConjugate::marginalLikelihoodVariance(const TWeightStyleVec& weightStyles,
                                                         const TDouble4Vec& weights) const {
    if (this->isNonInformative()) {
        return boost::numeric::bounds<double>::highest();
    }

    // We use the fact that E[X} = E_{R}[Var[X | R]]
    //                           = E_{R}[ R + (R - a/b)^2 ]
    //                           = "prior mean" + "prior variance"

    double varianceScale = 1.0;
    try {
        varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights) *
                        maths_t::countVarianceScale(weightStyles, weights);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to get variance scale: " << e.what());
    }
    return varianceScale * (this->priorMean() + this->priorVariance());
}

CPoissonMeanConjugate::TDoubleDoublePr
CPoissonMeanConjugate::marginalLikelihoodConfidenceInterval(double percentage,
                                                            const TWeightStyleVec& /*weightStyles*/,
                                                            const TDouble4Vec& /*weights*/) const {
    if (this->isNonInformative()) {
        return this->marginalLikelihoodSupport();
    }

    percentage /= 100.0;
    percentage = CTools::truncate(percentage, 0.0, 1.0);

    // We use the fact that the marginal likelihood function is
    // negative binomial.

    try {
        double r = m_Shape;
        double p = m_Rate / (m_Rate + 1.0);
        boost::math::negative_binomial_distribution<> negativeBinomial(r, p);
        double x1 = boost::math::quantile(negativeBinomial, (1.0 - percentage) / 2.0) - m_Offset;
        double x2 = percentage > 0.0
                        ? boost::math::quantile(negativeBinomial, (1.0 + percentage) / 2.0) - m_Offset
                        : x1;
        return std::make_pair(x1, x2);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute confidence interval: " << e.what());
    }

    return this->marginalLikelihoodSupport();
}

maths_t::EFloatingPointErrorStatus
CPoissonMeanConjugate::jointLogMarginalLikelihood(const TWeightStyleVec& weightStyles,
                                                  const TDouble1Vec& samples,
                                                  const TDouble4Vec1Vec& weights,
                                                  double& result) const {
    result = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute likelihood for empty sample set");
        return maths_t::E_FpFailed;
    }

    if (samples.size() != weights.size()) {
        LOG_ERROR(<< "Mismatch in samples '"
                  << core::CContainerPrinter::print(samples) << "' and weights '"
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

    // Compute the log likelihood function of the samples integrating
    // over the prior distribution for the mean of the Poisson process.
    // In particular, it can be shown that:
    //   log( likelihood(x) ) =
    //     log( Gamma(a + Sum_i( x(i) ))
    //          / (b + n) ^ (a + Sum_i( x(i) ))
    //          / Product_i{ x(i)! }
    //          / Gamma(a)
    //          * b ^ a ).
    //
    // Here,
    //   x = {x(1), x(2), ... , x(n)} the sample vector.
    //   n = |x| the number of elements in the sample vector.
    //   a is the prior gamma shape
    //   b is the prior gamma rate

    try {
        // Calculate the statistics we need for the calculation.
        double numberSamples = 0.0;
        double sampleSum = 0.0;
        double sampleLogFactorialSum = 0.0;

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double n = maths_t::countForUpdate(weightStyles, weights[i]);
            double x = samples[i] + m_Offset;
            if (x < 0.0) {
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
                return maths_t::E_FpOverflowed;
            }

            numberSamples += n;
            sampleSum += n * x;
            // Recall n! = Gamma(n + 1).
            sampleLogFactorialSum += n * boost::math::lgamma(x + 1.0);
        }

        // Get the implied shape parameter for the gamma distribution
        // including the samples.
        double impliedShape = m_Shape + sampleSum;
        double impliedRate = m_Rate + numberSamples;

        result = boost::math::lgamma(impliedShape) + m_Shape * std::log(m_Rate) -
                 impliedShape * std::log(impliedRate) - sampleLogFactorialSum -
                 boost::math::lgamma(m_Shape);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Error calculating marginal likelihood: " << e.what());
        return maths_t::E_FpFailed;
    }

    maths_t::EFloatingPointErrorStatus status = CMathsFuncs::fpStatus(result);
    if (status & maths_t::E_FpFailed) {
        LOG_ERROR(<< "Failed to compute log likelihood");
        LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR(<< "weights = " << core::CContainerPrinter::print(weights));
    }
    return status;
}

void CPoissonMeanConjugate::sampleMarginalLikelihood(std::size_t numberSamples,
                                                     TDouble1Vec& samples) const {
    samples.clear();

    if (numberSamples == 0 || this->isNonInformative()) {
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
    //   3) As the number of samples increase, each sample moment tends
    //      asymptotically to each corresponding distribution moment.
    //
    // X is negative binomial distributed, but for large mean it is well
    // approximated by a moment matched Gaussian. In the negative binomial
    // distribution limit we use the relationship:
    //                                                            x2
    //   E[ X * I{[x1,x2]} ] = [(1-p) / p * r * F(x - 1 | p, r+1)]
    //                                                            x1
    //
    // In the normal distribution limit we use the relationship:
    //   E[ X * I{[x1,x2]} ] =
    //     [a/b/2 * erf((x - a/b) * b / (2*a*(b+1))^(1/2))                     x2
    //      - (a*(b+1)/2/pi)^(1/2) / b * exp(-b^2 / (2*a*(b+1)) * (x - a/b)^2)]
    //                                                                         x1
    // where,
    //   a is the prior gamma shape.
    //   b is the prior gamma rate.
    //   erf(.) is the error function.

    samples.reserve(numberSamples);

    TDoubleDoublePr support = this->marginalLikelihoodSupport();

    boost::math::gamma_distribution<> gamma(m_Shape, 1.0 / m_Rate);
    double mean = boost::math::mean(gamma);
    double lastPartialExpectation = 0.0;

    if (mean > MINIMUM_GAUSSIAN_MEAN) {
        double variance = mean + this->priorVariance();

        LOG_TRACE(<< "mean = " << mean << ", variance = " << variance);

        try {
            boost::math::normal_distribution<> normal(mean, std::sqrt(variance));

            for (std::size_t i = 1u; i < numberSamples; ++i) {
                double q = static_cast<double>(i) / static_cast<double>(numberSamples);
                double xq = boost::math::quantile(normal, q);

                double partialExpectation = mean * q -
                                            variance * CTools::safePdf(normal, xq);

                double sample = static_cast<double>(numberSamples) *
                                    (partialExpectation - lastPartialExpectation) -
                                m_Offset;

                LOG_TRACE(<< "sample = " << sample);

                // Sanity check the sample: should be in the distribution support.
                if (sample >= support.first && sample <= support.second) {
                    samples.push_back(sample);
                } else {
                    LOG_ERROR(<< "Sample out of bounds: sample = " << sample << ", support = ["
                              << support.first << "," << support.second << "]"
                              << ", mean = " << mean << ", variance = " << variance
                              << ", q = " << q << ", x(q) = " << xq);
                }
                lastPartialExpectation = partialExpectation;
            }
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to sample: " << e.what() << ", mean = " << mean
                      << ", variance = " << variance);
        }
    } else {
        double r = m_Shape;
        double p = m_Rate / (m_Rate + 1.0);

        LOG_TRACE(<< "mean = " << mean << ", r = " << r << ", p = " << p);

        using boost::math::policies::discrete_quantile;
        using boost::math::policies::policy;
        using boost::math::policies::real;

        using TRealQuantilePolicy = policy<discrete_quantile<real>>;
        using TNegativeBinomialRealQuantile =
            boost::math::negative_binomial_distribution<double, TRealQuantilePolicy>;

        try {
            TNegativeBinomialRealQuantile negativeBinomial1(r, p);
            TNegativeBinomialRealQuantile negativeBinomial2(r + 1.0, p);

            for (std::size_t i = 1u; i < numberSamples; ++i) {
                double q = static_cast<double>(i) / static_cast<double>(numberSamples);
                double xq = boost::math::quantile(negativeBinomial1, q);

                double partialExpectation =
                    mean * boost::math::cdf(negativeBinomial2, std::max(xq - 1.0, 0.0));

                double sample = static_cast<double>(numberSamples) *
                                    (partialExpectation - lastPartialExpectation) -
                                m_Offset;

                LOG_TRACE(<< "sample = " << sample);

                // Sanity check the sample: should be in the distribution support.
                if (sample >= support.first && sample <= support.second) {
                    samples.push_back(sample);
                } else {
                    LOG_ERROR(<< "Sample out of bounds: sample = " << sample << ", support = ["
                              << support.first << "," << support.second << "]"
                              << ", mean = " << mean << ", r = " << r << ", p = " << p
                              << ", q = " << q << ", x(q) = " << xq);
                }
                lastPartialExpectation = partialExpectation;
            }
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to sample: " << e.what() << ", mean = " << mean
                      << ", r = " << r << ", p = " << p);
        }
    }

    double sample = static_cast<double>(numberSamples) * (mean - lastPartialExpectation) - m_Offset;

    LOG_TRACE(<< "sample = " << sample);

    // Sanity check the sample: should be in the distribution support.
    if (sample >= support.first && sample <= support.second) {
        samples.push_back(sample);
    } else {
        LOG_ERROR(<< "Sample out of bounds: sample = " << sample << ", mean = " << mean);
    }
}

bool CPoissonMeanConjugate::minusLogJointCdf(const TWeightStyleVec& weightStyles,
                                             const TDouble1Vec& samples,
                                             const TDouble4Vec1Vec& weights,
                                             double& lowerBound,
                                             double& upperBound) const {
    lowerBound = upperBound = 0.0;

    double value;
    if (!detail::evaluateFunctionOnJointDistribution(
            weightStyles, samples, weights, CTools::SMinusLogCdf(), detail::SPlusWeight(),
            m_Offset, this->isNonInformative(), m_Shape, m_Rate, value)) {
        LOG_ERROR(<< "Failed computing c.d.f. for "
                  << core::CContainerPrinter::print(samples));
        return false;
    }

    lowerBound = upperBound = value;
    return true;
}

bool CPoissonMeanConjugate::minusLogJointCdfComplement(const TWeightStyleVec& weightStyles,
                                                       const TDouble1Vec& samples,
                                                       const TDouble4Vec1Vec& weights,
                                                       double& lowerBound,
                                                       double& upperBound) const {
    lowerBound = upperBound = 0.0;

    double value;
    if (!detail::evaluateFunctionOnJointDistribution(
            weightStyles, samples, weights, CTools::SMinusLogCdfComplement(),
            detail::SPlusWeight(), m_Offset, this->isNonInformative(), m_Shape,
            m_Rate, value)) {
        LOG_ERROR(<< "Failed computing c.d.f. complement for "
                  << core::CContainerPrinter::print(samples));
        return false;
    }

    lowerBound = upperBound = value;
    return true;
}

bool CPoissonMeanConjugate::probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                           const TWeightStyleVec& weightStyles,
                                                           const TDouble1Vec& samples,
                                                           const TDouble4Vec1Vec& weights,
                                                           double& lowerBound,
                                                           double& upperBound,
                                                           maths_t::ETail& tail) const {
    lowerBound = upperBound = 0.0;
    tail = maths_t::E_UndeterminedTail;

    double value = 0.0;
    maths_t::ETail tail_ = maths_t::E_UndeterminedTail;

    CJointProbabilityOfLessLikelySamples probability;
    if (!detail::evaluateFunctionOnJointDistribution(
            weightStyles, samples, weights,
            boost::bind<double>(CTools::CProbabilityOfLessLikelySample(calculation),
                                _1, _2, boost::ref(tail_)),
            CJointProbabilityOfLessLikelySamples::SAddProbability(), m_Offset,
            this->isNonInformative(), m_Shape, m_Rate, probability) ||
        !probability.calculate(value)) {
        LOG_ERROR(<< "Failed computing probability for "
                  << core::CContainerPrinter::print(samples));
        return false;
    }

    lowerBound = upperBound = value;
    tail = tail_;

    return true;
}

bool CPoissonMeanConjugate::isNonInformative() const {
    return m_Rate == NON_INFORMATIVE_RATE;
}

void CPoissonMeanConjugate::print(const std::string& indent, std::string& result) const {
    result += core_t::LINE_ENDING + indent + "poisson ";
    if (this->isNonInformative()) {
        result += "non-informative";
        return;
    }
    result += "mean = " + core::CStringUtils::typeToStringPretty(this->marginalLikelihoodMean()) +
              " sd = " +
              core::CStringUtils::typeToStringPretty(
                  std::sqrt(this->marginalLikelihoodVariance()));
}

std::string CPoissonMeanConjugate::printJointDensityFunction() const {
    if (this->isNonInformative()) {
        // The non-informative prior is improper and effectively 0 everywhere.
        return std::string();
    }

    // We'll plot the density function for the process mean in the interval
    // [a/b - 3 * a^(1/2) / b, a/b + 3 * a^(1/2) / b].
    //
    // where,
    //   a/b is the mean of the prior distribution.
    //   a^(1/2) / b is the standard deviation of the prior distribution.

    static const double RANGE = 6.0;
    static const unsigned int POINTS = 51;

    // Construct our gamma distribution.
    boost::math::gamma_distribution<> gamma(m_Shape, 1.0 / m_Rate);

    // Calculate the first point and increment at which to plot the p.d.f.
    double mean = boost::math::mean(gamma);
    double dev = std::sqrt(boost::math::variance(gamma));
    double increment = RANGE * dev / (POINTS - 1.0);
    double x = std::max(mean - RANGE * dev / 2.0, 0.0);

    std::ostringstream coordinates;
    std::ostringstream pdf;
    coordinates << "x = [";
    pdf << "pdf = [";
    for (unsigned int i = 0u; i < POINTS; ++i, x += increment) {
        coordinates << x << " ";
        pdf << CTools::safePdf(gamma, x) << " ";
    }
    coordinates << "];" << core_t::LINE_ENDING;
    pdf << "];" << core_t::LINE_ENDING << "plot(x, pdf);";

    return coordinates.str() + pdf.str();
}

uint64_t CPoissonMeanConjugate::checksum(uint64_t seed) const {
    seed = this->CPrior::checksum(seed);
    seed = CChecksum::calculate(seed, m_Offset);
    seed = CChecksum::calculate(seed, m_Shape);
    return CChecksum::calculate(seed, m_Rate);
}

void CPoissonMeanConjugate::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CPoissonMeanConjugate");
}

std::size_t CPoissonMeanConjugate::memoryUsage() const {
    return 0;
}

std::size_t CPoissonMeanConjugate::staticSize() const {
    return sizeof(*this);
}

void CPoissonMeanConjugate::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(DECAY_RATE_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(OFFSET_TAG, m_Offset, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(SHAPE_TAG, m_Shape, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(RATE_TAG, m_Rate, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(NUMBER_SAMPLES_TAG, this->numberSamples(),
                         core::CIEEE754::E_SinglePrecision);
}

double CPoissonMeanConjugate::priorMean() const {
    if (this->isNonInformative()) {
        return 0.0;
    }

    try {
        boost::math::gamma_distribution<> gamma(m_Shape, 1.0 / m_Rate);
        return boost::math::mean(gamma);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to calculate mean: " << e.what()
                  << ", prior shape = " << m_Shape << ", prior rate = " << m_Rate);
    }

    return 0.0;
}

double CPoissonMeanConjugate::priorVariance() const {
    if (this->isNonInformative()) {
        return boost::numeric::bounds<double>::highest();
    }

    try {
        boost::math::gamma_distribution<> gamma(m_Shape, 1.0 / m_Rate);
        return boost::math::variance(gamma);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to calculate variance: " << e.what()
                  << ", prior shape = " << m_Shape << ", prior rate = " << m_Rate);
    }

    return boost::numeric::bounds<double>::highest();
}

CPoissonMeanConjugate::TDoubleDoublePr
CPoissonMeanConjugate::meanConfidenceInterval(double percentage) const {
    if (this->isNonInformative()) {
        return this->marginalLikelihoodSupport();
    }

    // Compute the symmetric confidence interval around the median of the
    // distribution from the percentiles, i.e. find the percentiles q(1)
    // and q(2) which satisfy:
    //   q(1) + q(2) = 1.0
    //   q(2) - q(1) = percentage

    percentage /= 100.0;
    double lowerPercentile = 0.5 * (1.0 - percentage);
    double upperPercentile = 0.5 * (1.0 + percentage);

    try {
        boost::math::gamma_distribution<> gamma(m_Shape, 1.0 / m_Rate);
        return std::make_pair(boost::math::quantile(gamma, lowerPercentile) - m_Offset,
                              boost::math::quantile(gamma, upperPercentile) - m_Offset);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute mean confidence interval: " << e.what()
                  << ", prior shape = " << m_Shape << ", prior rate = " << m_Rate);
    }

    return this->marginalLikelihoodSupport();
}

bool CPoissonMeanConjugate::equalTolerance(const CPoissonMeanConjugate& rhs,
                                           const TEqualWithTolerance& equal) const {
    LOG_DEBUG(<< m_Shape << " " << rhs.m_Shape << ", " << m_Rate << " " << rhs.m_Rate);
    return equal(m_Shape, rhs.m_Shape) && equal(m_Rate, rhs.m_Rate);
}

const double CPoissonMeanConjugate::NON_INFORMATIVE_SHAPE = 0.1;
const double CPoissonMeanConjugate::NON_INFORMATIVE_RATE = 0.0;
}
}
