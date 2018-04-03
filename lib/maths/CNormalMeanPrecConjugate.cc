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

#include <maths/CNormalMeanPrecConjugate.h>

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
#include <maths/CMathsFuncs.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
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

typedef CBasicStatistics::SSampleMeanVar<double>::TAccumulator TMeanVarAccumulator;

const double MINIMUM_GAUSSIAN_SHAPE = 100.0;

namespace detail {

typedef maths_t::TWeightStyleVec TWeightStyleVec;
typedef core::CSmallVector<double, 1> TDouble1Vec;
typedef core::CSmallVector<double, 4> TDouble4Vec;
typedef core::CSmallVector<TDouble4Vec, 1> TDouble4Vec1Vec;
typedef std::pair<double, double> TDoubleDoublePr;
typedef std::vector<TDoubleDoublePr> TDoubleDoublePrVec;

//! Adds "weight" x "right operand" to the "left operand".
struct SPlusWeight {
    double operator()(double lhs, double rhs, double weight = 1.0) const { return lhs + weight * rhs; }
};

//! Evaluate \p func on the joint predictive distribution for \p samples
//! (integrating over the prior for the normal mean and precision) and
//! aggregate the results using \p aggregate.
//!
//! \param weightStyles Controls the interpretation of the weights that
//! are associated with each sample. See maths_t::ESampleWeightStyle for
//! more details.
//! \param samples The weighted samples.
//! \param weights The weights of each sample in \p samples.
//! \param func The function to evaluate.
//! \param aggregate The function to aggregate the results of \p func.
//! \param isNonInformative True if the prior is non-informative.
//! \param offset The constant offset of the data, in particular it is
//! assumed that \p samples are distributed as Y - "offset", where Y
//! is a normally distributed R.V.
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
                                         double predictionMean,
                                         RESULT& result) {
    result = RESULT();

    if (samples.empty()) {
        LOG_ERROR("Can't compute distribution for empty sample set");
        return false;
    }

    // Computing the true joint marginal distribution of all the samples
    // by integrating the joint likelihood over the prior distribution
    // for the mean and precision is not tractable. We will approximate
    // the joint p.d.f. as follows:
    //   Integral{ Product_i{ L(x(i) | m,p) } * f(m,p) }dm*dp
    //      ~= Product_i{ Integral{ L(x(i) | m,p) * f(m,p) }dm*dp }.
    //
    // where,
    //   L(. | m,p) is the likelihood function and
    //   f(m,p) is the prior for the mean and precision.
    //
    // This becomes increasingly accurate as the prior distribution narrows.

    try {
        if (isNonInformative) {
            // The non-informative prior is improper and effectively 0 everywhere.
            // (It is acceptable to approximate all finite samples as at the median
            // of this distribution.)
            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double x = samples[i];
                double n = maths_t::count(weightStyles, weights[i]);
                if (!CMathsFuncs::isFinite(n)) {
                    LOG_ERROR("Bad count weight " << n);
                    return false;
                }
                result = aggregate(result, func(CTools::SImproperDistribution(), x), n);
            }
        } else if (shape > MINIMUM_GAUSSIAN_SHAPE) {
            // For large shape the marginal likelihood is very well approximated
            // by a moment matched Gaussian, i.e. N(m, (p+1)/p * b/a) where "m"
            // is the mean and "p" is the precision of the prior Gaussian and "a"
            // is the shape and "b" is the rate of the prior gamma distribution,
            // and the error function is significantly cheaper to compute.

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = maths_t::count(weightStyles, weights[i]);
                double seasonalScale = std::sqrt(maths_t::seasonalVarianceScale(weightStyles, weights[i]));
                double countVarianceScale = maths_t::countVarianceScale(weightStyles, weights[i]);

                double x = seasonalScale != 1.0 ? predictionMean + (samples[i] - predictionMean) / seasonalScale : samples[i];

                // Get the effective precision and rate of the sample.
                double scaledPrecision = countVarianceScale * precision;
                double scaledRate = countVarianceScale * rate;

                double deviation = std::sqrt((scaledPrecision + 1.0) / scaledPrecision * scaledRate / shape);
                boost::math::normal_distribution<> normal(mean, deviation);
                result = aggregate(result, func(normal, x + offset), n);
            }
        } else {
            // The marginal likelihood is a t distribution with 2*a degrees of
            // freedom, location m and scale ((p+1)/p * b/a) ^ (1/2). We can
            // compute the distribution by transforming the data as follows:
            //   x -> (x - m) / ((p+1)/p * b/a) ^ (1/2)
            //
            // and using the student's t distribution with 2*a degrees of freedom.

            boost::math::students_t_distribution<> students(2.0 * shape);

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = maths_t::count(weightStyles, weights[i]);
                double seasonalScale = std::sqrt(maths_t::seasonalVarianceScale(weightStyles, weights[i]));
                double countVarianceScale = maths_t::countVarianceScale(weightStyles, weights[i]);

                double x = seasonalScale != 1.0 ? predictionMean + (samples[i] - predictionMean) / seasonalScale : samples[i];

                // Get the effective precision and rate of the sample.
                double scaledPrecision = countVarianceScale * precision;
                double scaledRate = countVarianceScale * rate;

                double scale = std::sqrt((scaledPrecision + 1.0) / scaledPrecision * scaledRate / shape);
                double sample = (x + offset - mean) / scale;
                result = aggregate(result, func(students, sample), n);
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Error calculating joint distribution: " << e.what());
        return false;
    }

    LOG_TRACE("result = " << result);

    return true;
}

//! Evaluates a specified function object, which must be default constructible,
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
                       double mean,
                       double precision,
                       double shape,
                       double rate,
                       double predictionMean)
        : m_WeightStyles(weightStyles),
          m_Samples(samples),
          m_Weights(weights),
          m_IsNonInformative(isNonInformative),
          m_Mean(mean),
          m_Precision(precision),
          m_Shape(shape),
          m_Rate(rate),
          m_PredictionMean(predictionMean) {}

    bool operator()(double x, double& result) const {
        return evaluateFunctionOnJointDistribution(m_WeightStyles,
                                                   m_Samples,
                                                   m_Weights,
                                                   F(),
                                                   SPlusWeight(),
                                                   m_IsNonInformative,
                                                   x,
                                                   m_Shape,
                                                   m_Rate,
                                                   m_Mean,
                                                   m_Precision,
                                                   m_PredictionMean,
                                                   result);
    }

private:
    const TWeightStyleVec& m_WeightStyles;
    const TDouble1Vec& m_Samples;
    const TDouble4Vec1Vec& m_Weights;
    bool m_IsNonInformative;
    double m_Mean;
    double m_Precision;
    double m_Shape;
    double m_Rate;
    double m_PredictionMean;
};

//! Computes the probability of seeing less likely samples at a specified offset.
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
                                    double mean,
                                    double precision,
                                    double shape,
                                    double rate,
                                    double predictionMean)
        : m_Calculation(calculation),
          m_WeightStyles(weightStyles),
          m_Samples(samples),
          m_Weights(weights),
          m_IsNonInformative(isNonInformative),
          m_Mean(mean),
          m_Precision(precision),
          m_Shape(shape),
          m_Rate(rate),
          m_PredictionMean(predictionMean),
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
                x,
                m_Shape,
                m_Rate,
                m_Mean,
                m_Precision,
                m_PredictionMean,
                probability) ||
            !probability.calculate(result)) {
            LOG_ERROR("Failed to compute probability of less likely samples");
            return false;
        }

        m_Tail = m_Tail | tail;

        return true;
    }

    maths_t::ETail tail(void) const { return static_cast<maths_t::ETail>(m_Tail); }

private:
    maths_t::EProbabilityCalculation m_Calculation;
    const TWeightStyleVec& m_WeightStyles;
    const TDouble1Vec& m_Samples;
    const TDouble4Vec1Vec& m_Weights;
    bool m_IsNonInformative;
    double m_Mean;
    double m_Precision;
    double m_Shape;
    double m_Rate;
    double m_PredictionMean;
    mutable int m_Tail;
};

//! The log marginal likelihood function of the samples is the log of the
//! likelihood function for the data integrated over the prior distribution
//! for the mean and scale. It can be shown that:
//!   log( L(x | m, p, a, b) ) =
//!     log( 1 / (2 * pi) ^ (n/2)
//!          * (p / (p + n)) ^ (1/2)
//!          * b ^ a
//!          * Gamma(a + n/2)
//!          / Gamma(a)
//!          / (b + 1/2 * (n * var(x) + p * n * (mean(x) - m)^2 / (p + n))) ^ (a + n/2) ).
//!
//! Here,
//!   x = {x(i)} is the sample vector.
//!   n = |x| the number of elements in the sample vector.
//!   mean(x) is the sample mean.
//!   var(x) is the sample variance.
//!   m and p are the prior Gaussian mean and precision, respectively.
//!   a and b are the prior Gamma shape and rate, respectively.
class CLogMarginalLikelihood : core::CNonCopyable {
public:
    CLogMarginalLikelihood(const TWeightStyleVec& weightStyles,
                           const TDouble1Vec& samples,
                           const TDouble4Vec1Vec& weights,
                           double mean,
                           double precision,
                           double shape,
                           double rate,
                           double predictionMean)
        : m_Mean(mean),
          m_Precision(precision),
          m_Shape(shape),
          m_Rate(rate),
          m_NumberSamples(0.0),
          m_WeightedNumberSamples(0.0),
          m_SampleMean(0.0),
          m_SampleSquareDeviation(0.0),
          m_Constant(0.0),
          m_ErrorStatus(maths_t::E_FpNoErrors) {
        this->precompute(weightStyles, samples, weights, predictionMean);
    }

    //! Evaluate the log marginal likelihood at the offset \p x.
    bool operator()(double x, double& result) const {
        if (m_ErrorStatus & maths_t::E_FpFailed) {
            return false;
        }

        double sampleMean = m_SampleMean + x;
        double impliedShape = m_Shape + 0.5 * m_NumberSamples;
        double impliedRate = m_Rate + 0.5 * (m_SampleSquareDeviation + m_Precision * m_WeightedNumberSamples * (sampleMean - m_Mean) *
                                                                           (sampleMean - m_Mean) / (m_Precision + m_WeightedNumberSamples));
        result = m_Constant - impliedShape * std::log(impliedRate);

        return true;
    }

    //! Retrieve the error status for the integration.
    maths_t::EFloatingPointErrorStatus errorStatus(void) const { return m_ErrorStatus; }

private:
    static const double LOG_2_PI;

private:
    //! Compute all the constants in the integrand.
    void
    precompute(const TWeightStyleVec& weightStyles, const TDouble1Vec& samples, const TDouble4Vec1Vec& weights, double predictionMean) {
        m_NumberSamples = 0.0;
        TMeanVarAccumulator sampleMoments;
        double logVarianceScaleSum = 0.0;

        try {
            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = maths_t::countForUpdate(weightStyles, weights[i]);
                double seasonalScale = std::sqrt(maths_t::seasonalVarianceScale(weightStyles, weights[i]));
                double countVarianceScale = maths_t::countVarianceScale(weightStyles, weights[i]);
                double w = 1.0 / countVarianceScale;
                m_NumberSamples += n;
                if (seasonalScale != 1.0) {
                    sampleMoments.add(predictionMean + (samples[i] - predictionMean) / seasonalScale, n * w);
                    logVarianceScaleSum += 2.0 * std::log(seasonalScale);
                } else {
                    sampleMoments.add(samples[i], n * w);
                }
                if (countVarianceScale != 1.0) {
                    logVarianceScaleSum += std::log(countVarianceScale);
                }
            }
            m_WeightedNumberSamples = CBasicStatistics::count(sampleMoments);
            m_SampleMean = CBasicStatistics::mean(sampleMoments);
            m_SampleSquareDeviation = (m_WeightedNumberSamples - 1.0) * CBasicStatistics::variance(sampleMoments);

            double impliedShape = m_Shape + 0.5 * m_NumberSamples;
            double impliedPrecision = m_Precision + m_WeightedNumberSamples;

            m_Constant = 0.5 * (std::log(m_Precision) - std::log(impliedPrecision)) - 0.5 * m_NumberSamples * LOG_2_PI -
                         0.5 * logVarianceScaleSum + boost::math::lgamma(impliedShape) - boost::math::lgamma(m_Shape) +
                         m_Shape * std::log(m_Rate);
        } catch (const std::exception& e) {
            LOG_ERROR("Error calculating marginal likelihood: " << e.what());
            this->addErrorStatus(maths_t::E_FpFailed);
        }
    }

    //! Update the error status.
    void addErrorStatus(maths_t::EFloatingPointErrorStatus status) const {
        m_ErrorStatus = static_cast<maths_t::EFloatingPointErrorStatus>(m_ErrorStatus | status);
    }

private:
    double m_Mean;
    double m_Precision;
    double m_Shape;
    double m_Rate;
    double m_NumberSamples;
    double m_WeightedNumberSamples;
    double m_SampleMean;
    double m_SampleSquareDeviation;
    double m_Constant;
    mutable maths_t::EFloatingPointErrorStatus m_ErrorStatus;
};

const double CLogMarginalLikelihood::LOG_2_PI = std::log(boost::math::double_constants::two_pi);

} // detail::

// We use short field names to reduce the state size
const std::string GAUSSIAN_MEAN_TAG("a");
const std::string GAUSSIAN_PRECISION_TAG("b");
const std::string GAMMA_SHAPE_TAG("c");
const std::string GAMMA_RATE_TAG("d");
const std::string NUMBER_SAMPLES_TAG("e");
//const std::string MINIMUM_TAG("f"); No longer used
//const std::string MAXIMUM_TAG("g"); No longer used
const std::string DECAY_RATE_TAG("h");
const std::string EMPTY_STRING;
}

CNormalMeanPrecConjugate::CNormalMeanPrecConjugate(maths_t::EDataType dataType,
                                                   double gaussianMean,
                                                   double gaussianPrecision,
                                                   double gammaShape,
                                                   double gammaRate,
                                                   double decayRate /*= 0.0*/)
    : CPrior(dataType, decayRate),
      m_GaussianMean(gaussianMean),
      m_GaussianPrecision(gaussianPrecision),
      m_GammaShape(gammaShape),
      m_GammaRate(gammaRate) {
    this->numberSamples(gaussianPrecision);
}

CNormalMeanPrecConjugate::CNormalMeanPrecConjugate(maths_t::EDataType dataType, const TMeanVarAccumulator& moments, double decayRate)
    : CPrior(dataType, decayRate), m_GaussianMean(0.0), m_GaussianPrecision(0.0), m_GammaShape(0.0), m_GammaRate(0.0) {
    this->reset(dataType, moments, decayRate);
}

CNormalMeanPrecConjugate::CNormalMeanPrecConjugate(const SDistributionRestoreParams& params, core::CStateRestoreTraverser& traverser)
    : CPrior(params.s_DataType, params.s_DecayRate), m_GaussianMean(0.0), m_GaussianPrecision(0.0), m_GammaShape(0.0), m_GammaRate(0.0) {
    traverser.traverseSubLevel(boost::bind(&CNormalMeanPrecConjugate::acceptRestoreTraverser, this, _1));
}

bool CNormalMeanPrecConjugate::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(
            DECAY_RATE_TAG, double decayRate, core::CStringUtils::stringToType(traverser.value(), decayRate), this->decayRate(decayRate))
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

void CNormalMeanPrecConjugate::reset(maths_t::EDataType dataType, const TMeanVarAccumulator& moments, double decayRate) {
    this->dataType(dataType);
    this->decayRate(decayRate);

    double n = CBasicStatistics::count(moments);
    double mean = CBasicStatistics::mean(moments);
    double variance = CBasicStatistics::maximumLikelihoodVariance(moments);

    m_GaussianMean = NON_INFORMATIVE_MEAN + mean + (this->isInteger() ? 0.5 : 0.0);
    m_GaussianPrecision = NON_INFORMATIVE_PRECISION + n;
    m_GammaShape = NON_INFORMATIVE_SHAPE + n / 2.0;
    m_GammaRate = NON_INFORMATIVE_RATE + n / 2.0 * (variance + (this->isInteger() ? 1.0 / 12.0 : 0.0));

    // If the coefficient of variation of the data is too small we run
    // in to numerical problems. We truncate the variation by modeling
    // the impact of an actual variation (standard deviation divided by
    // mean) in the data of size MINIMUM_COEFFICIENT_OF_VARATION on the
    // prior parameters.

    if (m_GaussianPrecision > 1.5) {
        double truncatedMean = std::max(::fabs(m_GaussianMean), 1e-8);
        double minimumDeviation = truncatedMean * MINIMUM_COEFFICIENT_OF_VARIATION;
        double minimumRate = (m_GaussianPrecision - 1.0) * minimumDeviation * minimumDeviation;
        m_GammaRate = std::max(m_GammaRate, minimumRate);
    }

    this->CPrior::addSamples(n);
}

bool CNormalMeanPrecConjugate::needsOffset(void) const {
    return false;
}

CNormalMeanPrecConjugate CNormalMeanPrecConjugate::nonInformativePrior(maths_t::EDataType dataType, double decayRate /*= 0.0*/) {
    return CNormalMeanPrecConjugate(
        dataType, NON_INFORMATIVE_MEAN, NON_INFORMATIVE_PRECISION, NON_INFORMATIVE_SHAPE, NON_INFORMATIVE_RATE, decayRate);
}

CNormalMeanPrecConjugate::EPrior CNormalMeanPrecConjugate::type(void) const {
    return E_Normal;
}

CNormalMeanPrecConjugate* CNormalMeanPrecConjugate::clone(void) const {
    return new CNormalMeanPrecConjugate(*this);
}

void CNormalMeanPrecConjugate::setToNonInformative(double /*offset*/, double decayRate) {
    *this = nonInformativePrior(this->dataType(), decayRate);
}

double CNormalMeanPrecConjugate::adjustOffset(const TWeightStyleVec& /*weightStyles*/,
                                              const TDouble1Vec& /*samples*/,
                                              const TDouble4Vec1Vec& /*weights*/) {
    return 0.0;
}

double CNormalMeanPrecConjugate::offset(void) const {
    return 0.0;
}

void CNormalMeanPrecConjugate::addSamples(const TWeightStyleVec& weightStyles, const TDouble1Vec& samples, const TDouble4Vec1Vec& weights) {
    if (samples.empty()) {
        return;
    }

    if (samples.size() != weights.size()) {
        LOG_ERROR("Mismatch in samples '" << core::CContainerPrinter::print(samples) << "' and weights '"
                                          << core::CContainerPrinter::print(weights) << "'");
        return;
    }

    this->CPrior::addSamples(weightStyles, samples, weights);

    // If {x(i)} denotes the sample vector, the likelihood function is:
    //   likelihood(x | p', m') ~
    //       Product_i{ p'^(1/2) * exp(-p' * (x(i) - m')^2 / 2) }
    //
    // The conjugate joint prior for m' and p' is gamma-normal and the
    // update of the posterior with n independent samples comes from:
    //   likelihood(x | m', p') * prior(m', p' | m, p, a, b)       (1)
    //
    // where,
    //   prior(m', p' | m, p, a, b) ~
    //     (p' * p)^(1/2) * exp(-p' * p * (m' - m)^2 / 2)
    //     * p'^(a - 1) * exp(-b * p')
    //
    // i.e. that the condition distribution of the mean is Gaussian with
    // mean m and precision p' * p and the marginal distribution of p'
    // is gamma with shape a and rate b. Equation (1) implies that the
    // parameters of the prior distribution update as follows:
    //   m -> (p * m + n * mean(x)) / (p + n)
    //   p -> p + n
    //   a -> a + n/2
    //   b -> b + 1/2 * (n * var(x) + p * n * (mean(x) - m)^2 / (p + n))
    //
    // where,
    //   mean(x) is the sample mean.
    //   var(x) is the sample variance.
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
    // of samples n->inf. In this case, the law of large numbers give that:
    //   mean(x(i) + z(i))
    //     -> 1/n * Sum_i( x(i) ) + E[Z]
    //
    // and
    //   var(x(i) + z(i))
    //     = Sum_i( (x(i) + z(i) - 1/n * Sum_i( x(i) + z(i) ))^2 )
    //     -> Sum_i( (x(i) - 1/n * Sum_j( x(j) ) + z(i) - E[Z])^2 )
    //     -> var(x(i)) + n * E[(Z - E[Z])^2]
    //
    // Since Z is uniform on the interval [0,1]
    //   E[Z] = 1/2
    //   E[(Z - E[Z])^2] = 1/12

    double numberSamples = 0.0;
    TMeanVarAccumulator sampleMoments;
    try {
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double n = maths_t::countForUpdate(weightStyles, weights[i]);
            double varianceScale =
                maths_t::seasonalVarianceScale(weightStyles, weights[i]) * maths_t::countVarianceScale(weightStyles, weights[i]);
            numberSamples += n;
            sampleMoments.add(samples[i], n / varianceScale);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to update likelihood: " << e.what());
        return;
    }
    double scaledNumberSamples = CBasicStatistics::count(sampleMoments);
    double sampleMean = CBasicStatistics::mean(sampleMoments);
    double sampleSquareDeviation = (scaledNumberSamples - 1.0) * CBasicStatistics::variance(sampleMoments);

    if (this->isInteger()) {
        sampleMean += 0.5;
        sampleSquareDeviation += numberSamples / 12.0;
    }

    m_GammaShape += 0.5 * numberSamples;
    m_GammaRate += 0.5 * (sampleSquareDeviation + m_GaussianPrecision * scaledNumberSamples * (sampleMean - m_GaussianMean) *
                                                      (sampleMean - m_GaussianMean) / (m_GaussianPrecision + scaledNumberSamples));

    m_GaussianMean =
        (m_GaussianPrecision * m_GaussianMean + scaledNumberSamples * sampleMean) / (m_GaussianPrecision + scaledNumberSamples);
    m_GaussianPrecision += scaledNumberSamples;

    // If the coefficient of variation of the data is too small we run
    // in to numerical problems. We truncate the variation by modeling
    // the impact of an actual variation (standard deviation divided by
    // mean) in the data of size MINIMUM_COEFFICIENT_OF_VARATION on the
    // prior parameters.

    if (m_GaussianPrecision > 1.5) {
        double truncatedMean = std::max(::fabs(m_GaussianMean), 1e-8);
        double minimumDeviation = truncatedMean * MINIMUM_COEFFICIENT_OF_VARIATION;
        double minimumRate = (2.0 * m_GammaShape - 1.0) * minimumDeviation * minimumDeviation;
        m_GammaRate = std::max(m_GammaRate, minimumRate);
    }

    LOG_TRACE("sampleMean = " << sampleMean << ", sampleSquareDeviation = " << sampleSquareDeviation
                              << ", numberSamples = " << numberSamples << ", scaledNumberSamples = " << scaledNumberSamples
                              << ", m_GammaShape = " << m_GammaShape << ", m_GammaRate = " << m_GammaRate
                              << ", m_GaussianMean = " << m_GaussianMean << ", m_GaussianPrecision = " << m_GaussianPrecision);

    if (this->isBad()) {
        LOG_ERROR("Update failed (" << this->debug() << ")");
        LOG_ERROR("samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR("weights = " << core::CContainerPrinter::print(weights));
        this->setToNonInformative(this->offsetMargin(), this->decayRate());
    }
}

void CNormalMeanPrecConjugate::propagateForwardsByTime(double time) {
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
        LOG_ERROR("Bad propagation time " << time);
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

    LOG_TRACE("time = " << time << ", alpha = " << alpha << ", m_GaussianPrecision = " << m_GaussianPrecision << ", m_GammaShape = "
                        << m_GammaShape << ", m_GammaRate = " << m_GammaRate << ", numberSamples = " << this->numberSamples());
}

CNormalMeanPrecConjugate::TDoubleDoublePr CNormalMeanPrecConjugate::marginalLikelihoodSupport(void) const {
    return std::make_pair(boost::numeric::bounds<double>::lowest(), boost::numeric::bounds<double>::highest());
}

double CNormalMeanPrecConjugate::marginalLikelihoodMean(void) const {
    return this->isInteger() ? this->mean() - 0.5 : this->mean();
}

double CNormalMeanPrecConjugate::marginalLikelihoodMode(const TWeightStyleVec& /*weightStyles*/, const TDouble4Vec& /*weights*/) const {
    return this->marginalLikelihoodMean();
}

double CNormalMeanPrecConjugate::marginalLikelihoodVariance(const TWeightStyleVec& weightStyles, const TDouble4Vec& weights) const {
    if (this->isNonInformative() || m_GammaShape <= 1.0) {
        return boost::numeric::bounds<double>::highest();
    }

    // This is just E_{B}[Var(X | M, P)] where M and P are the mean and
    // precision priors. There is a complication due to the fact that
    // variance is a function of both X and the mean, which is a random
    // variable. We can write Var(X | B) as
    //   E[ (X - M)^2 + (M - m)^2 | M, P ]
    //
    // and use the fact that X conditioned on M and P is a normal. The
    // first term evaluates to 1 / P and the second term 1 / p / t whence...

    double varianceScale = 1.0;
    try {
        varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights) * maths_t::countVarianceScale(weightStyles, weights);
    } catch (const std::exception& e) { LOG_ERROR("Failed to get variance scale: " << e.what()); }
    double a = m_GammaShape;
    double b = m_GammaRate;
    double t = m_GaussianPrecision;
    return varianceScale * (1.0 + 1.0 / t) * b / (a - 1.0);
}

CNormalMeanPrecConjugate::TDoubleDoublePr
CNormalMeanPrecConjugate::marginalLikelihoodConfidenceInterval(double percentage,
                                                               const TWeightStyleVec& weightStyles,
                                                               const TDouble4Vec& weights) const {
    if (this->isNonInformative()) {
        return this->marginalLikelihoodSupport();
    }

    percentage /= 100.0;
    percentage = CTools::truncate(percentage, 0.0, 1.0);

    // We use the fact that the marginal likelihood is a t-distribution.

    try {
        double seasonalScale = std::sqrt(maths_t::seasonalVarianceScale(weightStyles, weights));
        double countVarianceScale = maths_t::countVarianceScale(weightStyles, weights);

        double scaledPrecision = countVarianceScale * m_GaussianPrecision;
        double scaledRate = countVarianceScale * m_GammaRate;
        double scale = std::sqrt((scaledPrecision + 1.0) / scaledPrecision * scaledRate / m_GammaShape);
        double m = this->marginalLikelihoodMean();

        if (m_GammaShape > MINIMUM_GAUSSIAN_SHAPE) {
            boost::math::normal_distribution<> normal(m_GaussianMean, scale);
            double x1 = boost::math::quantile(normal, (1.0 - percentage) / 2.0) - (this->isInteger() ? 0.5 : 0.0);
            x1 = seasonalScale != 1.0 ? m + seasonalScale * (x1 - m) : x1;
            double x2 = percentage > 0.0 ? boost::math::quantile(normal, (1.0 + percentage) / 2.0) - (this->isInteger() ? 0.5 : 0.0) : x1;
            x2 = seasonalScale != 1.0 ? m + seasonalScale * (x2 - m) : x2;
            LOG_TRACE("x1 = " << x1 << ", x2 = " << x2 << ", scale = " << scale);
            return std::make_pair(x1, x2);
        }
        boost::math::students_t_distribution<> students(2.0 * m_GammaShape);
        double x1 = m_GaussianMean + scale * boost::math::quantile(students, (1.0 - percentage) / 2.0) - (this->isInteger() ? 0.5 : 0.0);
        x1 = seasonalScale != 1.0 ? m + seasonalScale * (x1 - m) : x1;
        double x2 = percentage > 0.0 ? m_GaussianMean + scale * boost::math::quantile(students, (1.0 + percentage) / 2.0) -
                                           (this->isInteger() ? 0.5 : 0.0)
                                     : x1;
        x2 = seasonalScale != 1.0 ? m + seasonalScale * (x2 - m) : x2;
        LOG_TRACE("x1 = " << x1 << ", x2 = " << x2 << ", scale = " << scale);
        return std::make_pair(x1, x2);
    } catch (const std::exception& e) { LOG_ERROR("Failed to compute confidence interval: " << e.what()); }

    return this->marginalLikelihoodSupport();
}

maths_t::EFloatingPointErrorStatus CNormalMeanPrecConjugate::jointLogMarginalLikelihood(const TWeightStyleVec& weightStyles,
                                                                                        const TDouble1Vec& samples,
                                                                                        const TDouble4Vec1Vec& weights,
                                                                                        double& result) const {
    result = 0.0;

    if (samples.empty()) {
        LOG_ERROR("Can't compute likelihood for empty sample set");
        return maths_t::E_FpFailed;
    }

    if (samples.size() != weights.size()) {
        LOG_ERROR("Mismatch in samples '" << core::CContainerPrinter::print(samples) << "' and weights '"
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
        weightStyles, samples, weights, m_GaussianMean, m_GaussianPrecision, m_GammaShape, m_GammaRate, this->marginalLikelihoodMean());
    if (this->isInteger()) {
        CIntegration::logGaussLegendre<CIntegration::OrderThree>(logMarginalLikelihood, 0.0, 1.0, result);
    } else {
        logMarginalLikelihood(0.0, result);
    }

    maths_t::EFloatingPointErrorStatus status =
        static_cast<maths_t::EFloatingPointErrorStatus>(logMarginalLikelihood.errorStatus() | CMathsFuncs::fpStatus(result));
    if (status & maths_t::E_FpFailed) {
        LOG_ERROR("Failed to compute log likelihood (" << this->debug() << ")");
        LOG_ERROR("samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR("weights = " << core::CContainerPrinter::print(weights));
    } else if (status & maths_t::E_FpOverflowed) {
        LOG_TRACE("Log likelihood overflowed for (" << this->debug() << ")");
        LOG_TRACE("samples = " << core::CContainerPrinter::print(samples));
        LOG_TRACE("weights = " << core::CContainerPrinter::print(weights));
    }
    return status;
}

void CNormalMeanPrecConjugate::sampleMarginalLikelihood(std::size_t numberSamples, TDouble1Vec& samples) const {
    samples.clear();

    if (numberSamples == 0 || this->numberSamples() == 0.0) {
        return;
    }

    if (this->isNonInformative()) {
        // We can't sample the marginal likelihood directly. This should
        // only happen if we've had one sample so just return that sample.
        samples.push_back(m_GaussianMean);
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
    // X is t-distributed, but for large prior shape it very nearly normally
    // distributed. In the t-distribution limit we use the relationship:
    //   E[ X * I{[x1,x2]} ] =
    //     (F(b - m) - F(a - m)) * m                                               x2
    //     + N * [2 * (1 - p/(p+1)/b/2 * x) ^ -(a-1/2) / (p/(p+1)/b/2) / (2*a - 1)]
    //                                                                             x1
    //
    // In the normal distribution limit we use the relationship:
    //   E[ X * I{[x1,x2]} ] =
    //     [m/2 * erf((p/(p+1)*a/b/2) ^ (1/2) * (x - m))                          x2
    //      - 1/(2 * pi * (p/(p+1)*a/b)) ^ (1/2) * exp(p/(p+1)*a/b/2 * (x - m)^2)]
    //                                                                            x1
    //
    // where,
    //   m and p are the prior Gaussian mean and precision, respectively.
    //   a and b are the prior Gamma shape and rate, respectively.
    //   N is the normalization factor for the student's t-distribution with
    //   2*a degrees of freedom.
    //   F(.) is the c.d.f. of the student's t-distribution with 2*a degrees
    //   of freedom.
    //   erf(.) is the error function.

    samples.reserve(numberSamples);

    TDoubleDoublePr support = this->marginalLikelihoodSupport();

    double lastPartialExpectation = 0.0;

    if (m_GammaShape > MINIMUM_GAUSSIAN_SHAPE) {
        double variance = (m_GaussianPrecision + 1.0) / m_GaussianPrecision * m_GammaRate / m_GammaShape;

        LOG_TRACE("mean = " << m_GaussianMean << ", variance = " << variance << ", numberSamples = " << numberSamples);

        try {
            boost::math::normal_distribution<> normal(m_GaussianMean, std::sqrt(variance));

            for (std::size_t i = 1u; i < numberSamples; ++i) {
                double q = static_cast<double>(i) / static_cast<double>(numberSamples);
                double xq = boost::math::quantile(normal, q);

                double partialExpectation = m_GaussianMean * q - variance * CTools::safePdf(normal, xq);

                double sample = static_cast<double>(numberSamples) * (partialExpectation - lastPartialExpectation);

                LOG_TRACE("sample = " << sample);

                // Sanity check the sample: should be in the distribution support.
                if (sample >= support.first && sample <= support.second) {
                    samples.push_back(sample);
                } else {
                    LOG_ERROR("Sample out of bounds: sample = " << sample << ", gaussianMean = " << m_GaussianMean
                                                                << ", variance = " << variance << ", q = " << q << ", x(q) = " << xq);
                }

                lastPartialExpectation = partialExpectation;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to sample: " << e.what() << ", gaussianMean = " << m_GaussianMean << ", variance = " << variance);
        }
    } else {
        double degreesFreedom = 2.0 * m_GammaShape;

        try {
            boost::math::students_t_distribution<> students(degreesFreedom);

            double scale = std::sqrt((m_GaussianPrecision + 1.0) / m_GaussianPrecision * m_GammaRate / m_GammaShape);

            LOG_TRACE("degreesFreedom = " << degreesFreedom << ", mean = " << m_GaussianMean << ", scale = " << scale
                                          << ", numberSamples = " << numberSamples);

            double constant = CTools::safePdf(students, 0.0) * scale * degreesFreedom / (degreesFreedom - 1.0);

            for (std::size_t i = 1u; i < numberSamples; ++i) {
                double q = static_cast<double>(i) / static_cast<double>(numberSamples);
                double xq = boost::math::quantile(students, q);

                double residual = xq * xq / degreesFreedom;

                double partialExpectation =
                    m_GaussianMean * q - constant * std::exp(-(degreesFreedom - 1.0) / 2.0 * std::log(1.0 + residual));

                double sample = static_cast<double>(numberSamples) * (partialExpectation - lastPartialExpectation);

                LOG_TRACE("sample = " << sample);

                // Sanity check the sample: should be in the distribution support.
                if (sample >= support.first && sample <= support.second) {
                    samples.push_back(sample);
                } else {
                    LOG_ERROR("Sample out of bounds: sample = " << sample << ", gaussianMean = " << m_GaussianMean
                                                                << ", constant = " << constant << ", residual = " << residual
                                                                << ", q = " << q << ", x(q) = " << xq);
                }

                lastPartialExpectation = partialExpectation;
            }
        } catch (const std::exception& e) { LOG_ERROR("Failed to sample: " << e.what() << ", degreesFreedom = " << degreesFreedom); }
    }

    double sample = static_cast<double>(numberSamples) * (m_GaussianMean - lastPartialExpectation);

    LOG_TRACE("sample = " << sample);

    // Sanity check the sample: should be in the distribution support.
    if (sample >= support.first && sample <= support.second) {
        samples.push_back(sample);
    } else {
        LOG_ERROR("Sample out of bounds: sample = " << sample << ", gaussianMean = " << m_GaussianMean);
    }
}

bool CNormalMeanPrecConjugate::minusLogJointCdf(const TWeightStyleVec& weightStyles,
                                                const TDouble1Vec& samples,
                                                const TDouble4Vec1Vec& weights,
                                                double& lowerBound,
                                                double& upperBound) const {
    typedef detail::CEvaluateOnSamples<CTools::SMinusLogCdf> TMinusLogCdf;

    lowerBound = upperBound = 0.0;

    TMinusLogCdf minusLogCdf(weightStyles,
                             samples,
                             weights,
                             this->isNonInformative(),
                             m_GaussianMean,
                             m_GaussianPrecision,
                             m_GammaShape,
                             m_GammaRate,
                             this->marginalLikelihoodMean());

    if (this->isInteger()) {
        // If the data are discrete we compute the approximate expectation
        // w.r.t. to the hidden offset of the samples Z, which is uniform
        // on the interval [0,1].
        double value;
        if (!CIntegration::logGaussLegendre<CIntegration::OrderThree>(minusLogCdf, 0.0, 1.0, value)) {
            LOG_ERROR("Failed computing c.d.f. for " << core::CContainerPrinter::print(samples));
            return false;
        }

        lowerBound = upperBound = value;
        return true;
    }

    double value;
    if (!minusLogCdf(0.0, value)) {
        LOG_ERROR("Failed computing c.d.f. for " << core::CContainerPrinter::print(samples));
        return false;
    }

    lowerBound = upperBound = value;
    return true;
}

bool CNormalMeanPrecConjugate::minusLogJointCdfComplement(const TWeightStyleVec& weightStyles,
                                                          const TDouble1Vec& samples,
                                                          const TDouble4Vec1Vec& weights,
                                                          double& lowerBound,
                                                          double& upperBound) const {
    typedef detail::CEvaluateOnSamples<CTools::SMinusLogCdfComplement> TMinusLogCdfComplement;

    lowerBound = upperBound = 0.0;

    TMinusLogCdfComplement minusLogCdfComplement(weightStyles,
                                                 samples,
                                                 weights,
                                                 this->isNonInformative(),
                                                 m_GaussianMean,
                                                 m_GaussianPrecision,
                                                 m_GammaShape,
                                                 m_GammaRate,
                                                 this->marginalLikelihoodMean());

    if (this->isInteger()) {
        // If the data are discrete we compute the approximate expectation
        // w.r.t. to the hidden offset of the samples Z, which is uniform
        // on the interval [0,1].
        double value;
        if (!CIntegration::logGaussLegendre<CIntegration::OrderThree>(minusLogCdfComplement, 0.0, 1.0, value)) {
            LOG_ERROR("Failed computing c.d.f. complement for " << core::CContainerPrinter::print(samples));
            return false;
        }

        lowerBound = upperBound = value;
        return true;
    }

    double value;
    if (!minusLogCdfComplement(0.0, value)) {
        LOG_ERROR("Failed computing c.d.f. complement for " << core::CContainerPrinter::print(samples));
        return false;
    }

    lowerBound = upperBound = value;
    return true;
}

bool CNormalMeanPrecConjugate::probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
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
                                                        m_GaussianMean,
                                                        m_GaussianPrecision,
                                                        m_GammaShape,
                                                        m_GammaRate,
                                                        this->marginalLikelihoodMean());

    if (this->isInteger()) {
        // If the data are discrete we compute the approximate expectation
        // w.r.t. to the hidden offset of the samples Z, which is uniform
        // on the interval [0,1].
        double value;
        if (!CIntegration::gaussLegendre<CIntegration::OrderThree>(probability, 0.0, 1.0, value)) {
            LOG_ERROR("Failed computing probability for " << core::CContainerPrinter::print(samples));
            return false;
        }

        lowerBound = upperBound = value;
        tail = probability.tail();

        return true;
    }

    double value;
    if (!probability(0.0, value)) {
        LOG_ERROR("Failed computing probability for " << core::CContainerPrinter::print(samples));
        return false;
    }

    lowerBound = upperBound = value;
    tail = probability.tail();

    return true;
}

bool CNormalMeanPrecConjugate::isNonInformative(void) const {
    return m_GammaRate == NON_INFORMATIVE_RATE || m_GaussianPrecision == NON_INFORMATIVE_PRECISION;
}

void CNormalMeanPrecConjugate::print(const std::string& indent, std::string& result) const {
    result += core_t::LINE_ENDING + indent + "normal ";
    if (this->isNonInformative()) {
        result += "non-informative";
        return;
    }
    result += "mean = " + core::CStringUtils::typeToStringPretty(this->marginalLikelihoodMean()) +
              " sd = " + core::CStringUtils::typeToStringPretty(std::sqrt(this->marginalLikelihoodVariance()));
}

std::string CNormalMeanPrecConjugate::printJointDensityFunction(void) const {
    if (this->isNonInformative()) {
        // The non-informative prior is improper and effectively 0 everywhere.
        return std::string();
    }

    // We'll plot the prior over a range where most of the mass is.

    static const double RANGE = 0.99;
    static const unsigned int POINTS = 51;

    boost::math::gamma_distribution<> gamma(m_GammaShape, 1.0 / m_GammaRate);

    double precision = m_GaussianPrecision * this->precision();
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

uint64_t CNormalMeanPrecConjugate::checksum(uint64_t seed) const {
    seed = this->CPrior::checksum(seed);
    seed = CChecksum::calculate(seed, m_GaussianMean);
    seed = CChecksum::calculate(seed, m_GaussianPrecision);
    seed = CChecksum::calculate(seed, m_GammaShape);
    return CChecksum::calculate(seed, m_GammaRate);
}

void CNormalMeanPrecConjugate::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CNormalMeanPrecConjugate");
}

std::size_t CNormalMeanPrecConjugate::memoryUsage(void) const {
    return 0;
}

std::size_t CNormalMeanPrecConjugate::staticSize(void) const {
    return sizeof(*this);
}

void CNormalMeanPrecConjugate::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(DECAY_RATE_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(GAUSSIAN_MEAN_TAG, m_GaussianMean, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(GAUSSIAN_PRECISION_TAG, m_GaussianPrecision, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(GAMMA_SHAPE_TAG, m_GammaShape, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(GAMMA_RATE_TAG, m_GammaRate, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(NUMBER_SAMPLES_TAG, this->numberSamples(), core::CIEEE754::E_SinglePrecision);
}

double CNormalMeanPrecConjugate::mean(void) const {
    return m_GaussianMean;
}

double CNormalMeanPrecConjugate::precision(void) const {
    if (this->isNonInformative()) {
        return 0.0;
    }

    return m_GammaShape / m_GammaRate;
}

CNormalMeanPrecConjugate::TDoubleDoublePr CNormalMeanPrecConjugate::confidenceIntervalMean(double percentage) const {
    if (this->isNonInformative()) {
        return std::make_pair(boost::numeric::bounds<double>::lowest(), boost::numeric::bounds<double>::highest());
    }

    // Compute the symmetric confidence interval around the median of the
    // distribution from the percentiles, i.e. find the percentiles q(1)
    // and q(2) which satisfy:
    //   q(1) + q(2) = 1.0
    //   q(2) - q(1) = percentage
    //
    // We assume that the data are described by X, which is normally distributed.
    //
    // The marginal prior distribution for the mean, M, of X is a t distribution
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
    xLower = m_GaussianMean + xLower / std::sqrt(m_GaussianPrecision * m_GammaShape / m_GammaRate);
    double xUpper = boost::math::quantile(students, upperPercentile);
    xUpper = m_GaussianMean + xUpper / std::sqrt(m_GaussianPrecision * m_GammaShape / m_GammaRate);

    return std::make_pair(xLower, xUpper);
}

CNormalMeanPrecConjugate::TDoubleDoublePr CNormalMeanPrecConjugate::confidenceIntervalPrecision(double percentage) const {
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

bool CNormalMeanPrecConjugate::equalTolerance(const CNormalMeanPrecConjugate& rhs, const TEqualWithTolerance& equal) const {
    LOG_DEBUG(m_GaussianMean << " " << rhs.m_GaussianMean << ", " << m_GaussianPrecision << " " << rhs.m_GaussianPrecision << ", "
                             << m_GammaShape << " " << rhs.m_GammaShape << ", " << m_GammaRate << " " << rhs.m_GammaRate);

    return equal(m_GaussianMean, rhs.m_GaussianMean) && equal(m_GaussianPrecision, rhs.m_GaussianPrecision) &&
           equal(m_GammaShape, rhs.m_GammaShape) && equal(m_GammaRate, rhs.m_GammaRate);
}

bool CNormalMeanPrecConjugate::isBad(void) const {
    return !CMathsFuncs::isFinite(m_GaussianMean) || !CMathsFuncs::isFinite(m_GaussianPrecision) || !CMathsFuncs::isFinite(m_GammaShape) ||
           !CMathsFuncs::isFinite(m_GammaRate);
}

std::string CNormalMeanPrecConjugate::debug(void) const {
    std::ostringstream result;
    result << std::scientific << std::setprecision(15) << m_GaussianMean << " " << m_GaussianPrecision << " " << m_GammaShape << " "
           << m_GammaRate;
    return result.str();
}

const double CNormalMeanPrecConjugate::NON_INFORMATIVE_MEAN = 0.0;
const double CNormalMeanPrecConjugate::NON_INFORMATIVE_PRECISION = 0.0;
const double CNormalMeanPrecConjugate::NON_INFORMATIVE_SHAPE = 1.0;
const double CNormalMeanPrecConjugate::NON_INFORMATIVE_RATE = 0.0;
}
}
