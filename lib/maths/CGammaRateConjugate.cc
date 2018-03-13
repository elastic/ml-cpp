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

#include <maths/CGammaRateConjugate.h>

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
#include <maths/COrderings.h>
#include <maths/CRestoreParams.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/conversion/bounds.hpp>

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

#include <math.h>

namespace ml {
namespace maths {

namespace {
namespace detail {

typedef std::pair<double, double> TDoubleDoublePr;
typedef maths_t::TWeightStyleVec TWeightStyleVec;
typedef core::CSmallVector<double, 1> TDouble1Vec;
typedef core::CSmallVector<double, 4> TDouble4Vec;
typedef core::CSmallVector<TDouble4Vec, 1> TDouble4Vec1Vec;
typedef CBasicStatistics::SSampleMean<CDoublePrecisionStorage>::TAccumulator TMeanAccumulator;
typedef CBasicStatistics::SSampleMeanVar<CDoublePrecisionStorage>::TAccumulator TMeanVarAccumulator;

const double NON_INFORMATIVE_COUNT = 3.5;

//! Compute the coefficient of variance of the sample moments.
double minimumCoefficientOfVariation(bool isInteger, double mean) {
    return std::max(MINIMUM_COEFFICIENT_OF_VARIATION, isInteger ? ::sqrt(1.0 / 12.0) / mean : 0.0);
}

//! Apply the minimum coefficient of variation constraint to the sample
//! variance and log sample mean.
//!
//! \param[in] isInteger True if the data are integer and false otherwise.
//! \param[in,out] logMean The mean of the log of the data.
//! \param[in,out] moments The mean and variance of the data.
void truncateVariance(bool isInteger, TMeanAccumulator &logMean, TMeanVarAccumulator &moments) {
    if (CBasicStatistics::count(moments) > 1.5) {
        // The idea is to model the impact of a small coefficient of variation
        // on the variance and the log of samples samples. Note that:
        //   mean(log(x'(i)) = 1/n * Sum_i{ log(m + (x'(i) - m)) }
        //                   = 1/n * Sum_i{ log(m) + log(1 + d(i) / m) }
        //                   ~ 1/n * Sum_i{ log(m) + d(i) / m - (d(i)/m)^2 / 2 }
        //                                                                  (1)
        //
        // where x(i) are our true samples, which are all very nearly m
        // (since their variation is tiny by assumption) and x'(i) are our
        // samples assuming minimum coefficient of variation. Finally, we
        // note that:
        //   E[d(i)] = 0
        //   E[(d(i)/m)^2] = "coefficient variation"^2                      (2)
        //
        // Because we age the data we can't rely on the fact that the estimated
        // coefficient of variance is accurately reflected in the mean of the
        // logs. The calculation of the likelihood shape is poorly conditioned
        // if mean(log(x(i))) is too close to log(mean(x(i))). A variation of
        // Jensen's inequality gives that:
        //   mean(log(x(i))) <= log(mean(x(i)))                             (3)
        //
        // (i.e. use the fact that log is concave and so is less than its tangent
        // at every point:
        //   log(x0 + x - x0) <= log(x0) + 1/x0 * (x - x0)
        //
        // set x0 in above to mean(x(i)) and note that:
        //   Sum_i( x(i) - mean(x(i)) ) = 0.
        //
        // From which (3) follows.) Together with (1) and (2) this means that
        // we should bound the mean of log(x(i)) by:
        //   log(mean(x(i))) - 1/2 * "minimum coefficient variation"^2

        double sampleDeviation = ::sqrt(CBasicStatistics::variance(moments));
        double sampleMean = std::max(::fabs(CBasicStatistics::mean(moments)), 1e-8);
        double cov = sampleDeviation / sampleMean;
        double covMin = minimumCoefficientOfVariation(isInteger, sampleMean);
        if (cov < covMin) {
            double extraDeviation = sampleMean * (covMin - cov);
            moments.s_Moments[1] += extraDeviation * extraDeviation;
        }

        double maxLogMean = ::log(moments.s_Moments[0]) - covMin * covMin / 2.0;
        logMean.s_Moments[0] = std::min(double(logMean.s_Moments[0]), double(maxLogMean));
    }
}

//! Computes the derivative w.r.t. the shape of the marginal likelihood
//! function for gamma distributed data with known prior for the rate.
class CLikelihoodDerivativeFunction : public std::unary_function<double, double> {
public:
    CLikelihoodDerivativeFunction(double numberSamples, double target)
        : m_NumberSamples(numberSamples), m_Target(target) {}

    double operator()(double x) const {
        return boost::math::digamma(m_NumberSamples * x) - boost::math::digamma(x) - m_Target;
    }

private:
    double m_NumberSamples;
    double m_Target;
};

//! Compute the maximum likelihood posterior shape if possible otherwise
//! estimates the shape using the method of moments.
//!
//! \param[in] oldShape The value of the shape used to seed the root bracketing
//! loop to find the new maximum likelihood shape (this should correspond to
//! the maximum likelihood shape implied by \p oldLogMean and \p oldMoments).
//! \param[in] oldLogMean The mean of the logs of all previous samples.
//! \param[in] newLogMean The mean of the logs of all previous samples plus
//! new samples to be incorporated into the estimate.
//! \param[in] oldMoments The mean and variance of the all previous samples.
//! \param[in] newMoments The mean and variance of the all previous samples
//! plus new samples to be incorporated into the estimate.
double maximumLikelihoodShape(double oldShape,
                              const TMeanAccumulator &oldLogMean,
                              const TMeanAccumulator &newLogMean,
                              const TMeanVarAccumulator &oldMoments,
                              const TMeanVarAccumulator &newMoments) {
    if (CBasicStatistics::count(newMoments) < NON_INFORMATIVE_COUNT) {
        return oldShape;
    }

    static const double EPS = 1e-3;

    // Use large maximum growth factors for root bracketing because
    // overshooting is less costly than extra iterations to bracket
    // the root since we get cubic convergence in the solving loop.
    // The derivative of the digamma function is monotone decreasing
    // so we use a higher maximum growth factor on the upside.
    static const double MIN_DOWN_FACTOR = 0.25;
    static const double MAX_UP_FACTOR = 8.0;

    std::size_t maxIterations = 20u;

    double oldNumber = CBasicStatistics::count(oldMoments);
    double oldMean = CBasicStatistics::mean(oldMoments);

    double oldTarget = 0.0;
    if (oldNumber * oldMean > 0.0) {
        oldTarget = ::log(oldNumber * oldMean) - CBasicStatistics::mean(oldLogMean);
    }

    double newNumber = CBasicStatistics::count(newMoments);
    double newMean = CBasicStatistics::mean(newMoments);

    if (newNumber * newMean == 0.0) {
        return 0.0;
    }
    double target = ::log(newNumber * newMean) - CBasicStatistics::mean(newLogMean);

    // Fall back to method of moments if maximum-likelihood fails.
    double bestGuess = 1.0;
    if (CBasicStatistics::variance(newMoments) > 0.0) {
        bestGuess = newMean * newMean / CBasicStatistics::variance(newMoments);
    }

    // If we've estimated the shape before the old shape will typically
    // be a very good initial estimate. Otherwise, use the best guess.
    double x0 = bestGuess;
    if (oldNumber > NON_INFORMATIVE_COUNT) {
        x0 = oldShape;
    }

    TDoubleDoublePr bracket(x0, x0);

    double downFactor = 0.8;
    double upFactor = 1.4;

    if (oldNumber > NON_INFORMATIVE_COUNT) {
        // Compute, very approximately, minus the gradient of the function
        // at the old shape. We just use the chord from the origin to the
        // target value and truncate its value so the bracketing loop is
        // well behaved.
        double gradient = 1.0;
        if (oldShape > 0.0) {
            gradient = CTools::truncate(oldTarget / oldShape, EPS, 1.0);
        }

        // Choose the growth factors so we will typically bracket the root
        // in one iteration and not overshoot too much. Again we truncate
        // the values so that bracketing loop is well behaved.
        double dTarget = ::fabs(target - oldTarget);
        downFactor = CTools::truncate(1.0 - 2.0 * dTarget / gradient, MIN_DOWN_FACTOR, 1.0 - EPS);
        upFactor = CTools::truncate(1.0 + 2.0 * dTarget / gradient, 1.0 + EPS, MAX_UP_FACTOR);
    }

    CLikelihoodDerivativeFunction derivative(newNumber, target);
    double f0 = 0.0;
    TDoubleDoublePr fBracket(f0, f0);

    try {
        fBracket.first = fBracket.second = f0 = derivative(x0);

        if (f0 == 0.0) {
            // We're done.
            return x0;
        }

        // The target function is monotone decreasing. The rate at which we
        // change the down and up factors in this loop has been determined
        // empirically to give a good expected total number of evaluations
        // of the likelihood derivative function across a range of different
        // process gamma shapes and rates. In particular, the mean total
        // number of evaluations used by this function is around five.
        for (/**/; maxIterations > 0; --maxIterations) {
            if (fBracket.first < 0.0) {
                bracket.second = bracket.first;
                fBracket.second = fBracket.first;

                bracket.first *= downFactor;
                fBracket.first = derivative(bracket.first);

                downFactor = std::max(0.8 * downFactor, MIN_DOWN_FACTOR);
            } else if (fBracket.second > 0.0) {
                bracket.first = bracket.second;
                fBracket.first = fBracket.second;

                bracket.second *= upFactor;
                fBracket.second = derivative(bracket.second);

                upFactor = std::min(1.4 * upFactor, MAX_UP_FACTOR);
            } else {
                break;
            }
        }
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to bracket root: "
                  << e.what() << ", newNumber = " << newNumber << ", newMean = " << newMean
                  << ", newLogMean = " << newLogMean << ", x0 = " << x0 << ", f(x0) = " << f0
                  << ", bracket = " << core::CContainerPrinter::print(bracket) << ", f(bracket) = "
                  << core::CContainerPrinter::print(fBracket) << ", bestGuess = " << bestGuess);
        return bestGuess;
    }

    if (maxIterations == 0) {
        LOG_TRACE("Failed to bracket root:"
                  << " newNumber = " << newNumber << ", newMean = " << newMean
                  << ", newLogMean = " << newLogMean << ", x0 = " << x0 << ", f(x0) = " << f0
                  << ", bracket = " << core::CContainerPrinter::print(bracket) << ", f(bracket) = "
                  << core::CContainerPrinter::print(fBracket) << ", bestGuess = " << bestGuess);
        return bestGuess;
    }

    LOG_TRACE("newNumber = " << newNumber << ", newMean = " << newMean
                             << ", newLogMean = " << newLogMean << ", oldTarget = " << oldTarget
                             << ", target = " << target << ", upFactor = " << upFactor
                             << ", downFactor = " << downFactor << ", x0 = " << x0 << ", f(x0) = "
                             << f0 << ", bracket = " << core::CContainerPrinter::print(bracket)
                             << ", f(bracket) = " << core::CContainerPrinter::print(fBracket));

    try {
        CEqualWithTolerance<double> tolerance(CToleranceTypes::E_AbsoluteTolerance, EPS * x0);
        CSolvers::solve(bracket.first,
                        bracket.second,
                        fBracket.first,
                        fBracket.second,
                        derivative,
                        maxIterations,
                        tolerance,
                        bestGuess);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to solve: "
                  << e.what() << ", newNumber = " << newNumber << ", x0 = " << x0
                  << ", f(x0) = " << f0 << ", bracket = " << core::CContainerPrinter::print(bracket)
                  << ", f(bracket) = " << core::CContainerPrinter::print(fBracket)
                  << ", bestGuess = " << bestGuess);
        return bestGuess;
    }

    LOG_TRACE("bracket = " << core::CContainerPrinter::print(bracket));

    return (bracket.first + bracket.second) / 2.0;
}

//! Adds "weight" x "right operand" to the "left operand".
struct SPlusWeight {
    double operator()(double lhs, double rhs, double weight = 1.0) const {
        return lhs + weight * rhs;
    }
};

//! Evaluate \p func on the joint predictive distribution for \p samples
//! (integrating over the prior for the gamma rate) and aggregate the
//! results using \p aggregate.
//!
//! \param[in] weightStyles Controls the interpretation of the weight(s)
//! that are associated with each sample. See maths_t::ESampleWeightStyle
//! for more details.
//! \param[in] samples The weighted samples.
//! \param[in] func The function to evaluate.
//! \param[in] aggregate The function to aggregate the results of \p func.
//! \param[in] isNonInformative True if the prior is non-informative.
//! \param[in] offset The constant offset of the data, in particular it
//! is assumed that \p samples are distributed as Y - "offset", where Y
//! is a gamma distributed R.V.
//! \param[in] likelihoodShape The shape of the likelihood for \p samples.
//! \param[in] priorShape The shape of the gamma prior of the rate parameter
//! of the likelihood for \p samples.
//! \param[in] priorRate The rate of the gamma prior of the rate parameter
//! of the likelihood for \p samples.
//! \param[out] result Filled in with the aggregation of results of \p func.
template <typename FUNC, typename AGGREGATOR, typename RESULT>
bool evaluateFunctionOnJointDistribution(const TWeightStyleVec &weightStyles,
                                         const TDouble1Vec &samples,
                                         const TDouble4Vec1Vec &weights,
                                         FUNC func,
                                         AGGREGATOR aggregate,
                                         bool isNonInformative,
                                         double offset,
                                         double likelihoodShape,
                                         double priorShape,
                                         double priorRate,
                                         RESULT &result) {
    result = RESULT();

    if (samples.empty()) {
        LOG_ERROR("Can't compute distribution for empty sample set");
        return false;
    }

    // Computing the true joint marginal distribution of all the samples
    // by integrating the joint likelihood over the prior distribution
    // for the gamma rate is not tractable. We will approximate the joint
    // p.d.f. as follows:
    //   Integral{ Product_i{ L(x(i), a | b) } * f(b) }db
    //      ~= Product_i{ Integral{ L(x(i), a | b) * f(b) }db }.
    //
    // where,
    //   L(., a | b) is the likelihood function and
    //   f(b) is the prior for the gamma rate.
    //
    // This becomes increasingly accurate as the prior distribution narrows.

    static const double MINIMUM_GAMMA_SHAPE = 100.0;

    LOG_TRACE("likelihoodShape = " << likelihoodShape << ", priorShape = " << priorShape
                                   << ", priorRate = " << priorRate);

    try {
        if (isNonInformative) {
            // The non-informative prior is improper and effectively zero
            // everywhere. (It is acceptable to approximate all finite samples
            // as at the median of this distribution.)
            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = maths_t::count(weightStyles, weights[i]);
                double x = samples[i] + offset;
                result = aggregate(result, func(CTools::SImproperDistribution(), x), n);
            }
        } else if (priorShape > 2 && priorShape > likelihoodShape * MINIMUM_GAMMA_SHAPE) {
            // The marginal likelihood is well approximated by a moment matched
            // gamma distribution. By considering:
            //   E[ E[X | a, b] ] = E[ a' / B ]
            //   E[ E[(X - E[X | a, b])^2 | a, b] ] = E[ a' / B^2 ]
            //
            // where,
            //   a' is the likelihood shape.
            //   B is the likelihood rate and B ~ Gamma(a, b).
            //   a and b are the prior shape and rate parameters, respectively.
            //   The outer expectation E[.] is w.r.t. the prior on B.
            //
            // It is possible to show that the moment matched gamma distribution
            // has:
            //   shape = (a - 2) / (a - 1) * a'
            //   rate = (a - 2) / b.
            //
            // We use this approximation to avoid calculating the incomplete beta
            // function, which can be very slow particularly for large alpha and
            // beta.

            double shape = (priorShape - 2.0) / (priorShape - 1.0) * likelihoodShape;
            double rate = (priorShape - 2.0) / priorRate;
            LOG_TRACE("shape = " << shape << ", rate = " << rate);

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                // We assume the data are described by X = Y - u where, Y is
                // gamma distributed and u is a constant offset. This means
                // that {x(i) + u} are gamma distributed.

                double n = maths_t::count(weightStyles, weights[i]);
                double varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights[i]) *
                                       maths_t::countVarianceScale(weightStyles, weights[i]);

                double x = samples[i] + offset;
                LOG_TRACE("x = " << x);

                double scaledShape = shape / varianceScale;
                double scaledRate = rate / varianceScale;
                boost::math::gamma_distribution<> gamma(scaledShape, 1.0 / scaledRate);

                result = aggregate(result, func(gamma, x), n);
            }
        } else {
            // We use the fact that the random variable is Z = X / (b + X) is
            // beta distributed with parameters alpha equal to likelihoodShape
            // and beta equal to priorShape. Therefore, we can compute the
            // likelihoods by transforming the data as follows:
            //   x -> x / (b + x)
            //
            // and then using the beta distribution.

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                // We assume the data are described by X = Y - u where, Y is
                // gamma distributed and u is a constant offset. This means
                // that {x(i) + u} are gamma distributed.

                double n = maths_t::count(weightStyles, weights[i]);
                double varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights[i]) *
                                       maths_t::countVarianceScale(weightStyles, weights[i]);
                double x = samples[i] + offset;
                double scaledLikelihoodShape = likelihoodShape / varianceScale;
                double scaledPriorRate = varianceScale * priorRate;
                boost::math::beta_distribution<> beta(scaledLikelihoodShape, priorShape);
                double z = CTools::sign(x) * ::fabs(x / (scaledPriorRate + x));
                LOG_TRACE("x = " << x << ", z = " << z);

                result = aggregate(result, func(beta, z), n);
            }
        }
    } catch (const std::exception &e) {
        LOG_ERROR("Error calculating joint distribution: "
                  << e.what() << ", offset = " << offset
                  << ", likelihoodShape = " << likelihoodShape << ", priorShape = " << priorShape
                  << ", priorRate = " << priorRate
                  << ", samples = " << core::CContainerPrinter::print(samples));
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
template <typename F> class CEvaluateOnSamples : core::CNonCopyable {
public:
    CEvaluateOnSamples(const TWeightStyleVec &weightStyles,
                       const TDouble1Vec &samples,
                       const TDouble4Vec1Vec &weights,
                       bool isNonInformative,
                       double offset,
                       double likelihoodShape,
                       double priorShape,
                       double priorRate)
        : m_WeightStyles(weightStyles),
          m_Samples(samples),
          m_Weights(weights),
          m_IsNonInformative(isNonInformative),
          m_Offset(offset),
          m_LikelihoodShape(likelihoodShape),
          m_PriorShape(priorShape),
          m_PriorRate(priorRate) {}

    bool operator()(double x, double &result) const {
        return evaluateFunctionOnJointDistribution(m_WeightStyles,
                                                   m_Samples,
                                                   m_Weights,
                                                   F(),
                                                   SPlusWeight(),
                                                   m_IsNonInformative,
                                                   m_Offset + x,
                                                   m_LikelihoodShape,
                                                   m_PriorShape,
                                                   m_PriorRate,
                                                   result);
    }

private:
    const TWeightStyleVec &m_WeightStyles;
    const TDouble1Vec &m_Samples;
    const TDouble4Vec1Vec &m_Weights;
    bool m_IsNonInformative;
    double m_Offset;
    double m_LikelihoodShape;
    double m_PriorShape;
    double m_PriorRate;
};

//! Computes the probability of seeing less likely samples at a specified offset.
//!
//! This thin wrapper around the evaluateFunctionOnJointDistribution function
//! so that it can be integrated over the hidden variable representing the
//! actual value of a discrete datum which we assume is in the interval [n, n+1].
class CProbabilityOfLessLikelySamples : core::CNonCopyable {
public:
    CProbabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                    const TWeightStyleVec &weightStyles,
                                    const TDouble1Vec &samples,
                                    const TDouble4Vec1Vec &weights,
                                    bool isNonInformative,
                                    double offset,
                                    double likelihoodShape,
                                    double priorShape,
                                    double priorRate)
        : m_Calculation(calculation),
          m_WeightStyles(weightStyles),
          m_Samples(samples),
          m_Weights(weights),
          m_IsNonInformative(isNonInformative),
          m_Offset(offset),
          m_LikelihoodShape(likelihoodShape),
          m_PriorShape(priorShape),
          m_PriorRate(priorRate),
          m_Tail(0) {}

    bool operator()(double x, double &result) const {
        CJointProbabilityOfLessLikelySamples probability;
        maths_t::ETail tail = maths_t::E_UndeterminedTail;

        if (!evaluateFunctionOnJointDistribution(
                m_WeightStyles,
                m_Samples,
                m_Weights,
                boost::bind<double>(CTools::CProbabilityOfLessLikelySample(m_Calculation),
                                    _1,
                                    _2,
                                    boost::ref(tail)),
                CJointProbabilityOfLessLikelySamples::SAddProbability(),
                m_IsNonInformative,
                m_Offset + x,
                m_LikelihoodShape,
                m_PriorShape,
                m_PriorRate,
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
    const TWeightStyleVec &m_WeightStyles;
    const TDouble1Vec &m_Samples;
    const TDouble4Vec1Vec &m_Weights;
    bool m_IsNonInformative;
    double m_Offset;
    double m_LikelihoodShape;
    double m_PriorShape;
    double m_PriorRate;
    mutable int m_Tail;
};

//! Compute the joint marginal log likelihood function of a collection
//! of independent samples from the gamma process. This is obtained by
//! integrating over the prior distribution for the rate. In particular,
//! it can be shown that:
//!   log( L(x, a' | a, b) ) =
//!     log( Product_i{ x(i) }^(a' - 1)
//!          / Gamma(a') ^ n
//!          * b ^ a
//!          / Gamma(a)
//!          * Gamma(n * a' + a)
//!          / (b + Sum_i( x(i) ))^(n * a' + a) ).
//!
//! Here,
//!   x = {y(i) + u} and {y(i)} is the sample vector and u the constant offset.
//!   n = |x| the number of elements in the sample vector.
//!   a' is the (maximum) likelihood shape of the gamma process.
//!   a and b are the prior gamma shape and rate, respectively.
class CLogMarginalLikelihood : core::CNonCopyable {
public:
    CLogMarginalLikelihood(const TWeightStyleVec &weightStyles,
                           const TDouble1Vec &samples,
                           const TDouble4Vec1Vec &weights,
                           double offset,
                           double likelihoodShape,
                           double priorShape,
                           double priorRate)
        : m_WeightStyles(weightStyles),
          m_Samples(samples),
          m_Weights(weights),
          m_Offset(offset),
          m_LikelihoodShape(likelihoodShape),
          m_PriorShape(priorShape),
          m_PriorRate(priorRate),
          m_NumberSamples(0.0),
          m_ImpliedShape(0.0),
          m_Constant(0.0),
          m_ErrorStatus(maths_t::E_FpNoErrors) {
        this->precompute();
    }

    //! Evaluate the log marginal likelihood at the offset \p x.
    bool operator()(double x, double &result) const {
        if (m_ErrorStatus & maths_t::E_FpFailed) {
            return false;
        }

        double logSamplesSum = 0.0;
        double sampleSum = 0.0;
        double logSeasonalScaleSum = 0.0;

        try {
            for (std::size_t i = 0u; i < m_Samples.size(); ++i) {
                double n = maths_t::countForUpdate(m_WeightStyles, m_Weights[i]);
                double varianceScale =
                    maths_t::seasonalVarianceScale(m_WeightStyles, m_Weights[i]) *
                    maths_t::countVarianceScale(m_WeightStyles, m_Weights[i]);

                double sample = m_Samples[i] + x + m_Offset;

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
                logSamplesSum += n * (m_LikelihoodShape / varianceScale - 1.0) * ::log(sample);
                sampleSum += n / varianceScale * sample;
            }
        } catch (const std::exception &e) {
            LOG_ERROR("Failed to calculate likelihood: " << e.what());
            this->addErrorStatus(maths_t::E_FpFailed);
            return false;
        }

        result = m_Constant + logSamplesSum - m_ImpliedShape * ::log(m_PriorRate + sampleSum) -
                 logSeasonalScaleSum;

        return true;
    }

    //! Retrieve the error status for the integration.
    maths_t::EFloatingPointErrorStatus errorStatus(void) const { return m_ErrorStatus; }

private:
    //! Compute all the constants in the integrand.
    void precompute(void) {
        m_NumberSamples = 0.0;
        double logVarianceScaleSum = 0.0;
        double nResidual = 0.0;
        double logGammaScaledLikelihoodShape = 0.0;
        double scaledImpliedShape = 0.0;

        try {
            for (std::size_t i = 0u; i < m_Weights.size(); ++i) {
                double n = maths_t::countForUpdate(m_WeightStyles, m_Weights[i]);
                double varianceScale =
                    maths_t::seasonalVarianceScale(m_WeightStyles, m_Weights[i]) *
                    maths_t::countVarianceScale(m_WeightStyles, m_Weights[i]);
                m_NumberSamples += n;
                if (varianceScale != 1.0) {
                    logVarianceScaleSum -= m_LikelihoodShape / varianceScale * ::log(varianceScale);
                    logGammaScaledLikelihoodShape +=
                        n * boost::math::lgamma(m_LikelihoodShape / varianceScale);
                    scaledImpliedShape += n * m_LikelihoodShape / varianceScale;
                } else {
                    nResidual += n;
                }
            }

            m_ImpliedShape = scaledImpliedShape + nResidual * m_LikelihoodShape + m_PriorShape;

            LOG_TRACE("numberSamples = " << m_NumberSamples);

            m_Constant = m_PriorShape * ::log(m_PriorRate) - boost::math::lgamma(m_PriorShape) +
                         logVarianceScaleSum - logGammaScaledLikelihoodShape -
                         nResidual * boost::math::lgamma(m_LikelihoodShape) +
                         boost::math::lgamma(m_ImpliedShape);
        } catch (const std::exception &e) {
            LOG_ERROR("Error calculating marginal likelihood: " << e.what());
            this->addErrorStatus(maths_t::E_FpFailed);
        }
    }

    //! Update the error status.
    void addErrorStatus(maths_t::EFloatingPointErrorStatus status) const {
        m_ErrorStatus = static_cast<maths_t::EFloatingPointErrorStatus>(m_ErrorStatus | status);
    }

private:
    const TWeightStyleVec &m_WeightStyles;
    const TDouble1Vec &m_Samples;
    const TDouble4Vec1Vec &m_Weights;
    double m_Offset;
    double m_LikelihoodShape;
    double m_PriorShape;
    double m_PriorRate;
    double m_NumberSamples;
    double m_ImpliedShape;
    double m_Constant;
    mutable maths_t::EFloatingPointErrorStatus m_ErrorStatus;
};

}// detail::

// We use short field names to reduce the state size
const std::string OFFSET_TAG("a");
const std::string LIKELIHOOD_SHAPE_TAG("b");
const std::string LOG_SAMPLES_MEAN_TAG("c");
const std::string SAMPLE_MOMENTS_TAG("d");
const std::string PRIOR_SHAPE_TAG("e");
const std::string PRIOR_RATE_TAG("f");
const std::string NUMBER_SAMPLES_TAG("g");
// const std::string MINIMUM_TAG("h"); No longer used
// const std::string MAXIMUM_TAG("i"); No longer used
const std::string DECAY_RATE_TAG("j");
const std::string EMPTY_STRING;
}

CGammaRateConjugate::CGammaRateConjugate(maths_t::EDataType dataType,
                                         double offset,
                                         double shape,
                                         double rate,
                                         double decayRate,
                                         double offsetMargin)
    : CPrior(dataType, decayRate),
      m_Offset(offset),
      m_OffsetMargin(offsetMargin),
      m_LikelihoodShape(1.0),
      m_PriorShape(shape),
      m_PriorRate(rate) {}

CGammaRateConjugate::CGammaRateConjugate(const SDistributionRestoreParams &params,
                                         core::CStateRestoreTraverser &traverser,
                                         double offsetMargin)
    : CPrior(params.s_DataType, 0.0),
      m_Offset(0.0),
      m_OffsetMargin(offsetMargin),
      m_LikelihoodShape(1.0),
      m_PriorShape(0.0),
      m_PriorRate(0.0) {
    traverser.traverseSubLevel(boost::bind(&CGammaRateConjugate::acceptRestoreTraverser, this, _1));
}

bool CGammaRateConjugate::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name = traverser.name();
        RESTORE_SETUP_TEARDOWN(DECAY_RATE_TAG,
                               double decayRate,
                               core::CStringUtils::stringToType(traverser.value(), decayRate),
                               this->decayRate(decayRate))
        RESTORE_BUILT_IN(OFFSET_TAG, m_Offset)
        RESTORE_BUILT_IN(LIKELIHOOD_SHAPE_TAG, m_LikelihoodShape)
        RESTORE(LOG_SAMPLES_MEAN_TAG, m_LogSamplesMean.fromDelimited(traverser.value()))
        RESTORE(SAMPLE_MOMENTS_TAG, m_SampleMoments.fromDelimited(traverser.value()))
        RESTORE_BUILT_IN(PRIOR_SHAPE_TAG, m_PriorShape)
        RESTORE_BUILT_IN(PRIOR_RATE_TAG, m_PriorRate)
        RESTORE_SETUP_TEARDOWN(NUMBER_SAMPLES_TAG,
                               double numberSamples,
                               core::CStringUtils::stringToType(traverser.value(), numberSamples),
                               this->numberSamples(numberSamples))
    } while (traverser.next());

    return true;
}

CGammaRateConjugate CGammaRateConjugate::nonInformativePrior(maths_t::EDataType dataType,
                                                             double offset,
                                                             double decayRate,
                                                             double offsetMargin) {
    return CGammaRateConjugate(dataType,
                               offset + offsetMargin,
                               NON_INFORMATIVE_SHAPE,
                               NON_INFORMATIVE_RATE,
                               decayRate,
                               offsetMargin);
}

CGammaRateConjugate::EPrior CGammaRateConjugate::type(void) const { return E_Gamma; }

CGammaRateConjugate *CGammaRateConjugate::clone(void) const {
    return new CGammaRateConjugate(*this);
}

void CGammaRateConjugate::setToNonInformative(double offset, double decayRate) {
    *this = nonInformativePrior(
        this->dataType(), offset + this->offsetMargin(), decayRate, this->offsetMargin());
}

double CGammaRateConjugate::offsetMargin(void) const { return m_OffsetMargin; }

bool CGammaRateConjugate::needsOffset(void) const { return true; }

double CGammaRateConjugate::adjustOffset(const TWeightStyleVec &weightStyles,
                                         const TDouble1Vec &samples,
                                         const TDouble4Vec1Vec &weights) {
    COffsetCost cost(*this);
    CApplyOffset apply(*this);
    return this->adjustOffsetWithCost(weightStyles, samples, weights, cost, apply);
}

double CGammaRateConjugate::offset(void) const { return m_Offset; }

void CGammaRateConjugate::addSamples(const TWeightStyleVec &weightStyles,
                                     const TDouble1Vec &samples,
                                     const TDouble4Vec1Vec &weights) {
    if (samples.empty()) {
        return;
    }

    if (samples.size() != weights.size()) {
        LOG_ERROR("Mismatch in samples '" << core::CContainerPrinter::print(samples)
                                          << "' and weights '"
                                          << core::CContainerPrinter::print(weights) << "'");
        return;
    }

    this->adjustOffset(weightStyles, samples, weights);
    this->CPrior::addSamples(weightStyles, samples, weights);

    // We assume the data are described by X = Y - u where, Y is gamma
    // distributed and u is a constant offset.
    //
    // If y = {y(i)} denotes the sample vector, then x = {y(i) + u} are
    // gamma distributed with shape a' and rate b', and the likelihood
    // function is:
    //   likelihood(x, a' | b') ~
    //     Product_i{ b' ^ a' * x(i) ^ (a' - 1) * exp(-b' * x(i)) }.
    //
    // Note that we treat the likelihood as a function of the free parameter
    // a' for which we do not have prior distribution. Instead we estimate
    // this by maximizing the posterior marginal likelihood function for the
    // data. It can be shown that this is equivalent to solving:
    //   f(n * a' + 1) - f(a')
    //     = log( Sum_i(x(i)) ) - Sum_i( log(x(i)) ) / n             (1)
    //
    // where f(.) is the digamma function, i.e. the derivative of the log of
    // the gamma function. This means that sufficient statistics for estimating
    // a' are Sum_i( x(i) ), Sum_i( log(x(i)) ) and n the number of samples.
    // We maintain these statistics and compute a' by solving (1) after each
    // update.
    //
    // The conjugate prior for b' is gamma and the update of the posterior
    // with n independent samples comes from:
    //   likelihood(x, a' | b') * prior(b' | a, b)                   (2)
    //
    // where,
    //   prior(b' | a, b) ~ b'^(a - 1) * exp(-b * b')
    //
    // Equation (2) implies that the parameters of the prior distribution
    // update as follows:
    //   a -> a + n * a'
    //   b -> b + Sum_i( x(i) )
    //
    // Note that the weight of the sample x(i) is interpreted as its count,
    // i.e. n(i), so for example updating with {(x, 2)} is equivalent to
    // updating with {x, x}.
    //
    // Since these values can be computed on the fly from the sample mean and
    // number of samples we do not maintain separate variables for them.
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
    //     -> 1/n * Sum_i( (x(i) + 1) * log(x(i) + 1) - x(i) * log(x(i)) - 1 )
    //
    //   mean(x(i) + z(i))
    //     -> 1/n * Sum_i( x(i) ) + 1/2
    //
    // and
    //   var(x(i) + z(i))
    //     = Sum_i( (x(i) + z(i) - 1/n * Sum_i( x(i) + z(i) ))^2 )
    //     -> Sum_i( x(i) - 1/n * Sum_j( x(j) ) ) - n / 12

    TMeanAccumulator logSamplesMean = m_LogSamplesMean;
    TMeanVarAccumulator sampleMoments = m_SampleMoments;

    try {
        double shift = boost::math::digamma(m_LikelihoodShape);
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double n = maths_t::countForUpdate(weightStyles, weights[i]);
            double varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights[i]) *
                                   maths_t::countVarianceScale(weightStyles, weights[i]);

            double x = samples[i] + m_Offset;
            if (!CMathsFuncs::isFinite(x) || x <= 0.0) {
                LOG_ERROR("Discarding " << x << " it's not gamma");
                continue;
            }

            double shift_ = -shift + boost::math::digamma(m_LikelihoodShape / varianceScale) +
                            ::log(varianceScale);

            if (this->isInteger()) {
                double logxInvPlus1 = ::log(1.0 / x + 1.0);
                double logxPlus1 = ::log(x + 1.0);
                m_LogSamplesMean.add(x * logxInvPlus1 + logxPlus1 - 1.0 - shift_,
                                     n / varianceScale);
                m_SampleMoments.add(x + 0.5, n / varianceScale);
            } else {
                m_LogSamplesMean.add(::log(x) - shift_, n / varianceScale);
                m_SampleMoments.add(x, n / varianceScale);
            }
        }
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to update likelihood: " << e.what());
        return;
    }

    // If the coefficient of variation of the data is too small we run
    // in to numerical problems. We truncate the variation by modeling
    // the impact of an actual variation (standard deviation divided by
    // mean) in the data of size MINIMUM_COEFFICIENT_OF_VARATION on the
    // sample log mean and moments. Note that if the data are integer
    // then we limit the coefficient of variation to be:
    //   "s.t.d. + (1 / 12)^(1/2)" / mean
    //
    // This is equivalent to adding on the variance of the latent
    // variable.

    detail::truncateVariance(this->isInteger(), logSamplesMean, sampleMoments);
    detail::truncateVariance(this->isInteger(), m_LogSamplesMean, m_SampleMoments);

    m_LikelihoodShape = detail::maximumLikelihoodShape(
        m_LikelihoodShape, logSamplesMean, m_LogSamplesMean, sampleMoments, m_SampleMoments);

    LOG_TRACE("m_Offset = " << m_Offset << ", m_LikelihoodShape = " << m_LikelihoodShape
                            << ", m_LogSamplesMean = " << m_LogSamplesMean << ", m_SampleMoments = "
                            << m_SampleMoments << ", m_PriorShape = " << m_PriorShape
                            << ", m_PriorRate = " << m_PriorRate);

    if (this->isBad()) {
        LOG_ERROR("Update failed (" << this->debug() << ")");
        LOG_ERROR("samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR("weights = " << core::CContainerPrinter::print(weights));
        this->setToNonInformative(this->offsetMargin(), this->decayRate());
    }
}

void CGammaRateConjugate::propagateForwardsByTime(double time) {
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
        LOG_ERROR("Bad propagation time " << time);
        return;
    }

    if (this->isNonInformative()) {
        // Nothing to be done.
        return;
    }

    // We want to increase the variance of the gamma distribution while
    // holding its mean constant s.t. in the limit t -> inf var -> inf.
    // The mean and variance are a / b and a / b^2, respectively, for
    // shape a and rate b so choose a factor f in the range [0, 1] and
    // update as follows:
    //   a -> f * a
    //   b -> f * b
    //
    // Thus the mean is unchanged and variance is increased by 1 / f.
    //
    // This amounts to the following transformations of the sample moments:
    //   n -> f * n
    //   mean( x(i) ) -> mean( x(i) )
    //   mean( log(x(i) ) -> mean( log(x(i) )
    //   var( x(i) ) -> f / (f * n - 1) * var( x(i) )

    TMeanAccumulator logSamplesMean = m_LogSamplesMean;
    TMeanVarAccumulator sampleMoments = m_SampleMoments;

    double count = CBasicStatistics::count(m_LogSamplesMean);
    double alpha = ::exp(-this->decayRate() * time);
    alpha = count > detail::NON_INFORMATIVE_COUNT
                ? (alpha * count + (1.0 - alpha) * detail::NON_INFORMATIVE_COUNT) / count
                : 1.0;
    if (alpha < 1.0) {
        m_LogSamplesMean.age(alpha);
        m_SampleMoments.age(alpha);
        m_LikelihoodShape = detail::maximumLikelihoodShape(
            m_LikelihoodShape, logSamplesMean, m_LogSamplesMean, sampleMoments, m_SampleMoments);
    }

    this->numberSamples(this->numberSamples() * alpha);

    LOG_TRACE("m_LikelihoodShape = " << m_LikelihoodShape
                                     << ", m_LogSamplesMean = " << m_LogSamplesMean
                                     << ", m_SampleMoments = " << m_SampleMoments
                                     << ", numberSamples = " << this->numberSamples());
}

CGammaRateConjugate::TDoubleDoublePr CGammaRateConjugate::marginalLikelihoodSupport(void) const {
    return std::make_pair(-m_Offset, boost::numeric::bounds<double>::highest());
}

double CGammaRateConjugate::marginalLikelihoodMean(void) const {
    return this->isInteger() ? this->mean() - 0.5 : this->mean();
}

double CGammaRateConjugate::marginalLikelihoodMode(const TWeightStyleVec &weightStyles,
                                                   const TDouble4Vec &weights) const {
    double varianceScale = 1.0;
    try {
        varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights) *
                        maths_t::countVarianceScale(weightStyles, weights);
    } catch (const std::exception &e) { LOG_ERROR("Failed to get variance scale: " << e.what()); }

    if (!this->isNonInformative()) {
        // We use the fact that the marginal likelihood is the distribution
        // of the R.V. defined as:
        //   X = b * Z / (1 - Z) - u                                    (1)
        //
        // where,
        //   u is a constant offset.
        //   b is the prior rate.
        //   Z is beta distributed with alpha equal to m_LikelihoodShape
        //   and beta equal to m_PriorShape.
        //
        // So the mode occurs at the r.h.s. of (1) evaluated at the mode of Z.

        double scaledLikelihoodShape = m_LikelihoodShape / varianceScale;
        if (scaledLikelihoodShape > 1.0 && this->priorShape() > 1.0) {
            try {
                double scaledPriorRate = varianceScale * this->priorRate();
                boost::math::beta_distribution<> beta(scaledLikelihoodShape, this->priorShape());
                double mode = boost::math::mode(beta);
                return scaledPriorRate * mode / (1.0 - mode) - m_Offset;
            } catch (const std::exception &e) {
                LOG_ERROR("Failed to compute marginal likelihood mode: "
                          << e.what() << ", likelihood shape = " << m_LikelihoodShape
                          << ", prior shape = " << this->priorShape());
            }
        }
    }

    // We use the fact that for a gamma distribution:
    //   mean = a/b,
    //   mode = (a-1)/b and
    //   variance = a/b^2
    //
    // So provided mean isn't zero the mode is:
    //   (mean^2 - variance) / mean

    double mean = CBasicStatistics::mean(m_SampleMoments);
    double variance = varianceScale * CBasicStatistics::variance(m_SampleMoments);
    return std::max(mean == 0.0 ? 0.0 : mean - variance / mean, 0.0) - m_Offset;
}

double CGammaRateConjugate::marginalLikelihoodVariance(const TWeightStyleVec &weightStyles,
                                                       const TDouble4Vec &weights) const {
    if (this->isNonInformative()) {
        return boost::numeric::bounds<double>::highest();
    }

    // This is just E_{B}[Var(X | B)] where B is the rate prior. There is
    // a complication due to the fact that variance is a function of both
    // X and the mean, which is a random variable. We can write Var(X | B)
    // as
    //   E[ (X - M)^2 + (M - m)^2 | B ]
    //
    // and use the fact that X conditioned on B is gamma with shape equal
    // to the maximum likelihood shape and scale equal to B and m' = a' / B.
    // The first term evaluates to a' / B^2 and the expectation of 1 / B^2
    // w.r.t. the prior is b^2 / (a-1) / (a-2). Similarly, it is possible
    // to show that Var(a' / B) = a'^2 * E[ 1.0 / B^2 - (b / (a - 1))^2]
    // whence...

    double varianceScale = 1.0;
    try {
        varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights) *
                        maths_t::countVarianceScale(weightStyles, weights);
    } catch (const std::exception &e) { LOG_ERROR("Failed to get variance scale: " << e.what()); }
    double a = this->priorShape();
    if (a <= 2.0) {
        return varianceScale * CBasicStatistics::variance(m_SampleMoments);
    }
    double b = this->priorRate();
    return varianceScale * (1.0 + m_LikelihoodShape / (a - 1.0)) * m_LikelihoodShape * b * b /
           (a - 1.0) / (a - 2.0);
}

CGammaRateConjugate::TDoubleDoublePr
CGammaRateConjugate::marginalLikelihoodConfidenceInterval(double percentage,
                                                          const TWeightStyleVec &weightStyles,
                                                          const TDouble4Vec &weights) const {
    if (this->isNonInformative()) {
        return this->marginalLikelihoodSupport();
    }

    percentage /= 100.0;
    percentage = CTools::truncate(percentage, 0.0, 1.0);

    // We use the fact that the marginal likelihood is the distribution
    // of the R.V. defined as:
    //   X = b * Z / (1 - Z) - u
    //
    // where,
    //   u is a constant offset.
    //   b is the prior rate.
    //   Z is beta distributed with alpha equal to m_LikelihoodShape
    //   and beta equal to m_PriorShape.

    try {
        double varianceScale = maths_t::seasonalVarianceScale(weightStyles, weights) *
                               maths_t::countVarianceScale(weightStyles, weights);
        double scaledLikelihoodShape = m_LikelihoodShape / varianceScale;
        double scaledPriorRate = varianceScale * this->priorRate();
        boost::math::beta_distribution<> beta(scaledLikelihoodShape, this->priorShape());
        double x1 = boost::math::quantile(beta, (1.0 - percentage) / 2.0);
        x1 = scaledPriorRate * x1 / (1.0 - x1) - m_Offset - (this->isInteger() ? 0.5 : 0.0);
        double x2 = x1;
        if (percentage > 0.0) {
            x2 = boost::math::quantile(beta, (1.0 + percentage) / 2.0);
            x2 = scaledPriorRate * x2 / (1.0 - x2) - m_Offset - (this->isInteger() ? 0.5 : 0.0);
        }
        LOG_TRACE("x1 = " << x1 << ", x2 = " << x2);
        return std::make_pair(x1, x2);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to compute confidence interval: " << e.what());
    }

    return this->marginalLikelihoodSupport();
}

maths_t::EFloatingPointErrorStatus
CGammaRateConjugate::jointLogMarginalLikelihood(const TWeightStyleVec &weightStyles,
                                                const TDouble1Vec &samples,
                                                const TDouble4Vec1Vec &weights,
                                                double &result) const {
    result = 0.0;

    if (samples.empty()) {
        LOG_ERROR("Can't compute likelihood for empty sample set");
        return maths_t::E_FpFailed;
    }

    if (samples.size() != weights.size()) {
        LOG_ERROR("Mismatch in samples '" << core::CContainerPrinter::print(samples)
                                          << "' and weights '"
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

    maths_t::EFloatingPointErrorStatus status = maths_t::E_FpFailed;
    try {
        detail::CLogMarginalLikelihood logMarginalLikelihood(weightStyles,
                                                             samples,
                                                             weights,
                                                             m_Offset,
                                                             m_LikelihoodShape,
                                                             this->priorShape(),
                                                             this->priorRate());
        if (this->isInteger()) {
            // If the data are discrete we compute the approximate expectation
            // w.r.t. to the hidden offset of the samples Z, which is uniform
            // on the interval [0,1].
            CIntegration::logGaussLegendre<CIntegration::OrderThree>(
                logMarginalLikelihood, 0.0, 1.0, result);
        } else {
            logMarginalLikelihood(0.0, result);
        }

        status = static_cast<maths_t::EFloatingPointErrorStatus>(
            logMarginalLikelihood.errorStatus() | CMathsFuncs::fpStatus(result));
        if (status & maths_t::E_FpFailed) {
            LOG_ERROR("Failed to compute log likelihood (" << this->debug() << ")");
            LOG_ERROR("samples = " << core::CContainerPrinter::print(samples));
            LOG_ERROR("weights = " << core::CContainerPrinter::print(weights));
        } else if (status & maths_t::E_FpOverflowed) {
            LOG_TRACE("Log likelihood overflowed for (" << this->debug() << ")");
            LOG_TRACE("samples = " << core::CContainerPrinter::print(samples));
            LOG_TRACE("weights = " << core::CContainerPrinter::print(weights));
        }
    } catch (const std::exception &e) { LOG_ERROR("Failed to compute likelihood: " << e.what()); }
    return status;
}

void CGammaRateConjugate::sampleMarginalLikelihood(std::size_t numberSamples,
                                                   TDouble1Vec &samples) const {
    samples.clear();

    if (numberSamples == 0 || this->numberSamples() == 0.0) {
        return;
    }

    if (this->isNonInformative()) {
        // We can't sample the marginal likelihood directly so match sample
        // moments and sampled moments.

        numberSamples =
            std::min(numberSamples, static_cast<std::size_t>(this->numberSamples() + 0.5));
        double mean = CBasicStatistics::mean(m_SampleMoments) - m_Offset;
        double deviation = ::sqrt(CBasicStatistics::variance(m_SampleMoments));
        double root_two = boost::math::double_constants::root_two;

        switch (numberSamples) {
            case 1u:
                samples.push_back(mean);
                break;
            case 2u:
                samples.push_back(mean - deviation / root_two);
                samples.push_back(mean + deviation / root_two);
                break;
            default:
                samples.push_back(mean - deviation);
                samples.push_back(mean);
                samples.push_back(mean + deviation);
                break;
        }

        return;
    }

    // The sampling strategy is to split the marginal likelihood up into
    // equal quantiles and then compute the expectation on each quantile
    // and sample that point, i.e. effectively sample the points:
    //   { n * E[ X * I{[x_q_n(i), x_q_n((i+1))]} ] }
    //
    // where,
    //   X is a R.V. whose distribution is the marginal likelihood.
    //   I{.} is the indicator function.
    //   x_q_n(.) is the n'th quantile function.
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
    // We use the fact that X / (b + X) = Y where Y is beta distributed
    // with alpha = a' and beta = a to derive the relationship:
    //                                                         x2/(b+x2)
    //   E[ X * I{[x1,x2]} ] =  b*a'/(a-1) * [F(x | a'+1, a-1)]
    //                                                         x1/(b+x1)
    //
    // where,
    //   a and b are the prior gamma likelihood shape and rate, respectively.
    //   a' is the likelihood shape.
    //   F(. | a'+1, a-1) is the c.d.f. of a beta random variable with
    //   alpha = a'+1 and beta = a-1.

    samples.reserve(numberSamples);

    double mean = m_LikelihoodShape * this->priorRate() / (this->priorShape() - 1.0);

    try {
        boost::math::beta_distribution<> beta1(m_LikelihoodShape, this->priorShape());
        boost::math::beta_distribution<> beta2(m_LikelihoodShape + 1.0, this->priorShape() - 1.0);

        LOG_TRACE("mean = " << mean << ", numberSamples = " << numberSamples);

        TDoubleDoublePr support = this->marginalLikelihoodSupport();

        double lastPartialExpectation = 0.0;

        for (std::size_t i = 1u; i < numberSamples; ++i) {
            double q = static_cast<double>(i) / static_cast<double>(numberSamples);
            double xq = boost::math::quantile(beta1, q);

            double partialExpectation = mean * CTools::safeCdf(beta2, xq);

            double sample =
                static_cast<double>(numberSamples) * (partialExpectation - lastPartialExpectation) -
                m_Offset;

            LOG_TRACE("sample = " << sample);

            // Sanity check the sample: should be in the distribution support.
            if (sample >= support.first && sample <= support.second) {
                samples.push_back(sample);
            } else {
                LOG_ERROR("Sample out of bounds: sample = "
                          << sample << ", likelihoodShape = " << m_LikelihoodShape
                          << ", priorShape = " << this->priorShape() << ", q = " << q
                          << ", x(q) = " << xq << ", mean = " << mean);
            }

            lastPartialExpectation = partialExpectation;
        }

        double sample =
            static_cast<double>(numberSamples) * (mean - lastPartialExpectation) - m_Offset;

        LOG_TRACE("sample = " << sample);

        // Sanity check the sample: should be in the distribution support.
        if (sample >= support.first && sample <= support.second) {
            samples.push_back(sample);
        } else {
            LOG_ERROR("Sample out of bounds: sample = "
                      << sample << ", likelihoodShape = " << m_LikelihoodShape
                      << ", priorShape = " << this->priorShape() << ", mean = " << mean);
        }
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to sample: " << e.what() << ", likelihoodShape = " << m_LikelihoodShape
                                       << ", priorShape = " << this->priorShape()
                                       << ", mean = " << mean);
    }
}

bool CGammaRateConjugate::minusLogJointCdf(const TWeightStyleVec &weightStyles,
                                           const TDouble1Vec &samples,
                                           const TDouble4Vec1Vec &weights,
                                           double &lowerBound,
                                           double &upperBound) const {
    typedef detail::CEvaluateOnSamples<CTools::SMinusLogCdf> TMinusLogCdf;

    lowerBound = upperBound = 0.0;

    TMinusLogCdf minusLogCdf(weightStyles,
                             samples,
                             weights,
                             this->isNonInformative(),
                             m_Offset,
                             m_LikelihoodShape,
                             this->priorShape(),
                             this->priorRate());

    if (this->isInteger()) {
        // If the data are discrete we compute the approximate expectation
        // w.r.t. to the hidden offset of the samples Z, which is uniform
        // on the interval [0,1].
        double value;
        if (!CIntegration::logGaussLegendre<CIntegration::OrderThree>(
                minusLogCdf, 0.0, 1.0, value)) {
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

bool CGammaRateConjugate::minusLogJointCdfComplement(const TWeightStyleVec &weightStyles,
                                                     const TDouble1Vec &samples,
                                                     const TDouble4Vec1Vec &weights,
                                                     double &lowerBound,
                                                     double &upperBound) const {
    typedef detail::CEvaluateOnSamples<CTools::SMinusLogCdfComplement> TMinusLogCdfComplement;

    lowerBound = upperBound = 0.0;

    TMinusLogCdfComplement minusLogCdfComplement(weightStyles,
                                                 samples,
                                                 weights,
                                                 this->isNonInformative(),
                                                 m_Offset,
                                                 m_LikelihoodShape,
                                                 this->priorShape(),
                                                 this->priorRate());

    if (this->isInteger()) {
        // If the data are discrete we compute the approximate expectation
        // w.r.t. to the hidden offset of the samples Z, which is uniform
        // on the interval [0,1].
        double value;
        if (!CIntegration::logGaussLegendre<CIntegration::OrderThree>(
                minusLogCdfComplement, 0.0, 1.0, value)) {
            LOG_ERROR("Failed computing c.d.f. complement for "
                      << core::CContainerPrinter::print(samples));
            return false;
        }

        lowerBound = upperBound = value;
        return true;
    }

    double value;
    if (!minusLogCdfComplement(0.0, value)) {
        LOG_ERROR("Failed computing c.d.f. complement for "
                  << core::CContainerPrinter::print(samples));
        return false;
    }

    lowerBound = upperBound = value;
    return true;
}

bool CGammaRateConjugate::probabilityOfLessLikelySamples(
    maths_t::EProbabilityCalculation calculation,
    const TWeightStyleVec &weightStyles,
    const TDouble1Vec &samples,
    const TDouble4Vec1Vec &weights,
    double &lowerBound,
    double &upperBound,
    maths_t::ETail &tail) const {
    lowerBound = upperBound = 0.0;
    tail = maths_t::E_UndeterminedTail;

    detail::CProbabilityOfLessLikelySamples probability(calculation,
                                                        weightStyles,
                                                        samples,
                                                        weights,
                                                        this->isNonInformative(),
                                                        m_Offset,
                                                        m_LikelihoodShape,
                                                        this->priorShape(),
                                                        this->priorRate());

    if (this->isInteger()) {
        // If the data are discrete we compute the approximate expectation
        // w.r.t. to the hidden offset of the samples Z, which is uniform
        // on the interval [0,1].
        double value;
        if (!CIntegration::gaussLegendre<CIntegration::OrderThree>(probability, 0.0, 1.0, value)) {
            LOG_ERROR("Failed computing probability for "
                      << core::CContainerPrinter::print(samples));
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

bool CGammaRateConjugate::isNonInformative(void) const {
    return CBasicStatistics::count(m_SampleMoments) < detail::NON_INFORMATIVE_COUNT ||
           this->priorRate() == NON_INFORMATIVE_RATE;
}

void CGammaRateConjugate::print(const std::string &indent, std::string &result) const {
    result += core_t::LINE_ENDING + indent + "gamma ";
    if (this->isNonInformative()) {
        result += "non-informative";
        return;
    }

    try {
        if (this->priorShape() > 2.0) {
            double shape =
                (this->priorShape() - 2.0) / (this->priorShape() - 1.0) * m_LikelihoodShape;
            double rate = this->priorRate() / (this->priorShape() - 2.0);
            boost::math::gamma_distribution<> gamma(shape, rate);
            double mean = boost::math::mean(gamma);
            double deviation = boost::math::standard_deviation(gamma);
            result += "mean = " + core::CStringUtils::typeToStringPretty(mean - m_Offset) +
                      " sd = " + core::CStringUtils::typeToStringPretty(deviation);
            return;
        }
    } catch (const std::exception &) {}
    double mean = CBasicStatistics::mean(m_SampleMoments);
    double deviation = ::sqrt(CBasicStatistics::variance(m_SampleMoments));
    result += "mean = " + core::CStringUtils::typeToStringPretty(mean - m_Offset) +
              " sd = " + core::CStringUtils::typeToStringPretty(deviation);
}

std::string CGammaRateConjugate::printJointDensityFunction(void) const {
    if (this->isNonInformative()) {
        // The non-informative likelihood is improper 0 everywhere.
        return EMPTY_STRING;
    }

    // We'll plot the prior over a range where most of the mass is.

    static const double RANGE = 0.99;
    static const unsigned int POINTS = 51;

    boost::math::gamma_distribution<> gamma(this->priorShape(), 1.0 / this->priorRate());

    double xStart = boost::math::quantile(gamma, (1.0 - RANGE) / 2.0);
    double xEnd = boost::math::quantile(gamma, (1.0 + RANGE) / 2.0);
    double xIncrement = (xEnd - xStart) / (POINTS - 1.0);
    double x = xStart;

    std::ostringstream xCoordinates;
    std::ostringstream yCoordinates;
    xCoordinates << "x = [";
    for (unsigned int i = 0u; i < POINTS; ++i, x += xIncrement) {
        xCoordinates << x << " ";
    }
    xCoordinates << "];" << core_t::LINE_ENDING;

    std::ostringstream pdf;
    pdf << "pdf = [";
    x = xStart;
    for (unsigned int i = 0u; i < POINTS; ++i, x += xIncrement) {
        pdf << CTools::safePdf(gamma, x) << " ";
    }
    pdf << "];" << core_t::LINE_ENDING << "plot(x, pdf);";

    return xCoordinates.str() + yCoordinates.str() + pdf.str();
}

uint64_t CGammaRateConjugate::checksum(uint64_t seed) const {
    seed = this->CPrior::checksum(seed);
    seed = CChecksum::calculate(seed, m_Offset);
    seed = CChecksum::calculate(seed, m_LikelihoodShape);
    seed = CChecksum::calculate(seed, m_LogSamplesMean);
    seed = CChecksum::calculate(seed, m_SampleMoments);
    seed = CChecksum::calculate(seed, m_PriorShape);
    return CChecksum::calculate(seed, m_PriorRate);
}

void CGammaRateConjugate::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CGammaRateConjugate");
}

std::size_t CGammaRateConjugate::memoryUsage(void) const { return 0; }

std::size_t CGammaRateConjugate::staticSize(void) const { return sizeof(*this); }

void CGammaRateConjugate::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    inserter.insertValue(DECAY_RATE_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(OFFSET_TAG, m_Offset, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(
        LIKELIHOOD_SHAPE_TAG, m_LikelihoodShape, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(LOG_SAMPLES_MEAN_TAG, m_LogSamplesMean.toDelimited());
    inserter.insertValue(SAMPLE_MOMENTS_TAG, m_SampleMoments.toDelimited());
    inserter.insertValue(PRIOR_SHAPE_TAG, m_PriorShape, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(PRIOR_RATE_TAG, m_PriorRate, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(
        NUMBER_SAMPLES_TAG, this->numberSamples(), core::CIEEE754::E_SinglePrecision);
}

double CGammaRateConjugate::likelihoodShape(void) const { return m_LikelihoodShape; }

double CGammaRateConjugate::likelihoodRate(void) const {
    if (this->isNonInformative()) {
        return 0.0;
    }

    try {
        boost::math::gamma_distribution<> gamma(this->priorShape(), 1.0 / this->priorRate());
        return boost::math::mean(gamma);
    } catch (std::exception &e) {
        LOG_ERROR("Failed to compute likelihood rate: " << e.what()
                                                        << ", prior shape = " << this->priorShape()
                                                        << ", prior rate = " << this->priorRate());
    }

    return 0.0;
}

CGammaRateConjugate::TDoubleDoublePr
CGammaRateConjugate::confidenceIntervalRate(double percentage) const {
    if (this->isNonInformative()) {
        return std::make_pair(boost::numeric::bounds<double>::lowest(),
                              boost::numeric::bounds<double>::highest());
    }

    percentage /= 100.0;
    double lowerPercentile = 0.5 * (1.0 - percentage);
    double upperPercentile = 0.5 * (1.0 + percentage);

    try {
        // The prior distribution for the rate is gamma.
        boost::math::gamma_distribution<> gamma(this->priorShape(), 1.0 / this->priorRate());
        return std::make_pair(boost::math::quantile(gamma, lowerPercentile),
                              boost::math::quantile(gamma, upperPercentile));
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to compute confidence interval: "
                  << e.what() << ", prior shape = " << this->priorShape()
                  << ", prior rate = " << this->priorRate());
    }

    return std::make_pair(boost::numeric::bounds<double>::lowest(),
                          boost::numeric::bounds<double>::highest());
}

bool CGammaRateConjugate::equalTolerance(const CGammaRateConjugate &rhs,
                                         const TEqualWithTolerance &equal) const {
    LOG_DEBUG(m_LikelihoodShape << " " << rhs.m_LikelihoodShape << ", " << this->priorShape() << " "
                                << rhs.priorShape() << ", " << this->priorRate() << " "
                                << rhs.priorRate());
    return equal(m_LikelihoodShape, rhs.m_LikelihoodShape) &&
           equal(this->priorShape(), rhs.priorShape()) && equal(this->priorRate(), rhs.priorRate());
}

double CGammaRateConjugate::mean(void) const {
    if (this->isNonInformative()) {
        return CBasicStatistics::mean(m_SampleMoments);
    }

    // This is just E_{B}[E[X | B]] - u where B is the rate prior
    // and u is the offset. Note that X conditioned on B is gamma
    // with shape equal to the maximum likelihood shape and scale
    // equal to B. It's expectation is therefore a / B and the
    // expectation of 1 / B w.r.t. the prior is b / (a-1).

    double a = this->priorShape();
    if (a <= 1.0) {
        return CBasicStatistics::mean(m_SampleMoments) - m_Offset;
    }
    double b = this->priorRate();
    return m_LikelihoodShape * b / (a - 1.0) - m_Offset;
}

double CGammaRateConjugate::priorShape(void) const {
    return m_PriorShape +
           RATE_VARIANCE_SCALE * CBasicStatistics::count(m_SampleMoments) * m_LikelihoodShape;
}

double CGammaRateConjugate::priorRate(void) const {
    return m_PriorRate + RATE_VARIANCE_SCALE * CBasicStatistics::count(m_SampleMoments) *
                             CBasicStatistics::mean(m_SampleMoments);
}

bool CGammaRateConjugate::isBad(void) const {
    return !CMathsFuncs::isFinite(m_Offset) || !CMathsFuncs::isFinite(m_LikelihoodShape) ||
           !CMathsFuncs::isFinite(CBasicStatistics::count(m_LogSamplesMean)) ||
           !CMathsFuncs::isFinite(CBasicStatistics::moment<0>(m_LogSamplesMean)) ||
           !CMathsFuncs::isFinite(CBasicStatistics::count(m_SampleMoments)) ||
           !CMathsFuncs::isFinite(CBasicStatistics::moment<0>(m_SampleMoments)) ||
           !CMathsFuncs::isFinite(CBasicStatistics::moment<1>(m_SampleMoments)) ||
           !CMathsFuncs::isFinite(m_PriorShape) || !CMathsFuncs::isFinite(m_PriorRate);
}

std::string CGammaRateConjugate::debug(void) const {
    std::ostringstream result;
    result << std::scientific << std::setprecision(15) << m_Offset << " " << m_LikelihoodShape
           << " " << m_LogSamplesMean << " " << m_SampleMoments << " " << m_PriorShape << " "
           << m_PriorRate;
    return result.str();
}

const double CGammaRateConjugate::NON_INFORMATIVE_SHAPE = 1.0;
const double CGammaRateConjugate::NON_INFORMATIVE_RATE = 0.0;
const double CGammaRateConjugate::RATE_VARIANCE_SCALE = 0.23;
}
}
