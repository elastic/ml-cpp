/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CMultimodalPriorUtils_h
#define INCLUDED_ml_maths_CMultimodalPriorUtils_h

#include <core/CNonInstantiatable.h>
#include <core/CSmallVector.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCompositeFunctions.h>
#include <maths/Constants.h>
#include <maths/CMultimodalPriorMode.h>
#include <maths/CSampling.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>
#include <maths/MathsTypes.h>
#include <maths/ProbabilityAggregators.h>

#include <boost/bind.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace ml
{
namespace maths
{

//! \brief Assorted utility functions and objects used by our multimodal
//! and mixture priors.
class MATHS_EXPORT CMultimodalPriorUtils : private core::CNonInstantiatable
{
    public:
        using TDoubleDoublePr = std::pair<double, double>;
        using TDoubleVec = std::vector<double>;
        using TDouble1Vec = core::CSmallVector<double, 1>;
        using TDouble4Vec = core::CSmallVector<double, 4>;
        using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
        using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
        using TWeights = CConstantWeights;

        //! Get the mode of the marginal likelihood function.
        template<typename T>
        static TDoubleDoublePr marginalLikelihoodSupport(const std::vector<SMultimodalPriorMode<T> > &modes)
        {
            if (modes.size() == 0)
            {
                return std::make_pair(boost::numeric::bounds<double>::lowest(),
                                      boost::numeric::bounds<double>::highest());
            }
            if (modes.size() == 1)
            {
                return modes[0].s_Prior->marginalLikelihoodSupport();
            }

            TDoubleDoublePr result(boost::numeric::bounds<double>::highest(),
                                   boost::numeric::bounds<double>::lowest());

            // We define this is as the union of the mode supports.
            for (std::size_t i = 0u; i < modes.size(); ++i)
            {
                TDoubleDoublePr s = modes[i].s_Prior->marginalLikelihoodSupport();
                result.first  = std::min(result.first, s.first);
                result.second = std::max(result.second, s.second);
            }

            return result;
        }

        //! Get the mean of the marginal likelihood function.
        template<typename T>
        static double marginalLikelihoodMean(const std::vector<SMultimodalPriorMode<T> > &modes)
        {
            if (modes.size() == 0)
            {
                return 0.0;
            }
            if (modes.size() == 1)
            {
                return modes[0].s_Prior->marginalLikelihoodMean();
            }

            // By linearity we have that:
            //   Integral{ x * Sum_i{ w(i) * f(x | i) } }
            //     = Sum_i{ w(i) * Integral{ x * f(x | i) } }
            //     = Sum_i{ w(i) * mean(i) }

            TMeanAccumulator result;
            for (std::size_t i = 0u; i < modes.size(); ++i)
            {
                const SMultimodalPriorMode<T> &mode = modes[i];
                double w = mode.weight();
                result.add(mode.s_Prior->marginalLikelihoodMean(), w);
            }
            return CBasicStatistics::mean(result);
        }

        //! Get the mode of the marginal likelihood function.
        template<typename T>
        static double marginalLikelihoodMode(const std::vector<SMultimodalPriorMode<T> > &modes,
                                             const maths_t::TWeightStyleVec &weightStyles,
                                             const TDouble4Vec &weights)
        {
            if (modes.size() == 0)
            {
                return 0.0;
            }
            if (modes.size() == 1)
            {
                return modes[0].s_Prior->marginalLikelihoodMode(weightStyles, weights);
            }

            using TMaxAccumulator = CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double> >;

            // We'll approximate this as the maximum likelihood mode (mode).
            double result = 0.0;

            double seasonalScale = 1.0;
            double countVarianceScale = 1.0;
            try
            {
                seasonalScale = ::sqrt(maths_t::seasonalVarianceScale(weightStyles, weights));
                countVarianceScale = maths_t::countVarianceScale(weightStyles, weights);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to get variance scale " << e.what());
            }

            // Declared outside the loop to minimize number of times they
            // are created.
            TDouble1Vec mode(1);
            TDouble4Vec1Vec weight(1, TDouble4Vec(1, countVarianceScale));

            TMaxAccumulator maxLikelihood;
            for (std::size_t i = 0u; i < modes.size(); ++i)
            {
                double w = modes[i].weight();
                const T &prior = modes[i].s_Prior;
                mode[0] = prior->marginalLikelihoodMode(TWeights::COUNT_VARIANCE, weight[0]);
                double likelihood;
                if (  prior->jointLogMarginalLikelihood(TWeights::COUNT_VARIANCE, mode, weight, likelihood)
                    & (maths_t::E_FpFailed | maths_t::E_FpOverflowed))
                {
                    continue;
                }
                if (maxLikelihood.add(::log(w) + likelihood))
                {
                    result = mode[0];
                }
            }

            if (maths_t::hasSeasonalVarianceScale(weightStyles, weights))
            {
                double mean = marginalLikelihoodMean(modes);
                result = mean + seasonalScale * (result - mean);
            }

            return result;
        }

        //! Get the variance of the marginal likelihood.
        template<typename T>
        static double marginalLikelihoodVariance(const std::vector<SMultimodalPriorMode<T> > &modes,
                                                 const maths_t::TWeightStyleVec &weightStyles,
                                                 const TDouble4Vec &weights)
        {
            if (modes.size() == 0)
            {
                return boost::numeric::bounds<double>::highest();
            }
            if (modes.size() == 1)
            {
                return modes[0].s_Prior->marginalLikelihoodVariance(weightStyles, weights);
            }

            // By linearity we have that:
            //   Integral{ (x - m)^2 * Sum_i{ w(i) * f(x | i) } }
            //     = Sum_i{ w(i) * (Integral{ x^2 * f(x | i) } - m^2) }
            //     = Sum_i{ w(i) * ((mi^2 + vi) - m^2) }

            double varianceScale = 1.0;
            try
            {
                varianceScale =  maths_t::seasonalVarianceScale(weightStyles, weights)
                               * maths_t::countVarianceScale(weightStyles, weights);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to get variance scale " << e.what());
            }

            double mean = marginalLikelihoodMean(modes);

            TMeanAccumulator result;
            for (std::size_t i = 0u; i < modes.size(); ++i)
            {
                const SMultimodalPriorMode<T> &mode = modes[i];
                double w = mode.weight();
                double mm = mode.s_Prior->marginalLikelihoodMean();
                double mv = mode.s_Prior->marginalLikelihoodVariance();
                result.add((mm - mean) * (mm + mean) + mv, w);
            }

            return std::max(varianceScale * CBasicStatistics::mean(result), 0.0);
        }

        //! Get the \p percentage symmetric confidence interval for the marginal
        //! likelihood function, i.e. the values \f$a\f$ and \f$b\f$ such that:
        //! <pre class="fragment">
        //!   \f$P([a,m]) = P([m,b]) = p / 100 / 2\f$
        //! </pre>
        //!
        //! where \f$m\f$ is the median of the distribution and \f$p\f$ is the
        //! the percentage of interest \p percentage.
        template<typename PRIOR, typename MODE>
        static TDoubleDoublePr marginalLikelihoodConfidenceInterval(const PRIOR &prior,
                                                                    const std::vector<MODE> &modes,
                                                                    double percentage,
                                                                    const maths_t::TWeightStyleVec &weightStyles,
                                                                    const TDouble4Vec &weights)
        {
            TDoubleDoublePr support = marginalLikelihoodSupport(modes);

            if (isNonInformative(modes))
            {
                return support;
            }

            if (modes.size() == 1)
            {
                return modes[0].s_Prior->marginalLikelihoodConfidenceInterval(percentage, weightStyles, weights);
            }

            percentage /= 100.0;
            percentage = CTools::truncate(percentage, 0.0, 1.0);
            if (percentage == 1.0)
            {
                return support;
            }

            double p1 = ::log((1.0 - percentage) / 2.0);
            double p2 = ::log((1.0 + percentage) / 2.0);

            CLogCdf<PRIOR> fl(CLogCdf<PRIOR>::E_Lower, prior, weightStyles, weights);
            CLogCdf<PRIOR> fu(CLogCdf<PRIOR>::E_Upper, prior, weightStyles, weights);

            CCompositeFunctions::CMinusConstant<const CLogCdf<PRIOR>&> f1(fl, p1);
            CCompositeFunctions::CMinusConstant<const CLogCdf<PRIOR>&> f2(fu, p2);

            static const std::size_t MAX_ITERATIONS = 30u;
            static const double EPS = 1e-3;

            TDoubleDoublePr result;

            double x0 = marginalLikelihoodMode(modes, weightStyles, weights);

            try
            {
                double f10 = f1(x0);
                double a = x0, b = x0, fa = f10, fb = f10;
                LOG_TRACE("(a,b) = (" << a << "," << b << ")"
                          << ", (f(a),f(b)) = (" << fa << "," << fb << ")");

                std::size_t maxIterations = MAX_ITERATIONS;
                if (   (f10 < 0  && !CSolvers::rightBracket(a, b, fa, fb, f1, maxIterations))
                    || (f10 >= 0 && !CSolvers::leftBracket(a, b, fa, fb, f1, maxIterations)))
                {
                    LOG_ERROR("Unable to bracket left percentile = " << p1
                              << ", (a,b) = (" << a << "," << b << ")"
                              << ", (f(a),f(b)) = (" << fa << "," << fb << ")");
                    result.first = support.first;
                }
                else
                {
                    LOG_TRACE("(a,b) = (" << a << "," << b << ")"
                              << ", (f(a),f(b)) = (" << fa << "," << fb << ")");
                    maxIterations = MAX_ITERATIONS - maxIterations;
                    CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance,
                                                      std::min(std::numeric_limits<double>::epsilon() * b,
                                                               EPS * p1 / std::max(fa, fb)));
                    CSolvers::solve(a, b, fa, fb, f1, maxIterations, equal, result.first);
                    LOG_TRACE("p1 = " << p1
                              << ", x = " << result.first
                              << ", f(x) = " << fl(result.first));
                }

                result.second = result.first;
                double f20 = f2(x0);
                a = x0; b = x0; fa = f20; fb = f20;
                maxIterations = MAX_ITERATIONS;
                if (percentage == 0.0)
                {
                    // Fall: nothing to do.
                }
                else if (   (f20 < 0  && !CSolvers::rightBracket(a, b, fa, fb, f2, maxIterations))
                         || (f20 >= 0 && !CSolvers::leftBracket(a, b, fa, fb, f2, maxIterations)))
                {
                    LOG_ERROR("Unable to bracket right percentile = " << p2
                              << ", (a,b) = (" << a << "," << b << ")"
                              << ", (f(a),f(b)) = (" << fa << "," << fb << ")");
                    result.second = support.second;
                }
                else
                {
                    LOG_TRACE("(a,b) = [" << a << "," << b << "], "
                              << ", (f(a),f(b)) = [" << fa << "," << fb << "]");

                    maxIterations = MAX_ITERATIONS - maxIterations;
                    CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance,
                                                      std::min(std::numeric_limits<double>::epsilon() * b,
                                                               EPS * p2 / std::max(fa, fb)));
                    CSolvers::solve(a, b, fa, fb, f2, maxIterations, equal, result.second);
                    LOG_TRACE("p2 = " << p2
                              << ", x = " << result.second
                              << ", f(x) = " << fu(result.second));
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Unable to find left percentile: " << e.what()
                          << ", percentiles = [" << p1 << "," << p2 << "]"
                          << ", x0 = " << x0);
                return support;
            }

            return result;
        }

        //! Calculate the log marginal likelihood function integrating over
        //! the prior density function.
        template<typename T>
        static maths_t::EFloatingPointErrorStatus
            jointLogMarginalLikelihood(const std::vector<SMultimodalPriorMode<T> > &modes,
                                       const maths_t::TWeightStyleVec &weightStyles,
                                       const TDouble1Vec &samples,
                                       const TDouble4Vec1Vec &weights,
                                       double &result)
        {
            // The likelihood can be computed from the conditional likelihood
            // that a sample is from each mode. In particular, the likelihood
            // of a sample x is:
            //   L(x) = Sum_m{ L(x | m) * p(m) }
            //
            // where,
            //   L(x | m) is the likelihood the sample is from the m'th mode,
            //   p(m) is the probability a sample is from the m'th mode.
            //
            // We compute the combined likelihood by taking the product of the
            // individual likelihoods. Note, this brushes over the fact that the
            // joint marginal likelihood that a collection of samples is from
            // the i'th mode is not just the product of the likelihoods that the
            // individual samples are from the i'th mode since we're integrating
            // over a prior. Really, we should compute likelihoods over all
            // possible assignments of the samples to the modes and use the fact
            // that:
            //   P(a) = Product_i{ Sum_m{ p(m) * I{a(i) = m} } }
            //
            // where,
            //   P(a) is the probability of a given assignment,
            //   p(m) is the probability a sample is from the m'th mode,
            //   I{.} is the indicator function.
            //
            // The approximation is increasingly accurate as the prior distribution
            // on each mode narrows.

            using TSizeDoublePr = std::pair<std::size_t, double>;
            using TSizeDoublePr5Vec = core::CSmallVector<TSizeDoublePr, 5>;

            result = 0.0;

            // Declared outside the loop to minimize number of times it is created.
            TDouble1Vec sample(1);
            TSizeDoublePr5Vec modeLogLikelihoods;
            modeLogLikelihoods.reserve(modes.size());

            double mean = maths_t::hasSeasonalVarianceScale(weightStyles, weights) ?
                          marginalLikelihoodMean(modes) : 0.0;
            TDouble4Vec1Vec weight(1, TDouble4Vec(1, 1.0));
            try
            {
                for (std::size_t i = 0u; i < samples.size(); ++i)
                {
                    double n = maths_t::countForUpdate(weightStyles, weights[i]);
                    double seasonalScale = ::sqrt(maths_t::seasonalVarianceScale(weightStyles, weights[i]));
                    double logSeasonalScale = seasonalScale != 1.0 ? ::log(seasonalScale) : 0.0;

                    sample[0] = mean + (samples[i] - mean) / seasonalScale;
                    weight[0][0] = maths_t::countVarianceScale(weightStyles, weights[i]);

                    // We re-normalize so that the maximum log likelihood is one
                    // to avoid underflow.
                    modeLogLikelihoods.clear();
                    double maxLogLikelihood = boost::numeric::bounds<double>::lowest();

                    for (std::size_t j = 0u; j < modes.size(); ++j)
                    {
                        double modeLogLikelihood;
                        maths_t::EFloatingPointErrorStatus status =
                                modes[j].s_Prior->jointLogMarginalLikelihood(TWeights::COUNT_VARIANCE,
                                                                             sample,
                                                                             weight,
                                                                             modeLogLikelihood);
                        if (status & maths_t::E_FpFailed)
                        {
                            // Logging handled at a lower level.
                            return status;
                        }
                        if (!(status & maths_t::E_FpOverflowed))
                        {
                            modeLogLikelihoods.push_back(std::make_pair(j, modeLogLikelihood));
                            maxLogLikelihood = std::max(maxLogLikelihood, modeLogLikelihood);
                        }
                    }

                    if (modeLogLikelihoods.empty())
                    {
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

                    LOG_TRACE("modeLogLikelihoods = "
                              << core::CContainerPrinter::print(modeLogLikelihoods));

                    double sampleLikelihood = 0.0;
                    double Z = 0.0;

                    for (std::size_t j = 0u; j < modeLogLikelihoods.size(); ++j)
                    {
                        double w = modes[modeLogLikelihoods[j].first].weight();
                        // Divide through by the largest value to avoid underflow.
                        sampleLikelihood += w * ::exp(modeLogLikelihoods[j].second - maxLogLikelihood);
                        Z += w;
                    }

                    sampleLikelihood /= Z;
                    double sampleLogLikelihood = n * (::log(sampleLikelihood) + maxLogLikelihood);

                    LOG_TRACE("sample = " << core::CContainerPrinter::print(sample)
                              << ", maxLogLikelihood = " << maxLogLikelihood
                              << ", sampleLogLikelihood = " << sampleLogLikelihood);

                    result += sampleLogLikelihood - n * logSeasonalScale;
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to compute likelihood: " << e.what());
                return maths_t::E_FpFailed;
            }

            maths_t::EFloatingPointErrorStatus status = CMathsFuncs::fpStatus(result);
            if (status & maths_t::E_FpFailed)
            {
                LOG_ERROR("Failed to compute likelihood (" << SMultimodalPriorMode<T>::debugWeights(modes) << ")");
                LOG_ERROR("samples = " << core::CContainerPrinter::print(samples));
                LOG_ERROR("weights = " << core::CContainerPrinter::print(weights));
            }
            LOG_TRACE("Joint log likelihood = " << result);
            return status;
        }

        //! Sample the marginal likelihood function.
        template<typename T>
        static void sampleMarginalLikelihood(const std::vector<SMultimodalPriorMode<T> > &modes,
                                             std::size_t numberSamples,
                                             TDouble1Vec &samples)
        {
            samples.clear();

            if (modes.size() == 1)
            {
                modes[0].s_Prior->sampleMarginalLikelihood(numberSamples, samples);
                return;
            }

            // We sample each mode according to its weight.

            TDoubleVec normalizedWeights;
            normalizedWeights.reserve(modes.size());
            double Z = 0.0;

            for (std::size_t i = 0u; i < modes.size(); ++i)
            {
                double weight = modes[i].weight();
                normalizedWeights.push_back(weight);
                Z += weight;
            }
            for (std::size_t i = 0u; i < normalizedWeights.size(); ++i)
            {
                normalizedWeights[i] /= Z;
            }

            CSampling::TSizeVec sampling;
            CSampling::weightedSample(numberSamples, normalizedWeights, sampling);
            LOG_TRACE("normalizedWeights = " << core::CContainerPrinter::print(normalizedWeights)
                      << ", sampling = " << core::CContainerPrinter::print(sampling));

            if (sampling.size() != modes.size())
            {
                LOG_ERROR("Failed to sample marginal likelihood");
                return;
            }

            samples.reserve(numberSamples);
            TDouble1Vec modeSamples;
            for (std::size_t i = 0u; i < modes.size(); ++i)
            {
                modes[i].s_Prior->sampleMarginalLikelihood(sampling[i], modeSamples);
                LOG_TRACE("modeSamples = " << core::CContainerPrinter::print(modeSamples));
                std::copy(modeSamples.begin(), modeSamples.end(), std::back_inserter(samples));
            }
            LOG_TRACE("samples = " << core::CContainerPrinter::print(samples));
        }

        //! Calculate minus the log of the joint c.d.f. of the marginal
        //! likelihood for a collection of independent samples from the
        //! variable.
        template<typename T>
        static bool minusLogJointCdf(const std::vector<SMultimodalPriorMode<T> > &modes,
                                     const maths_t::TWeightStyleVec &weightStyles,
                                     const TDouble1Vec &samples,
                                     const TDouble4Vec1Vec &weights,
                                     double &lowerBound,
                                     double &upperBound)
        {
            return minusLogJointCdf(modes, CMinusLogJointCdf(),
                                    weightStyles, samples, weights, lowerBound, upperBound);
        }

        //! Compute minus the log of the one minus the joint c.d.f. of the
        //! marginal likelihood at \p samples without losing precision due
        //! to cancellation errors at one, i.e. the smallest non-zero value
        //! this can return is the minimum double rather than epsilon.
        template<typename T>
        static bool minusLogJointCdfComplement(const std::vector<SMultimodalPriorMode<T> > &modes,
                                               const maths_t::TWeightStyleVec &weightStyles,
                                               const TDouble1Vec &samples,
                                               const TDouble4Vec1Vec &weights,
                                               double &lowerBound,
                                               double &upperBound)
        {
            return minusLogJointCdf(modes, CMinusLogJointCdfComplement(),
                                    weightStyles, samples, weights, lowerBound, upperBound);
        }

        //! Calculate the joint probability of seeing a lower likelihood
        //! collection of independent samples from the variable integrating
        //! over the prior density function.
        template<typename PRIOR, typename MODE>
        static bool probabilityOfLessLikelySamples(const PRIOR &prior,
                                                   const std::vector<MODE> &modes,
                                                   maths_t::EProbabilityCalculation calculation,
                                                   const maths_t::TWeightStyleVec &weightStyles,
                                                   const TDouble1Vec &samples,
                                                   const TDouble4Vec1Vec &weights,
                                                   double &lowerBound,
                                                   double &upperBound,
                                                   maths_t::ETail &tail)
        {
            lowerBound = upperBound = 1.0;
            tail = maths_t::E_UndeterminedTail;

            if (samples.empty())
            {
                LOG_ERROR("Can't compute distribution for empty sample set");
                return false;
            }

            if (isNonInformative(modes))
            {
                return true;
            }

            if (modes.size() == 1)
            {
                return modes[0].s_Prior->probabilityOfLessLikelySamples(calculation,
                                                                        weightStyles,
                                                                        samples,
                                                                        weights,
                                                                        lowerBound, upperBound, tail);
            }

            // Ideally we'd find the probability of the set of samples whose
            // total likelihood is less than or equal to that of the specified
            // samples, i.e. the probability of the set
            //   R = { y | L(y) < L(x) }
            //
            // where,
            //   x = {x(1), x(2), ..., x(n)} is the sample vector.
            //   y is understood to be a vector quantity.
            //
            // This is not *trivially* related to the probability that the
            // probabilities of the sets
            //   R(i) = { y | L(y) < L(x(i)) }
            //
            // since the joint conditional likelihood must be integrated over
            // priors for the parameters. However, we'll approximate this as
            // the joint probability (of a collection of standard normal R.Vs.)
            // having probabilities {P(R(i))}. This becomes increasingly accurate
            // as the prior distribution narrows.
            //
            // For the two sided calculation, we use the fact that the likelihood
            // function decreases monotonically away from the interval [a, b]
            // whose end points are the leftmost and rightmost modes' modes
            // since all component likelihoods decrease away from this interval.
            //
            // To evaluate the probability in the interval [a, b] we relax
            // the hard constraint that regions where f > f(x) contribute
            // zero probability. In particular, we note that we can write
            // the probability as:
            //   P = Integral{ I(f(s) < f(x)) * f(s) }ds
            //
            // and that:
            //   I(f(s) < f(x)) = lim_{k->inf}{ exp(-k * (f(s)/f(x) - 1))
            //                                  / (1 + exp(-k * (f(s)/f(x) - 1))) }
            //
            // We evaluate a smoother integral, i.e. smaller p, initially
            // to find out which regions contribute the most to P and then
            // re-evaluate those regions we need with higher resolution
            // using the fact that the maximum error in the approximation
            // of I(f(s) < f(x)) is 0.5.

            switch (calculation)
            {
            case maths_t::E_OneSidedBelow:
                if (!minusLogJointCdf(modes, weightStyles, samples, weights, upperBound, lowerBound))
                {
                    LOG_ERROR("Failed computing probability of less likely samples: "
                              << core::CContainerPrinter::print(samples));
                    return false;
                }
                lowerBound = ::exp(-lowerBound);
                upperBound = ::exp(-upperBound);
                tail = maths_t::E_LeftTail;
                break;

            case maths_t::E_TwoSided:
                {
                    static const double EPS = 1000.0 * std::numeric_limits<double>::epsilon();
                    static const std::size_t MAX_ITERATIONS = 20u;

                    CJointProbabilityOfLessLikelySamples lowerBoundCalculator;
                    CJointProbabilityOfLessLikelySamples upperBoundCalculator;

                    TDoubleDoublePr support = marginalLikelihoodSupport(modes);
                    support.first  = (1.0 + (support.first > 0.0 ? EPS : -EPS)) * support.first;
                    support.second = (1.0 + (support.first > 0.0 ? EPS : -EPS)) * support.second;
                    double mean = marginalLikelihoodMean(modes);

                    double a = boost::numeric::bounds<double>::highest();
                    double b = boost::numeric::bounds<double>::lowest();
                    double Z = 0.0;
                    for (const auto &mode : modes)
                    {
                        double mode_ = mode.s_Prior->marginalLikelihoodMode();
                        a = std::min(a, mode_);
                        b = std::max(b, mode_);
                        Z += mode.weight();
                    }
                    a = CTools::truncate(a, support.first, support.second);
                    b = CTools::truncate(b, support.first, support.second);
                    LOG_TRACE("a = " << a << ", b = " << b << ", Z = " << Z);

                    std::size_t svi = static_cast<std::size_t>(
                                              std::find(weightStyles.begin(),
                                                        weightStyles.end(),
                                                        maths_t::E_SampleSeasonalVarianceScaleWeight)
                                            - weightStyles.begin());

                    // Declared outside the loop to minimize the number of times
                    // they are created.
                    TDouble4Vec1Vec weight(1);
                    TDouble1Vec wt(1);

                    int tail_ = 0;
                    for (std::size_t i = 0u; i < samples.size(); ++i)
                    {
                        double x = samples[i];
                        weight[0] = weights[i];

                        if (svi < weight.size())
                        {
                            x = mean + (x - mean) / std::sqrt(weights[i][svi]);
                            weight[0][svi] = 1.0;
                        }

                        double fx;
                        maths_t::EFloatingPointErrorStatus status =
                                jointLogMarginalLikelihood(modes, weightStyles, {x}, weight, fx);
                        if (status & maths_t::E_FpFailed)
                        {
                            LOG_ERROR("Unable to compute likelihood for " << x);
                            return false;
                        }
                        if (status & maths_t::E_FpOverflowed)
                        {
                            lowerBound = upperBound = 0.0;
                            return true;
                        }
                        LOG_TRACE("x = " << x << ", f(x) = " << fx);

                        CPrior::CLogMarginalLikelihood logLikelihood(prior, weightStyles, weight);

                        CTools::CMixtureProbabilityOfLessLikelySample calculator(modes.size(), x, fx, a, b);
                        for (const auto &mode : modes)
                        {
                            double w = mode.weight() / Z;
                            double centre = mode.s_Prior->marginalLikelihoodMode(weightStyles, weight[0]);
                            double spread = ::sqrt(mode.s_Prior->marginalLikelihoodVariance(weightStyles, weight[0]));
                            calculator.addMode(w, centre, spread);
                            tail_ = tail_ | (x < centre ? maths_t::E_LeftTail : maths_t::E_RightTail);
                        }

                        double sampleLowerBound = 0.0;
                        double sampleUpperBound = 0.0;

                        double lb, ub;

                        double l;
                        CEqualWithTolerance<double> lequal(CToleranceTypes::E_AbsoluteTolerance, EPS * a);
                        if (calculator.leftTail(logLikelihood, MAX_ITERATIONS, lequal, l))
                        {
                            wt[0] = l;
                            minusLogJointCdf(modes, weightStyles, wt, weight, lb, ub);
                            sampleLowerBound += ::exp(std::min(-lb, -ub));
                            sampleUpperBound += ::exp(std::max(-lb, -ub));
                        }
                        else
                        {
                            wt[0] = l;
                            minusLogJointCdf(modes, weightStyles, wt, weight, lb, ub);
                            sampleUpperBound += ::exp(std::max(-lb, -ub));
                        }

                        double r;
                        CEqualWithTolerance<double> requal(CToleranceTypes::E_AbsoluteTolerance, EPS * b);
                        if (calculator.rightTail(logLikelihood, MAX_ITERATIONS, requal, r))
                        {
                            wt[0] = r;
                            minusLogJointCdfComplement(modes, weightStyles, wt, weight, lb, ub);
                            sampleLowerBound += ::exp(std::min(-lb, -ub));
                            sampleUpperBound += ::exp(std::max(-lb, -ub));
                        }
                        else
                        {
                            wt[0] = r;
                            minusLogJointCdfComplement(modes, weightStyles, wt, weight, lb, ub);
                            sampleUpperBound += ::exp(std::max(-lb, -ub));
                        }

                        double p = 0.0;
                        if (a < b)
                        {
                            p = calculator.calculate(logLikelihood, sampleLowerBound);
                        }

                        LOG_TRACE("sampleLowerBound = " << sampleLowerBound
                                  << ", sampleUpperBound = " << sampleUpperBound
                                  << " p = " << p);

                        lowerBoundCalculator.add(CTools::truncate(sampleLowerBound + p, 0.0, 1.0));
                        upperBoundCalculator.add(CTools::truncate(sampleUpperBound + p, 0.0, 1.0));
                    }

                    if (   !lowerBoundCalculator.calculate(lowerBound)
                        || !upperBoundCalculator.calculate(upperBound))
                    {
                        LOG_ERROR("Couldn't compute probability of less likely samples:"
                                  << " " << lowerBoundCalculator
                                  << " " << upperBoundCalculator);
                        return false;
                    }
                    tail = static_cast<maths_t::ETail>(tail_);
                }
                break;

            case maths_t::E_OneSidedAbove:
                if (!minusLogJointCdfComplement(modes, weightStyles, samples, weights, upperBound, lowerBound))
                {
                    LOG_ERROR("Failed computing probability of less likely samples: "
                              << core::CContainerPrinter::print(samples));
                    return false;
                }
                lowerBound = ::exp(-lowerBound);
                upperBound = ::exp(-upperBound);
                tail = maths_t::E_RightTail;
                break;
            }

            return true;
        }

        //! Check if this is a non-informative prior.
        template<typename T>
        static bool isNonInformative(const std::vector<SMultimodalPriorMode<T> > &modes)
        {
            return modes.empty() || (modes.size() == 1 && modes[0].s_Prior->isNonInformative());
        }

        //! Get a human readable description of the prior.
        template<typename T>
        static void print(const std::vector<SMultimodalPriorMode<T> > &modes,
                          const std::string &indent,
                          std::string &result)
        {
            result += "\n" + indent + "multimodal";
            if (isNonInformative(modes))
            {
                result += " non-informative";
                return;
            }

            double Z = 0.0;
            for (std::size_t i = 0u; i < modes.size(); ++i)
            {
                Z += modes[i].weight();
            }
            result += ":";
            for (std::size_t i = 0u; i < modes.size(); ++i)
            {
                double weight = modes[i].weight() / Z;
                std::string indent_ = indent + " weight "
                                      + core::CStringUtils::typeToStringPretty(weight) + "  ";
                modes[i].s_Prior->print(indent_, result);
            }
        }

    private:
        //! \brief Wrapper to call the -log(c.d.f) of a prior object.
        class CMinusLogJointCdf
        {
            public:
                template<typename T>
                bool operator()(const T &prior,
                                const maths_t::TWeightStyleVec &weightStyles,
                                const TDouble1Vec &samples,
                                const TDouble4Vec1Vec &weights,
                                double &lowerBound,
                                double &upperBound) const
                {
                    return prior->minusLogJointCdf(weightStyles, samples, weights, lowerBound, upperBound);
                }
        };

        //! \brief Wrapper to call the log(1 - c.d.f) of a prior object.
        class CMinusLogJointCdfComplement
        {
            public:
                template<typename T>
                bool operator()(const T &prior,
                                const maths_t::TWeightStyleVec &weightStyles,
                                const TDouble1Vec &samples,
                                const TDouble4Vec1Vec &weights,
                                double &lowerBound,
                                double &upperBound) const
                {
                    return prior->minusLogJointCdfComplement(weightStyles, samples, weights, lowerBound, upperBound);
                }
        };

        //! \brief Wrapper of CMultimodalPrior::minusLogJointCdf function
        //! for use with our solver.
        template<typename PRIOR>
        class CLogCdf
        {
            public:
                using result_type = double;

                enum EStyle
                {
                    E_Lower,
                    E_Upper,
                    E_Mean
                };

            public:
                CLogCdf(EStyle style,
                        const PRIOR &prior,
                        const maths_t::TWeightStyleVec &weightStyles,
                        const TDouble4Vec &weights) :
                    m_Style(style),
                    m_Prior(&prior),
                    m_WeightStyles(&weightStyles),
                    m_Weights(1, weights),
                    m_X(1u, 0.0)
                {}

                double operator()(double x) const
                {
                    m_X[0] = x;
                    double lowerBound, upperBound;
                    if (!m_Prior->minusLogJointCdf(*m_WeightStyles, m_X, m_Weights, lowerBound, upperBound))
                    {
                        throw std::runtime_error("Unable to compute c.d.f. at "
                                                 + core::CStringUtils::typeToString(x));
                    }
                    switch (m_Style)
                    {
                    case E_Lower: return -lowerBound;
                    case E_Upper: return -upperBound;
                    case E_Mean:  return -(lowerBound + upperBound) / 2.0;
                    }
                    return -(lowerBound + upperBound) / 2.0;
                }

            private:
                EStyle m_Style;
                const PRIOR *m_Prior;
                const maths_t::TWeightStyleVec *m_WeightStyles;
                TDouble4Vec1Vec m_Weights;
                //! Avoids creating the vector argument to minusLogJointCdf
                //! more than once.
                mutable TDouble1Vec m_X;
        };

    private:
        //! Implementation of log of the joint c.d.f. of the marginal
        //! likelihood.
        template<typename T, typename CDF>
        static bool minusLogJointCdf(const std::vector<SMultimodalPriorMode<T> > &modes,
                                     CDF minusLogCdf,
                                     const maths_t::TWeightStyleVec &weightStyles,
                                     const TDouble1Vec &samples,
                                     const TDouble4Vec1Vec &weights,
                                     double &lowerBound,
                                     double &upperBound)
        {
            lowerBound = upperBound = 0.0;

            if (samples.empty())
            {
                LOG_ERROR("Can't compute c.d.f. for empty sample set");
                return false;
            }

            if (modes.size() == 1)
            {
                return minusLogCdf(modes[0].s_Prior, weightStyles, samples, weights, lowerBound, upperBound);
            }

            using TMinAccumulator = CBasicStatistics::COrderStatisticsStack<double, 1>;

            // The c.d.f. of the marginal likelihood is the weighted sum
            // of the c.d.fs of each mode since:
            //   cdf(x) = Integral{ L(u) }du
            //          = Integral{ Sum_m{ L(u | m) p(m) } }du
            //          = Sum_m{ Integral{ L(u | m) ) p(m) }du }

            // Declared outside the loop to minimize the number of times
            // they are created.
            TDouble1Vec sample(1);
            TDouble4Vec1Vec weight(1, TDouble4Vec(1, 1.0));
            TDouble4Vec modeLowerBounds;
            TDouble4Vec modeUpperBounds;
            modeLowerBounds.reserve(modes.size());
            modeUpperBounds.reserve(modes.size());

            try
            {
                double mean = maths_t::hasSeasonalVarianceScale(weightStyles, weights) ?
                              marginalLikelihoodMean(modes) : 0.0;

                for (std::size_t i = 0; i < samples.size(); ++i)
                {
                    double n = maths_t::count(weightStyles, weights[i]);
                    double seasonalScale = ::sqrt(maths_t::seasonalVarianceScale(weightStyles, weights[i]));
                    double countVarianceScale = maths_t::countVarianceScale(weightStyles, weights[i]);

                    if (isNonInformative(modes))
                    {
                        lowerBound -= n * ::log(CTools::IMPROPER_CDF);
                        upperBound -= n * ::log(CTools::IMPROPER_CDF);
                        continue;
                    }

                    sample[0] = seasonalScale != 1.0 ? mean + (samples[i] - mean) / seasonalScale : samples[i];
                    weight[0][0] = countVarianceScale;

                    // We re-normalize so that the maximum log c.d.f. is one
                    // to avoid underflow.
                    TMinAccumulator minLowerBound;
                    TMinAccumulator minUpperBound;
                    modeLowerBounds.clear();
                    modeUpperBounds.clear();

                    for (std::size_t j = 0u; j < modes.size(); ++j)
                    {
                        double modeLowerBound;
                        double modeUpperBound;
                        if (!minusLogCdf(modes[j].s_Prior,
                                         TWeights::COUNT_VARIANCE,
                                         sample, weight,
                                         modeLowerBound, modeUpperBound))
                        {
                            LOG_ERROR("Unable to compute c.d.f. for "
                                      << core::CContainerPrinter::print(samples));
                            return false;
                        }
                        minLowerBound.add(modeLowerBound);
                        minUpperBound.add(modeUpperBound);
                        modeLowerBounds.push_back(modeLowerBound);
                        modeUpperBounds.push_back(modeUpperBound);
                    }

                    TMeanAccumulator sampleLowerBound;
                    TMeanAccumulator sampleUpperBound;

                    for (std::size_t j = 0u; j < modes.size(); ++j)
                    {
                        LOG_TRACE("Mode -log(c.d.f.) = [" << modeLowerBounds[j]
                                  << "," << modeUpperBounds[j] << "]");
                        double w = modes[j].weight();
                        // Divide through by the largest value to avoid underflow.
                        // Remember we are working with minus logs so the largest
                        // value corresponds to the smallest log.
                        sampleLowerBound.add(::exp(-(modeLowerBounds[j] - minLowerBound[0])), w);
                        sampleUpperBound.add(::exp(-(modeUpperBounds[j] - minUpperBound[0])), w);
                    }

                    lowerBound += n * std::max(minLowerBound[0] - ::log(CBasicStatistics::mean(sampleLowerBound)), 0.0);
                    upperBound += n * std::max(minUpperBound[0] - ::log(CBasicStatistics::mean(sampleUpperBound)), 0.0);

                    LOG_TRACE("sample = " << core::CContainerPrinter::print(sample)
                              << ", sample -log(c.d.f.) = ["
                              << sampleLowerBound << "," << sampleUpperBound << "]");
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to calculate c.d.f.: " << e.what());
                return false;
            }

            LOG_TRACE("Joint -log(c.d.f.) = [" << lowerBound << "," << upperBound << "]");

            return true;
        }

};

}
}

#endif // INCLUDED_ml_maths_CMultimodalPriorUtils_h
