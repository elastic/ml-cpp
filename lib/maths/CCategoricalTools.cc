/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CCategoricalTools.h>

#include <core/CLogger.h>
#include <core/Constants.h>

#include <maths/CSampling.h>
#include <maths/CTools.h>
#include <maths/MathsTypes.h>

#include <boost/math/distributions/binomial.hpp>
#include <boost/numeric/conversion/bounds.hpp>

#include <cmath>
#include <iterator>
#include <numeric>

namespace ml {
namespace maths {

namespace {

const double LOG_TWO = std::log(2.0);

//! A fast lower bound for the binomial probability of \p m
//! successes for \p n trials and probability of success \p p.
inline maths_t::EFloatingPointErrorStatus
logBinomialProbabilityFastLowerBound(std::size_t n, double p, std::size_t m, double& result) {
    double n_ = static_cast<double>(n);
    double m_ = static_cast<double>(m);

    result = 0.0;

    if (!(p >= 0.0 || p <= 1.0)) {
        LOG_ERROR(<< "Bad probability: " << p);
        return maths_t::E_FpFailed;
    }
    if (p == 0.0) {
        if (m > 0) {
            result = boost::numeric::bounds<double>::lowest();
            return maths_t::E_FpOverflowed;
        }
        return maths_t::E_FpNoErrors;
    }
    if (p == 1.0) {
        if (m < n) {
            result = boost::numeric::bounds<double>::lowest();
            return maths_t::E_FpOverflowed;
        }
        return maths_t::E_FpNoErrors;
    }
    if (m == 0) {
        result = n_ * std::log(1.0 - p);
        return maths_t::E_FpNoErrors;
    }
    if (m == n) {
        result = n_ * std::log(p);
        return maths_t::E_FpNoErrors;
    }

    // This uses Stirling's approximation for the log of the
    // factorial to approximate the binomial coefficient, i.e.
    //   log(n!) >= log(2*pi) + (n+1/2) * log(n) - n
    //   log(n!) <= 1 + (n+1/2) * log(n) - n

    static const double CONSTANT = std::log(boost::math::double_constants::root_two_pi) - 2.0;

    double p_ = m_ / n_;
    result = -0.5 * std::log(n_ * (1.0 - p_) * p_) + m_ * std::log(p / p_) +
             (n_ - m_) * std::log((1.0 - p) / (1.0 - p_)) + CONSTANT;
    return maths_t::E_FpNoErrors;
}

//! Get a lower bound of the log of right tail probability of a
//! binomial, i.e. the probability of seeing m or a larger value
//! from a binomial with \p trials and probability of success
//! \p p.
maths_t::EFloatingPointErrorStatus
logRightTailProbabilityUpperBound(std::size_t n, double p, std::size_t m, double& result) {
    if (m > n) {
        LOG_ERROR(<< "Invalid sample: " << m << " > " << n);
        result = boost::numeric::bounds<double>::lowest();
        return maths_t::E_FpOverflowed;
    }

    result = 0.0;

    if (n == 0) {
        return maths_t::E_FpNoErrors;
    }
    if (!(p >= 0.0 && p <= 1.0)) {
        LOG_ERROR(<< "Bad probability: " << p);
        return maths_t::E_FpFailed;
    }
    if (p == 0.0) {
        if (m > 0) {
            result = boost::numeric::bounds<double>::lowest();
            return maths_t::E_FpOverflowed;
        }
        return maths_t::E_FpNoErrors;
    }
    if (p == 1.0) {
        return maths_t::E_FpNoErrors;
    }

    // The following uses the Chernoff bound to obtain an upper
    // bound for the right tail probabilities, since the binomial
    // distribution is the sum of independent Bernoulli RV.

    double m_ = static_cast<double>(m);
    double n_ = static_cast<double>(n);

    try {
        boost::math::binomial_distribution<> binomial(n_, p);

        if (m_ <= boost::math::median(binomial)) {
            return maths_t::E_FpNoErrors;
        }

        double eps = (m_ - n_ * p) / n_;
        double q = p + eps;
        double chernoff = m_ * (q * std::log(p / q) +
                                (1.0 - q) * std::log((1.0 - p) / (1.0 - q)));
        result = std::min(chernoff + LOG_TWO, 0.0);
        return maths_t::E_FpNoErrors;
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to calculate c.d.f. complement: " << e.what()
                  << ", n = " << n << ", p = " << p);
    }

    return maths_t::E_FpOverflowed;
}

//! Get a lower bound of the log of right tail probability of a
//! binomial, i.e. the probability of seeing m or a larger value
//! from a binomial with \p trials and probability of success
//! \p p.
maths_t::EFloatingPointErrorStatus
logRightTailProbabilityLowerBound(std::size_t n, double p, std::size_t m, double& result) {
    if (m > n) {
        LOG_ERROR(<< "Invalid sample: " << m << " > " << n);
        result = boost::numeric::bounds<double>::lowest();
        return maths_t::E_FpOverflowed;
    }

    result = 0.0;

    if (n == 0) {
        return maths_t::E_FpNoErrors;
    }
    if (!(p >= 0.0 && p <= 1.0)) {
        LOG_ERROR(<< "Bad probability: " << p);
        return maths_t::E_FpFailed;
    }
    if (p == 0.0) {
        if (m > 0) {
            result = boost::numeric::bounds<double>::lowest();
            return maths_t::E_FpOverflowed;
        }
        return maths_t::E_FpNoErrors;
    }
    if (p == 1.0) {
        return maths_t::E_FpNoErrors;
    }

    // We obtain a lower bound for the right tail probability as
    // follows:
    //   1 - F(m) = Sum_{m<=k<=n}{ n! / (k!(n-k)!) * p^k * (1-p)^(n-k) }
    //            = f(m) * (1 + Sum_{m+1<=k<=n}{ m!(n-m)!/k!/(n-k)! * (p/(1-p))^(k-m) }
    //
    // where F(.) and f(.) are the c.d.f. and p.d.f. of the binomial
    // B(n,p), respectively.
    //
    // Observe that
    //       k! / m!     = k * (k-1) * ... * (m+1) and
    //   (n-m)! / (n-k)! = (n-m) * (n-m-1) * ... * (n-k+1)
    //
    // So the coefficients in the summand have the form
    //   (n-m)/(m+1) * Prod_{1<=x<=k-(m+1)}{ (n-m-x) / (m+1+x) }
    //
    // Since the denominator is less than n we can lower bound this
    // by
    //   >= ((n-m)/n)^(k-(m+1)) * Prod_{0<=x<=k-(m+1)}{ 1 - x/(n-m) }
    //
    // Note that (1 - x/(n-m)) / (1 / (x+1)) >= 1 for x <= n-(m+1).
    // So,
    //   >= ((n-m)/n)^(k-(m+1)) / (k-m)!
    //
    // Finally, substituting back into the summation we get
    //   1 - F(m) >= f(m) * (1 + n/(m+1) * (exp( p/(1-p)*(n-m)/n) - (1 + R)))
    //
    // The remainder can be calculated from the Lagrange form to
    // be less than
    //   max_{0<=x<=p}{ e^x * x^(n-m+1) / (n-m+1)! }
    //
    // for all p and m this is essentially zero so we just ignore
    // it here.

    double m_ = static_cast<double>(m);
    double n_ = static_cast<double>(n);

    try {
        boost::math::binomial_distribution<> binomial(n_, p);

        if (m_ <= boost::math::median(binomial)) {
            return maths_t::E_FpNoErrors;
        }

        double logf;
        maths_t::EFloatingPointErrorStatus status =
            logBinomialProbabilityFastLowerBound(n, p, m, logf);
        if (status & maths_t::E_FpAllErrors) {
            result = logf;
            return status;
        }
        double bound =
            logf + std::log(1.0 + n_ / (m_ + 1.0) *
                                      (std::exp(p / (1.0 - p) * (n_ - m_) / n_) - 1.0));
        result = std::min(bound + LOG_TWO, 0.0);
        return maths_t::E_FpNoErrors;
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to calculate c.d.f. complement: " << e.what()
                  << ", n = " << n << ", p = " << p);
    }

    return maths_t::E_FpFailed;
}

//! Get the log of right tail probability, i.e. the probability
//! of seeing m or a larger value from a binomial with \p trials
//! and probability of success \p p.
maths_t::EFloatingPointErrorStatus
logRightTailProbability(std::size_t n, double p, std::size_t m, double& result) {
    if (m > n) {
        LOG_ERROR(<< "Invalid sample: " << m << " > " << n);
        result = boost::numeric::bounds<double>::lowest();
        return maths_t::E_FpOverflowed;
    }

    result = 0.0;

    if (n == 0) {
        return maths_t::E_FpNoErrors;
    }
    if (!(p >= 0.0 && p <= 1.0)) {
        LOG_ERROR(<< "Bad probability: " << p);
        return maths_t::E_FpFailed;
    }
    if (p == 0.0) {
        if (m > 0) {
            result = boost::numeric::bounds<double>::lowest();
            return maths_t::E_FpOverflowed;
        }
        return maths_t::E_FpNoErrors;
    }
    if (p == 1.0) {
        return maths_t::E_FpNoErrors;
    }

    double n_ = static_cast<double>(n);
    double m_ = static_cast<double>(m);

    try {
        boost::math::binomial_distribution<> binomial(n_, p);

        if (m_ <= boost::math::median(binomial)) {
            return maths_t::E_FpNoErrors;
        }

        // Note that the lower bound is much sharper than the
        // upper bound.

        double lb, ub;
        maths_t::EFloatingPointErrorStatus status =
            logRightTailProbabilityLowerBound(n, p, m, lb);
        if (status & maths_t::E_FpAllErrors) {
            result = status == maths_t::E_FpOverflowed
                         ? boost::numeric::bounds<double>::lowest()
                         : 0.0;
            return status;
        }

        status = logRightTailProbabilityUpperBound(n, p, m, ub);
        if (status & maths_t::E_FpAllErrors) {
            result = status == maths_t::E_FpOverflowed
                         ? boost::numeric::bounds<double>::lowest()
                         : 0.0;
            return status;
        }

        if (ub <= core::constants::LOG_MIN_DOUBLE) {
            result = lb;
            return maths_t::E_FpNoErrors;
        }

        double oneMinusF = CTools::safeCdfComplement(binomial, m_);
        if (oneMinusF == 0.0) {
            result = lb;
            return maths_t::E_FpNoErrors;
        }

        double logf;
        status = CCategoricalTools::logBinomialProbability(n, p, m, logf);
        if (status == maths_t::E_FpFailed) {
            return maths_t::E_FpFailed;
        }

        double f = status == maths_t::E_FpOverflowed ? 0.0 : std::exp(logf);
        result = std::min(std::log(oneMinusF + f) + LOG_TWO, 0.0);
        return maths_t::E_FpNoErrors;
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to calculate c.d.f. complement: " << e.what()
                  << ", n = " << n << ", p = " << p);
    }

    return maths_t::E_FpFailed;
}
}

bool CCategoricalTools::probabilityOfLessLikelyMultinomialSample(const TDoubleVec& /*probabilities*/,
                                                                 const TSizeVec& i,
                                                                 const TSizeVec& ni,
                                                                 double& result) {
    result = 1.0;

    if (i.size() != ni.size()) {
        LOG_ERROR(<< "Inconsistent categories and counts: "
                  << core::CContainerPrinter::print(i) << " "
                  << core::CContainerPrinter::print(ni));
        return false;
    }

    //std::size_t n = std::accumulate(ni.begin(), ni.end(), std::size_t(0));

    // TODO

    return 0.0;
}

double CCategoricalTools::probabilityOfCategory(std::size_t n, const double probability) {
    if (n == 0) {
        return 0.0;
    }

    if (probability > 0.0 && probability < 1.0) {
        boost::math::binomial_distribution<> binomial(static_cast<double>(n), probability);
        return boost::math::cdf(boost::math::complement(binomial, 0.0));
    }

    return probability;
}

bool CCategoricalTools::expectedDistinctCategories(const TDoubleVec& probabilities,
                                                   const double n,
                                                   double& result) {
    // We imagine drawing n samples from a multinomial random variable
    // with m categories. We'd like to calculate how many distinct
    // categories we'd expect in this sample of n. This quantity is
    // given by the summation
    //   n' = Sum_{all combination}{ k * m!/Product_{i=[k]}{ n(i)! } p(i)^(Sum_{i=[k]}{ n(i) }) }
    //                                                              (1)
    //
    // We note that the quantity k can be written as
    //   k(x) = Sum_i{ H(n(i)) }                                    (2)
    //
    // Here, H(.) denotes the Heavyside unit step function. It should
    // be clear that this is equivalent to
    //   |{n(i) : n(i) >= 1 }|
    //
    // i.e. the count of distinct values. Substituting into the summation
    // (1), we note that we can reorder the summations in (1) and (2) so
    // that:
    //   n' = Sum_i{ H(n(i)) * "marginal of category i" }
    //
    // The marginal of a multinomial is binomial, just group all the other
    // categories together, with parameters n and p(i). So,
    //   n' = Sum_i{ 1 - F_i(1) }
    //
    // Here, F_i(.) is the c.d.f. of the binomial R.V. B(n, p(i)).

    result = 0.0;

    if (probabilities.size() == 0) {
        return false;
    }

    for (std::size_t i = 0u; i < probabilities.size(); ++i) {
        if (probabilities[i] > 0.0 && probabilities[i] < 1.0) {
            boost::math::binomial_distribution<> binomial(n, probabilities[i]);
            result += boost::math::cdf(boost::math::complement(binomial, 0.0));
        } else if (probabilities[i] == 1.0) {
            result += 1.0;
        }
    }

    return true;
}

double CCategoricalTools::logBinomialCoefficient(std::size_t n, std::size_t m) {
    if (m == n || m == 0) {
        return 0.0;
    }
    double n_ = static_cast<double>(n);
    double m_ = static_cast<double>(m);
    return std::lgamma(n_ + 1.0) - std::lgamma(m_ + 1.0) - std::lgamma(n_ - m_ + 1.0);
}

double CCategoricalTools::binomialCoefficient(std::size_t n, std::size_t m) {
    return std::exp(logBinomialCoefficient(n, m));
}

bool CCategoricalTools::probabilityOfLessLikelyCategoryCount(TDoubleVec& probabilities,
                                                             const TSizeVec& i,
                                                             const TSizeVec& ni,
                                                             TDoubleVec& result,
                                                             std::size_t trials) {
    result.clear();

    if (i.size() != ni.size()) {
        LOG_ERROR(<< "Inconsistent categories and counts: "
                  << core::CContainerPrinter::print(i) << " "
                  << core::CContainerPrinter::print(ni));
        return false;
    }

    // The quantity that we are interested in calculating is
    // probability of seeing a more extreme right tail event.
    //
    // The marginal distribution of each category is binomial.
    // Therefore, the probability of seeing counts larger than
    // r in the j'th category is 1 - F(r ; n, p(j)). Suppose we
    // observe a count of n(i) for the i'th category. Then its
    // right tail probability is 1 - F(n(i) ; n, p(i)). However,
    // if there are a large number of categories with similar
    // probability then this quantity will be small for every
    // random sample of categories. What we want to calculate
    // is the probability of seeing at least as an extreme right
    // tail probability for a random sample from the distribution.
    //
    // This quantity has to be calculated by Monte-Carlo. In
    // particular, for a number of independent random samples
    // from the multinomial distribution with number of trials
    // n = Sum_i{ n(i) } and probabilities {p} we calculate the
    // minimum right tail probability we see for any category
    // in each sample. This is used to estimate the distribution
    // of the minimum right tail probability. Specifically, we
    // use a curve fit, G, to the cumulative distribution generated
    // by these probabilities. The quantity we are interested in
    // for each category i is approximately G(1 - F(n(i) ; n, p(i)).

    std::size_t n = std::accumulate(ni.begin(), ni.end(), std::size_t(0));

    TDoubleVec probabilities_;
    probabilities_.reserve(i.size());
    for (std::size_t i_ = 0u; i_ < i.size(); ++i_) {
        if (i[i_] >= probabilities.size()) {
            LOG_ERROR(<< "Bad category: " << i[i_] << " out of range");
            return false;
        }
        probabilities_.push_back(probabilities[i[i_]]);
    }

    std::sort(probabilities.begin(), probabilities.end(), std::greater<double>());

    // Declared outside the loop to minimize the number of times
    // it is created.
    TSizeVec sample;

    double n_ = static_cast<double>(n);

    TDoubleVec g;
    g.reserve(trials);

    for (std::size_t i_ = 0u; i_ < trials; ++i_) {
        sample.clear();
        CSampling::multinomialSampleFast(probabilities, n, sample, true);

        double logPMin = 0.0;
        for (std::size_t j = 0u; j < sample.size(); ++j) {
            // We check the sample is in the right tail because
            // we are interested in unusually large values.
            double pj = probabilities[j];
            if (sample[j] > static_cast<std::size_t>((n_ + 1.0) * pj)) {
                std::size_t nj = sample[j];
                double lowerBound;
                if (logRightTailProbabilityLowerBound(n, pj, nj, lowerBound) &
                    maths_t::E_FpAllErrors) {
                    continue;
                }
                if (logPMin > lowerBound) {
                    continue;
                }
                double logP;
                if (logRightTailProbability(n, pj, nj, logP) & maths_t::E_FpAllErrors) {
                    continue;
                }
                logPMin = std::min(logPMin, logP);
            }
        }
        g.push_back(logPMin);
    }

    std::sort(g.begin(), g.end());

    // TODO interpolation.

    return 0.0;
}

maths_t::EFloatingPointErrorStatus
CCategoricalTools::logBinomialProbability(std::size_t n, double p, std::size_t m, double& result) {
    if (m > n) {
        result = boost::numeric::bounds<double>::lowest();
        return maths_t::E_FpOverflowed;
    }

    result = 0.0;

    if (!(p >= 0.0 && p <= 1.0)) {
        LOG_ERROR(<< "Bad probability: " << p);
        return maths_t::E_FpFailed;
    }
    if (p == 0.0) {
        if (m > 0) {
            result = boost::numeric::bounds<double>::lowest();
            return maths_t::E_FpOverflowed;
        }
        return maths_t::E_FpNoErrors;
    }
    if (p == 1.0) {
        if (m < n) {
            result = boost::numeric::bounds<double>::lowest();
            return maths_t::E_FpFailed;
        }
        return maths_t::E_FpNoErrors;
    }

    double n_ = static_cast<double>(n);
    double m_ = static_cast<double>(m);

    double logGammaNPlusOne = 0.0;
    double logGammaMPlusOne = 0.0;
    double logGammaNMinusMPlusOne = 0.0;

    if ((CTools::lgamma(n_ + 1.0, logGammaNPlusOne, true) &&
         CTools::lgamma(m_ + 1.0, logGammaMPlusOne, true) &&
         CTools::lgamma(n_ - m_ + 1.0, logGammaNMinusMPlusOne, true)) == false) {

        return maths_t::E_FpOverflowed;
    }

    result = std::min(logGammaNPlusOne - logGammaMPlusOne - logGammaNMinusMPlusOne +
                          m_ * std::log(p) + (n_ - m_) * std::log(1.0 - p),
                      0.0);
    return maths_t::E_FpNoErrors;
}

maths_t::EFloatingPointErrorStatus
CCategoricalTools::logMultinomialProbability(const TDoubleVec& probabilities,
                                             const TSizeVec& ni,
                                             double& result) {
    result = 0.0;

    if (probabilities.size() != ni.size()) {
        LOG_ERROR(<< "Inconsistent categories and counts: "
                  << core::CContainerPrinter::print(probabilities) << " "
                  << core::CContainerPrinter::print(ni));
        return maths_t::E_FpFailed;
    }

    std::size_t n = std::accumulate(ni.begin(), ni.end(), std::size_t(0));
    if (n == 0) {
        return maths_t::E_FpNoErrors;
    }

    double n_ = static_cast<double>(n);
    double logP = 0.0;

    if (CTools::lgamma(n_ + 1.0, logP, true) == false) {
        return maths_t::E_FpOverflowed;
    }

    for (std::size_t i = 0u; i < ni.size(); ++i) {
        double ni_ = static_cast<double>(ni[i]);
        if (ni_ > 0.0) {
            double pi_ = probabilities[i];
            if (!(pi_ >= 0.0 || pi_ <= 1.0)) {
                LOG_ERROR(<< "Bad probability: " << pi_);
                return maths_t::E_FpFailed;
            }
            if (pi_ == 0.0) {
                result = boost::numeric::bounds<double>::lowest();
                return maths_t::E_FpOverflowed;
            }

            double logGammaNiPlusOne = 0.0;
            if (CTools::lgamma(ni_ + 1.0, logGammaNiPlusOne, true) == false) {
                return maths_t::E_FpOverflowed;
            }

            logP += ni_ * std::log(pi_) - logGammaNiPlusOne;
        }
    }

    result = std::min(logP, 0.0);
    return maths_t::E_FpNoErrors;
}
}
}
