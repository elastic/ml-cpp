/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CCategoricalTools_h
#define INCLUDED_ml_maths_CCategoricalTools_h

#include <core/CNonCopyable.h>
#include <core/CNonInstantiatable.h>

#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <vector>

namespace ml {
namespace maths {

class MATHS_EXPORT CCategoricalTools : core::CNonInstantiatable, core::CNonCopyable {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;

public:
    //! Compute the probability of seeing a less likely sample from
    //! the multinomial distribution with category counts \p ni and
    //! category probabilities \p probabilities.
    //!
    //! This calculates the probability of seeing a less likely sample
    //! than \f$\(\{n_i\}\)\f$ from the multinomial distribution
    //! <pre class="fragment">
    //!   \f$\(\displaystyle f(\{n_j\}) = \frac{n!}{\prod_i{n_j}}\prod_i{p_j^{n_j}}\)\f$
    //! </pre>
    //!
    //! where \f$\(n = \sum_i{n_i}\)\f$, i.e. the sum
    //! <pre class="fragment">
    //!   \f$\(\displaystyle \sum_{\{n_j : f(\{n_j\}) \leq f(\{n_i\})\} }{ f(\{n_j\}) }\)\f$
    //! </pre>
    //!
    //! This summation is estimated using the Gaussian approximation
    //! to the multinomial distribution.
    //!
    //! \param[in] probabilities The category probabilities, which
    //! should be normalized.
    //! \param[in] i The categories.
    //! \param[in] ni The category counts.
    //! \param[out] result Filled in with an estimate of the probability
    //! of seeing a less likely sample than category counts \p ni.
    static bool probabilityOfLessLikelyMultinomialSample(const TDoubleVec& probabilities,
                                                         const TSizeVec& i,
                                                         const TSizeVec& ni,
                                                         double& result);

    //! Compute the probability of seeing less likely counts than \p ni
    //! independently for each category in \p i whose probabilities are
    //! \p probabilities.
    //!
    //! \param[in] probabilities The category probabilities, which
    //! should be normalized.
    //! \param[in] i The categories.
    //! \param[in] ni The category counts.
    //! \param[out] result Filled in with an estimate of the probability
    //! of seeing a less likely count than nj in \p ni for each category
    //! j in \p i.
    static bool probabilityOfLessLikelyCategoryCount(TDoubleVec& probabilities,
                                                     const TSizeVec& i,
                                                     const TSizeVec& ni,
                                                     TDoubleVec& result,
                                                     std::size_t trials = 100);

    //! Compute the probability that a category will occur in \p n
    //! samples from a multinomial with category probability \p probability.
    //!
    //! For a category \f$k\f$, this is:
    //! <pre class="fragment">
    //!   \f$\displaystyle \sum_{\{\{n_j\}:\sum{n_j}=n\}} H(n_k) n! \prod_{j=1}^m\frac{p_j^{n_j}}{n_j!}\f$
    //! </pre>
    //! where \f$H(.)\f$ denotes the Heavyside function.\n\n
    //! Summing over all other categories it is clear that this is just
    //! the expectation of \f$H(.)\f$ w.r.t. the marginal of the category
    //! \f$k\f$ which is binomial.
    static double probabilityOfCategory(std::size_t n, double probability);

    //! \brief Computes the expected number of distinct categories
    //! in \p n samples from a multinomial random variable with m
    //! categories.
    //!
    //! This computes the expectation of:
    //! <pre class="fragment">
    //!   \f$\displaystyle E[unique(Y)]=\sum_{\{\{n_j\}:n_j>0,\sum{n_j}=n\}}kn!\prod_{j=1}^k{\frac{p_j^{n_j}}{n_j!}} \f$
    //! </pre>
    //! Here, \f$Y\f$ denotes the set of \p n random samples from a
    //! multinomial with m categories.
    //!
    //! We calculate this summation by noting that we can write \f$k\f$
    //! as a sum of Heavyside functions and reorder the summations so we
    //! end up computing a sum of the expectation of these functions w.r.t.
    //! the marginal distributions of the multinomial.
    //!
    //! \warning It is the callers responsibility to ensure that the
    //! probabilities are normalized.
    static bool
    expectedDistinctCategories(const TDoubleVec& probabilities, double n, double& result);

    //! Get the log of the binomial coefficient \f$\binom{n}{m}\f$.
    static double logBinomialCoefficient(std::size_t n, std::size_t m);

    //! Get the binomial coefficient \f$\binom{n}{m}\f$.
    static double binomialCoefficient(std::size_t n, std::size_t m);

    //! Compute the log of the probability of a count of \p m from
    //! a binomial with \p n trials and \p probability of success
    //! \p p.
    //!
    //! This is
    //! <pre class="fragment">
    //!   \f$\(\displaystyle \log\left(\frac{n!}{m!(n-m)!}p^m(1-p)^{n-m}\right)\)\f$
    //! </pre>
    //!
    //! for \f$\(m \leq n\)\f$ and minus maximum double otherwise.
    //!
    //! \param[in] n The number of trials.
    //! \param[in] p The probability of success.
    //! \param[in] m The number of successes.
    //! \param[out] result Filled in with the log probability.
    static maths_t::EFloatingPointErrorStatus
    logBinomialProbability(std::size_t n, double p, std::size_t m, double& result);

    //! Compute the log of the probability of a sample of \p ni counts
    //! of categories from the multinomial with number of trials equal
    //! to the the sum of \p ni and category probabilities \p probabilities.
    //!
    //! This is
    //! <pre class="fragment">
    //!   \f$\(\displaystyle \log\left(\frac{n!}{ \prod_i{n_i!} }p^n_i\right)\)\f$
    //! </pre>
    //!
    //! for \f$\(n = sum_i{n_i}\)\f$.
    //!
    //! \param[in] probabilities The category probabilities.
    //! \param[in] ni The category counts.
    //! \param[out] result Filled in with the log probability.
    static maths_t::EFloatingPointErrorStatus
    logMultinomialProbability(const TDoubleVec& probabilities, const TSizeVec& ni, double& result);
};
}
}

#endif // INCLUDED_ml_maths_CCategoricalTools_h
