/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CMultinomialConjugate.h>

#include <core/CContainerPrinter.h>
#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CMathsFuncs.h>
#include <maths/COrderings.h>
#include <maths/CRestoreParams.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>
#include <maths/ProbabilityAggregators.h>

#include <boost/math/distributions/beta.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tuple/tuple_io.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>

namespace ml {
namespace maths {

namespace {
using TDoubleVec = std::vector<double>;
using TDoubleVecCItr = TDoubleVec::const_iterator;
using TDouble7Vec = core::CSmallVector<double, 7>;

namespace detail {
using TDoubleDoublePr = std::pair<double, double>;

//! Truncate \p to fit into a signed integer.
int truncate(std::size_t x) {
    return x > static_cast<std::size_t>(std::numeric_limits<int>::max())
               ? std::numeric_limits<int>::max()
               : static_cast<int>(x);
}

//! This computes the cumulative density function of the predictive
//! distribution for a single sample from the multinomial. The predictive
//! distribution is obtained by integrating over the Dirichlet prior
//! for all the probabilities of the multinomial. In particular,
//! <pre class="fragment">
//!   \f$\displaystyle \int{L(x_i\ |\ p)f(p)}dp = \Gamma(\sum_j{a_j})\int_{p\in S}{p_i\prod_j{\frac {p_j^{a_j-1}}{\Gamma(a_j)}}}dp_j \f$
//! </pre class="fragment">
//!
//! Here, the integral is over the simplex defined by:
//! <pre class="fragment">
//!   \f$\displaystyle R = \{p\ |\ \left\|p\right\|_1 = 1\}\f$
//! </pre class="fragment">
//!
//! This reduces to:
//! <pre class="fragment">
//!   \f$\displaystyle \frac{\Gamma(a_0)\Gamma(a_i + 1)}{\Gamma(a_0 + 1)\Gamma(a_i)} = \frac{a_i}{a_0}\f$
//! </pre class="fragment">
//!
//! where, \f$\displaystyle a_0 = \sum_i{a_i}\f$. We see that the c.d.f.
//! is thus given by the sum:
//! <pre class="fragment">
//!   \f$\displaystyle F(x) = \frac{1}{a_0}\sum_{\{i : x_i \leq x\}}{a_i}\f$
//! </pre class="fragment">
//!
//! In the case that the model has overflowed there are clear sharp upper
//! and lower bounds: the upper bound assumes that all the additional mass
//! is to the left of every sample, the lower bound assumes that all the
//! additional mass is to the right of every sample.
class CCdf : core::CNonCopyable {
public:
    CCdf(const TDoubleVec& categories, const TDoubleVec& concentrations, double totalConcentration)
        : m_Categories(categories), m_Cdf(), m_Pu(0.0) {
        m_Cdf.reserve(m_Categories.size() + 2u);

        // Construct the c.d.f.
        m_Cdf.push_back(0.0);
        double r = 1.0 / static_cast<double>(concentrations.size());
        for (std::size_t i = 0u; i < concentrations.size(); ++i) {
            double p = concentrations[i] / totalConcentration;
            m_Cdf.push_back(m_Cdf.back() + p);
            m_Pu += r - p;
        }
        m_Cdf.push_back(m_Cdf.back());
    }

    void operator()(double x, double& lowerBound, double& upperBound) const {
        std::size_t category =
            std::upper_bound(m_Categories.begin(), m_Categories.end(), x) -
            m_Categories.begin();

        lowerBound = m_Cdf[category];
        upperBound = m_Cdf[category] + m_Pu;
    }

    void dump(TDoubleVec& lowerBounds, TDoubleVec& upperBounds) const {
        lowerBounds.reserve(m_Cdf.size() - 2u);
        upperBounds.reserve(m_Cdf.size() - 2u);
        for (std::size_t i = 1u; i < m_Cdf.size() - 1u; ++i) {
            lowerBounds.push_back(m_Cdf[i]);
            upperBounds.push_back(m_Cdf[i] + m_Pu);
        }
    }

private:
    const TDoubleVec& m_Categories;
    TDoubleVec m_Cdf;
    double m_Pu;
};

//! This computes the complement of the cumulative density function
//! of the predictive. This is:
//! <pre class="fragment">
//!   \f$\displaystyle F^c(x) = \frac{1}{a_0}\sum_{\{i : x_i \geq x\}}{a_i}\f$
//! </pre class="fragment">
//!
//! Note that \f$F^c(x) \neq 1 - F(x)\f$ (which is the standard definition)
//! since they differ at the points \f${x_i}\f$. This is intentional since
//! we use this in the calculation of the probability of less likely samples
//! where the definition above is appropriate.
//! \see CCdf for details.
//!
//! In the case that the model has overflowed there are clear sharp upper
//! and lower bounds: the upper bound assumes that all the additional mass
//! is to the right of every sample, the lower bound assumes that all the
//! additional mass is to the left of every sample.
class CCdfComplement : core::CNonCopyable {
public:
    CCdfComplement(const TDoubleVec& categories, const TDoubleVec& concentrations, double totalConcentration)
        : m_Categories(categories), m_CdfComplement(), m_Pu(0.0) {
        m_CdfComplement.reserve(m_Categories.size() + 2u);

        // Construct the c.d.f.
        m_CdfComplement.push_back(0.0);
        double r = 1.0 / static_cast<double>(concentrations.size());
        for (std::size_t i = concentrations.size(); i > 0; --i) {
            double p = concentrations[i - 1] / totalConcentration;
            m_CdfComplement.push_back(m_CdfComplement.back() + p);
            m_Pu += r - p;
        }
        m_CdfComplement.push_back(m_CdfComplement.back());
    }

    void operator()(double x, double& lowerBound, double& upperBound) const {
        std::size_t category =
            std::lower_bound(m_Categories.begin(), m_Categories.end(), x) -
            m_Categories.begin();

        lowerBound = m_CdfComplement[category + 1];
        upperBound = m_CdfComplement[category + 1] + m_Pu;
    }

    void dump(TDoubleVec& lowerBounds, TDoubleVec& upperBounds) const {
        lowerBounds.reserve(m_CdfComplement.size() - 2u);
        upperBounds.reserve(m_CdfComplement.size() - 2u);
        for (std::size_t i = 1u; i < m_CdfComplement.size() - 1u; ++i) {
            lowerBounds.push_back(m_CdfComplement[i]);
            upperBounds.push_back(m_CdfComplement[i] + m_Pu);
        }
    }

private:
    const TDoubleVec& m_Categories;
    TDoubleVec m_CdfComplement;
    double m_Pu;
};

//! Get the number of samples of the marginal priors to use as a
//! function of the total concentration in the prior.
//!
//! This was determined, empirically, to give reasonable errors
//! in the calculation of the less likely probabilities.
std::size_t numberPriorSamples(double x) {
    static const double THRESHOLDS[] = {100.0, 1000.0, 10000.0,
                                        boost::numeric::bounds<double>::highest()};
    static const std::size_t NUMBERS[] = {7u, 5u, 3u, 1u};
    return NUMBERS[std::lower_bound(boost::begin(THRESHOLDS), boost::end(THRESHOLDS), x) - boost::begin(THRESHOLDS)];
}

//! Generate \p numberSamples samples of a beta R.V. with alpha \p a
//! and beta \p b.
//!
//! \param[in] a The alpha shape parameter, i.e. \f$\alpha\f$.
//! \param[in] b The beta shape parameter, i.e. \f$\beta\f$.
//! \param[in] numberSamples The number of samples to generate.
//! \param[out] samples Filled in with \p numberSamples of a beta
//! R.V. with distribution:
//! <pre class="fragment">
//!   \f$\displaystyle \frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{beta-1}\f$
//! </pre>
void generateBetaSamples(double a, double b, std::size_t numberSamples, TDouble7Vec& samples) {
    samples.clear();
    samples.reserve(numberSamples);

    double mean = a / (a + b);
    if (numberSamples == 1 || a == 0.0 || b == 0.0) {
        samples.push_back(mean);
        return;
    }

    try {
        boost::math::beta_distribution<> beta(a, b);
        boost::math::beta_distribution<> betaAlphaPlus1(a + 1, b);
        double dq = 1.0 / static_cast<double>(numberSamples);
        double q = dq;
        double f = 0.0;
        mean /= dq;
        for (std::size_t i = 1u; i < numberSamples; ++i, q += dq) {
            double xq = boost::math::quantile(beta, q);
            double fq = boost::math::cdf(betaAlphaPlus1, xq);
            samples.push_back(mean * (fq - f));
            f = fq;
        }
        samples.push_back(mean * (1.0 - f));
    } catch (const std::exception&) {
        samples.clear();
        samples.push_back(mean);
    }
}

} // detail::

using TDoubleDoubleMap = std::map<double, double>;
using TDoubleDoubleMapCItr = TDoubleDoubleMap::const_iterator;
using TDoubleDoubleSizeTr = boost::tuples::tuple<double, double, std::size_t>;
using TDoubleDoubleSizeTrVec = std::vector<TDoubleDoubleSizeTr>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

// We use short field names to reduce the state size
const std::string NUMBER_AVAILABLE_CATEGORIES_TAG("a");
const std::string CATEGORY_TAG("b");
const std::string CONCENTRATION_TAG("c");
const std::string TOTAL_CONCENTRATION_TAG("d");
const std::string NUMBER_SAMPLES_TAG("e");
//const std::string MINIMUM_TAG("f"); No longer used
//const std::string MAXIMUM_TAG("g"); No longer used
const std::string DECAY_RATE_TAG("h");
const std::string EMPTY_STRING;
}

CMultinomialConjugate::CMultinomialConjugate()
    : m_NumberAvailableCategories(0), m_TotalConcentration(0.0) {
}

CMultinomialConjugate::CMultinomialConjugate(std::size_t maximumNumberOfCategories,
                                             const TDoubleVec& categories,
                                             const TDoubleVec& concentrations,
                                             double decayRate)
    : CPrior(maths_t::E_DiscreteData, decayRate),
      m_NumberAvailableCategories(detail::truncate(maximumNumberOfCategories) -
                                  detail::truncate(categories.size())),
      m_Categories(categories), m_Concentrations(concentrations),
      m_TotalConcentration(0.0) {
    m_Concentrations.resize(m_Categories.size(), NON_INFORMATIVE_CONCENTRATION);
    m_TotalConcentration =
        std::accumulate(m_Concentrations.begin(), m_Concentrations.end(), 0.0);
    this->numberSamples(m_TotalConcentration);
}

CMultinomialConjugate::CMultinomialConjugate(const SDistributionRestoreParams& params,
                                             core::CStateRestoreTraverser& traverser)
    : CPrior(maths_t::E_DiscreteData, params.s_DecayRate),
      m_NumberAvailableCategories(0), m_TotalConcentration(0.0) {
    traverser.traverseSubLevel(
        boost::bind(&CMultinomialConjugate::acceptRestoreTraverser, this, _1));
}

bool CMultinomialConjugate::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(DECAY_RATE_TAG, double decayRate,
                               core::CStringUtils::stringToType(traverser.value(), decayRate),
                               this->decayRate(decayRate))
        RESTORE_BUILT_IN(NUMBER_AVAILABLE_CATEGORIES_TAG, m_NumberAvailableCategories)
        if (!name.empty() && name[0] == CATEGORY_TAG[0]) {
            // Categories have been split across multiple fields b0, b1, etc
            if (core::CPersistUtils::fromString(
                    traverser.value(), m_Categories, core::CPersistUtils::DELIMITER,
                    core::CPersistUtils::PAIR_DELIMITER, true) == false) {
                LOG_ERROR(<< "Invalid categories in split " << traverser.value());
                return false;
            }
            continue;
        }
        if (!name.empty() && name[0] == CONCENTRATION_TAG[0]) {
            // Concentrations have been split across multiple fields c0, c1, c2, etc
            if (core::CPersistUtils::fromString(
                    traverser.value(), m_Concentrations, core::CPersistUtils::DELIMITER,
                    core::CPersistUtils::PAIR_DELIMITER, true) == false) {
                LOG_ERROR(<< "Invalid concentrations in split " << traverser.value());
                return false;
            }
            continue;
        }
        RESTORE_BUILT_IN(TOTAL_CONCENTRATION_TAG, m_TotalConcentration)
        RESTORE_SETUP_TEARDOWN(NUMBER_SAMPLES_TAG, double numberSamples,
                               core::CStringUtils::stringToType(traverser.value(), numberSamples),
                               this->numberSamples(numberSamples))
    } while (traverser.next());

    this->shrink();

    return true;
}

void CMultinomialConjugate::swap(CMultinomialConjugate& other) {
    this->CPrior::swap(other);
    std::swap(m_NumberAvailableCategories, other.m_NumberAvailableCategories);
    m_Categories.swap(other.m_Categories);
    m_Concentrations.swap(other.m_Concentrations);
    std::swap(m_TotalConcentration, other.m_TotalConcentration);
}

CMultinomialConjugate CMultinomialConjugate::nonInformativePrior(std::size_t maximumNumberOfCategories,
                                                                 double decayRate) {
    return CMultinomialConjugate(maximumNumberOfCategories, TDoubleVec(),
                                 TDoubleVec(), decayRate);
}

CMultinomialConjugate::EPrior CMultinomialConjugate::type() const {
    return E_Multinomial;
}

CMultinomialConjugate* CMultinomialConjugate::clone() const {
    return new CMultinomialConjugate(*this);
}

void CMultinomialConjugate::setToNonInformative(double /*offset*/, double decayRate) {
    *this = nonInformativePrior(
        m_NumberAvailableCategories + detail::truncate(m_Categories.size()), decayRate);
}

bool CMultinomialConjugate::needsOffset() const {
    return false;
}

double CMultinomialConjugate::adjustOffset(const TDouble1Vec& /*samples*/,
                                           const TDoubleWeightsAry1Vec& /*weights*/) {
    return 1.0;
}

double CMultinomialConjugate::offset() const {
    return 0.0;
}

void CMultinomialConjugate::addSamples(const TDouble1Vec& samples,
                                       const TDoubleWeightsAry1Vec& weights) {
    if (samples.empty()) {
        return;
    }
    if (samples.size() != weights.size()) {
        LOG_ERROR(<< "Mismatch in samples '"
                  << core::CContainerPrinter::print(samples) << "' and weights '"
                  << core::CContainerPrinter::print(weights) << "'");
        return;
    }

    this->CPrior::addSamples(samples, weights);

    // If x = {x(i)} denotes the sample vector, then x are multinomially
    // distributed with probabilities {p(i)}. Let n(i) denote the counts
    // of each category in x, then the likelihood function is:
    //   likelihood(x | {p(i)}) ~
    //     Product_i{ 1 / n(i)! * p(i)^n(i) }.
    //
    // This factorizes as f({n(i)}) * g({p(i)}) so the conjugate joint
    // prior for {p(i)} and is Dirichlet and the update of the posterior
    // with n independent samples comes from:
    //   likelihood(x | {p(i)}) * prior({p(i)} | {a(i)})             (1)
    //
    // where,
    //   prior({p(i)} | {a(i)}) = 1/B({a(i)}) * Product_i{ p(i)^a(i) }.
    //   B({a(i)}) is the multinomial beta function.
    //   {a(i)} are the concentration parameters of the Dirichlet distribution.
    //
    // Equation (1) implies that the parameters of the prior distribution
    // update as follows:
    //   a(i) -> a(i) + n(i)
    //
    // Note that the weight of the sample x(i) is interpreted as its count,
    // i.e. n(i), so for example updating with {(x, 2)} is equivalent to
    // updating with {x, x}.

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        double x = samples[i];
        if (CMathsFuncs::isNan(x)) {
            LOG_ERROR(<< "Discarding " << x);
            continue;
        }
        double n = maths_t::countForUpdate(weights[i]);
        if (!CMathsFuncs::isFinite(n)) {
            LOG_ERROR(<< "Bad count weight " << n);
            continue;
        }

        m_TotalConcentration += n;

        std::size_t category =
            std::lower_bound(m_Categories.begin(), m_Categories.end(), x) -
            m_Categories.begin();
        if (category == m_Categories.size() || m_Categories[category] != x) {
            m_NumberAvailableCategories = std::max(m_NumberAvailableCategories - 1, -1);
            if (m_NumberAvailableCategories < 0) {
                continue;
            }

            // This is infrequent so the amortized cost is low.

            m_Categories.insert(m_Categories.begin() + category, x);
            m_Concentrations.insert(m_Concentrations.begin() + category,
                                    NON_INFORMATIVE_CONCENTRATION);
            this->shrink();
        }

        m_Concentrations[category] += n;
    }

    LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples)
              << ", m_NumberAvailableCategories = " << m_NumberAvailableCategories
              << ", m_Categories = " << core::CContainerPrinter::print(m_Categories)
              << ", m_Concentrations = " << core::CContainerPrinter::print(m_Concentrations)
              << ", m_TotalConcentration = " << m_TotalConcentration);
}

void CMultinomialConjugate::propagateForwardsByTime(double time) {
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
        LOG_ERROR(<< "Can't propagate model backwards in time");
        return;
    }

    if (this->isNonInformative()) {
        // Nothing to be done.
        return;
    }

    double alpha = std::exp(-this->decayRate() * time);

    // We want to increase the variance of each category while holding
    // its mean constant s.t. in the limit t -> inf var -> inf. The mean
    // and variance of the i'th category are:
    //   a(i) / a0
    //   a(i) * (a0 - a(i)) / a0^2 / (a0 + 1),
    //
    // where, a0 = Sum_i{ a(i) } so we choose a factor f in the range
    // [0, 1] and as follows:
    //   a(i) -> f * a(i)
    //
    // Thus the mean is unchanged and for large a0 the variance is
    // increased by very nearly 1 / f.

    double factor = std::min(
        (alpha * m_TotalConcentration + (1.0 - alpha) * NON_INFORMATIVE_CONCENTRATION) / m_TotalConcentration,
        1.0);

    for (std::size_t i = 0u; i < m_Concentrations.size(); ++i) {
        m_Concentrations[i] *= factor;
    }

    m_TotalConcentration *= factor;

    this->numberSamples(this->numberSamples() * factor);

    LOG_TRACE(<< "factor = " << factor
              << ", m_Concentrations = " << core::CContainerPrinter::print(m_Concentrations)
              << ", m_TotalConcentration = " << m_TotalConcentration
              << ", numberSamples = " << this->numberSamples());
}

CMultinomialConjugate::TDoubleDoublePr CMultinomialConjugate::marginalLikelihoodSupport() const {

    // Strictly speaking for a particular likelihood this is the
    // set of discrete values or categories, but we are interested
    // in the support for the possible discrete values which can
    // be any real numbers.

    return {boost::numeric::bounds<double>::lowest(),
            boost::numeric::bounds<double>::highest()};
}

double CMultinomialConjugate::marginalLikelihoodMean() const {

    if (this->isNonInformative()) {
        return 0.0;
    }

    // This is just E_{p(i)}[E[X | p(i)]] and
    //   E[X | p(i)] = Sum_i( c(i) * p(i) )
    //
    // By linearity of expectation
    //   E[{p(i)}[E[X | p(i)]] = Sum_i{ c(i) * E[p(i)] }

    TMeanAccumulator result;
    TDoubleVec probabilities = this->probabilities();
    for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
        result.add(m_Categories[i], probabilities[i]);
    }
    return CBasicStatistics::mean(result);
}

double CMultinomialConjugate::marginalLikelihoodMode(const TDoubleWeightsAry& /*weights*/) const {

    if (this->isNonInformative()) {
        return 0.0;
    }

    // This is just the category with the maximum concentration.

    std::size_t mode = 0u;
    for (std::size_t i = 1u; i < m_Concentrations.size(); ++i) {
        if (m_Concentrations[i] > m_Concentrations[mode]) {
            mode = i;
        }
    }

    return m_Categories[mode];
}

double CMultinomialConjugate::marginalLikelihoodVariance(const TDoubleWeightsAry& /*weights*/) const {

    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    if (this->isNonInformative()) {
        return boost::numeric::bounds<double>::highest();
    }

    // This is just E_{p(i)}[Var[X | p(i)]] and
    //   Var[X | p(i)] = Sum_i( (c(i) - m)^2 * p(i) )
    //
    // By linearity of expectation
    //   E[{p(i)}[E[X | p(i)]] = Sum_i{ (c(i) - m)^2 * E[p(i)] }

    TMeanVarAccumulator result;
    TDoubleVec probabilities = this->probabilities();
    for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
        result.add(m_Categories[i], probabilities[i]);
    }
    return CBasicStatistics::variance(result);
}

CMultinomialConjugate::TDoubleDoublePr
CMultinomialConjugate::marginalLikelihoodConfidenceInterval(double percentage,
                                                            const TDoubleWeightsAry& /*weights*/) const {

    if (this->isNonInformative()) {
        return this->marginalLikelihoodSupport();
    }

    percentage /= 100.0;
    percentage = CTools::truncate(percentage, 0.0, 1.0);

    // We use the fact that the marginal likelihood is multinomial.

    TDoubleVec quantiles;
    quantiles.reserve(m_Concentrations.size());
    double pU = 0.0;
    double pCumulative = 0.0;
    for (std::size_t i = 0u; i < m_Concentrations.size(); ++i) {
        double p = m_Concentrations[i] / m_TotalConcentration;
        pCumulative += p;
        quantiles.push_back(pCumulative);
        pU += 1.0 / static_cast<double>(m_Concentrations.size()) - p;
    }
    double q1 = (1.0 - percentage) / 2.0;
    ptrdiff_t i1 = std::lower_bound(quantiles.begin(), quantiles.end(), q1 - pU) -
                   quantiles.begin();
    double x1 = m_Categories[i1];
    double x2 = x1;
    if (percentage > 0.0) {
        double q2 = (1.0 + percentage) / 2.0;
        ptrdiff_t i2 =
            std::min(std::lower_bound(quantiles.begin(), quantiles.end(), q2 + pU) -
                         quantiles.begin(),
                     static_cast<ptrdiff_t>(quantiles.size()) - 1);
        x2 = m_Categories[i2];
    }
    LOG_TRACE(<< "x1 = " << x1 << ", x2 = " << x2)
    LOG_TRACE(<< "quantiles = " << core::CContainerPrinter::print(quantiles));
    LOG_TRACE(<< "            " << core::CContainerPrinter::print(m_Categories));

    return {x1, x2};
}

maths_t::EFloatingPointErrorStatus
CMultinomialConjugate::jointLogMarginalLikelihood(const TDouble1Vec& samples,
                                                  const TDoubleWeightsAry1Vec& weights,
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

    // We assume the data are described by X, where X is multinomially
    // distributed, i.e. for the vector {x(i)}
    //   f(x(i)) = n! / Product_i{ n(i)! } * Product_i{ p(i)^n(i) }
    //
    // where, n is the number of elements in the sample vector and n(i)
    // is the count of occurrences of category x(i) in the vector.
    //
    // The marginal likelihood function of the samples is the likelihood
    // function for the data integrated over the prior distribution for
    // the probabilities {p(i)}. It can be shown that:
    //   log( L(x | {p(i)}) ) =
    //     log( n! / Product_i{ n(i)! } * B(a') / B(a) ).
    //
    // Here,
    //   B(a) is the multinomial beta function.
    //   a = {a(i)} is the vector of concentration parameters.
    //   a' = {a(i) + n(i)}.
    //
    // Note that any previously unobserved category has concentration
    // parameter equal to 0.

    TDoubleDoubleMap categoryCounts;
    double numberSamples = 0.0;

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        double n = maths_t::countForUpdate(weights[i]);
        numberSamples += n;
        categoryCounts[samples[i]] += n;
    }

    try {
        LOG_TRACE(<< "# samples = " << numberSamples
                  << ", total concentration = " << m_TotalConcentration);

        result = std::lgamma(numberSamples + 1.0) + std::lgamma(m_TotalConcentration) -
                 std::lgamma(m_TotalConcentration + numberSamples);

        for (TDoubleDoubleMapCItr countItr = categoryCounts.begin();
             countItr != categoryCounts.end(); ++countItr) {
            double category = countItr->first;
            double count = countItr->second;
            LOG_TRACE(<< "category = " << category << ", count = " << count);

            result -= std::lgamma(countItr->second + 1.0);

            std::size_t index = std::lower_bound(m_Categories.begin(),
                                                 m_Categories.end(), category) -
                                m_Categories.begin();
            if (index < m_Categories.size() && m_Categories[index] == category) {
                LOG_TRACE(<< "concentration = " << m_Concentrations[index]);
                result += std::lgamma(m_Concentrations[index] + count) -
                          std::lgamma(m_Concentrations[index]);
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Unable to compute joint log likelihood: " << e.what()
                  << ", samples = " << core::CContainerPrinter::print(samples)
                  << ", categories = " << core::CContainerPrinter::print(m_Categories)
                  << ", concentrations = " << core::CContainerPrinter::print(m_Concentrations));
        return maths_t::E_FpFailed;
    }

    LOG_TRACE(<< "result = " << result);

    maths_t::EFloatingPointErrorStatus status = CMathsFuncs::fpStatus(result);
    if (status & maths_t::E_FpFailed) {
        LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR(<< "weights = " << core::CContainerPrinter::print(weights));
    }
    return status;
}

void CMultinomialConjugate::sampleMarginalLikelihood(std::size_t numberSamples,
                                                     TDouble1Vec& samples) const {

    samples.clear();

    if (numberSamples == 0 || this->isNonInformative()) {
        return;
    }

    // We sample each category according to its probability.
    // In particular, the expected probability of the i'th category
    // is:
    //   p(i) = E[P(i)] = a(i) / Sum_j{ a(j) }
    //
    // where, {a(i)} are the concentration parameters. Note that
    // there is a mild complication if we have observed more
    // categories than we are permitted in this case:
    //   Sum_i{ p(i) } < 1.0.
    //
    // We handle this by rounding the number of samples to the
    // nearest integer to n * Sum_i{ p(i) }.

    LOG_TRACE(<< "Number samples = " << numberSamples);

    TDoubleVec probabilities;
    probabilities.reserve(m_Categories.size());
    for (std::size_t i = 0u; i < m_Concentrations.size(); ++i) {
        double p = m_Concentrations[i] / m_TotalConcentration;
        probabilities.push_back(p);
    }

    CSampling::TSizeVec sampling;
    CSampling::weightedSample(numberSamples, probabilities, sampling);

    if (sampling.size() != m_Categories.size()) {
        LOG_ERROR(<< "Failed to sample marginal likelihood");
        return;
    }

    samples.reserve(numberSamples);
    for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
        std::fill_n(std::back_inserter(samples), sampling[i], m_Categories[i]);
    }
    LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));
}

bool CMultinomialConjugate::minusLogJointCdf(const TDouble1Vec& samples,
                                             const TDoubleWeightsAry1Vec& weights,
                                             double& lowerBound,
                                             double& upperBound) const {
    lowerBound = upperBound = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute distribution for empty sample set");
        return false;
    }

    // We assume that the samples are independent.
    //
    // Computing the true joint marginal distribution of all the samples
    // by integrating the joint likelihood over the prior distribution for
    // the multinomial probabilities is not tractable. We will approximate
    // the joint p.d.f. as follows:
    //   Integral{ Product_i{ L(x(i) | p) } * f(p) }dp
    //      ~= Product_i{ Integral{ L(x(i) | p) * f(p) }dp }.
    //
    // where,
    //   L(. | p) is the likelihood function.
    //   f(.) is the Dirichlet prior.
    //   p = {p(i)} is the vector of likelihood probabilities.
    //
    // This becomes increasingly accurate as the prior distribution narrows.
    // The marginal c.d.f. of a single sample is just:
    //   E[ Sum_{x(i) <= x}{ p(i) } ]
    //
    // where the expectation is taken over the prior for the probabilities
    // of each category x(i), i.e. p(i). By linearity of expectation this
    // is just:
    //   Sum_{x(i) <= x}{ E[p(i)] }
    //
    // Here, E[p(i)] is just the mean of the i'th category's probability,
    // which is a(i) / Sum_j{a(j)} where {a(i)} are the prior concentrations.

    detail::CCdf cdf(m_Categories, m_Concentrations, m_TotalConcentration);

    static const double MAX_DOUBLE = boost::numeric::bounds<double>::highest();

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        double x = samples[i];
        double n = maths_t::count(weights[i]);

        double sampleLowerBound;
        double sampleUpperBound;
        cdf(x, sampleLowerBound, sampleUpperBound);

        // We need to handle the case that the c.d.f. is zero and hence
        // the log blows up.
        lowerBound = sampleLowerBound == 0.0 || lowerBound == MAX_DOUBLE
                         ? MAX_DOUBLE
                         : lowerBound - n * std::log(sampleLowerBound);
        upperBound = sampleUpperBound == 0.0 || upperBound == MAX_DOUBLE
                         ? MAX_DOUBLE
                         : upperBound - n * std::log(sampleUpperBound);
    }

    return true;
}

bool CMultinomialConjugate::minusLogJointCdfComplement(const TDouble1Vec& samples,
                                                       const TDoubleWeightsAry1Vec& weights,
                                                       double& lowerBound,
                                                       double& upperBound) const {

    // See minusLogJointCdf for the rationale behind this approximation.

    detail::CCdfComplement cdfComplement(m_Categories, m_Concentrations, m_TotalConcentration);

    static const double MAX_DOUBLE = boost::numeric::bounds<double>::highest();

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        double x = samples[i];
        double n = maths_t::count(weights[i]);

        double sampleLowerBound;
        double sampleUpperBound;
        cdfComplement(x, sampleLowerBound, sampleUpperBound);

        // We need to handle the case that the c.d.f. is zero and hence
        // the log blows up.
        lowerBound = sampleLowerBound == 0.0 || lowerBound == MAX_DOUBLE
                         ? MAX_DOUBLE
                         : lowerBound - n * std::log(sampleLowerBound);
        upperBound = sampleUpperBound == 0.0 || upperBound == MAX_DOUBLE
                         ? MAX_DOUBLE
                         : upperBound - n * std::log(sampleUpperBound);
    }

    return true;
}

bool CMultinomialConjugate::probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                           const TDouble1Vec& samples,
                                                           const TDoubleWeightsAry1Vec& weights,
                                                           double& lowerBound,
                                                           double& upperBound,
                                                           maths_t::ETail& tail) const {
    lowerBound = upperBound = 0.0;
    tail = maths_t::E_UndeterminedTail;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute distribution for empty sample set");
        return false;
    }

    // To compute the overall joint probability we use our approximate
    // aggregation scheme which interprets the probabilities as arising
    // from jointly independent Gaussians. Note that trying to compute
    // the exact aggregation requires us to calculate:
    //   P(R) = P({ y | f(y | p) <= f(x | p) })
    //
    // where,
    //   f(. | p) = n! * Product_i{ p(i)^n(i) / n(i)! }, i.e. the
    //   multinomial distribution function.
    //   n(i) are the counts of the i'th category in vector argument.
    //
    // This is computationally hard. (We might be able to make head way
    // on the relaxation of the problem for the case that {n(i)} are
    // no longer integer, but must sum to one. In which case the values
    // live on a k-dimensional simplex. We would be interested in contours
    // of constant probability, which might be computable by the method
    // of Lagrange multipliers.)

    switch (calculation) {
    case maths_t::E_OneSidedBelow: {
        // See minusLogJointCdf for a discussion of the calculation
        // of the marginal c.d.f. for a single sample.

        CJointProbabilityOfLessLikelySamples jointLowerBound;
        CJointProbabilityOfLessLikelySamples jointUpperBound;

        detail::CCdf cdf(m_Categories, m_Concentrations, m_TotalConcentration);
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double x = samples[i];
            double n = maths_t::count(weights[i]);
            double sampleLowerBound, sampleUpperBound;
            cdf(x, sampleLowerBound, sampleUpperBound);
            jointLowerBound.add(sampleLowerBound, n);
            jointUpperBound.add(sampleUpperBound, n);
        }

        if (!jointLowerBound.calculate(lowerBound) || !jointUpperBound.calculate(upperBound)) {
            LOG_ERROR(<< "Unable to compute probability for "
                      << core::CContainerPrinter::print(samples) << ": "
                      << jointLowerBound << ", " << jointUpperBound);
            return false;
        }
        tail = maths_t::E_LeftTail;
    } break;

    case maths_t::E_TwoSided: {
        // The probability of a less likely category is given by:
        //   E[ Sum_{p(j) <= p(i)}{ p(j) } ]
        //
        // where the expectation is taken over the prior for the distribution
        // probabilities. This is hard to compute because the terms in the sum
        // vary as the probabilities vary. We approximate this by taking the
        // expectation over the marginal for each p(i). In particular, we compute:
        //   P(i) = Sum_{E[p(j)] <= E[p(i)]}{ E[p(j)] }
        //
        // Here, P(i) is the probability of less likely category and the
        // expected probability of the i'th category is:
        //   E[p(i)] = a(i) / Sum_j{ a(j) }                                (1)
        //
        // where, a(i) are the concentration parameters of the prior. So, (1)
        // reduces to:
        //   Sum_{j:a(j)<=a(i)}{ E[p(j)] }
        //
        // We can think of P(.) as a function of probability, i.e. if the
        // probability of a category were p then its corresponding P(.) would
        // be:
        //   P(argmin_{i : E[p(i)] >= p}{ E[p(i)] })
        //
        // Given this definition we can compute:
        //   E[ P(p) ]                                                     (2)
        //
        // where the expectation is taken over the marginal for each probability.
        // This can be computed exactly by noting that marginal for p(i) is
        // beta distributed with alpha = a(i) and beta = Sum_j{ a(j) } - a(i).
        // However, this requires us to compute quantiles at every E[p(i)] which
        // would be expensive for a large number of categories. Instead we
        // approximate the probability by using a fixed number of samples from
        // the marginal and computing the probability by taking the mean of these.
        // Finally, note that if E[p(i)] and E[p(j)] are very close we want P(i)
        // and P(j) to be close, but they can in fact be very different if there
        // are many probabilities E[p(k)] which satisfy:
        //   E[p(i)] <= E[p(k)] <= E[p(j)],
        //
        // To avoid this problem we scale all probabilities by (1 + eps), for
        // small eps > 0, when computing (2).
        //
        // In the case that the number of categories has overflowed we derive a
        // sharp lower bound by considering the case that all the additional
        // mass is in one category which is not in the sample set. In this case
        // we have three sets of categories:
        //   U = "Uncounted categories"
        //   L = {i : i is counted and P(i) < P(U)}
        //   M = {i : i is counted and P(i) >= P(U)}
        //
        // where, clearly P(U) is the extra mass. If X denotes the sample set
        // and N the total concentration then for this case:
        //   P(i in XU) >= 1 / N
        //   P(i in XL) >= P(i)
        //   P(i in XM) >= P(i) + P(U)
        //
        // If |X| = 1 then a sharp upper bound is easy to compute:
        //   P(i in XU) <= P(U) + P(L)
        //   P(i in X\U) <= P(i) + P(U)                                    (3)
        //
        // For the case that X contains a mixture of values that are and aren't
        // in U then a sharp upper bound is difficult to compute: it depends on
        // the number of different categories in XU, P(U) and the probabilities
        // P(i in L). In this case we just fall back to using (3) which isn't
        // sharp.

        using TSizeVec = std::vector<std::size_t>;

        tail = maths_t::E_MixedOrNeitherTail;

        TDoubleDoubleSizeTrVec pCategories;
        pCategories.reserve(m_Categories.size());
        double pU = 0.0;
        double pmin = 1.0 / m_TotalConcentration;

        {
            double r = 1.0 / static_cast<double>(m_Concentrations.size());
            for (std::size_t i = 0u; i < m_Concentrations.size(); ++i) {
                double p = m_Concentrations[i] / m_TotalConcentration;
                pCategories.emplace_back(p, p, i);
                pU += r - p;
            }
        }
        std::sort(pCategories.begin(), pCategories.end());
        LOG_TRACE(<< "p = " << core::CContainerPrinter::print(pCategories));

        double pl = 0.0;
        {
            // Get the index of largest probability less than or equal to P(U).
            std::size_t l = pCategories.size();
            if (pU > 0.0) {
                l = std::lower_bound(pCategories.begin(), pCategories.end(),
                                     TDoubleDoubleSizeTr(pU, pU, 0)) -
                    pCategories.begin();
            }

            // Compute probabilities of less likely categories.
            double pCumulative = 0.0;
            for (std::size_t i = 0u, j = 0u; i < pCategories.size(); /**/) {
                // Find the probability equal range [i, j).
                double p = pCategories[i].get<1>();
                pCumulative += p;
                while (++j < pCategories.size() && pCategories[j].get<1>() == p) {
                    pCumulative += p;
                }

                // Update the equal range probabilities [i, j).
                for (/**/; i < j; ++i) {
                    pCategories[i].get<1>() = pCumulative;
                }
            }

            if (l < pCategories.size()) {
                pl = pCategories[l].get<1>();
            }
        }

        std::size_t nSamples = detail::numberPriorSamples(m_TotalConcentration);
        LOG_TRACE(<< "n = " << nSamples);
        if (nSamples > 1) {
            // Extract the indices of the categories we want.
            TSizeVec categoryIndices;
            categoryIndices.reserve(samples.size());
            for (std::size_t i = 0u; i < samples.size(); ++i) {
                std::size_t index = std::lower_bound(m_Categories.begin(),
                                                     m_Categories.end(), samples[i]) -
                                    m_Categories.begin();
                if (index < m_Categories.size() && m_Categories[index] == samples[i]) {
                    categoryIndices.push_back(index);
                }
            }
            std::sort(categoryIndices.begin(), categoryIndices.end());

            for (std::size_t i = 0u; i < pCategories.size(); ++i) {
                // For all categories that we actually want compute the
                // average probability over a set of independent samples
                // from the marginal prior for this category, which by the
                // law of large numbers converges to E[ P(p) ] w.r.t. to
                // marginal for p. The constants a and b are a(i) and
                // Sum_j( a(j) ) - a(i), respectively.

                std::size_t j = pCategories[i].get<2>();
                if (std::binary_search(categoryIndices.begin(), categoryIndices.end(), j)) {
                    TDouble7Vec marginalSamples;
                    double a = m_Concentrations[j];
                    double b = m_TotalConcentration - m_Concentrations[j];
                    detail::generateBetaSamples(a, b, nSamples, marginalSamples);
                    LOG_TRACE(<< "E[p] = " << pCategories[i].get<0>()
                              << ", mean = " << CBasicStatistics::mean(marginalSamples)
                              << ", samples = " << marginalSamples);

                    TMeanAccumulator pAcc;
                    for (std::size_t k = 0u; k < marginalSamples.size(); ++k) {
                        TDoubleDoubleSizeTr x(1.05 * marginalSamples[k], 0.0, 0);
                        ptrdiff_t r = std::min(
                            std::upper_bound(pCategories.begin(), pCategories.end(), x) -
                                pCategories.begin(),
                            static_cast<ptrdiff_t>(pCategories.size()) - 1);

                        double fl = r > 0 ? pCategories[r - 1].get<0>() : 0.0;
                        double fr = pCategories[r].get<0>();
                        double pl_ = r > 0 ? pCategories[r - 1].get<1>() : 0.0;
                        double pr_ = pCategories[r].get<1>();
                        double alpha = std::min(
                            (fr - fl == 0.0) ? 0.0 : (x.get<0>() - fl) / (fr - fl), 1.0);
                        double px = (1.0 - alpha) * pl_ + alpha * pr_;
                        LOG_TRACE(<< "E[p(l)] = " << fl << ", P(l) = " << pl_
                                  << ", E[p(r)] = " << fr << ", P(r) = " << pr_
                                  << ", alpha = " << alpha << ", p = " << px);

                        pAcc.add(px);
                    }
                    pCategories[i].get<1>() = CBasicStatistics::mean(pAcc);
                }
            }
        }

        LOG_TRACE(<< "pCategories = " << core::CContainerPrinter::print(pCategories));
        LOG_TRACE(<< "P(U) = " << pU << ", P(l) = " << pl);

        // We can use radix sort to reorder the probabilities in O(n).
        // To understand the following loop note that on each iteration
        // at least one extra probability is in its correct position
        // so it will necessarily terminate in at most n iterations.
        for (std::size_t i = 0; i < pCategories.size(); /**/) {
            if (i == pCategories[i].get<2>()) {
                ++i;
            } else {
                std::swap(pCategories[i], pCategories[pCategories[i].get<2>()]);
            }
        }

        LOG_TRACE(<< "pCategories = " << core::CContainerPrinter::print(pCategories));

        if (samples.size() == 1) {
            // No special aggregation is required if there is a single sample.
            std::size_t index = std::lower_bound(m_Categories.begin(),
                                                 m_Categories.end(), samples[0]) -
                                m_Categories.begin();

            if (index < m_Categories.size() && m_Categories[index] == samples[0]) {
                double p = pCategories[index].get<1>();
                lowerBound = p + (p >= pU ? pU : 0.0);
                upperBound = p + pU;
            } else {
                lowerBound = pmin;
                upperBound = std::max(pU + pl, pmin);
            }

            return true;
        }

        TDoubleDoubleMap categoryCounts;

        // Count the occurrences of each category in the sample set.
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double x = samples[i];
            double n = maths_t::count(weights[i]);
            categoryCounts[x] += n;
        }

        LOG_TRACE(<< "categoryCounts = " << core::CContainerPrinter::print(categoryCounts));

        CJointProbabilityOfLessLikelySamples jointLowerBound;
        CJointProbabilityOfLessLikelySamples jointUpperBound;

        for (TDoubleDoubleMapCItr countItr = categoryCounts.begin();
             countItr != categoryCounts.end(); ++countItr) {
            double category = countItr->first;
            double count = countItr->second;
            LOG_TRACE(<< "category = " << category << ", count = " << count);

            std::size_t index = std::lower_bound(m_Categories.begin(),
                                                 m_Categories.end(), category) -
                                m_Categories.begin();

            double p = pCategories[index].get<1>();
            if (index < m_Categories.size() && m_Categories[index] == category) {
                jointLowerBound.add(p + (p >= pU ? pU : 0.0), count);
                jointUpperBound.add(p + pU, count);
            } else {
                jointLowerBound.add(pmin, count);
                jointUpperBound.add(std::max(pU + pl, pmin), count);
            }
        }

        if (!jointLowerBound.calculate(lowerBound) || !jointUpperBound.calculate(upperBound)) {
            LOG_ERROR(<< "Unable to compute probability for "
                      << core::CContainerPrinter::print(samples) << ": "
                      << jointLowerBound << ", " << jointUpperBound);
            return false;
        }

        LOG_TRACE(<< "probability = [" << lowerBound << ", " << upperBound << "]");
    } break;

    case maths_t::E_OneSidedAbove: {
        // See minusLogJointCdf for a discussion of the calculation
        // of the marginal c.d.f. for a single sample.

        CJointProbabilityOfLessLikelySamples jointLowerBound;
        CJointProbabilityOfLessLikelySamples jointUpperBound;

        detail::CCdfComplement cdfComplement(m_Categories, m_Concentrations, m_TotalConcentration);
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double x = samples[i];
            double n = maths_t::count(weights[i]);
            double sampleLowerBound, sampleUpperBound;
            cdfComplement(x, sampleLowerBound, sampleUpperBound);
            jointLowerBound.add(sampleLowerBound, n);
            jointUpperBound.add(sampleUpperBound, n);
        }

        if (!jointLowerBound.calculate(lowerBound) || !jointUpperBound.calculate(upperBound)) {
            LOG_ERROR(<< "Unable to compute probability for "
                      << core::CContainerPrinter::print(samples) << ": "
                      << jointLowerBound << ", " << jointUpperBound);
            return false;
        }
        tail = maths_t::E_RightTail;
    } break;
    }

    return true;
}

bool CMultinomialConjugate::isNonInformative() const {
    return m_TotalConcentration <= NON_INFORMATIVE_CONCENTRATION;
}

void CMultinomialConjugate::print(const std::string& indent, std::string& result) const {
    result += core_t::LINE_ENDING + indent + "multinomial " +
              (this->isNonInformative()
                   ? std::string("non-informative")
                   : std::string("categories ") +
                         core::CContainerPrinter::print(m_Categories) + " concentrations " +
                         core::CContainerPrinter::print(m_Concentrations));
}

std::string CMultinomialConjugate::printMarginalLikelihoodFunction(double /*weight*/) const {
    // This is infinite at the categories and zero elsewhere.
    return "Not supported";
}

std::string CMultinomialConjugate::printJointDensityFunction() const {
    static const double RANGE = 0.999;
    static const unsigned int POINTS = 51;

    std::ostringstream result;

    result << "hold off" << core_t::LINE_ENDING;

    // We show the marginals for each category plotted on the same axes.
    for (std::size_t i = 0u; i < m_Concentrations.size(); ++i) {
        double a = m_Concentrations[i];
        double b = m_TotalConcentration - m_Concentrations[i];
        boost::math::beta_distribution<> beta(a, b);

        double xStart = boost::math::quantile(beta, (1.0 - RANGE) / 2.0);
        double xEnd = boost::math::quantile(beta, (1.0 + RANGE) / 2.0);
        double xIncrement = (xEnd - xStart) / (POINTS - 1.0);
        double x = xStart;

        result << "x = [";
        for (unsigned int j = 0u; j < POINTS; ++j, x += xIncrement) {
            result << x << " ";
        }
        result << "];" << core_t::LINE_ENDING;

        result << "pdf = [";
        x = xStart;
        for (unsigned int j = 0u; j < POINTS; ++j, x += xIncrement) {
            result << CTools::safePdf(beta, x) << " ";
        }
        result << "];" << core_t::LINE_ENDING;
        result << "plot(x, pdf);" << core_t::LINE_ENDING;
        result << "hold on;" << core_t::LINE_ENDING;
    }

    result << "hold off;" << core_t::LINE_ENDING;

    return result.str();
}

uint64_t CMultinomialConjugate::checksum(uint64_t seed) const {
    seed = this->CPrior::checksum(seed);
    seed = CChecksum::calculate(seed, m_NumberAvailableCategories);
    seed = CChecksum::calculate(seed, m_Categories);
    seed = CChecksum::calculate(seed, m_Concentrations);
    return CChecksum::calculate(seed, m_TotalConcentration);
}

void CMultinomialConjugate::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CMultinomialConjugate");
    core::CMemoryDebug::dynamicSize("m_Categories", m_Categories, mem);
    core::CMemoryDebug::dynamicSize("m_Concentrations", m_Concentrations, mem);
}

std::size_t CMultinomialConjugate::memoryUsage() const {
    std::size_t mem = core::CMemory::dynamicSize(m_Categories);
    mem += core::CMemory::dynamicSize(m_Concentrations);
    return mem;
}

std::size_t CMultinomialConjugate::staticSize() const {
    return sizeof(*this);
}

void CMultinomialConjugate::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(DECAY_RATE_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(NUMBER_AVAILABLE_CATEGORIES_TAG, m_NumberAvailableCategories);
    inserter.insertValue(CATEGORY_TAG, core::CPersistUtils::toString(m_Categories));
    inserter.insertValue(CONCENTRATION_TAG, core::CPersistUtils::toString(m_Concentrations));
    inserter.insertValue(TOTAL_CONCENTRATION_TAG, m_TotalConcentration,
                         core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(NUMBER_SAMPLES_TAG, this->numberSamples(),
                         core::CIEEE754::E_SinglePrecision);
}

void CMultinomialConjugate::removeCategories(TDoubleVec categoriesToRemove) {
    if (categoriesToRemove.empty()) {
        return;
    }

    std::size_t end = 0u;

    std::sort(categoriesToRemove.begin(), categoriesToRemove.end());
    categoriesToRemove.push_back(boost::numeric::bounds<double>::highest());
    for (std::size_t i = 0u, j = 0u; i < m_Categories.size(); /**/) {
        if (m_Categories[i] < categoriesToRemove[j]) {
            std::swap(m_Categories[end], m_Categories[i]);
            std::swap(m_Concentrations[end], m_Concentrations[i]);
            ++i;
            ++end;
        } else {
            m_Categories[i] > categoriesToRemove[j] ? ++j : ++i;
        }
    }
    m_NumberAvailableCategories += static_cast<int>(m_Categories.size() - end);

    m_Categories.erase(m_Categories.begin() + end, m_Categories.end());
    m_Concentrations.erase(m_Concentrations.begin() + end, m_Concentrations.end());
    m_TotalConcentration =
        std::accumulate(m_Concentrations.begin(), m_Concentrations.end(), 0.0);
    LOG_TRACE(<< "categories     = " << core::CContainerPrinter::print(m_Categories));
    LOG_TRACE(<< "concentrations = " << core::CContainerPrinter::print(m_Concentrations));

    this->numberSamples(m_TotalConcentration);
}

bool CMultinomialConjugate::index(double category, std::size_t& result) const {
    result = std::numeric_limits<std::size_t>::max();

    TDoubleVecCItr categoryItr =
        std::lower_bound(m_Categories.begin(), m_Categories.end(), category);
    if (categoryItr == m_Categories.end() || *categoryItr != category) {
        return false;
    }

    result = categoryItr - m_Categories.begin();
    return true;
}

const CMultinomialConjugate::TDoubleVec& CMultinomialConjugate::categories() const {
    return m_Categories;
}

const CMultinomialConjugate::TDoubleVec& CMultinomialConjugate::concentrations() const {
    return m_Concentrations;
}

bool CMultinomialConjugate::concentration(double category, double& result) const {
    result = 0.0;

    std::size_t i;
    if (!this->index(category, i)) {
        return false;
    }

    result = m_Concentrations[i];
    return true;
}

double CMultinomialConjugate::totalConcentration() const {
    return m_TotalConcentration;
}

bool CMultinomialConjugate::probability(double category, double& result) const {
    result = 0.0;

    double concentration;
    if (!this->concentration(category, concentration)) {
        return false;
    }

    result = concentration / m_TotalConcentration;
    return true;
}

CMultinomialConjugate::TDoubleVec CMultinomialConjugate::probabilities() const {
    TDoubleVec result(m_Concentrations);
    for (std::size_t i = 0u; i < result.size(); ++i) {
        result[i] /= m_TotalConcentration;
    }
    return result;
}

void CMultinomialConjugate::probabilitiesOfLessLikelyCategories(maths_t::EProbabilityCalculation calculation,
                                                                TDoubleVec& lowerBounds,
                                                                TDoubleVec& upperBounds) const {
    // See probabilityOfLessLikelySamples for an explanation of these
    // calculations.

    lowerBounds.clear();
    upperBounds.clear();

    switch (calculation) {
    case maths_t::E_OneSidedBelow: {
        detail::CCdf cdf(m_Categories, m_Concentrations, m_TotalConcentration);
        cdf.dump(lowerBounds, upperBounds);
    } break;

    case maths_t::E_TwoSided: {
        TDoubleDoubleSizeTrVec pCategories;
        pCategories.reserve(m_Categories.size());
        double pU = 0.0;

        {
            double r = 1.0 / static_cast<double>(m_Concentrations.size());
            for (std::size_t i = 0u; i < m_Concentrations.size(); ++i) {
                double p = m_Concentrations[i] / m_TotalConcentration;
                pCategories.emplace_back(p, p, i);
                pU += r - p;
            }
        }
        std::sort(pCategories.begin(), pCategories.end());
        LOG_TRACE(<< "pCategories = " << core::CContainerPrinter::print(pCategories));

        // Get the index of largest probability less than or equal to P(U).
        double pl = 0.0;
        {
            std::size_t l = pCategories.size();
            if (pU > 0.0) {
                l = std::lower_bound(pCategories.begin(), pCategories.end(),
                                     TDoubleDoubleSizeTr(pU, pU, 0)) -
                    pCategories.begin();
            }

            // Compute probabilities of less likely categories.
            double pCumulative = 0.0;
            for (std::size_t i = 0u, j = 0u; i < pCategories.size(); /**/) {
                // Find the probability equal range [i, j).
                double p = pCategories[i].get<1>();
                pCumulative += p;
                while (++j < pCategories.size() && pCategories[j].get<1>() == p) {
                    pCumulative += p;
                }

                // Update the equal range probabilities [i, j).
                for (/**/; i < j; ++i) {
                    pCategories[i].get<1>() = pCumulative;
                }
            }

            if (l < pCategories.size()) {
                pl = pCategories[l].get<1>();
            }
        }

        LOG_TRACE(<< "pCategories = " << core::CContainerPrinter::print(pCategories));
        LOG_TRACE(<< "P(U) = " << pU << ", P(l) = " << pl);

        lowerBounds.resize(pCategories.size(), 0.0);
        upperBounds.resize(pCategories.size(), 0.0);

        double p = 0.0;
        double pLast = -1.0;
        std::size_t n = detail::numberPriorSamples(m_TotalConcentration);
        LOG_TRACE(<< "n = " << n);
        for (std::size_t i = 0u; i < pCategories.size(); ++i) {
            std::size_t j = pCategories[i].get<2>();

            // We compute the average probability over a set of
            // independent samples from the marginal prior for this
            // category, which by the law of large numbers converges
            // to E[ P(p) ] w.r.t. to marginal for p. The constants
            // a and b are a(i) and Sum_j( a(j) ) - a(i), respectively.
            // See confidenceIntervalProbabilities for a discussion.

            if (pCategories[i].get<0>() != pLast) {
                TDouble7Vec samples;
                double a = m_Concentrations[j];
                double b = m_TotalConcentration - m_Concentrations[j];
                detail::generateBetaSamples(a, b, n, samples);
                LOG_TRACE(<< "E[p] = " << pCategories[i].get<0>()
                          << ", mean = " << CBasicStatistics::mean(samples)
                          << ", samples = " << core::CContainerPrinter::print(samples));

                TMeanAccumulator pAcc;
                for (std::size_t k = 0u; k < samples.size(); ++k) {
                    TDoubleDoubleSizeTr x(1.05 * samples[k], 0.0, 0);
                    ptrdiff_t r = std::min(
                        std::upper_bound(pCategories.begin(), pCategories.end(), x) -
                            pCategories.begin(),
                        static_cast<ptrdiff_t>(pCategories.size()) - 1);

                    double fl = r > 0 ? pCategories[r - 1].get<0>() : 0.0;
                    double fr = pCategories[r].get<0>();
                    double pl_ = r > 0 ? pCategories[r - 1].get<1>() : 0.0;
                    double pr_ = pCategories[r].get<1>();
                    double alpha = std::min(
                        (fr - fl == 0.0) ? 0.0 : (x.get<0>() - fl) / (fr - fl), 1.0);
                    double px = (1.0 - alpha) * pl_ + alpha * pr_;
                    LOG_TRACE(<< "E[p(l)] = " << fl << ", P(l) = " << pl_
                              << ", E[p(r)] = " << fr << ", P(r) = " << pr_
                              << ", alpha = " << alpha << ", p = " << px);

                    pAcc.add(px);
                }
                p = CBasicStatistics::mean(pAcc);
                pLast = pCategories[i].get<0>();
            }

            LOG_TRACE(<< "p = " << p);
            lowerBounds[j] = p + (p >= pU ? pU : 0.0);
            upperBounds[j] = p + pU;
        }
    } break;

    case maths_t::E_OneSidedAbove: {
        detail::CCdfComplement cdfComplement(m_Categories, m_Concentrations, m_TotalConcentration);
        cdfComplement.dump(lowerBounds, upperBounds);
    } break;
    }
}

CMultinomialConjugate::TDoubleDoublePrVec
CMultinomialConjugate::confidenceIntervalProbabilities(double percentage) const {
    if (this->isNonInformative()) {
        return TDoubleDoublePrVec(m_Concentrations.size(), {0.0, 1.0});
    }

    // The marginal distribution over each probability is beta.
    // In particular,
    //   f(p(i)) = p(i)^(a(i)-1)
    //             * Intergal_{R(p(i))}{ 1/B(a) * Product_{j!=i}{ p^(a(j)-1) }dp(j) }
    //
    // where,
    //   R(p(i)) is the simplex Sum_{j!=i}{ p(j) } = 1 - p(i).
    //   a = {a(i)} is the concentration vector.
    //
    // Change variables to p(j)' = p(j) / (1 - p(i)) and note that R(p(i))
    // is now equivalent to the simplex Sum_{j!=i}{ p(j)' } = 1 and the
    // integral represents the normalization constant of a Dirichlet
    // distribution over the reduced set {p(i) : i!=j}. After simplification,
    // this gives that:
    //  f(p(i)) = Gamma(a0)/Gamma(a0 - a(i))/Gamma(a(i))
    //              * (1 - p(i))^(a0-a(i)-1) * p(i)^(a(i)-1)
    //
    // Here, a0 = Sum_i{ a(i) }. We recognize this as the p.d.f. of a Beta
    // R.V. with:
    //   alpha = a(i)
    //   beta = a0 - a(i)

    TDoubleDoublePrVec result;
    result.reserve(m_Concentrations.size());

    percentage /= 100.0;
    double lowerPercentile = 0.5 * (1.0 - percentage);
    double upperPercentile = 0.5 * (1.0 + percentage);

    for (std::size_t i = 0u; i < m_Concentrations.size(); ++i) {
        double a = m_Concentrations[i];
        double b = m_TotalConcentration - m_Concentrations[i];
        boost::math::beta_distribution<> beta(a, b);
        TDoubleDoublePr percentiles(boost::math::quantile(beta, lowerPercentile),
                                    boost::math::quantile(beta, upperPercentile));
        result.push_back(percentiles);
    }

    return result;
}

bool CMultinomialConjugate::equalTolerance(const CMultinomialConjugate& rhs,
                                           const TEqualWithTolerance& equal) const {
    LOG_DEBUG(<< m_NumberAvailableCategories << " " << rhs.m_NumberAvailableCategories);
    LOG_DEBUG(<< core::CContainerPrinter::print(m_Categories) << " "
              << core::CContainerPrinter::print(rhs.m_Categories));
    LOG_DEBUG(<< core::CContainerPrinter::print(m_Concentrations) << " "
              << core::CContainerPrinter::print(rhs.m_Concentrations));
    LOG_DEBUG(<< m_TotalConcentration << " " << rhs.m_TotalConcentration);

    return m_NumberAvailableCategories == rhs.m_NumberAvailableCategories &&
           m_Categories == rhs.m_Categories &&
           std::equal(m_Concentrations.begin(), m_Concentrations.end(),
                      rhs.m_Concentrations.begin(), equal) &&
           equal(m_TotalConcentration, rhs.m_TotalConcentration);
}

void CMultinomialConjugate::shrink() {
    // Note that the vectors are only ever shrunk once.

    using std::swap;

    if (m_Categories.capacity() > m_Categories.size() + m_NumberAvailableCategories) {
        TDoubleVec categories(m_Categories);
        swap(categories, m_Categories);
        m_Categories.reserve(m_Categories.size() + m_NumberAvailableCategories);
    }

    if (m_Concentrations.capacity() > m_Concentrations.size() + m_NumberAvailableCategories) {
        TDoubleVec concentrationParameters(m_Concentrations);
        swap(concentrationParameters, m_Concentrations);
        m_Concentrations.reserve(m_Concentrations.size() + m_NumberAvailableCategories);
    }
}

const double CMultinomialConjugate::NON_INFORMATIVE_CONCENTRATION = 0.0;
}
}
