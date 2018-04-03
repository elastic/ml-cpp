/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CCooccurrences.h>

#include <core/CAllocationStrategy.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>

#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>

#include <boost/math/distributions/chi_squared.hpp>

#include <string>

namespace ml
{
namespace maths
{

namespace
{

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrUSet = boost::unordered_set<TSizeSizePr>;
using TPoint = CVector<double>;
using TPointVec = std::vector<TPoint>;
using TPackedBitVectorVec = std::vector<CPackedBitVector>;

//! \brief Counts the (co-)occurrences of two variables.
struct SCooccurrence
{
    SCooccurrence(void) :
        s_Nxy(0.0), s_Nx(0.0), s_Ny(0.0), s_X(0), s_Y(0)
    {}
    SCooccurrence(double nxy, double nx, double ny, std::size_t x, std::size_t y) :
        s_Nxy(nxy), s_Nx(nx), s_Ny(ny), s_X(x), s_Y(y)
    {}

    bool operator<(const SCooccurrence &rhs) const
    {
        return  s_Nxy * static_cast<double>(rhs.s_X) * static_cast<double>(rhs.s_Y)
              < rhs.s_Nxy * s_Nx * s_Ny;
    }

    double s_Nxy, s_Nx, s_Ny;
    std::size_t s_X, s_Y;
};

using TMostSignificant = CBasicStatistics::COrderStatisticsHeap<SCooccurrence>;

//! Compute \p x * \p x.
double pow2(double x)
{
    return x * x;
}

//! Generate a random projection in the positive orthant.
//!
//! \param[in] dimension The dimension.
//! \param[out] result Filled in with the projection.
void generateProjection(std::size_t dimension, CPackedBitVector &result)
{
    if (dimension == 0)
    {
        return;
    }

    // We are interested in the case that events occur close in time. Rather than
    // generating components of the projection vector uniformly at random we use
    // a Markov process with low transition probability but equal equilibrium
    // probabilities. Any transition matrix of the form
    //
    //   [ 1 - p    p   ]
    //   [   p    1 - p ]
    //
    // works for this purpose.

    static const double TRANSITION_PROBABILITY = 0.1;

    TDoubleVec uniform01;
    CSampling::uniformSample(0.0, 1.0, dimension, uniform01);

    bool last = (uniform01[0] < 0.5);
    result.extend(last);
    for (std::size_t i = 1; i < uniform01.size(); ++i)
    {
        if (uniform01[i] < TRANSITION_PROBABILITY)
        {
            last = !last;
        }
        result.extend(last);
    }
}

//! Generate p random projections of the event indicators.
//!
//! \param[in] indicators The indicator vectors for the events.
//! \param[in] lengths The Euclidean lengths of the indicator vectors.
//! \param[in] mask A mask of events to consider.
//! \param[in] result Filled in with the p projections of indicator vectors.
void generateProjections(const TPackedBitVectorVec &indicators,
                         const TDoubleVec &lengths,
                         const TSizeVec &mask,
                         TDoubleVecVec &result)
{
    std::size_t dimension = indicators[0].dimension();
    for (std::size_t i = 0u; i < result.size(); ++i)
    {
        CPackedBitVector projection;
        generateProjection(dimension, projection);
        double length = projection.euclidean();
        for (std::size_t j = 0u; j < mask.size(); ++j)
        {
            std::size_t k = mask[j];
            result[i][j] = indicators[k].inner(projection) / lengths[k] / length;
        }
    }
}

//! Test to see if we should include the co-occurrence of \p x and \p y.
//!
//! \param[in] indicators The indicator vectors for the events.
//! \param[in] lengths The Euclidean lengths of the indicator vectors.
//! \param[in] x An event index of interest.
//! \param[in] y Another event index of interest.
//! \param[out] added If (\p x, \p y) is added to result the pair is added
//! to this set.
//! \param[out] mostSignificant Maybe updated to include the co-occurrence
//! of \p x and \p y.
void testCooccurrence(const TPackedBitVectorVec &indicators,
                      const TDoubleVec &lengths,
                      std::size_t x,
                      std::size_t y,
                      TSizeSizePrUSet &added,
                      TMostSignificant &mostSignificant)
{
    if (x > y)
    {
        std::swap(x, y);
    }
    if (added.count(std::make_pair(x, y)) == 0)
    {
        double nxy = indicators[x].inner(indicators[y]);
        std::size_t count = mostSignificant.count();
        std::size_t u = mostSignificant.biggest().s_X;
        std::size_t v = mostSignificant.biggest().s_Y;
        if (mostSignificant.add(SCooccurrence(nxy, lengths[x], lengths[y], x, y)))
        {
            if (mostSignificant.count() == count)
            {
                added.erase(std::make_pair(u, v));
            }
            added.insert(std::make_pair(x, y));
        }
    }
}

//! Create a reasonable initialization of the most significantly co-occurring
//! events.
//!
//! \param[in] indicators The indicator vectors for the events.
//! \param[in] lengths The Euclidean lengths of the indicator vectors.
//! \param[in] mask A mask of events to consider.
//! \param[in] projected The p projections of indicator vectors.
//! \param[out] added Filled in with ordered pairs of the seed most
//! significantly co-occurring event indices.
//! \param[out] mostSignificant Filled in with the most seed significantly
//! co-occurring event pairs.
void seed(const TPackedBitVectorVec &indicators,
          const TDoubleVec &lengths,
          TSizeVec mask,
          const TDoubleVecVec &projected,
          TSizeSizePrUSet &added,
          TMostSignificant &mostSignificant)
{
    std::size_t n = mask.size();
    TDoubleVec theta(n, 0.0);
    for (std::size_t i = 0u; i < n; ++i)
    {
        for (std::size_t j = 0u; j < projected.size(); ++j)
        {
            theta[i] += pow2(projected[j][i]);
        }
        theta[i] = ::acos(::sqrt(theta[i]));
    }
    COrderings::simultaneousSort(theta, mask);
    for (std::size_t i = 1u; i < n; ++i)
    {
        testCooccurrence(indicators, lengths, mask[i-1], mask[i], added, mostSignificant);
    }
}

//! Generate a filter based on the bounds on the significance.
//!
//! \param[in] mask A mask of events to consider.
//! \param[in] theta The corresponding event angles.
//! \param[in] i The index into \p theta for which to compute the filter.
//! \param[in] bound The largest angularly separated event to include.
//! \param[out] result The indices of the events in the filter.
void computeFilter(const TSizeVec &mask,
                   const TDoubleVec &theta,
                   std::size_t i,
                   double bound,
                   TSizeVec &result)
{
    result.clear();
    ptrdiff_t start = std::lower_bound(theta.begin(), theta.end(),
                                       theta[i] - bound) - theta.begin();
    ptrdiff_t end   = std::upper_bound(theta.begin(), theta.end(),
                                       theta[i] + bound) - theta.begin();
    result.reserve(end - start);
    result.insert(result.end(), mask.begin() + start, mask.begin() + i);
    result.insert(result.end(), mask.begin() + i + 1, mask.begin() + end);
    std::sort(result.begin(), result.end());
}

//! Apply \p filter to \p result (set intersection).
void applyFilter(const TSizeVec &filter,
                 TSizeVec &placeholder,
                 TSizeVec &result)
{
    placeholder.clear();
    std::set_intersection(result.begin(), result.end(),
                          filter.begin(), filter.end(), std::back_inserter(placeholder));
    result.swap(placeholder);
}

//! A branch and bound search for the most significant co-occurring events.
//!
//! \param[in] indicators The indicator vectors for the events.
//! \param[in] lengths The Euclidean lengths of the indicator vectors.
//! \param[in] mask A mask of events to consider.
//! \param[in] p The number of projections to use.
//! \param[in] mostSignificant Filled in with the most significant co-occurring
//! events.
void searchForMostSignificantCooccurrences(const TPackedBitVectorVec &indicators,
                                           const TDoubleVec &lengths,
                                           const TSizeVec &mask,
                                           std::size_t p,
                                           TMostSignificant &mostSignificant)
{
    // This uses the fact that after projecting the values using
    // f : x -> (||p^t x|| / ||x||, ||(1 - p p^t) x|| / || x ||) the
    // Euclidean separation ||f(x) - f(y)||^2 = 2 ( 1 - x^t y / ( ||x|| ||y|| ) )
    // and the significance is a monotonic increasing function of
    // x^t y / ( ||x|| ||y|| ). Since p is normalized, this projection
    // is into the unit circle so we can express all distances in terms
    // of the angle around the circle.

    std::size_t n = mask.size();

    TDoubleVecVec thetas(p, TDoubleVec(n));
    generateProjections(indicators, lengths, mask, thetas);

    TSizeSizePrUSet added;
    seed(indicators, lengths, mask, thetas, added, mostSignificant);

    TSizeVecVec masks(p, mask);
    for (std::size_t i = 0u; i < p; ++i)
    {
        for (std::size_t j = 0u; j < n; ++j)
        {
            thetas[i][j] = ::acos(thetas[i][j]);
        }
        COrderings::simultaneousSort(thetas[i], masks[i]);
    }

    TSizeVec candidates;
    TSizeVec filter;
    TSizeVec placeholder;

    for (std::size_t i = 0u; i < n; ++i)
    {
        double lambda =   mostSignificant.biggest().s_Nxy
                       / (mostSignificant.biggest().s_Nx * mostSignificant.biggest().s_Ny);

        double bound = 2.0 * ::asin(1.0 - lambda);

        computeFilter(masks[0], thetas[0], i, bound, candidates);
        for (std::size_t j = 1u; !candidates.empty() && j < p; ++j)
        {
            computeFilter(masks[j], thetas[j], i, bound, filter);
            applyFilter(filter, placeholder, candidates);
        }

        for (std::size_t j = 0u; j < candidates.size(); ++j)
        {
            testCooccurrence(indicators, lengths, mask[i], candidates[j], added, mostSignificant);
        }
    }
}

//! Compute the significance of the coincidence of x and y as the approximate
//! probability they are independent.
//!
//! \param[in] nxy The count of co-occurrences of x and y.
//! \param[in] nx The count of occurrences of x.
//! \param[in] ny The count of occurrences of y.
//! \param[in] n The total sample size.
double significance(double nxy, double nx, double ny, double n)
{
    // Here we test a nested composite hypothesis.
    //
    // Our null hypothesis H0 is that the probability of seeing x is independent
    // of y. Our alternative hypothesis H1 is that the conditional probabilities
    // are different for seeing y depending on whether or not we have seen x.
    //
    // We will use the generalized likelihood ratio test and make use of Wilk's
    // theorem to calculate the asymptotic size of the test. In order to compute
    // the likelihood we note that for each of the n samples the variable whose
    // categories are { xy, x~y, ~xy, ~x~y } is categorical with probabilities
    // { px * py, px * (1-py), (x-px) * py, (1-px) * (1-py) } under the null
    // hypothesis and { px * py|x, px * (1-py|x), (1-px) * py|~x, (1-px) * (1-py|~x) }.
    // Note it is clear that the hypotheses are composite since the parameter space
    // of H0 is the plane subset py|x = py|~x of the cube of H1. By a standard
    // result, the likelihood of a collection of counts for the different categories
    // of n samples of a categorical variable is multinomial. Using N = { nxy, nx~y,
    // n~xy, n~x~y } to denote these counts we have under the null hypothesis
    //   L(N | H0) =  n! / (nxy! * nx~y! * n~xy! * n~x~y!)
    //              * (px * py)^nxy * (px * (1-py))^nx~y
    //              * ((1-px) * py)^n~xy * ((1-px) * (1-py))^n~x~y
    //
    // Under the alternative hypothesis
    //   L(N | H1) =  n! / (nxy! * nx~y! * n~xy! * n~x~y!)
    //              * (px * py|x)^nxy * (px * (1-py|x))^nx~y
    //              * ((1-px) * py|~x)^n~xy * ((1-px) * (1-py|~x))^n~x~y
    //
    // The generalized likelihood ratio is
    //   R(N) = sup_{px,py}{ L(N | H0) } / sup_{px,py|x,py|~x}{ L(N | H1) }
    //
    // The maximum likelihood estimate for a multinomial distribution's probabilities
    // are just the respective frequencies. So noting that the normalization constants
    // cancel we have that
    //   log(R(N)) =   nxy * log(nx/n * ny/n) + nx~y * log(nx/n * (1-ny/n))
    //              + n~xy * log((1-nx/n) * ny/n) + n~x~y * log((1-nx/n) * (1-ny/n))
    //              -  nxy * log(nx/n * nxy/nx) - nx~y * log(nx/n * (1-nxy/nx))
    //              - n~xy * log((1-nx/n) * (ny-nxy)/(n-nx)) - n~x~y * log((1-nx/n) * (1-(ny-nxy)/(n-nx)))
    //
    // It is convenient to express this in slightly different terms. In particular,
    // we are interested in the case that nxy is significantly bigger than expected
    // by independence. In this case we have that nxy is expected to be nx*ny / n.
    // So define g = nxy * n / (nx*ny). We will be interested in the case that g > 1.
    // We also have that n~xy = ny - nxy, nx~y = nx - nxy and n~x~y = n - nx - ny + nxy.
    // Finally, define p_x as nx/n and p_y as ny/n. After some algebra we have
    //   log(R(N)) = n * (                   -g*p_x*p_y * log(g)
    //                    +             p_x*(1 - g*p_y) * log((1 - p_y)/(1 - g*p_y))
    //                    +             p_y*(1 - g*p_x) * log((1 - p_x)/(1 - g*p_x))
    //                    + (1 - p_x - p_y - g*p_x*p_y) * log((1 - p_x)*(1 - p_y) / (1 - p_x - p_y + g*p_x*p_y)))
    //
    // and, by Wilk's theorem, we have asymptotically
    //   -2*log(R(N)) ~ chi_1^2
    //
    // which gives us the size of the test.

    if (nx == 0.0 || ny == 0.0)
    {
        return 1.0;
    }

    double g = (nxy * n) / (nx * ny);

    if (g > 1.0)
    {
        double px = nx / n;
        double py = ny / n;

        double lambda = n * (  -g * px * py * ::log(g)
                             + px * (1.0 - g * py) * ::log((1.0 - py) / (1.0 - g * py))
                             + py * (1.0 - g * px) * ::log((1.0 - px) / (1.0 - g * px))
                             + (1.0 - px - py + g*px*py) * ::log((1.0 - px) * (1.0 - py) / (1.0 - px - py + g*px*py)));

        boost::math::chi_squared_distribution<> chi(1.0);

        return CTools::safeCdfComplement(chi, -2.0 * lambda);
    }

    return 1.0;
}

std::string LENGTH_TAG("a");
std::string OFFSET_TAG("b");
std::string CURRENT_INDICATOR_TAG("c");
std::string INDICATOR_TAG("d");

}

CCooccurrences::CCooccurrences(std::size_t maximumLength, std::size_t indicatorWidth) :
        m_MaximumLength(maximumLength),
        m_Length(0),
        m_IndicatorWidth(indicatorWidth),
        m_Offset(0)
{
}

bool CCooccurrences::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name = traverser.name();
        if (   name == LENGTH_TAG
            && core::CStringUtils::stringToType(traverser.value(), m_Length) == false)
        {
            LOG_ERROR("Invalid length in " << traverser.value());
            return false;
        }
        if (   name == OFFSET_TAG
            && core::CStringUtils::stringToType(traverser.value(), m_Offset) == false)
        {
            LOG_ERROR("Invalid offset in " << traverser.value());
            return false;
        }
        if (core::CPersistUtils::restore(CURRENT_INDICATOR_TAG, m_CurrentIndicators, traverser) == false)
        {
            LOG_ERROR("Invalid indicators in " << traverser.value());
            return false;
        }
        if (core::CPersistUtils::restore(INDICATOR_TAG, m_Indicators, traverser) == false)
        {
            LOG_ERROR("Invalid indicators in " << traverser.value());
            return false;
        }
    }
    while (traverser.next());
    return true;
}

void CCooccurrences::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(LENGTH_TAG, m_Length);
    inserter.insertValue(OFFSET_TAG, m_Offset);
    core::CPersistUtils::persist(CURRENT_INDICATOR_TAG, m_CurrentIndicators, inserter);
    core::CPersistUtils::persist(INDICATOR_TAG, m_Indicators, inserter);
}

void CCooccurrences::topNBySignificance(std::size_t X,
                                        std::size_t /*n*/,
                                        TSizeSizePrVec &/*top*/,
                                        TDoubleVec &/*significances*/) const
{
    if (X >= m_Indicators.size())
    {
        LOG_ERROR("Unexpected event " << X);
        return;
    }

    // TODO
}

void CCooccurrences::topNBySignificance(std::size_t n,
                                        TSizeSizePrVec &top,
                                        TDoubleVec &significances) const
{
    top.clear();
    significances.clear();

    std::size_t N = m_Indicators.size();

    if (N == 0)
    {
        return;
    }

    std::size_t dimension = m_Indicators[0].dimension();

    TDoubleVec lengths(N);
    TSizeVec mask;
    mask.reserve(N);

    for (std::size_t i = 0u; i < N; ++i)
    {
        lengths[i] = m_Indicators[i].euclidean();
        if (lengths[i] > 0.0)
        {
            mask.push_back(i);
        }
    }

    std::size_t p = static_cast<std::size_t>(std::max(::sqrt(static_cast<double>(dimension)), 1.0) + 0.5);

    TMostSignificant mostSignificant(n);
    searchForMostSignificantCooccurrences(m_Indicators, lengths, mask, p, mostSignificant);

    mostSignificant.sort();

    top.reserve(mostSignificant.count());
    significances.reserve(mostSignificant.count());
    for (std::size_t i = 0u; i < mostSignificant.count(); ++i)
    {
        const SCooccurrence &co = mostSignificant[i];
        double nxy = static_cast<double>(co.s_Nxy);
        double nx  = static_cast<double>(co.s_Nx);
        double ny  = static_cast<double>(co.s_Ny);
        top.emplace_back(co.s_X, co.s_Y);
        significances.push_back(significance(nxy, nx, ny, static_cast<double>(dimension)));
    }
}

void CCooccurrences::addEventStreams(std::size_t n)
{
    if (n > m_Indicators.size())
    {
        core::CAllocationStrategy::resize(m_Indicators, n, CPackedBitVector(m_Length, false));
    }
}

void CCooccurrences::removeEventStreams(const TSizeVec &remove)
{
    for (std::size_t i = 0u; i < remove.size(); ++i)
    {
        std::size_t X = remove[i];
        if (X < m_Indicators.size())
        {
            m_Indicators[X] = CPackedBitVector();
        }
    }
}

void CCooccurrences::recycleEventStreams(const TSizeVec &recycle)
{
    for (std::size_t i = 0u; i < recycle.size(); ++i)
    {
        std::size_t X = recycle[i];
        if (X < m_Indicators.size())
        {
            m_Indicators[X] = CPackedBitVector(m_Length, false);
        }
    }
}

void CCooccurrences::add(std::size_t X)
{
    if (X >= m_Indicators.size())
    {
        LOG_ERROR("Unexpected event " << X);
        return;
    }
    m_CurrentIndicators.insert(X);
}

void CCooccurrences::capture(void)
{
    if (++m_Offset < m_IndicatorWidth)
    {
        return;
    }

    m_Offset = 0;
    m_Length = std::min(m_Length + 1, m_MaximumLength);

    for (std::size_t X = 0u; X < m_Indicators.size(); ++X)
    {
        CPackedBitVector &indicator = m_Indicators[X];
        indicator.extend(m_CurrentIndicators.count(X) > 0);
        while (indicator.dimension() > m_MaximumLength)
        {
            indicator.contract();
        }
    }
    m_CurrentIndicators.clear();
}

uint64_t CCooccurrences::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_MaximumLength);
    seed = CChecksum::calculate(seed, m_Length);
    seed = CChecksum::calculate(seed, m_IndicatorWidth);
    seed = CChecksum::calculate(seed, m_Offset);
    seed = CChecksum::calculate(seed, m_CurrentIndicators);
    return CChecksum::calculate(seed, m_Indicators);
}

void CCooccurrences::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CCooccurrences");
    core::CMemoryDebug::dynamicSize("m_CurrentIndicators", m_CurrentIndicators, mem);
    core::CMemoryDebug::dynamicSize("m_Indicators", m_Indicators, mem);
}

std::size_t CCooccurrences::memoryUsage(void) const
{
    std::size_t mem = core::CMemory::dynamicSize(m_CurrentIndicators);
    mem += core::CMemory::dynamicSize(m_Indicators);
    return mem;
}

}
}
