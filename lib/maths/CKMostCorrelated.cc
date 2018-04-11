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

#include <maths/CKMostCorrelated.h>

#include <core/CAllocationStrategy.h>
#include <core/CPersistUtils.h>
#include <core/CStringUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CSampling.h>

#include <boost/array.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/adapted/boost_array.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/unordered_set.hpp>

#include <cmath>
#include <functional>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
namespace bgm = boost::geometry::model;

namespace ml {
namespace maths {

namespace {

using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrUSet = boost::unordered_set<TSizeSizePr>;
using TPoint = boost::array<double, CKMostCorrelated::NUMBER_PROJECTIONS>;
using TPointSizePr = std::pair<TPoint, std::size_t>;
using TPointSizePrVec = std::vector<TPointSizePr>;

//! \brief Unary predicate to check variables, corresponding
//! to labeled points, are not equal to a specified variable.
class CNotEqual : public std::unary_function<TPointSizePr, bool> {
public:
    CNotEqual(std::size_t X) : m_X(X) {}

    bool operator()(const TPointSizePr& y) const {
        std::size_t Y = y.second;
        return m_X != Y;
    }

private:
    std::size_t m_X;
};

//! \brief Unary predicate to check if one specified variable
//! and others, corresponding to labeled points, are in a
//! specified collection pairs of variables.
class CPairNotIn : public std::unary_function<TPointSizePr, bool> {
public:
    CPairNotIn(const TSizeSizePrUSet& lookup, std::size_t X)
        : m_Lookup(&lookup), m_X(X) {}

    bool operator()(const TPointSizePr& y) const {
        std::size_t Y = y.second;
        return m_Lookup->count(std::make_pair(std::min(m_X, Y), std::max(m_X, Y))) == 0;
    }

private:
    const TSizeSizePrUSet* m_Lookup;
    std::size_t m_X;
};

//! \brief Unary predicate to check if a point is closer,
//! in square Euclidean distance, to a specified point than
//! a specified threshold.
class CCloserThan : public std::unary_function<TPointSizePr, bool> {
public:
    CCloserThan(double threshold, const TPoint& x)
        : m_Threshold(threshold), m_X(x) {}

    bool operator()(const TPointSizePr& y) const {
        return pow2(bg::distance(m_X, y.first)) < m_Threshold;
    }

private:
    static double pow2(double x) { return x * x; }

private:
    double m_Threshold;
    TPoint m_X;
};

const std::string PROJECTIONS_TAG("a");
const std::string CURRENT_PROJECTED_TAG("b");
const std::string PROJECTED_TAG("c");
const std::string MAXIMUM_COUNT_TAG("d");
const std::string MOMENTS_TAG("e");
const std::string MOST_CORRELATED_TAG("f");
const std::string RNG_TAG("g");
// Nested tags.
const std::string CORRELATION_TAG("a");
const std::string X_TAG("b");
const std::string Y_TAG("c");

const double MINIMUM_FREQUENCY = 0.25;

} // unnamed::

CKMostCorrelated::CKMostCorrelated(std::size_t k, double decayRate, bool initialize)
    : m_K(k), m_DecayRate(decayRate), m_MaximumCount(0.0) {
    if (initialize) {
        this->nextProjection();
    }
}

bool CKMostCorrelated::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    m_Projections.clear();
    m_CurrentProjected.clear();
    m_Projected.clear();
    m_Moments.clear();
    m_MostCorrelated.clear();

    do {
        const std::string& name = traverser.name();
        RESTORE(RNG_TAG, m_Rng.fromString(traverser.value()))
        RESTORE(PROJECTIONS_TAG,
                core::CPersistUtils::restore(PROJECTIONS_TAG, m_Projections, traverser))
        RESTORE(CURRENT_PROJECTED_TAG,
                core::CPersistUtils::restore(CURRENT_PROJECTED_TAG, m_CurrentProjected, traverser))
        RESTORE(PROJECTED_TAG,
                core::CPersistUtils::restore(PROJECTED_TAG, m_Projected, traverser))
        RESTORE_BUILT_IN(MAXIMUM_COUNT_TAG, m_MaximumCount)
        RESTORE(MOMENTS_TAG, core::CPersistUtils::restore(MOMENTS_TAG, m_Moments, traverser))
        RESTORE(MOST_CORRELATED_TAG,
                core::CPersistUtils::restore(MOST_CORRELATED_TAG, m_MostCorrelated, traverser))
    } while (traverser.next());

    return true;
}

void CKMostCorrelated::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(RNG_TAG, m_Rng.toString());
    core::CPersistUtils::persist(PROJECTIONS_TAG, m_Projections, inserter);
    core::CPersistUtils::persist(CURRENT_PROJECTED_TAG, m_CurrentProjected, inserter);
    core::CPersistUtils::persist(PROJECTED_TAG, m_Projected, inserter);
    inserter.insertValue(MAXIMUM_COUNT_TAG, m_MaximumCount);
    core::CPersistUtils::persist(MOMENTS_TAG, m_Moments, inserter);
    core::CPersistUtils::persist(MOST_CORRELATED_TAG, m_MostCorrelated, inserter);
}

void CKMostCorrelated::mostCorrelated(TSizeSizePrVec& result) const {
    result.clear();
    std::size_t N = std::min(m_K, m_MostCorrelated.size());
    if (N > 0) {
        result.reserve(N);
        for (std::size_t i = 0u; i < N; ++i) {
            result.emplace_back(m_MostCorrelated[i].s_X, m_MostCorrelated[i].s_Y);
        }
    }
}

void CKMostCorrelated::mostCorrelated(std::size_t n,
                                      TSizeSizePrVec& correlates,
                                      TDoubleVec* pearson) const {
    correlates.clear();
    if (pearson) {
        pearson->clear();
    }
    n = std::min(n, m_MostCorrelated.size());
    if (n > 0) {
        correlates.reserve(n);
        if (pearson) {
            pearson->reserve(n);
        }
        for (std::size_t i = 0u; i < n; ++i) {
            correlates.emplace_back(m_MostCorrelated[i].s_X, m_MostCorrelated[i].s_Y);
            if (pearson) {
                pearson->push_back(CBasicStatistics::mean(m_MostCorrelated[i].s_Correlation));
            }
        }
    }
}

void CKMostCorrelated::correlations(TDoubleVec& result) const {
    result.clear();
    std::size_t N = std::min(m_K, m_MostCorrelated.size());
    if (N > 0) {
        result.reserve(N);
        for (std::size_t i = 0u; i < N; ++i) {
            result.push_back(CBasicStatistics::mean(m_MostCorrelated[i].s_Correlation));
        }
    }
}

void CKMostCorrelated::correlations(std::size_t n, TDoubleVec& result) const {
    result.clear();
    n = std::min(n, m_MostCorrelated.size());
    if (n > 0) {
        result.reserve(n);
        for (std::size_t i = 0u; i < n; ++i) {
            result.push_back(CBasicStatistics::mean(m_MostCorrelated[i].s_Correlation));
        }
    }
}

void CKMostCorrelated::addVariables(std::size_t n) {
    core::CAllocationStrategy::resize(m_Moments, std::max(n, m_Moments.size()));
}

void CKMostCorrelated::removeVariables(const TSizeVec& remove) {
    LOG_TRACE(<< "removing = " << core::CContainerPrinter::print(remove));
    for (std::size_t i = 0u; i < remove.size(); ++i) {
        if (remove[i] < m_Moments.size()) {
            m_Moments[remove[i]] = TMeanVarAccumulator();
            m_Projected.erase(remove[i]);
            m_MostCorrelated.erase(std::remove_if(m_MostCorrelated.begin(),
                                                  m_MostCorrelated.end(),
                                                  CMatches(remove[i])),
                                   m_MostCorrelated.end());
        }
    }
}

bool CKMostCorrelated::changed() const {
    return m_Projections.size() == PROJECTION_DIMENSION;
}

void CKMostCorrelated::add(std::size_t X, double x) {
    if (X >= m_Moments.size()) {
        LOG_ERROR(<< "Invalid variable " << X);
        return;
    }

    TMeanVarAccumulator& moments = m_Moments[X];
    moments.add(x);
    TVector projected(0.0);
    if (CBasicStatistics::count(moments) > 2.0) {
        double m = CBasicStatistics::mean(moments);
        double sd = std::sqrt(CBasicStatistics::variance(moments));
        if (sd > 10.0 * std::numeric_limits<double>::epsilon() * std::fabs(m)) {
            projected = m_Projections.back() * (x - m) / sd;
            m_CurrentProjected[X] += projected;
        }
    }
}

void CKMostCorrelated::capture() {
    m_MaximumCount += 1.0;

    for (TSizeVectorUMapCItr i = m_CurrentProjected.begin();
         i != m_CurrentProjected.end(); ++i) {
        std::size_t X = i->first;
        TSizeVectorPackedBitVectorPrUMapItr j = m_Projected.find(X);
        if (j == m_Projected.end()) {
            TVector zero(0.0);
            CPackedBitVector indicator(PROJECTION_DIMENSION - m_Projections.size(), false);
            j = m_Projected
                    .emplace(boost::unordered::piecewise_construct,
                             boost::make_tuple(X), boost::make_tuple(zero, indicator))
                    .first;
        }
        j->second.first += i->second;
    }
    for (TSizeVectorPackedBitVectorPrUMapItr i = m_Projected.begin();
         i != m_Projected.end(); ++i) {
        i->second.second.extend(m_CurrentProjected.count(i->first) > 0);
    }

    m_Projections.pop_back();
    m_CurrentProjected.clear();

    if (m_Projections.empty()) {
        LOG_TRACE(<< "# projections = " << m_Projected.size());

        // For existing indices in the "most correlated" collection
        // compute the updated statistics.
        for (std::size_t i = 0u; i < m_MostCorrelated.size(); ++i) {
            m_MostCorrelated[i].update(m_Projected);
        }
        std::stable_sort(m_MostCorrelated.begin(), m_MostCorrelated.end());

        // Remove any variables for which the correlation will necessarily be zero.
        for (TSizeVectorPackedBitVectorPrUMapItr i = m_Projected.begin();
             i != m_Projected.end();
             /**/) {
            const CPackedBitVector& indicator = i->second.second;
            if (indicator.manhattan() <=
                MINIMUM_FREQUENCY * static_cast<double>(indicator.dimension())) {
                i = m_Projected.erase(i);
            } else {
                ++i;
            }
        }
        LOG_TRACE(<< "# projections = " << m_Projected.size());

        // Find the "most correlated" collection for the current
        // projections.
        TCorrelationVec add;
        this->mostCorrelated(add);

        std::size_t N = m_MostCorrelated.size();
        std::size_t n = add.size();
        std::size_t desired = 2 * m_K;
        std::size_t added = N < desired ? std::min(desired - N, n) : 0;
        LOG_TRACE(<< "N = " << N << ", n = " << n << ", desired = " << desired
                  << ", added = " << added);

        if (added > 0) {
            m_MostCorrelated.insert(m_MostCorrelated.end(), add.end() - added, add.end());
        }
        if (n > added) {
            // When deciding which values to replace from the set [m_K, N) we
            // do so at random with probability proportional to 1 - absolute
            // correlation.

            LOG_TRACE(<< "add = " << core::CContainerPrinter::print(add));

            std::size_t vunerable = std::max(m_K, N - 3 * n);

            TDoubleVec p;
            p.reserve(std::min(N - m_K, 3 * n));
            double Z = 0.0;
            for (std::size_t i = vunerable; i < N; ++i) {
                double oneMinusCorrelation = 1.0 - m_MostCorrelated[i].absCorrelation();
                p.push_back(oneMinusCorrelation);
                Z += oneMinusCorrelation;
            }
            if (Z > 0.0) {
                for (std::size_t i = 0u; i < p.size(); ++i) {
                    p[i] /= Z;
                }
                LOG_TRACE(<< "p = " << core::CContainerPrinter::print(p));

                TSizeVec replace;
                CSampling::categoricalSampleWithoutReplacement(m_Rng, p, n - added, replace);

                for (std::size_t i = 1u; i <= n - added; ++i) {
                    m_MostCorrelated[vunerable + replace[i - 1]] = add[n - added - i];
                }
            }
        }

        this->nextProjection();
    }
}

uint64_t CKMostCorrelated::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_K);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_Projections);
    seed = CChecksum::calculate(seed, m_CurrentProjected);
    seed = CChecksum::calculate(seed, m_Projected);
    seed = CChecksum::calculate(seed, m_MaximumCount);
    seed = CChecksum::calculate(seed, m_Moments);
    return CChecksum::calculate(seed, m_MostCorrelated);
}

void CKMostCorrelated::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CKMostCorrelated");
    core::CMemoryDebug::dynamicSize("m_Projections", m_Projections, mem);
    core::CMemoryDebug::dynamicSize("m_CurrentProjected", m_CurrentProjected, mem);
    core::CMemoryDebug::dynamicSize("m_Projected", m_Projected, mem);
    core::CMemoryDebug::dynamicSize("m_Moments", m_Moments, mem);
    core::CMemoryDebug::dynamicSize("m_MostCorrelated", m_MostCorrelated, mem);
}

std::size_t CKMostCorrelated::memoryUsage() const {
    std::size_t mem = core::CMemory::dynamicSize(m_Projections);
    mem += core::CMemory::dynamicSize(m_CurrentProjected);
    mem += core::CMemory::dynamicSize(m_Projected);
    mem += core::CMemory::dynamicSize(m_Moments);
    mem += core::CMemory::dynamicSize(m_MostCorrelated);
    return mem;
}

void CKMostCorrelated::mostCorrelated(TCorrelationVec& result) const {
    using TMaxDoubleAccumulator =
        CBasicStatistics::COrderStatisticsStack<double, 2, std::greater<double>>;
    using TMaxCorrelationAccumulator = CBasicStatistics::COrderStatisticsHeap<SCorrelation>;
    using TPointRTree = bgi::rtree<TPointSizePr, bgi::quadratic<16>>;

    result.clear();

    std::size_t N = m_MostCorrelated.size();
    std::size_t V = m_Projected.size();
    std::size_t desired = 2 * m_K;
    LOG_TRACE(<< "N = " << N << ", V = " << V << ", desired = " << desired);
    if (V == 1) {
        return;
    }

    TSizeSizePrUSet lookup;
    for (std::size_t i = 0u; i < m_MostCorrelated.size(); ++i) {
        std::size_t X = m_MostCorrelated[i].s_X;
        std::size_t Y = m_MostCorrelated[i].s_Y;
        lookup.insert(std::make_pair(std::min(X, Y), std::max(X, Y)));
    }

    std::size_t replace = std::max(
        static_cast<std::size_t>(REPLACE_FRACTION * static_cast<double>(desired) + 0.5),
        std::max(desired - N, std::size_t(1)));
    LOG_TRACE(<< "replace = " << replace);

    TMaxCorrelationAccumulator mostCorrelated(replace);

    if (10 * replace > V * (V - 1)) {
        LOG_TRACE(<< "Exhaustive search");

        for (TSizeVectorPackedBitVectorPrUMapCItr x = m_Projected.begin();
             x != m_Projected.end(); ++x) {
            std::size_t X = x->first;
            TSizeVectorPackedBitVectorPrUMapCItr y = x;
            while (++y != m_Projected.end()) {
                std::size_t Y = y->first;
                if (lookup.count(std::make_pair(std::min(X, Y), std::max(X, Y))) == 0) {
                    SCorrelation cxy(X, x->second.first, x->second.second, Y,
                                     y->second.first, y->second.second);
                    mostCorrelated.add(cxy);
                }
            }
        }
    } else {
        LOG_TRACE(<< "Nearest neighbour search");

        // 1) Build an r-tree,
        // 2) Lookup up V / replace nearest neighbours of each point
        //    and its negative to initialise search,
        // 3) Create a predicate with separation corresponding to the
        //    smallest correlation,
        // 4) Search for neighbours of each point and its negative for
        //    points in range updating the predicate in the loop with
        //    the new least correlated variable.

        // Bound the correlation based on the sparsity of the metric.
        TMaxDoubleAccumulator fmax;
        double dimension = 0.0;
        for (TSizeVectorPackedBitVectorPrUMapCItr i = m_Projected.begin();
             i != m_Projected.end(); ++i) {
            const CPackedBitVector& ix = i->second.second;
            dimension = static_cast<double>(ix.dimension());
            fmax.add(ix.manhattan() / dimension);
        }
        fmax.sort();
        if (fmax[1] <= MINIMUM_FREQUENCY) {
            return;
        }
        double amax = fmax[1] * dimension;

        TPointSizePrVec points;
        points.reserve(m_Projected.size());
        for (TSizeVectorPackedBitVectorPrUMapCItr i = m_Projected.begin();
             i != m_Projected.end(); ++i) {
            points.emplace_back(i->second.first.to<double>().toBoostArray(), i->first);
        }
        LOG_TRACE(<< "# points = " << points.size());

        unsigned int k = static_cast<unsigned int>(replace / V + 1);
        LOG_TRACE(<< "k = " << k);

        // The nearest neighbour search is very slow compared with
        // the search over the smallest correlation box predicate
        // so we use a small number of seed variables if V is large
        // compared to the number to replace.
        TSizeVec seeds;
        if (2 * replace < V) {
            CSampling::uniformSample(m_Rng, 0, V, 2 * replace, seeds);
            std::sort(seeds.begin(), seeds.end());
            seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
        } else {
            seeds.reserve(V);
            seeds.assign(boost::counting_iterator<std::size_t>(0),
                         boost::counting_iterator<std::size_t>(V));
        }

        try {
            TPointRTree rtree(points);
            TPointSizePrVec nearest;
            for (std::size_t i = 0u; i < seeds.size(); ++i) {
                std::size_t X = points[seeds[i]].second;
                const TVectorPackedBitVectorPr& px = m_Projected.at(X);

                nearest.clear();
                bgi::query(rtree,
                           bgi::satisfies(CNotEqual(X)) &&
                               bgi::satisfies(CPairNotIn(lookup, X)) &&
                               bgi::nearest((px.first.to<double>()).toBoostArray(), k),
                           std::back_inserter(nearest));
                bgi::query(rtree,
                           bgi::satisfies(CNotEqual(X)) &&
                               bgi::satisfies(CPairNotIn(lookup, X)) &&
                               bgi::nearest((-px.first.to<double>()).toBoostArray(), k),
                           std::back_inserter(nearest));

                for (std::size_t j = 0u; j < nearest.size(); ++j) {
                    std::size_t n = mostCorrelated.count();
                    std::size_t S = n == desired ? mostCorrelated.biggest().s_X : 0;
                    std::size_t T = n == desired ? mostCorrelated.biggest().s_Y : 0;
                    std::size_t Y = nearest[j].second;
                    const TVectorPackedBitVectorPr& py = m_Projected.at(Y);
                    SCorrelation cxy(X, px.first, px.second, Y, py.first, py.second);
                    if (lookup.count(std::make_pair(cxy.s_X, cxy.s_Y)) > 0) {
                        continue;
                    }
                    if (mostCorrelated.add(cxy)) {
                        if (n == desired) {
                            lookup.erase(std::make_pair(S, T));
                        }
                        lookup.insert(std::make_pair(cxy.s_X, cxy.s_Y));
                    }
                }
            }
            LOG_TRACE(<< "# seeds = " << mostCorrelated.count());
            LOG_TRACE(<< "seed most correlated = " << mostCorrelated);

            for (std::size_t i = 0u; i < points.size(); ++i) {
                const SCorrelation& biggest = mostCorrelated.biggest();
                double threshold = biggest.distance(amax);
                LOG_TRACE(<< "threshold = " << threshold);

                std::size_t X = points[i].second;
                const TVectorPackedBitVectorPr& px = m_Projected.at(X);

                TVector width(std::sqrt(threshold));
                nearest.clear();
                {
                    bgm::box<TPoint> box((px.first - width).to<double>().toBoostArray(),
                                         (px.first + width).to<double>().toBoostArray());
                    bgi::query(rtree,
                               bgi::within(box) && bgi::satisfies(CNotEqual(X)) &&
                                   bgi::satisfies(CCloserThan(
                                       threshold, px.first.to<double>().toBoostArray())) &&
                                   bgi::satisfies(CPairNotIn(lookup, X)),
                               std::back_inserter(nearest));
                }
                {
                    bgm::box<TPoint> box((-px.first - width).to<double>().toBoostArray(),
                                         (-px.first + width).to<double>().toBoostArray());
                    bgi::query(rtree,
                               bgi::within(box) && bgi::satisfies(CNotEqual(X)) &&
                                   bgi::satisfies(CCloserThan(
                                       threshold, (-px.first).to<double>().toBoostArray())) &&
                                   bgi::satisfies(CPairNotIn(lookup, X)),
                               std::back_inserter(nearest));
                }
                LOG_TRACE(<< "# candidates = " << nearest.size());

                for (std::size_t j = 0u; j < nearest.size(); ++j) {
                    std::size_t n = mostCorrelated.count();
                    std::size_t S = n == desired ? mostCorrelated.biggest().s_X : 0;
                    std::size_t T = n == desired ? mostCorrelated.biggest().s_Y : 0;
                    std::size_t Y = nearest[j].second;
                    const TVectorPackedBitVectorPr& py = m_Projected.at(Y);
                    SCorrelation cxy(X, px.first, px.second, Y, py.first, py.second);
                    if (lookup.count(std::make_pair(cxy.s_X, cxy.s_Y)) > 0) {
                        continue;
                    }
                    if (mostCorrelated.add(cxy)) {
                        if (n == desired) {
                            lookup.erase(std::make_pair(S, T));
                        }
                        lookup.insert(std::make_pair(cxy.s_X, cxy.s_Y));
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to compute most correlated " << e.what());
            return;
        }
    }

    mostCorrelated.sort();
    result.assign(mostCorrelated.begin(), mostCorrelated.end());
    LOG_TRACE(<< "most correlated " << core::CContainerPrinter::print(result));
}

void CKMostCorrelated::nextProjection() {
    TDoubleVec uniform01;
    CSampling::uniformSample(m_Rng, 0.0, 1.0,
                             NUMBER_PROJECTIONS * PROJECTION_DIMENSION, uniform01);
    m_Projections.reserve(PROJECTION_DIMENSION);
    m_Projections.resize(PROJECTION_DIMENSION);
    for (std::size_t i = 0u, j = 0u; i < PROJECTION_DIMENSION; ++i) {
        for (std::size_t k = 0u; k < NUMBER_PROJECTIONS; ++j, ++k) {
            m_Projections[i](k) = uniform01[j] < 0.5 ? -1.0 : 1.0;
        }
    }

    m_Projected.clear();

    double factor = std::exp(-m_DecayRate);
    m_MaximumCount *= factor;
    for (std::size_t i = 0u; i < m_Moments.size(); ++i) {
        m_Moments[i].age(factor);
    }
    for (std::size_t i = 0u; i < m_MostCorrelated.size(); ++i) {
        m_MostCorrelated[i].s_Correlation.age(factor);
    }
}

const CKMostCorrelated::TVectorVec& CKMostCorrelated::projections() const {
    return m_Projections;
}

const CKMostCorrelated::TSizeVectorPackedBitVectorPrUMap&
CKMostCorrelated::projected() const {
    return m_Projected;
}

const CKMostCorrelated::TCorrelationVec& CKMostCorrelated::correlations() const {
    return m_MostCorrelated;
}

const CKMostCorrelated::TMeanVarAccumulatorVec& CKMostCorrelated::moments() const {
    return m_Moments;
}

const std::size_t CKMostCorrelated::PROJECTION_DIMENSION = 20u;
const double CKMostCorrelated::MINIMUM_SPARSENESS = 0.5;
const double CKMostCorrelated::REPLACE_FRACTION = 0.1;

CKMostCorrelated::SCorrelation::SCorrelation()
    : s_X(std::numeric_limits<std::size_t>::max()),
      s_Y(std::numeric_limits<std::size_t>::max()) {
}

CKMostCorrelated::SCorrelation::SCorrelation(std::size_t X,
                                             const TVector& px,
                                             const CPackedBitVector& ix,
                                             std::size_t Y,
                                             const TVector& py,
                                             const CPackedBitVector& iy)
    : s_X(std::min(X, Y)), s_Y(std::max(X, Y)) {
    s_Correlation.add(correlation(px, ix, py, iy));
}

bool CKMostCorrelated::SCorrelation::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == CORRELATION_TAG) {
            if (s_Correlation.fromDelimited(traverser.value()) == false) {
                LOG_ERROR(<< "Invalid correlation in " << traverser.value());
                return false;
            }
        } else if (name == X_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), s_X) == false) {
                LOG_ERROR(<< "Invalid variable in " << traverser.value());
                return false;
            }
        } else if (name == Y_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), s_Y) == false) {
                LOG_ERROR(<< "Invalid variable in " << traverser.value());
                return false;
            }
        }
    } while (traverser.next());
    return true;
}

void CKMostCorrelated::SCorrelation::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(CORRELATION_TAG, s_Correlation.toDelimited());
    inserter.insertValue(X_TAG, s_X);
    inserter.insertValue(Y_TAG, s_Y);
}

bool CKMostCorrelated::SCorrelation::operator<(const SCorrelation& rhs) const {
    return COrderings::lexicographical_compare(
        -this->absCorrelation(), s_X, s_Y, -rhs.absCorrelation(), rhs.s_X, rhs.s_Y);
}

void CKMostCorrelated::SCorrelation::update(const TSizeVectorPackedBitVectorPrUMap& projected) {
    TSizeVectorPackedBitVectorPrUMapCItr x = projected.find(s_X);
    TSizeVectorPackedBitVectorPrUMapCItr y = projected.find(s_Y);
    if (x != projected.end() && y != projected.end()) {
        const TVector& px = x->second.first;
        const TVector& py = y->second.first;
        const CPackedBitVector& ix = x->second.second;
        const CPackedBitVector& iy = y->second.second;
        s_Correlation.add(correlation(px, ix, py, iy));
    }
}

double CKMostCorrelated::SCorrelation::distance(double amax) const {
    return static_cast<double>(NUMBER_PROJECTIONS) * amax * 2.0 *
           (1.0 - std::fabs(CBasicStatistics::mean(s_Correlation)));
}

double CKMostCorrelated::SCorrelation::absCorrelation() const {
    return std::fabs(CBasicStatistics::mean(s_Correlation)) -
           (1.0 / std::max(CBasicStatistics::count(s_Correlation), 2.0) +
            std::sqrt(CBasicStatistics::variance(s_Correlation)));
}

double CKMostCorrelated::SCorrelation::correlation(const TVector& px,
                                                   const CPackedBitVector& ix,
                                                   const TVector& py,
                                                   const CPackedBitVector& iy) {
    double result = 0.0;

    double nx = ix.manhattan() / static_cast<double>(ix.dimension());
    double ny = iy.manhattan() / static_cast<double>(iy.dimension());
    if (nx <= MINIMUM_FREQUENCY && ny <= MINIMUM_FREQUENCY) {
        return result;
    }

    double axy = ix.inner(iy, CPackedBitVector::E_AND);
    double oxy = ix.inner(iy, CPackedBitVector::E_OR);
    double cxy = axy / oxy;

    if (cxy > MINIMUM_FREQUENCY) {
        // The following uses the method of moments noting that
        //     E[S] = 2 (1 + cov(X,Y))
        //     E[D] = 2 (1 - cov(X,Y))
        //   var[S] = 8 (1 + cov(X,Y))^2
        //   var[D] = 8 (1 - cov(X,Y))^2
        //
        // Note that if the variables are strongly positively
        // (negatively) correlated the variance of the D (S)
        // estimators are much smaller than the others. We trap
        // this case, as best we can, and only use appropriate
        // ones.

        TMeanVarAccumulator dmv;
        TMeanVarAccumulator smv;
        for (std::size_t i = 0u; i < px.dimension(); ++i) {
            dmv.add((px(i) - py(i)) * (px(i) - py(i)));
            smv.add((px(i) + py(i)) * (px(i) + py(i)));
        }

        double dm = CBasicStatistics::mean(dmv);
        double dv = CBasicStatistics::variance(dmv);
        double sm = CBasicStatistics::mean(smv);
        double sv = CBasicStatistics::variance(smv);
        double cdm = 1.0 - 0.5 * dm / axy;
        double csm = 0.5 * sm / axy - 1.0;

        result = (cxy - MINIMUM_FREQUENCY) / (1.0 - MINIMUM_FREQUENCY);

        if (3.0 * dv < sv) {
            result *= std::max(cdm, 0.0);
        } else if (3.0 * sv < dv) {
            result *= std::min(csm, 0.0);
        } else {
            double lambda = dv == 0 ? 1.0 : sv / dv;
            double a = (2.0 + lambda - 1.0) / 4.0;
            double b = (2.0 + 1.0 - lambda) / 4.0;
            result *= a * std::max(cdm, 0.0) + b * std::min(csm, 0.0);
        }
    }

    return result;
}

uint64_t CKMostCorrelated::SCorrelation::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, s_Correlation);
    seed = CChecksum::calculate(seed, s_X);
    return CChecksum::calculate(seed, s_Y);
}

std::string CKMostCorrelated::SCorrelation::print() const {
    return CBasicStatistics::print(s_Correlation) + ' ' +
           core::CStringUtils::typeToString(s_X) + ' ' +
           core::CStringUtils::typeToString(s_Y);
}

CKMostCorrelated::CMatches::CMatches(std::size_t x) : m_X(x) {
}

bool CKMostCorrelated::CMatches::operator()(const SCorrelation& correlation) const {
    return correlation.s_X == m_X || correlation.s_Y == m_X;
}
}
}
