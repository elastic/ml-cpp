/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <maths/common/CQuantileSketch.h>

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/common/CChecksum.h>
#include <maths/common/COrderings.h>

#include <boost/operators.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace ml {
namespace maths {
namespace common {
namespace {

using TFloatFloatPr = CQuantileSketch::TFloatFloatPr;
using TFloatFloatPrVec = CQuantileSketch::TFloatFloatPrVec;

//! \brief An iterator over just the unique knot values.
// clang-format off
class CUniqueIterator : private boost::addable2<CUniqueIterator, std::ptrdiff_t,
                                boost::subtractable2<CUniqueIterator, std::ptrdiff_t,
                                boost::equality_comparable<CUniqueIterator>>> {
    // clang-format on
public:
    CUniqueIterator(TFloatFloatPrVec& knots, std::size_t i)
        : m_Knots(&knots), m_I(i) {}

    bool operator==(const CUniqueIterator& rhs) const {
        return m_I == rhs.m_I && m_Knots == rhs.m_Knots;
    }

    TFloatFloatPr& operator*() const { return (*m_Knots)[m_I]; }
    TFloatFloatPr* operator->() const { return &(*m_Knots)[m_I]; }

    const CUniqueIterator& operator++() {
        CFloatStorage x{(*m_Knots)[m_I].first};
        std::ptrdiff_t n{static_cast<std::ptrdiff_t>(m_Knots->size())};
        while (++m_I < n && (*m_Knots)[m_I].first == x) {
        }
        return *this;
    }

    const CUniqueIterator& operator--() {
        CFloatStorage x{(*m_Knots)[m_I].first};
        while (--m_I >= 0 && (*m_Knots)[m_I].first == x) {
        }
        return *this;
    }

    const CUniqueIterator& operator+=(std::ptrdiff_t i) {
        while (--i >= 0) {
            this->operator++();
        }
        return *this;
    }

    const CUniqueIterator& operator-=(std::ptrdiff_t i) {
        while (--i >= 0) {
            this->operator--();
        }
        return *this;
    }

    std::ptrdiff_t index() const { return m_I; }

private:
    TFloatFloatPrVec* m_Knots;
    std::ptrdiff_t m_I;
};

std::ptrdiff_t previousDifferent(TFloatFloatPrVec& knots, std::size_t index) {
    CUniqueIterator previous{knots, index};
    --previous;
    return previous.index();
}

std::ptrdiff_t nextDifferent(TFloatFloatPrVec& knots, std::size_t index) {
    CUniqueIterator next{knots, index};
    ++next;
    return next.index();
}

const auto EPS = static_cast<double>(std::numeric_limits<float>::epsilon());
const std::size_t MINIMUM_MAX_SIZE{3};
const core::TPersistenceTag UNSORTED_TAG{"a", "unsorted"};
const core::TPersistenceTag KNOTS_TAG{"b", "knots"};
const core::TPersistenceTag COUNT_TAG{"c", "count"};
}

CQuantileSketch::CQuantileSketch(EInterpolation interpolation, std::size_t size)
    : m_Interpolation(interpolation),
      m_MaxSize(std::max(size, MINIMUM_MAX_SIZE)), m_Unsorted(0), m_Count(0.0) {
    m_Knots.reserve(m_MaxSize + 1);
}

bool CQuantileSketch::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(UNSORTED_TAG, m_Unsorted)
        RESTORE(KNOTS_TAG, core::CPersistUtils::fromString(traverser.value(), m_Knots))
        RESTORE_BUILT_IN(COUNT_TAG, m_Count)
    } while (traverser.next());
    return true;
}

void CQuantileSketch::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(UNSORTED_TAG, m_Unsorted);
    inserter.insertValue(KNOTS_TAG, core::CPersistUtils::toString(m_Knots));
    inserter.insertValue(COUNT_TAG, m_Count, core::CIEEE754::E_SinglePrecision);
}

const CQuantileSketch& CQuantileSketch::operator+=(const CQuantileSketch& rhs) {
    m_Knots.insert(m_Knots.end(), rhs.m_Knots.begin(), rhs.m_Knots.end());
    std::sort(m_Knots.begin(), m_Knots.end());
    m_Unsorted = 0;
    m_Count += rhs.m_Count;
    LOG_TRACE(<< "knots = " << m_Knots);

    this->reduce(m_MaxSize);

    TFloatFloatPrVec values(m_Knots.begin(), m_Knots.end());
    m_Knots.swap(values);

    return *this;
}

void CQuantileSketch::add(double x, double n) {
    ++m_Unsorted;
    m_Knots.emplace_back(x, n);
    m_Count += n;
    if (m_Knots.size() > m_MaxSize) {
        this->fastReduce();
    }
}

void CQuantileSketch::age(double factor) {
    for (auto& knot : m_Knots) {
        knot.second *= factor;
    }
    m_Count *= factor;
}

bool CQuantileSketch::cdf(double x_, double& result) const {
    result = 0.0;
    if (m_Knots.empty()) {
        LOG_ERROR(<< "No values added to quantile sketch");
        return false;
    }

    if (m_Unsorted > 0 || m_Knots.size() > this->fastReduceTargetSize()) {
        // It is critical to match the size selected by the call to reduce in
        // the add method or the unmerged values can cause issues estimating
        // the cdf close to 0 or 1.
        const_cast<CQuantileSketch*>(this)->reduce(this->fastReduceTargetSize());
    }

    CFloatStorage x = x_;
    std::ptrdiff_t n = m_Knots.size();
    if (n == 1) {
        result = x < m_Knots[0].first ? 0.0 : (x > m_Knots[0].first ? 1.0 : 0.5);
        return true;
    }

    std::ptrdiff_t k = std::lower_bound(m_Knots.begin(), m_Knots.end(), x,
                                        COrderings::SFirstLess()) -
                       m_Knots.begin();
    LOG_TRACE(<< "k = " << k);

    switch (m_Interpolation) {
    case E_Linear: {
        if (k == 0) {
            double xl = m_Knots[0].first;
            double xr = m_Knots[1].first;
            double f = m_Knots[0].second / m_Count;
            LOG_TRACE(<< "xl = " << xl << ", xr = " << xr << ", f = " << f);
            result = f * std::max(x - 1.5 * xl + 0.5 * xr, 0.0) / (xr - xl);
        } else if (k == n) {
            double xl = m_Knots[n - 2].first;
            double xr = m_Knots[n - 1].first;
            double f = m_Knots[n - 1].second / m_Count;
            LOG_TRACE(<< "xl = " << xl << ", xr = " << xr << ", f = " << f);
            result = 1.0 - f * std::max(1.5 * xr - 0.5 * xl - x, 0.0) / (xr - xl);
        } else {
            double xl = m_Knots[k - 1].first;
            double xr = m_Knots[k].first;
            bool left = (2 * k < n);
            bool loc = (2.0 * x < xl + xr);
            double partial = 0.0;
            for (std::ptrdiff_t i = left ? 0 : (loc ? k : k + 1),
                                m = left ? (loc ? k - 1 : k) : n;
                 i < m; ++i) {
                partial += m_Knots[i].second;
            }
            partial /= m_Count;
            double dn;
            if (loc) {
                double xll = k > 1 ? static_cast<double>(m_Knots[k - 2].first)
                                   : 2.0 * xl - xr;
                xr = 0.5 * (xl + xr);
                xl = 0.5 * (xll + xl);
                dn = m_Knots[k - 1].second / m_Count;
            } else {
                double xrr = k + 1 < n ? static_cast<double>(m_Knots[k + 1].first)
                                       : 2.0 * xr - xl;
                xl = 0.5 * (xl + xr);
                xr = 0.5 * (xr + xrr);
                dn = m_Knots[k].second / m_Count;
            }
            LOG_TRACE(<< "left = " << left << ", loc = " << loc << ", partial = " << partial
                      << ", xl = " << xl << ", xr = " << xr << ", dn = " << dn);
            result = left ? partial + dn * (x - xl) / (xr - xl)
                          : 1.0 - partial - dn * (xr - x) / (xr - xl);
        }
        return true;
    }
    case E_PiecewiseConstant: {
        if (k == 0) {
            double f = m_Knots[0].second / m_Count;
            result = x < m_Knots[0].first ? 0.0 : 0.5 * f;
        } else if (k == n) {
            double f = m_Knots[n - 1].second / m_Count;
            result = x > m_Knots[0].first ? 1.0 : 1.0 - 0.5 * f;
        } else {
            bool left = (2 * k < n);
            double partial = x < m_Knots[0].first ? 0.0 : 0.5 * m_Knots[0].second;
            for (std::ptrdiff_t i = left ? 0 : k + 1, m = left ? k : n; i < m; ++i) {
                partial += m_Knots[i].second;
            }
            partial /= m_Count;
            LOG_TRACE(<< "left = " << left << ", partial = " << partial);
            result = left ? partial : 1.0 - partial;
        }
        return true;
    }
    }
    return true;
}

bool CQuantileSketch::minimum(double& result) const {
    if (m_Knots.empty()) {
        LOG_ERROR(<< "No values added to quantile sketch");
        return false;
    }

    result = m_Knots[0].first;
    return true;
}

bool CQuantileSketch::maximum(double& result) const {
    if (m_Knots.empty()) {
        LOG_ERROR(<< "No values added to quantile sketch");
        return false;
    }

    result = m_Knots.back().first;
    return true;
}

bool CQuantileSketch::mad(double& result) const {
    if (m_Knots.empty()) {
        LOG_ERROR(<< "No values added to quantile sketch");
        return false;
    }

    double median;
    quantile(E_Linear, m_Knots, m_Count, 50.0, median);
    LOG_TRACE(<< "median = " << median);

    TFloatFloatPrVec knots(m_Knots);
    std::for_each(knots.begin(), knots.end(), [median](TFloatFloatPr& knot) {
        knot.first = std::fabs(knot.first - median);
    });
    std::sort(knots.begin(), knots.end(), COrderings::SFirstLess());
    LOG_TRACE(<< "knots = " << knots);

    quantile(E_Linear, knots, m_Count, 50.0, result);

    return true;
}

bool CQuantileSketch::quantile(double percentage, double& result) const {
    if (m_Knots.empty()) {
        LOG_ERROR(<< "No values added to quantile sketch");
        return false;
    }
    if (m_Unsorted > 0 || m_Knots.size() > this->fastReduceTargetSize()) {
        // It is critical to match the size selected by the call to reduce in
        // the add method or the unmerged values can cause issues estimating
        // percentiles close to 0 or 100.
        const_cast<CQuantileSketch*>(this)->reduce(this->fastReduceTargetSize());
    }
    if (percentage < 0.0 || percentage > 100.0) {
        LOG_ERROR(<< "Invalid percentile " << percentage);
        return false;
    }

    quantile(m_Interpolation, m_Knots, m_Count, percentage, result);

    return true;
}

const CQuantileSketch::TFloatFloatPrVec& CQuantileSketch::knots() const {
    return m_Knots;
}

double CQuantileSketch::count() const {
    return m_Count;
}

std::uint64_t CQuantileSketch::checksum(std::uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_MaxSize);
    seed = CChecksum::calculate(seed, m_Knots);
    return CChecksum::calculate(seed, m_Count);
}

void CQuantileSketch::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CQuantileSketch");
    core::CMemoryDebug::dynamicSize("m_Knots", m_Knots, mem);
}

std::size_t CQuantileSketch::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Knots);
}

std::size_t CQuantileSketch::staticSize() const {
    return sizeof(*this);
}

bool CQuantileSketch::checkInvariants() const {
    if (m_Knots.size() > m_MaxSize) {
        LOG_ERROR(<< "Too many knots: " << m_Knots.size() << " > " << m_MaxSize);
        return false;
    }
    if (m_Unsorted > m_Knots.size()) {
        LOG_ERROR(<< "Invalid unsorted count: " << m_Unsorted << "/" << m_Knots.size());
        return false;
    }
    if (std::is_sorted(m_Knots.begin(), m_Knots.end() - m_Unsorted) == false) {
        LOG_ERROR(<< "Unordered knots: "
                  << core::CContainerPrinter::print(m_Knots.begin(), m_Knots.end() - m_Unsorted));
        return false;
    }
    double count = 0.0;
    for (const auto& knot : m_Knots) {
        count += knot.second;
    }
    if (std::fabs(m_Count - count) > 10.0 * EPS * m_Count) {
        LOG_ERROR(<< "Count mismatch: error " << std::fabs(m_Count - count) << "/" << m_Count);
        return false;
    }
    return true;
}

std::string CQuantileSketch::print() const {
    return core::CContainerPrinter::print(m_Knots);
}

void CQuantileSketch::quantile(EInterpolation interpolation,
                               const TFloatFloatPrVec& knots,
                               double count,
                               double percentage,
                               double& result) {
    std::size_t n = knots.size();

    percentage /= 100.0;

    double partial = 0.0;
    double cutoff = percentage * count;
    for (std::size_t i = 0; i < n; ++i) {
        partial += knots[i].second;
        if (partial >= cutoff - count * EPS) {
            switch (interpolation) {
            case E_Linear:
                if (n == 1) {
                    result = knots[0].first;
                } else {
                    double x0 = knots[0].first;
                    double x1 = knots[1].first;
                    double xa = i == 0 ? 2.0 * x0 - x1
                                       : static_cast<double>(knots[i - 1].first);
                    double xb = knots[i].first;
                    double xc = i + 1 == n ? 2.0 * xb - xa
                                           : static_cast<double>(knots[i + 1].first);
                    xa += 0.5 * (xb - xa);
                    xb += 0.5 * (xc - xb);
                    double dx = (xb - xa);
                    double nb = knots[i].second;
                    double m = nb / dx;
                    result = xb + (cutoff - partial) / m;
                }
                return;

            case E_PiecewiseConstant:
                if (i + 1 == n || partial > cutoff + count * EPS) {
                    result = knots[i].first;
                } else {
                    result = (knots[i].first + knots[i + 1].first) / 2.0;
                }
                return;
            }
        }
    }

    result = knots[n - 1].second;
}

std::size_t CQuantileSketch::fastReduceTargetSize() const {
    return static_cast<std::size_t>(0.9 * static_cast<double>(m_MaxSize) + 1.0);
}

void CQuantileSketch::fastReduce() {
    this->reduce(this->fastReduceTargetSize());
}

void CQuantileSketch::reduce(std::size_t target) {

    this->orderAndDeduplicate();

    if (m_Knots.size() > target) {
        TFloatFloatPrVec mergeCosts;
        TSizeVec mergeCandidates;
        TBoolVec stale(m_Knots.size(), false);
        mergeCosts.reserve(m_Knots.size());
        mergeCandidates.reserve(m_Knots.size());
        CPRNG::CXorOShiro128Plus rng(static_cast<std::uint64_t>(m_Count));
        std::uniform_real_distribution<double> u01{0.0, 1.0};
        for (std::size_t i = 0; i + 3 < m_Knots.size(); ++i) {
            mergeCosts.emplace_back(cost(m_Knots[i + 1], m_Knots[i + 2]), u01(rng));
            mergeCandidates.push_back(i);
        }
        LOG_TRACE(<< "merge costs = " << mergeCosts);
        this->reduceWithSuppliedCosts(target, mergeCosts, mergeCandidates, stale);
    }
}

void CQuantileSketch::reduceWithSuppliedCosts(std::size_t target,
                                              TFloatFloatPrVec& mergeCosts,
                                              TSizeVec& mergeCandidates,
                                              TBoolVec& stale) {

    auto mergeCostGreater = [&mergeCosts](std::size_t lhs, std::size_t rhs) {
        return COrderings::lexicographical_compare(
            -mergeCosts[lhs].first, mergeCosts[lhs].second,
            -mergeCosts[rhs].first, mergeCosts[rhs].second);
    };
    std::make_heap(mergeCandidates.begin(), mergeCandidates.end(), mergeCostGreater);

    std::size_t numberToMerge{m_Knots.size() - target};
    std::ptrdiff_t numberMergeCandidates{static_cast<std::ptrdiff_t>(m_Knots.size()) - 3};

    while (numberToMerge > 0) {
        LOG_TRACE(<< "merge candidates = " << mergeCandidates);

        std::size_t l{mergeCandidates.front() + 1};
        std::pop_heap(mergeCandidates.begin(), mergeCandidates.end(), mergeCostGreater);
        mergeCandidates.pop_back();

        LOG_TRACE(<< "stale = " << stale);
        if (stale[l] == false) {
            std::size_t r{static_cast<std::size_t>(nextDifferent(m_Knots, l))};
            LOG_TRACE(<< "Merging " << l << " and " << r
                      << ", cost = " << mergeCosts[l - 1].first);

            // Note that mergeCosts[l - 1].second isn't truly random because
            // it is used for selecting the merge order, but it's good enough.
            auto mergedKnot = this->mergedKnot(l, r, mergeCosts[l - 1].second);

            // Find the points that have been merged with xl and xr if any.
            std::ptrdiff_t ll{previousDifferent(m_Knots, l)};
            std::ptrdiff_t rr{nextDifferent(m_Knots, r) - 1};
            std::fill_n(m_Knots.begin() + ll + 1, rr - ll, mergedKnot);
            LOG_TRACE(<< "merged = " << mergedKnot);
            LOG_TRACE(<< "right  = " << m_Knots[rr + 1]);

            if (ll > 0) {
                stale[ll] = true;
            }
            if (rr < numberMergeCandidates) {
                stale[rr] = true;
            }
            --numberToMerge;
        } else {
            CUniqueIterator ll(m_Knots, l);
            CUniqueIterator rr{ll};
            ++rr;
            mergeCosts[l - 1].first = cost(*ll, *rr);
            mergeCandidates.push_back(l - 1);
            std::push_heap(mergeCandidates.begin(), mergeCandidates.end(), mergeCostGreater);
            stale[l] = false;
        }
    }

    m_Knots.erase(std::unique(m_Knots.begin(), m_Knots.end()), m_Knots.end());
    LOG_TRACE(<< "final = " << m_Knots);
}

void CQuantileSketch::orderAndDeduplicate() {
    if (m_Unsorted > 0) {
        std::sort(m_Knots.end() - m_Unsorted, m_Knots.end());
        std::inplace_merge(m_Knots.begin(), m_Knots.end() - m_Unsorted, m_Knots.end());
    }
    LOG_TRACE(<< "sorted = " << m_Knots);

    // Combine any duplicate points.
    auto end = m_Knots.begin();
    for (std::size_t i = 1; i <= m_Knots.size(); ++end, ++i) {
        TFloatFloatPr& knot = *end;
        knot = m_Knots[i - 1];
        CFloatStorage x{knot.first};
        for (/**/; i < m_Knots.size() && m_Knots[i].first == x; ++i) {
            knot.second += m_Knots[i].second;
        }
    }
    m_Knots.erase(end, m_Knots.end());
    LOG_TRACE(<< "de-duplicated = " << m_Knots);

    m_Unsorted = 0;
}

CQuantileSketch::TFloatFloatPr
CQuantileSketch::mergedKnot(std::size_t l, std::size_t r, double tieBreaker) const {
    double xl{m_Knots[l].first};
    double nl{m_Knots[l].second};
    double xr{m_Knots[r].first};
    double nr{m_Knots[r].second};
    LOG_TRACE(<< "(xl,nl) = (" << xl << "," << nl << "), (xr,nr) = (" << xr
              << "," << nr << ")");

    TFloatFloatPr mergedKnot;
    switch (m_Interpolation) {
    case E_Linear:
        mergedKnot.first = (nl * xl + nr * xr) / (nl + nr);
        mergedKnot.second = nl + nr;
        break;
    case E_PiecewiseConstant:
        mergedKnot.first = nl < nr ? xr : (nl > nr ? xl : tieBreaker < 0.5 ? xl : xr);
        mergedKnot.second = nl + nr;
        break;
    }

    return mergedKnot;
}

double CQuantileSketch::cost(const TFloatFloatPr& vl, const TFloatFloatPr& vr) {
    // Interestingly, minimizing the approximation error (area between
    // curve before and after merging) produces good summary for the
    // piecewise constant objective, but a very bad summary for the linear
    // objective. Basically, an empirically good strategy is to target
    // the piecewise objective when sketching and then perform unbiased
    // linear interpolation by linearly interpolating between the mid-
    // points of the steps when computing quantiles.

    double xl{vl.first};
    double xr{vr.first};
    double nl{vl.second};
    double nr{vr.second};

    return std::min(nl, nr) * (xr - xl);
}

std::uint64_t CFastQuantileSketch::checksum(std::uint64_t seed) const {
    seed = this->CQuantileSketch::checksum(seed);
    return CChecksum::calculate(seed, m_ReductionFraction);
}

void CFastQuantileSketch::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CFastQuantileSketch");
    core::CMemoryDebug::dynamicSize("m_Knots", this->knots(), mem);
    core::CMemoryDebug::dynamicSize("m_MergeCosts", m_MergeCosts, mem);
    core::CMemoryDebug::dynamicSize("m_MergeCandidates", m_MergeCandidates, mem);
}

std::size_t CFastQuantileSketch::memoryUsage() const {
    std::size_t mem{this->CQuantileSketch::memoryUsage()};
    mem += core::CMemory::dynamicSize(m_MergeCosts);
    mem += core::CMemory::dynamicSize(m_MergeCandidates);
    return mem;
}

std::size_t CFastQuantileSketch::staticSize() const {
    return sizeof(*this);
}

std::size_t CFastQuantileSketch::fastReduceTargetSize() const {
    return static_cast<std::size_t>(
        m_ReductionFraction * static_cast<double>(this->maxSize()) + 1.0);
}

void CFastQuantileSketch::fastReduce() {

    this->orderAndDeduplicate();

    TFloatFloatPrVec& knots{this->writeableKnots()};
    if (knots.size() > this->fastReduceTargetSize()) {
        std::size_t n{m_MergeCosts.size()};
        m_MergeCosts.resize(knots.size() - 3);
        std::uniform_real_distribution<double> u01{0.0, 1.0};
        for (std::size_t i = n; i < m_MergeCosts.size(); ++i) {
            m_MergeCosts[i].second = u01(m_Rng);
        }
        m_MergeCandidates.resize(knots.size() - 3);
        for (std::size_t i = 0; i + 3 < knots.size(); ++i) {
            m_MergeCosts[i].first = cost(knots[i + 1], knots[i + 2]);
            m_MergeCandidates[i] = i;
        }
        LOG_TRACE(<< "merge costs = " << m_MergeCosts);

        std::size_t numberToMerge{knots.size() - this->fastReduceTargetSize()};

        // This is pseudo greedy unlike CQuantileSketch which is greedy: we
        // simply ignore the fact that we have slightly different costs after
        // each merge. This allows us to avoid using a heap altogether which
        // dominates the computational cost of reduce.

        std::nth_element(m_MergeCandidates.begin(), m_MergeCandidates.begin() + numberToMerge,
                         m_MergeCandidates.end(), [&](auto lhs, auto rhs) {
                             return m_MergeCosts[lhs] < m_MergeCosts[rhs];
                         });
        for (std::size_t i = 0; i < numberToMerge; ++i) {
            std::size_t l{m_MergeCandidates[i] + 1};
            std::size_t r{static_cast<std::size_t>(nextDifferent(knots, l))};
            auto mergedKnot = this->mergedKnot(l, r, m_MergeCosts[l - 1].second);

            // Find the points that have been merged with xl and xr if any.
            std::ptrdiff_t ll{previousDifferent(knots, l)};
            std::ptrdiff_t rr{nextDifferent(knots, r) - 1};
            std::fill_n(knots.begin() + ll + 1, rr - ll, mergedKnot);
            LOG_TRACE(<< "merged = " << mergedKnot);
            LOG_TRACE(<< "right  = " << knots[rr + 1]);
        }
        knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    }
}
}
}
}
