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
#include <core/CMemoryDef.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/common/CChecksum.h>
#include <maths/common/COrderings.h>

#include <boost/operators.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>

#if defined(__SSE__)

#include <xmmintrin.h>

#define ml_unaligned_load_128 _mm_loadu_ps
#define ml_unaligned_store_128 _mm_storeu_ps
#define ml_minimum_128 _mm_min_ps
#define ml_subtract_128 _mm_sub_ps
#define ml_multiply_128 _mm_mul_ps
#define ml_shuffle_mask(w, x, y, z) _MM_SHUFFLE(w, x, y, z)
#define ml_shuffle_128 _mm_shuffle_ps
#define ml_rotate_128(x) _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 3, 2, 1))

#elif defined(__ARM_NEON__)

#include <type_traits>

#include <arm_neon.h>

// clang-format off
#define ml_unaligned_load_128(x) vld1q_f32(x)
#define ml_unaligned_store_128(x, y) vst1q_f32(x, y)
#define ml_minimum_128(x, y) vminq_f32(x, y)
#define ml_subtract_128(x, y) vsubq_f32(x, y)
#define ml_multiply_128(x, y) vmulq_f32(x, y)
#define ml_shuffle_mask(w, x, y, z)                                             \
    std::integral_constant<int, (((w) << 6) | ((x) << 4) | ((y) << 2) | (z))>{}
// clang-format on

template<typename MASK>
inline __attribute__((always_inline)) auto
ml_shuffle_128(float32x4_t a, float32x4_t b, MASK) {
    float32x4_t result;
    result = vmovq_n_f32(vgetq_lane_f32(a, MASK::value & 0x3));
    result = vsetq_lane_f32(vgetq_lane_f32(a, (MASK::value >> 2) & 0x3), result, 1);
    result = vsetq_lane_f32(vgetq_lane_f32(b, (MASK::value >> 4) & 0x3), result, 2);
    result = vsetq_lane_f32(vgetq_lane_f32(b, (MASK::value >> 6) & 0x3), result, 3);
    return result;
}

inline __attribute__((always_inline)) auto ml_rotate_128(float32x4_t x) {
    float32x2_t x21{vget_high_f32(vextq_f32(x, x, 3))};
    float32x2_t x03{vget_low_f32(vextq_f32(x, x, 3))};
    return vcombine_f32(x21, x03);
}

#else

// clang-format off
#define ml_unaligned_load_128(x)                                               \
    std::array<float, 4>{*(x), *((x) + 1), *((x) + 2), *((x) + 3)};
#define ml_unaligned_store_128(x, y)                                           \
    *(x)       = (y)[0];                                                       \
    *((x) + 1) = (y)[1];                                                       \
    *((x) + 2) = (y)[2];                                                       \
    *((x) + 3) = (y)[3]
#define ml_minimum_128(x, y)                                                   \
    std::array<float, 4>{                                                      \
        std::min((x)[0], (y)[0]), std::min((x)[1], (y)[1]),                    \
        std::min((x)[2], (y)[2]), std::min((x)[3], (y)[3])                     \
    }
#define ml_subtract_128(x, y)                                                  \
    std::array<float, 4>{                                                      \
        (x)[0] - (y)[0], (x)[1] - (y)[1], (x)[2] - (y)[2], (x)[3] - (y)[3]     \
    }
#define ml_multiply_128(x, y)                                                  \
    std::array<float, 4>{                                                      \
        (x)[0] * (y)[0], (x)[1] * (y)[1],                                      \
        (x)[2] * (y)[2], (x)[3] * (y)[3]                                       \
    }
#define ml_shuffle_mask(w, x, y, z) (((w) << 6) | ((x) << 4) | ((y) << 2) | (z))
#define ml_shuffle_128(x, y, mask)                                             \
    std::array<float, 4>{                                                      \
        (x)[(mask) & 0x3], (x)[((mask) >> 2) & 0x3],                           \
        (y)[((mask) >> 4) & 0x3], (y)[((mask) >> 6) & 0x3]                     \
    }
#define ml_rotate_128(x)                                                       \
    std::array<float, 4>{(x)[1], (x)[2], (x)[3], (x)[0]};
// clang-format on

#endif

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

std::size_t fastSketchSize(double reductionFactor, std::size_t size) {
    size = static_cast<std::size_t>(
        static_cast<double>(size) * CQuantileSketch::REDUCTION_FACTOR / reductionFactor + 0.5);
    return size + (3 - (size + 1) % 3) % 3;
}

const auto EPS = static_cast<double>(std::numeric_limits<float>::epsilon());
const core::TPersistenceTag UNSORTED_TAG{"a", "unsorted"};
const core::TPersistenceTag KNOTS_TAG{"b", "knots"};
const core::TPersistenceTag COUNT_TAG{"c", "count"};
}

CQuantileSketch::CQuantileSketch(const TFloatVec& centres, const TFloatVec& counts)
    : m_MaxSize{std::max(2 * centres.size(), MINIMUM_MAX_SIZE)},
      m_Count{std::accumulate(counts.begin(), counts.end(), 0.0)} {
    m_Knots.reserve(centres.size());
    for (std::size_t i = 0; i < centres.size(); ++i) {
        m_Knots.emplace_back(centres[i], counts[i]);
    }
}

CQuantileSketch::CQuantileSketch(std::size_t size)
    : m_MaxSize(std::max(size, MINIMUM_MAX_SIZE)) {
    m_Knots.reserve(m_MaxSize + 1);
}

CQuantileSketch::~CQuantileSketch() = default;

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
    m_Unsorted = m_Knots.size();
    m_Count += rhs.m_Count;
    this->reduce(m_MaxSize + 1);
    m_Knots.shrink_to_fit();
    LOG_TRACE(<< "knots = " << m_Knots);
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

bool CQuantileSketch::cdf(double x_, double& result, TOptionalInterpolation interpolation) const {
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

    // This must make the same assumptions as quantile regarding the distribution
    // of values for each histogram bucket. See that function for more details.

    switch (interpolation.value_or(this->cdfAndQuantileInterpolation())) {
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
            partial = (left ? (partial + (loc ? 1.0 * m_Knots[k - 1].second : 0.0))
                            : (partial + (loc ? 0.0 : 1.0 * m_Knots[k].second))) /
                      m_Count;
            double dn{0.5 * m_Knots[loc ? k - 1 : k].second / m_Count};
            LOG_TRACE(<< "left = " << left << ", loc = " << loc << ", partial = " << partial
                      << ", xl = " << xl << ", xr = " << xr << ", dn = " << dn);
            result = left ? partial + dn * (2.0 * x - xl - xr) / (xr - xl)
                          : 1.0 - partial - dn * (xl + xr - 2.0 * x) / (xr - xl);
            LOG_TRACE(<< "result = " << result << " "
                      << dn * (2.0 * x - xl - xr) / (xr - xl));
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

bool CQuantileSketch::quantile(double percentage,
                               double& result,
                               TOptionalInterpolation interpolation) const {
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

    quantile(interpolation.value_or(this->cdfAndQuantileInterpolation()),
             m_Knots, m_Count, percentage, result);

    return true;
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
    core::memory_debug::dynamicSize("m_Knots", m_Knots, mem);
}

std::size_t CQuantileSketch::memoryUsage() const {
    return core::memory::dynamicSize(m_Knots);
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
    for (std::size_t i = 1; i + m_Unsorted < m_Knots.size(); ++i) {
        if (m_Knots[i].first == m_Knots[i - 1].first) {
            LOG_ERROR(<< "Duplicate values: " << m_Knots[i - 1] << " and " << m_Knots[i]);
            return false;
        }
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

    // For linear interpolation we have to make some assumptions about how the
    // merged bucket values are distributed.
    //
    //  We make the following assumptions:
    //   1. The bucket centre bisects (in weight) the values it represents,
    //   2. The bucket start and end point are the midpoints between the bucket
    //      centre and the preceding and succeeding bucket centres, respectively,
    //   3. Values are uniformly distributed on the intervals between the bucket
    //      endpoints and its centre.
    //
    // Assumption 1 is consistent with how bucket centres are computed on merge
    // and assumption 2 is consistent with how we decide to merge buckets. This
    // scheme also has the highly desireable property that if the sketch contains
    // the raw data, i.e. that the number of distinct values is less than the
    // sketch size, it computes quantiles exactly.

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
                    double xa = i == 0 ? 2.0 * knots[0].first - knots[1].first
                                       : static_cast<double>(knots[i - 1].first);
                    double xb = knots[i].first;
                    double xc = i + 1 == n ? 2.0 * xb - xa
                                           : static_cast<double>(knots[i + 1].first);
                    double nb = knots[i].second;
                    partial -= 0.5 * nb;
                    result = xb + (cutoff - partial) * (cutoff - partial < 0.0
                                                            ? (xb - xa) / nb
                                                            : (xc - xb) / nb);
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

    result = knots[n - 1].first;
}

CQuantileSketch::EInterpolation CQuantileSketch::cdfAndQuantileInterpolation() const {
    // If the number of knots is less than the target size for reduce we must
    // never have combined any distinct values into a single bucket and the
    // quantile and empircal cdf are computed exactly using piecewise constant
    // interpolation.
    return m_Knots.size() < this->fastReduceTargetSize() ? E_PiecewiseConstant : E_Linear;
}

std::size_t CQuantileSketch::fastReduceTargetSize() const {
    return static_cast<std::size_t>(REDUCTION_FACTOR * static_cast<double>(m_MaxSize) + 1.0);
}

void CQuantileSketch::fastReduce() {
    this->reduce(this->fastReduceTargetSize());
}

void CQuantileSketch::reduce(std::size_t target) {

    this->order();
    this->deduplicate(m_Knots.begin(), m_Knots.end());

    if (m_Knots.size() > target) {
        TFloatFloatPrVec mergeCosts;
        TSizeVec mergeCandidates;
        TBoolVec stale(m_Knots.size(), false);
        mergeCosts.reserve(m_Knots.size());
        mergeCandidates.reserve(m_Knots.size());
        CPRNG::CXorOShiro128Plus rng(static_cast<std::uint64_t>(m_Count));
        std::uniform_real_distribution<double> u01{0.0, 1.0};
        for (std::size_t i = 0; i + 3 < m_Knots.size(); ++i) {
            mergeCosts.emplace_back(mergeCost(m_Knots[i + 1], m_Knots[i + 2]), u01(rng));
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
            auto mergedKnot = this->mergedKnot(l, r);

            // Find the points that have been merged with xl and xr if any.
            std::ptrdiff_t ll{previousDifferent(m_Knots, l)};
            std::ptrdiff_t rr{nextDifferent(m_Knots, r) - 1};
            std::fill_n(m_Knots.begin() + ll + 1, rr - ll, mergedKnot);
            stale[ll] = true;
            stale[rr] = true;
            --numberToMerge;
            LOG_TRACE(<< "merged = " << mergedKnot);
            LOG_TRACE(<< "right  = " << m_Knots[rr + 1]);
        } else {
            CUniqueIterator ll(m_Knots, l);
            CUniqueIterator rr{ll};
            ++rr;
            mergeCosts[l - 1].first = mergeCost(*ll, *rr);
            mergeCandidates.push_back(l - 1);
            std::push_heap(mergeCandidates.begin(), mergeCandidates.end(), mergeCostGreater);
            stale[l] = false;
        }
    }

    m_Knots.erase(std::unique(m_Knots.begin(), m_Knots.end()), m_Knots.end());
    LOG_TRACE(<< "final = " << m_Knots);
}

void CQuantileSketch::order() {
    if (m_Unsorted > 0) {
        std::sort(m_Knots.end() - m_Unsorted, m_Knots.end(), COrderings::SFirstLess());

        // Deduplicate points before merging.
        std::size_t removed{m_Knots.size()};
        this->deduplicate(m_Knots.end() - m_Unsorted, m_Knots.end());
        removed -= m_Knots.size();
        m_Unsorted -= removed;

        std::inplace_merge(m_Knots.begin(), m_Knots.end() - m_Unsorted,
                           m_Knots.end(), COrderings::SFirstLess());
    }
    m_Unsorted = 0;
}

void CQuantileSketch::deduplicate(TFloatFloatPrVecItr begin, TFloatFloatPrVecItr end) {
    // Deduplicate new values.
    for (auto i = begin + 1; i <= end; ++i, ++begin) {
        if (begin != i - 1) {
            *begin = *(i - 1);
        }
        CFloatStorage x{begin->first};
        for (/**/; i != end && i->first == x; ++i) {
            begin->second += i->second;
        }
    }
    m_Knots.erase(begin, end);
    LOG_TRACE(<< "de-duplicated = " << m_Knots);
}

CQuantileSketch::TFloatFloatPr CQuantileSketch::mergedKnot(std::size_t l, std::size_t r) const {
    auto[xl, nl] = m_Knots[l];
    auto[xr, nr] = m_Knots[r];
    return {(nl * xl + nr * xr) / (nl + nr), nl + nr};
}

double CQuantileSketch::mergeCost(const TFloatFloatPr& l, const TFloatFloatPr& r) {
    // Interestingly, minimizing the approximation error (area between
    // curve before and after merging) produces good summary for the
    // piecewise constant objective, but a very bad summary for the linear
    // objective. Basically, an empirically good strategy is to target
    // the piecewise objective when sketching and then perform unbiased
    // linear interpolation by linearly interpolating between the mid-
    // points of the steps when computing quantiles.
    auto[xl, nl] = l;
    auto[xr, nr] = r;
    return std::min(nl, nr) * (xr - xl);
}

CFastQuantileSketch::CFastQuantileSketch(std::size_t size,
                                         CPRNG::CXorOShiro128Plus rng,
                                         TOptionalDouble reductionFraction)
    : CQuantileSketch{fastSketchSize(reductionFraction.value_or(REDUCTION_FACTOR), size)},
      m_ReductionFactor{reductionFraction.value_or(REDUCTION_FACTOR)} {
    size = this->maxSize();
    LOG_TRACE(<< "size = " << size);
    m_Tiebreakers.resize(size + 1);
    m_MergeCosts.reserve(size - 1);
    m_MergeCandidates.reserve(size - 2);
    std::uniform_real_distribution<double> u01{0.0, 1.0};
    std::generate_n(m_Tiebreakers.begin(), size + 1, [&] { return u01(rng); });
}

std::uint64_t CFastQuantileSketch::checksum(std::uint64_t seed) const {
    seed = this->CQuantileSketch::checksum(seed);
    return CChecksum::calculate(seed, m_ReductionFactor);
}

void CFastQuantileSketch::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CFastQuantileSketch");
    core::memory_debug::dynamicSize("m_Knots", this->knots(), mem);
    core::memory_debug::dynamicSize("m_Tiebreakers", m_Tiebreakers, mem);
    core::memory_debug::dynamicSize("m_MergeCosts", m_MergeCosts, mem);
    core::memory_debug::dynamicSize("m_MergeCandidates", m_MergeCandidates, mem);
}

std::size_t CFastQuantileSketch::memoryUsage() const {
    std::size_t mem{this->CQuantileSketch::memoryUsage()};
    mem += core::memory::dynamicSize(m_MergeCosts);
    mem += core::memory::dynamicSize(m_Tiebreakers);
    mem += core::memory::dynamicSize(m_MergeCandidates);
    return mem;
}

std::size_t CFastQuantileSketch::staticSize() const {
    return sizeof(*this);
}

std::size_t CFastQuantileSketch::fastReduceTargetSize() const {
    return static_cast<std::size_t>(
        m_ReductionFactor * static_cast<double>(this->maxSize()) + 1.0);
}

void CFastQuantileSketch::fastReduce() {

    this->order();

    if (this->knots().size() > this->fastReduceTargetSize()) {
        TFloatFloatPrVec knots{std::move(this->knots())};
        std::size_t numberToMerge{knots.size() - this->fastReduceTargetSize()};

        this->computeMergeCosts(knots);
        this->computeMergeCandidates(numberToMerge);

        m_MergeCandidates[numberToMerge] = static_cast<std::uint32_t>(knots.size() - 1);
        LOG_TRACE(<< "merge candidates = " << m_MergeCandidates);

        LOG_TRACE(<< "knots before merge = " << knots);

        std::uint32_t back{m_MergeCandidates[0] + 1};
        std::uint32_t next{back};
        for (std::size_t i = 1; i <= numberToMerge; ++i) {
            std::uint32_t l{next};
            std::uint32_t r{l + 1};
            for (next = m_MergeCandidates[i] + 1; next == r;
                 next = m_MergeCandidates[++i] + 1, ++r) {
            }
            LOG_TRACE(<< "left = " << l << ", right = " << r << ", next = " << next);

            CFloatStorage centre{0.0};
            CFloatStorage count{0.0};
            for (std::uint32_t j = l; j <= r; ++j) {
                auto[x, n] = knots[j];
                centre += n * x;
                count += n;
            }
            centre /= count;
            knots[back] = TFloatFloatPr{centre, count};
            LOG_TRACE(<< "merged knot = " << knots[back]);

            for (std::size_t j = back + 1, k = r + 1; k < next; ++j, ++k) {
                knots[j] = knots[k];
            }
            back += next - r;
            LOG_TRACE(<< "knots = " << knots << ", back = " << back);
        }

        knots.resize(back);
        LOG_TRACE(<< "knots after merge = " << knots);

        this->knots() = std::move(knots);
    }
}

void CFastQuantileSketch::computeMergeCosts(TFloatFloatPrVec& knots) {
    // This is equivalent to:
    //
    // for (std::size_t i = 0; i + 3 < knots.size(); ++i) {
    //     m_MergeCosts[i] = mergeCost(knots[i + 1], knots[i + 2]);
    // }
    //
    // If we let the compiler do its thing on this it works out about 3X slower
    // than the following vectorised version. Note that the latency of the loads
    // are the dominant cost so we choose to compute three values for two loads
    // rather than four values for three loads.

    // The following loop requires that knots size is a multiple of three.
    std::size_t n{knots.size()};
    knots.resize(3 * ((knots.size() + 2) / 3));

    m_MergeCosts.resize(knots.size() - 2);

    for (std::size_t i = 0; i + 3 < knots.size(); i += 3) {
        auto knots12 = ml_unaligned_load_128(&knots[i + 1].first.cstorage());
        auto knots34 = ml_unaligned_load_128(&knots[i + 3].first.cstorage());
        auto centres1 = ml_shuffle_128(knots12, knots34, ml_shuffle_mask(2, 0, 2, 0));
        auto counts1 = ml_shuffle_128(knots12, knots34, ml_shuffle_mask(3, 1, 3, 1));
        auto centres2 = ml_rotate_128(centres1);
        auto counts2 = ml_rotate_128(counts1);
        auto countsMin = ml_minimum_128(counts1, counts2);
        auto centresDiff = ml_subtract_128(centres2, centres1);
        auto costs = ml_multiply_128(countsMin, centresDiff);
        ml_unaligned_store_128(&m_MergeCosts[i].storage(), costs);
    }
    m_MergeCosts.resize(n - 3);
    knots.resize(n);

    LOG_TRACE(<< "merge costs = " << m_MergeCosts << ", merge candidates = " << m_MergeCandidates);
}

void CFastQuantileSketch::computeMergeCandidates(std::size_t numberToMerge) {
    // This is pseudo greedy unlike CQuantileSketch which is greedy: we
    // simply ignore the fact that we have slightly different costs after
    // each merge. This allows us to avoid using a heap altogether which
    // dominates the computational cost of reduce.

    m_MergeCandidates.resize(m_MergeCosts.size());
    std::iota(m_MergeCandidates.begin(), m_MergeCandidates.end(), 0);
    std::nth_element(m_MergeCandidates.begin(), m_MergeCandidates.begin() + numberToMerge,
                     m_MergeCandidates.end(), [&](auto lhs, auto rhs) {
                         return m_MergeCosts[lhs] == m_MergeCosts[rhs]
                                    ? m_Tiebreakers[lhs] < m_Tiebreakers[rhs]
                                    : m_MergeCosts[lhs] < m_MergeCosts[rhs];
                     });
    std::sort(m_MergeCandidates.begin(), m_MergeCandidates.begin() + numberToMerge);
}
}
}
}
