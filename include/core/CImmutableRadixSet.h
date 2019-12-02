/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CImmutableRadixSet_h
#define INCLUDED_ml_core_CImmutableRadixSet_h

#include <core/CContainerPrinter.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

namespace ml {
namespace core {

//! \brief An immutable sorted set which provides very fast lookup.
//!
//! DESCRIPTION:\n
//! This supports lower bound and look up by index as well as a subset of the non
//! modifying interface of std::set. Its main purpose is to provide much faster
//! lookup. To this end it subdivides the range of sorted values into buckets.
//! In the case that the values are uniformly distributed lowerBound will be O(1)
//! with low constant. Otherwise, it is worst case O(log(n)).
template<typename T>
class CImmutableRadixSet {
public:
    using TVec = std::vector<T>;
    using TCItr = typename std::vector<T>::const_iterator;

public:
    // We only need to support floating point types at present (although it
    // could easily extended to support any numeric type).
    static_assert(std::is_floating_point<T>::value, "Only supports floating point types");

public:
    CImmutableRadixSet() = default;
    explicit CImmutableRadixSet(std::initializer_list<T> values)
        : m_Values{std::move(values)} {
        this->initialize();
    }
    explicit CImmutableRadixSet(TVec values) : m_Values{std::move(values)} {
        this->initialize();
    }

    // This is movable only because we hold iterators to the underlying container.
    CImmutableRadixSet(const CImmutableRadixSet&) = delete;
    CImmutableRadixSet& operator=(const CImmutableRadixSet&) = delete;
    CImmutableRadixSet(CImmutableRadixSet&&) = default;
    CImmutableRadixSet& operator=(CImmutableRadixSet&&) = default;

    //! \name Capacity
    //@{
    bool empty() const { return m_Values.size(); }
    std::size_t size() const { return m_Values.size(); }
    //@}

    //! \name Iterators
    //@{
    TCItr begin() const { m_Values.begin(); }
    TCItr end() const { m_Values.end(); }
    //@}

    //! \name Lookup
    //@{
    const T& operator[](std::size_t i) const { return m_Values[i]; }
    std::ptrdiff_t upperBound(const T& value) const {
        // This branch is predictable so essentially free.
        if (m_Values.size() < 2) {
            return std::distance(m_Values.begin(),
                                 std::upper_bound(m_Values.begin(), m_Values.end(), value));
        }

        std::ptrdiff_t bucket{static_cast<std::ptrdiff_t>(m_Scale * (value - m_Min))};
        if (bucket < 0) {
            return 0;
        }
        if (bucket >= static_cast<std::ptrdiff_t>(m_Buckets.size())) {
            return static_cast<std::ptrdiff_t>(m_Values.size());
        }
        TCItr beginBucket;
        TCItr endBucket;
        std::tie(beginBucket, endBucket) = m_Buckets[bucket];
        return std::distance(m_Values.begin(),
                             std::upper_bound(beginBucket, endBucket, value));
    }
    //@}

    std::string print() const {
        return core::CContainerPrinter::print(m_Values);
    }

private:
    using TCItrCItrPr = std::pair<TCItr, TCItr>;
    using TCItrCItrPrVec = std::vector<TCItrCItrPr>;
    using TPtrdiffVec = std::vector<std::ptrdiff_t>;

private:
    void initialize() {
        std::sort(m_Values.begin(), m_Values.end());
        m_Values.erase(std::unique(m_Values.begin(), m_Values.end()), m_Values.end());
        if (m_Values.size() > 1) {
            std::size_t numberBuckets{m_Values.size()};
            m_Min = m_Values[0];
            m_Scale = static_cast<T>(numberBuckets) / (m_Values.back() - m_Min);
            m_Buckets.reserve(numberBuckets);
            T bucket{1};
            T bucketClose{m_Min + bucket / m_Scale};
            auto start = m_Values.begin();
            for (auto i = m_Values.begin(); i != m_Values.end(); ++i) {
                if (*i > bucketClose) {
                    m_Buckets.emplace_back(start, i);
                    bucket += T{1};
                    bucketClose = m_Min + bucket / m_Scale;
                    start = i;
                    while (*i > bucketClose) {
                        m_Buckets.emplace_back(start, i + 1);
                        bucket += T{1};
                        bucketClose = m_Min + bucket / m_Scale;
                    }
                }
            }
            if (m_Buckets.size() < numberBuckets) {
                m_Buckets.emplace_back(start, m_Values.end());
            }
        }
    }

private:
    T m_Min = T{0};
    T m_Scale = T{0};
    TCItrCItrPrVec m_Buckets;
    TVec m_Values;
};
}
}

#endif // INCLUDED_ml_core_CImmutableRadixSet_h
