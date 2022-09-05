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

#ifndef INCLUDED_ml_maths_common_CSetTools_h
#define INCLUDED_ml_maths_common_CSetTools_h

#include <maths/common/ImportExport.h>

#include <cstddef>
#include <set>
#include <variant>
#include <vector>

namespace ml {
namespace maths {
namespace common {
//! \brief Collection of set utility functions not provided by the STL.
//!
//! DESCRIPTION:\n
//! This implements various set related functionality which can't be
//! implemented efficiently using the STL. For example, a function for
//! computing set difference in-place, and functions for counting
//! elements in set differences and unions. Common measures of set
//! similarity such as the Jaccard index are also implemented.
class MATHS_COMMON_EXPORT CSetTools {
public:
    //! \brief Checks if an indexed object is in a specified collection
    //! of indices.
    class MATHS_COMMON_EXPORT CIndexInSet {
    public:
        using TSizeSet = std::set<std::size_t>;

    public:
        explicit CIndexInSet(std::size_t index) : m_IndexSet(index) {}
        explicit CIndexInSet(const TSizeSet& indexSet) : m_IndexSet(indexSet) {}

        template<typename T>
        bool operator()(const T& indexedObject) const {
            const std::size_t* index = std::get_if<std::size_t>(&m_IndexSet);
            if (index) {
                return indexedObject.s_Index == *index;
            }
            const auto& indexSet = std::get<TSizeSet>(m_IndexSet);
            return indexSet.count(indexedObject.s_Index) > 0;
        }

    private:
        using TSizeOrSizeSet = std::variant<std::size_t, TSizeSet>;

    private:
        TSizeOrSizeSet m_IndexSet;
    };

    //! Remove all instances of \p keys for which \p pred is true and
    //! corresponding values of \p others.
    //!
    //! \note All containers must have the same size.
    //! \tparam K must support size(), begin(), end() and erase().
    //! \tparam V each must support size(), begin(), end() and erase().
    //! \tparam PRED must support operator() taking key value type.
    template<typename PRED, typename K, typename... V>
    static bool simultaneousRemoveIf(PRED pred, K& keys, V&... others) {
        if (((others.size() != keys.size()) && ...)) {
            return false;
        }
        auto removeIf = [&](auto& other) {
            auto i = other.begin();
            auto j = keys.begin();
            for (auto k = other.begin(); k != other.end(); ++j, (void)++k) {
                if (i != k) {
                    *i = std::move(*k);
                }
                if (pred(*j) == false) {
                    ++i;
                }
            }
            other.erase(i, other.end());
        };
        (removeIf(others), ...);
        keys.erase(std::remove_if(keys.begin(), keys.end(), pred), keys.end());
        return true;
    }

    //! Compute the number of elements in the intersection of the
    //! ranges [\p beginLhs, \p endLhs) and [\p beginRhs, \p endRhs).
    template<typename ITR1, typename ITR2>
    static std::size_t
    setIntersectSize(ITR1 beginLhs, ITR1 endLhs, ITR2 beginRhs, ITR2 endRhs) {
        std::size_t result = 0;
        while (beginLhs != endLhs && beginRhs != endRhs) {
            if (*beginLhs < *beginRhs) {
                ++beginLhs;
            } else if (*beginRhs < *beginLhs) {
                ++beginRhs;
            } else {
                ++beginLhs;
                ++beginRhs;
                ++result;
            }
        }
        return result;
    }

    //! Compute the number of elements in the union of the ranges
    //! [\p beginLhs, \p endLhs) and [\p beginRhs, \p endRhs).
    template<typename ITR1, typename ITR2>
    static std::size_t setUnionSize(ITR1 beginLhs, ITR1 endLhs, ITR2 beginRhs, ITR2 endRhs) {
        std::size_t result = 0;
        while (beginLhs != endLhs && beginRhs != endRhs) {
            if (*beginLhs < *beginRhs) {
                ++beginLhs;
            } else if (*beginRhs < *beginLhs) {
                ++beginRhs;
            } else {
                ++beginLhs;
                ++beginRhs;
            }
            ++result;
        }
        return result + std::distance(beginLhs, endLhs) + std::distance(beginRhs, endRhs);
    }

    //! Compute the Jaccard index of the elements of the ranges
    //! [\p beginLhs, \p endLhs) and [\p beginRhs, \p endRhs).
    //!
    //! This is defined as \f$\frac{|A\cap B|}{|A\cup B|}\f$.
    template<typename ITR1, typename ITR2>
    static double jaccard(ITR1 beginLhs, ITR1 endLhs, ITR2 beginRhs, ITR2 endRhs) {
        std::size_t numer = 0;
        std::size_t denom = 0;
        while (beginLhs != endLhs && beginRhs != endRhs) {
            if (*beginLhs < *beginRhs) {
                ++beginLhs;
            } else if (*beginRhs < *beginLhs) {
                ++beginRhs;
            } else {
                ++beginLhs;
                ++beginRhs;
                ++numer;
            }
            ++denom;
        }
        denom += std::distance(beginLhs, endLhs) + std::distance(beginRhs, endRhs);
        return denom == 0 ? 0.0 : static_cast<double>(numer) / static_cast<double>(denom);
    }

    //! Compute the overlap coefficient (or, Szymkiewicz-Simpson
    //! coefficient) of the elements of the ranges
    //! [\p beginLhs, \p endLhs) and [\p beginRhs, \p endRhs).
    //!
    //! This is defined as \f$\frac{|A\cap B|}{\min(|A|,|B|)}\f$.
    template<typename ITR1, typename ITR2>
    static double overlap(ITR1 beginLhs, ITR1 endLhs, ITR2 beginRhs, ITR2 endRhs) {
        std::size_t numer = 0;
        std::size_t nl = 0;
        std::size_t nr = 0;
        while (beginLhs != endLhs && beginRhs != endRhs) {
            if (*beginLhs < *beginRhs) {
                ++beginLhs;
                ++nl;
            } else if (*beginRhs < *beginLhs) {
                ++beginRhs;
                ++nr;
            } else {
                ++beginLhs;
                ++beginRhs;
                ++numer;
                ++nl;
                ++nr;
            }
        }
        nl += std::distance(beginLhs, endLhs);
        nr += std::distance(beginRhs, endRhs);
        double denom = static_cast<double>(std::min(nl, nr));
        return denom == 0 ? 0.0 : static_cast<double>(numer) / static_cast<double>(denom);
    }
};
}
}
}

#endif // INCLUDED_ml_maths_common_CSetTools_h
