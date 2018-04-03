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

#ifndef INCLUDED_ml_maths_CSetTools_h
#define INCLUDED_ml_maths_CSetTools_h

#include <maths/ImportExport.h>

#include <boost/variant.hpp>

#include <cstddef>
#include <set>
#include <vector>

namespace ml
{
namespace maths
{

//! \brief Collection of set utility functions not provided by the STL.
//!
//! DESCRIPTION:\n
//! This implements various set related functionality which can't be
//! implemented efficiently using the STL. For example, a function for
//! computing set difference in-place, and functions for counting
//! elements in set differences and unions. Common measures of set
//! similarity such as the Jaccard index are also implemented.
class MATHS_EXPORT CSetTools
{
    public:
        //! \brief Checks if an indexed object is in a specified collection
        //! of indices.
        class CIndexInSet
        {
            public:
                using TSizeSet = std::set<std::size_t>;

            public:
                CIndexInSet(std::size_t index) : m_IndexSet(index) {}
                CIndexInSet(const TSizeSet &indexSet) : m_IndexSet(indexSet) {}

                template<typename T>
                bool operator()(const T &indexedObject) const
                {
                    const std::size_t *index = boost::get<std::size_t>(&m_IndexSet);
                    if (index)
                    {
                        return indexedObject.s_Index == *index;
                    }
                    const TSizeSet &indexSet = boost::get<TSizeSet>(m_IndexSet);
                    return indexSet.count(indexedObject.s_Index) > 0;
                }

            private:
                using TSizeOrSizeSet = boost::variant<std::size_t, TSizeSet>;

            private:
                TSizeOrSizeSet m_IndexSet;
        };

        //! Compute the difference between \p S and [\p begin, \p end).
        template<typename T, typename ITR>
        static void inplace_set_difference(std::vector<T> &S, ITR begin, ITR end)
        {
            typename std::vector<T>::iterator i = S.begin(), last = i;
            for (ITR j = begin; i != S.end() && j != end; /**/)
            {
                if (*i < *j)
                {
                    if (last != i)
                    {
                        std::iter_swap(last, i);
                    }
                    ++i; ++last;
                }
                else if (*j < *i)
                {
                    ++j;
                }
                else
                {
                    ++i; ++j;
                }
            }
            if (last != i)
            {
                S.erase(std::swap_ranges(i, S.end(), last), S.end());
            }
        }

#define SIMULTANEOUS_REMOVE_IF_IMPL using std::swap; \
        std::size_t last{0u};                        \
        std::size_t n{values1.size()};               \
        for (std::size_t i = 0u; i < n; ++i)         \
        {                                            \
            if (last != i)                           \
            {                                        \
                CUSTOM_SWAP_VALUES                   \
            }                                        \
            if (!pred(values1[last]))                \
            {                                        \
                ++last;                              \
            }                                        \
        }                                            \
        if (last < n)                                \
        {                                            \
            CUSTOM_ERASE_VALUES                      \
            return true;                             \
        }                                            \
        return false;

#define CUSTOM_SWAP_VALUES swap(values1[i], values1[last]); \
                           swap(values2[i], values2[last]);
#define CUSTOM_ERASE_VALUES values1.erase(values1.begin() + last, values1.end()); \
                            values2.erase(values2.begin() + last, values2.end());

        //! Remove all instances of \p values1 for which \p pred is true
        //! and corresponding values of \p values2.
        template<typename T1, typename T2, typename F>
        static bool simultaneousRemoveIf(std::vector<T1> &values1,
                                         std::vector<T2> &values2,
                                         const F &pred)
        {
            if (values1.size() != values2.size())
            {
                return false;
            }

            SIMULTANEOUS_REMOVE_IF_IMPL
        }

#undef CUSTOM_SWAP_VALUES
#undef CUSTOM_ERASE_VALUES

#define CUSTOM_SWAP_VALUES swap(values1[i], values1[last]); \
                           swap(values2[i], values2[last]); \
                           swap(values3[i], values3[last]);
#define CUSTOM_ERASE_VALUES values1.erase(values1.begin() + last, values1.end()); \
                            values2.erase(values2.begin() + last, values2.end()); \
                            values3.erase(values3.begin() + last, values3.end());

        //! Remove all instances of \p values1 for which \p pred is true
        //! and corresponding values of \p values2 and \p values3.
        template<typename T1, typename T2, typename T3, typename F>
        static bool simultaneousRemoveIf(std::vector<T1> &values1,
                                         std::vector<T2> &values2,
                                         std::vector<T3> &values3,
                                         const F &pred)
        {
            if (   values1.size() != values2.size()
                || values2.size() != values3.size())
            {
                return false;
            }

            SIMULTANEOUS_REMOVE_IF_IMPL
        }

#undef CUSTOM_SWAP_VALUES
#undef CUSTOM_ERASE_VALUES
#undef SIMULTANEOUS_REMOVE_IF_IMPL

        //! Compute the number of elements in the intersection of the
        //! ranges [\p beginLhs, \p endLhs) and [\p beginRhs, \p endRhs).
        template<typename ITR1, typename ITR2>
        static std::size_t setIntersectSize(ITR1 beginLhs, ITR1 endLhs, ITR2 beginRhs, ITR2 endRhs)
        {
            std::size_t result = 0u;
            while (beginLhs != endLhs && beginRhs != endRhs)
            {
                if (*beginLhs < *beginRhs)
                {
                    ++beginLhs;
                }
                else if (*beginRhs < *beginLhs)
                {
                    ++beginRhs;
                }
                else
                {
                    ++beginLhs; ++beginRhs; ++result;
                }
            }
            return result;
        }

        //! Compute the number of elements in the union of the ranges
        //! [\p beginLhs, \p endLhs) and [\p beginRhs, \p endRhs).
        template<typename ITR1, typename ITR2>
        static std::size_t setUnionSize(ITR1 beginLhs, ITR1 endLhs, ITR2 beginRhs, ITR2 endRhs)
        {
            std::size_t result = 0u;
            while (beginLhs != endLhs && beginRhs != endRhs)
            {
                if (*beginLhs < *beginRhs)
                {
                    ++beginLhs;
                }
                else if (*beginRhs < *beginLhs)
                {
                    ++beginRhs;
                }
                else
                {
                    ++beginLhs; ++beginRhs;
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
        static double jaccard(ITR1 beginLhs, ITR1 endLhs, ITR2 beginRhs, ITR2 endRhs)
        {
            std::size_t numer = 0u;
            std::size_t denom = 0u;
            while (beginLhs != endLhs && beginRhs != endRhs)
            {
                if (*beginLhs < *beginRhs)
                {
                    ++beginLhs;
                }
                else if (*beginRhs < *beginLhs)
                {
                    ++beginRhs;
                }
                else
                {
                    ++beginLhs; ++beginRhs; ++numer;
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
        static double overlap(ITR1 beginLhs, ITR1 endLhs, ITR2 beginRhs, ITR2 endRhs)
        {
            std::size_t numer = 0u;
            std::size_t nl = 0u;
            std::size_t nr = 0u;
            while (beginLhs != endLhs && beginRhs != endRhs)
            {
                if (*beginLhs < *beginRhs)
                {
                    ++beginLhs; ++nl;
                }
                else if (*beginRhs < *beginLhs)
                {
                    ++beginRhs; ++nr;
                }
                else
                {
                    ++beginLhs; ++beginRhs; ++numer; ++nl; ++nr;
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

#endif // INCLUDED_ml_maths_CSetTools_h
