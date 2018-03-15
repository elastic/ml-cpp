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

#ifndef INCLUDED_ml_maths_COrderings_h
#define INCLUDED_ml_maths_COrderings_h

#include <core/CNonInstantiatable.h>
#include <core/CStoredStringPtr.h>
#include <core/CVectorRange.h>

#include <boost/algorithm/cxx11/is_sorted.hpp>
#include <boost/optional/optional_fwd.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

//! \brief A collection of useful functionality to order collections
//! of objects.
//!
//! DESCRIPTION:\n
//! This implements some generic commonly occurring ordering functionality.
//! In particular,
//!   -# Lexicographical compare for small collections of objects with
//!      distinct types (std::lexicographical_compare only supports a
//!      single type).
//!   -# Fast (partial) orderings on pairs which only compare the first
//!      or second element of the pair.
//!   -# Efficiently, O(N log(N)), simultaneously sorting multiple vectors
//!      using one of the vectors to provide the ordering.
class COrderings : private core::CNonInstantiatable {
public:
    //! \brief Orders two optional values such that non-null are
    //! less than null values.
    //! less than null values and otherwise compares using the type
    //! operator <.
    struct SOptionalLess {
        typedef bool result_type;

        //! \note U and V must be convertible to T or optional<T>
        //! for some type T and T must support operator <.
        template<typename U, typename V>
        inline bool operator()(const U& lhs, const V& rhs) const {
            return less(lhs, rhs);
        }

        template<typename T>
        static inline bool less(const boost::optional<T>& lhs, const boost::optional<T>& rhs) {
            bool lInitialized(lhs);
            bool rInitialized(rhs);
            return lInitialized && rInitialized ? boost::unwrap_ref(*lhs) < boost::unwrap_ref(*rhs)
                                                : rInitialized < lInitialized;
        }
        template<typename T>
        static inline bool less(const T& lhs, const boost::optional<T>& rhs) {
            return !rhs ? true : boost::unwrap_ref(lhs) < boost::unwrap_ref(*rhs);
        }
        template<typename T>
        static inline bool less(const boost::optional<T>& lhs, const T& rhs) {
            return !lhs ? false : boost::unwrap_ref(*lhs) < boost::unwrap_ref(rhs);
        }
    };

    //! \brief Orders two optional values such that null are greater
    //! than non-null values and otherwise compares using the type
    //! operator >.
    struct SOptionalGreater {
        typedef bool result_type;

        //! \note U and V must be convertible to T or optional<T>
        //! for some type T and T must support operator >.
        template<typename U, typename V>
        inline bool operator()(const U& lhs, const V& rhs) const {
            return greater(lhs, rhs);
        }

        template<typename T>
        static inline bool greater(const boost::optional<T>& lhs, const boost::optional<T>& rhs) {
            bool lInitialized(lhs);
            bool rInitialized(rhs);
            return lInitialized && rInitialized ? boost::unwrap_ref(*lhs) > boost::unwrap_ref(*rhs)
                                                : rInitialized > lInitialized;
        }
        template<typename T>
        static inline bool greater(const T& lhs, const boost::optional<T>& rhs) {
            return !rhs ? false : boost::unwrap_ref(lhs) > boost::unwrap_ref(*rhs);
        }
        template<typename T>
        static inline bool greater(const boost::optional<T>& lhs, const T& rhs) {
            return !lhs ? true : boost::unwrap_ref(*lhs) > boost::unwrap_ref(rhs);
        }
    };

    //! \brief Orders two pointers such that non-null are less
    //! than null values and otherwise compares using the type
    //! operator <.
    struct SPtrLess {
        typedef bool result_type;

        template<typename T>
        inline bool operator()(const T* lhs, const T* rhs) const {
            return less(lhs, rhs);
        }

        template<typename T>
        static inline bool less(const T* lhs, const T* rhs) {
            bool lInitialized(lhs != 0);
            bool rInitialized(rhs != 0);
            return lInitialized && rInitialized ? boost::unwrap_ref(*lhs) < boost::unwrap_ref(*rhs)
                                                : rInitialized < lInitialized;
        }
    };

    //! \brief Orders two pointers such that null are greater
    //! than non-null values and otherwise compares using
    //! the type operator >.
    struct SPtrGreater {
        typedef bool result_type;

        template<typename T>
        inline bool operator()(const T* lhs, const T* rhs) const {
            return greater(lhs, rhs);
        }

        template<typename T>
        static inline bool greater(const T* lhs, const T* rhs) {
            bool lInitialized(lhs != 0);
            bool rInitialized(rhs != 0);
            return lInitialized && rInitialized ? boost::unwrap_ref(*lhs) > boost::unwrap_ref(*rhs)
                                                : rInitialized > lInitialized;
        }
    };

    //! \brief Orders two reference wrapped objects which are
    //! comparable with operator <.
    struct SReferenceLess {
        typedef bool result_type;

        template<typename U, typename V>
        inline bool operator()(const U& lhs, const V& rhs) const {
            return less(lhs, rhs);
        }

        template<typename U, typename V>
        static inline bool less(const U& lhs, const V& rhs) {
            return boost::unwrap_ref(lhs) < boost::unwrap_ref(rhs);
        }
    };

    //! \brief Orders two reference wrapped objects which are
    //! comparable with operator >.
    struct SReferenceGreater {
        typedef bool result_type;

        template<typename U, typename V>
        inline bool operator()(const U& lhs, const V& rhs) const {
            return greater(lhs, rhs);
        }

        template<typename U, typename V>
        static inline bool greater(const U& lhs, const V& rhs) {
            return boost::unwrap_ref(lhs) > boost::unwrap_ref(rhs);
        }
    };

    //! \name Mixed Type Lexicographical Comparison
    //!
    //! This is equivalent to std::lexicographical_compare but allows
    //! for the type of each value in the collection to be different.
    //! Each type must define operator<.
    //@{
    //! Lexicographical comparison of \p l1 and \p r1.
    template<typename T1, typename COMP>
    static bool lexicographical_compare(const T1& l1, const T1& r1, COMP comp) {
        return comp(l1, r1);
    }
    template<typename T1>
    static bool lexicographical_compare(const T1& l1, const T1& r1) {
        return lexicographical_compare(l1, r1, SReferenceLess());
    }
#define COMPARE(l, r)                                                                              \
    if (comp(l, r)) {                                                                              \
        return true;                                                                               \
    } else if (comp(r, l)) {                                                                       \
        return false;                                                                              \
    }
    //! Lexicographical comparison of (\p l1, \p l2) and (\p r1, \p r2).
    template<typename T1, typename T2, typename COMP>
    static bool
    lexicographical_compare(const T1& l1, const T2& l2, const T1& r1, const T2& r2, COMP comp) {
        COMPARE(l1, r1);
        return comp(l2, r2);
    }
    template<typename T1, typename T2>
    static bool lexicographical_compare(const T1& l1, const T2& l2, const T1& r1, const T2& r2) {
        return lexicographical_compare(l1, l2, r1, r2, SReferenceLess());
    }
    //! Lexicographical comparison of (\p l1, \p l2, \p l3) and (\p r1, \p r2, \p r3).
    template<typename T1, typename T2, typename T3, typename COMP>
    static bool lexicographical_compare(const T1& l1,
                                        const T2& l2,
                                        const T3& l3,
                                        const T1& r1,
                                        const T2& r2,
                                        const T3& r3,
                                        COMP comp) {
        COMPARE(l1, r1);
        COMPARE(l2, r2);
        return comp(l3, r3);
    }
    template<typename T1, typename T2, typename T3>
    static bool lexicographical_compare(const T1& l1,
                                        const T2& l2,
                                        const T3& l3,
                                        const T1& r1,
                                        const T2& r2,
                                        const T3& r3) {
        return lexicographical_compare(l1, l2, l3, r1, r2, r3, SReferenceLess());
    }
    //! Lexicographical comparison of (\p l1, \p l2, \p l3, \p l4) and
    //! (\p r1, \p r2, \p r3, \p r4).
    template<typename T1, typename T2, typename T3, typename T4, typename COMP>
    static bool lexicographical_compare(const T1& l1,
                                        const T2& l2,
                                        const T3& l3,
                                        const T4& l4,
                                        const T1& r1,
                                        const T2& r2,
                                        const T3& r3,
                                        const T4& r4,
                                        COMP comp) {
        COMPARE(l1, r1);
        COMPARE(l2, r2);
        COMPARE(l3, r3);
        return comp(l4, r4);
    }
    template<typename T1, typename T2, typename T3, typename T4>
    static bool lexicographical_compare(const T1& l1,
                                        const T2& l2,
                                        const T3& l3,
                                        const T4& l4,
                                        const T1& r1,
                                        const T2& r2,
                                        const T3& r3,
                                        const T4& r4) {
        return lexicographical_compare(l1, l2, l3, l4, r1, r2, r3, r4, SReferenceLess());
    }
    //! Lexicographical comparison of (\p l1, \p l2, \p l3, \p l4, \p l5) and
    //! (\p r1, \p r2, \p r3, \p r4, \p r5).
    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename COMP>
    static bool lexicographical_compare(const T1& l1,
                                        const T2& l2,
                                        const T3& l3,
                                        const T4& l4,
                                        const T5& l5,
                                        const T1& r1,
                                        const T2& r2,
                                        const T3& r3,
                                        const T4& r4,
                                        const T5& r5,
                                        COMP comp) {
        COMPARE(l1, r1);
        COMPARE(l2, r2);
        COMPARE(l3, r3);
        COMPARE(l4, r4);
        return comp(l5, r5);
    }
    template<typename T1, typename T2, typename T3, typename T4, typename T5>
    static bool lexicographical_compare(const T1& l1,
                                        const T2& l2,
                                        const T3& l3,
                                        const T4& l4,
                                        const T5& l5,
                                        const T1& r1,
                                        const T2& r2,
                                        const T3& r3,
                                        const T4& r4,
                                        const T5& r5) {
        return lexicographical_compare(l1, l2, l3, l4, l5, r1, r2, r3, r4, r5, SReferenceLess());
    }
#undef COMPARE
    //@}

    //! \brief Wrapper around various less than comparisons.
    struct SLess {
        typedef bool result_type;

        template<typename T>
        bool operator()(const boost::optional<T>& lhs, const boost::optional<T>& rhs) const {
            return SOptionalLess::less(lhs, rhs);
        }

        template<typename T>
        bool operator()(const T* lhs, const T* rhs) const {
            return SPtrLess::less(lhs, rhs);
        }

        template<typename T>
        bool operator()(T* lhs, T* rhs) const {
            return SPtrLess::less(lhs, rhs);
        }

        template<typename U, typename V>
        bool operator()(const U& lhs, const V& rhs) const {
            return SReferenceLess::less(lhs, rhs);
        }

        bool operator()(const core::CStoredStringPtr& lhs, const core::CStoredStringPtr& rhs) {
            return SPtrLess::less(lhs.get(), rhs.get());
        }

        template<typename T>
        bool operator()(const boost::shared_ptr<T>& lhs, const boost::shared_ptr<T>& rhs) {
            return SPtrLess::less(lhs.get(), rhs.get());
        }

        template<typename U, typename V>
        bool operator()(const std::pair<U, V>& lhs, const std::pair<U, V>& rhs) const {
            return lexicographical_compare(lhs.first, lhs.second, rhs.first, rhs.second, *this);
        }
        SReferenceLess s_Less;
    };

    //! \brief Wrapper around various less than comparisons.
    struct SGreater {
        typedef bool result_type;

        template<typename T>
        bool operator()(const boost::optional<T>& lhs, const boost::optional<T>& rhs) const {
            return SOptionalGreater::greater(lhs, rhs);
        }

        template<typename T>
        bool operator()(const T* lhs, const T* rhs) const {
            return SPtrGreater::greater(lhs, rhs);
        }

        template<typename T>
        bool operator()(T* lhs, T* rhs) const {
            return SPtrGreater::greater(lhs, rhs);
        }

        template<typename U, typename V>
        bool operator()(const U& lhs, const V& rhs) const {
            return SReferenceGreater::greater(lhs, rhs);
        }

        bool operator()(const core::CStoredStringPtr& lhs, const core::CStoredStringPtr& rhs) {
            return SPtrGreater::greater(lhs.get(), rhs.get());
        }

        template<typename T>
        bool operator()(const boost::shared_ptr<T>& lhs, const boost::shared_ptr<T>& rhs) {
            return SPtrGreater::greater(lhs.get(), rhs.get());
        }

        template<typename U, typename V>
        bool operator()(const std::pair<U, V>& lhs, const std::pair<U, V>& rhs) const {
            return lexicographical_compare(lhs.first, lhs.second, rhs.first, rhs.second, *this);
        }

        SReferenceGreater s_Greater;
    };

    //! Lexicographical comparison of various common types.
    //!
    //! IMPLEMENTATION DECISIONS:\n
    //! Although these objects provide their own comparison operators
    //! This also tuples of handles reference wrapped types.
    struct SLexicographicalCompare {
        template<typename T1, typename T2>
        inline bool operator()(const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) const {
            return lexicographical_compare(lhs.first, lhs.second, rhs.first, rhs.second, s_Less);
        }

        template<typename T1, typename T2, typename T3>
        inline bool operator()(const boost::tuple<T1, T2, T3>& lhs,
                               const boost::tuple<T1, T2, T3>& rhs) const {
            return lexicographical_compare(lhs.template get<0>(),
                                           lhs.template get<1>(),
                                           lhs.template get<2>(),
                                           rhs.template get<0>(),
                                           rhs.template get<1>(),
                                           rhs.template get<2>(),
                                           s_Less);
        }

        template<typename T1, typename T2, typename T3, typename T4>
        inline bool operator()(const boost::tuple<T1, T2, T3, T4>& lhs,
                               const boost::tuple<T1, T2, T3, T4>& rhs) const {
            return lexicographical_compare(lhs.template get<0>(),
                                           lhs.template get<1>(),
                                           lhs.template get<2>(),
                                           lhs.template get<3>(),
                                           rhs.template get<0>(),
                                           rhs.template get<1>(),
                                           rhs.template get<2>(),
                                           rhs.template get<3>(),
                                           s_Less);
        }

        template<typename T1, typename T2, typename T3, typename T4, typename T5>
        inline bool operator()(const boost::tuple<T1, T2, T3, T4, T5>& lhs,
                               const boost::tuple<T1, T2, T3, T4, T5>& rhs) const {
            return lexicographical_compare(lhs.template get<0>(),
                                           lhs.template get<1>(),
                                           lhs.template get<2>(),
                                           lhs.template get<3>(),
                                           lhs.template get<4>(),
                                           rhs.template get<0>(),
                                           rhs.template get<1>(),
                                           rhs.template get<2>(),
                                           rhs.template get<3>(),
                                           rhs.template get<4>(),
                                           s_Less);
        }

        SLess s_Less;
    };

    //! \brief Partial ordering of std::pairs and some boost::tuples based
    //! on smaller first element.
    //!
    //! \note That while this functionality can be implemented by boost
    //! bind, since it overloads the comparison operators, the resulting
    //! code is more than an order of magnitude slower than this version.
    struct SFirstLess {
        template<typename U, typename V>
        inline bool operator()(const std::pair<U, V>& lhs, const std::pair<U, V>& rhs) const {
            return s_Less(lhs.first, rhs.first);
        }

        template<typename U, typename V>
        inline bool operator()(const U& lhs, const std::pair<U, V>& rhs) const {
            return s_Less(lhs, rhs.first);
        }

        template<typename U, typename V>
        inline bool operator()(const std::pair<U, V>& lhs, const U& rhs) const {
            return s_Less(lhs.first, rhs);
        }

#define TUPLE_FIRST_LESS                                                                           \
    template<TEMPLATE_ARGS_DECL>                                                                   \
    inline bool operator()(const boost::tuple<TEMPLATE_ARGS>& lhs,                                 \
                           const boost::tuple<TEMPLATE_ARGS>& rhs) const {                         \
        return s_Less(lhs.template get<0>(), rhs.template get<0>());                               \
    }                                                                                              \
    template<TEMPLATE_ARGS_DECL>                                                                   \
    inline bool operator()(const T1& lhs, const boost::tuple<TEMPLATE_ARGS>& rhs) const {          \
        return s_Less(lhs, rhs.template get<0>());                                                 \
    }                                                                                              \
    template<TEMPLATE_ARGS_DECL>                                                                   \
    inline bool operator()(const boost::tuple<TEMPLATE_ARGS>& lhs, const T1& rhs) const {          \
        return s_Less(lhs.template get<0>(), rhs);                                                 \
    }

#define TEMPLATE_ARGS_DECL typename T1, typename T2, typename T3
#define TEMPLATE_ARGS T1, T2, T3
        TUPLE_FIRST_LESS
#undef TEMPLATE_ARGS
#undef TEMPLATE_ARGS_DECL
#define TEMPLATE_ARGS_DECL typename T1, typename T2, typename T3, typename T4
#define TEMPLATE_ARGS T1, T2, T3, T4
        TUPLE_FIRST_LESS
#undef TEMPLATE_ARGS
#undef TEMPLATE_ARGS_DECL
#define TEMPLATE_ARGS_DECL typename T1, typename T2, typename T3, typename T4, typename T5
#define TEMPLATE_ARGS T1, T2, T3, T4, T5
        TUPLE_FIRST_LESS
#undef TEMPLATE_ARGS
#undef TEMPLATE_ARGS_DECL
#undef TUPLE_FIRST_LESS

        SLess s_Less;
    };

    //! \brief Partial ordering of std::pairs and some boost::tuples based
    //! on larger first element.
    //!
    //! \note That while this functionality can be implemented by boost
    //! bind, since it overloads the comparison operators, the resulting
    //! code is more than an order of magnitude slower than this version.
    struct SFirstGreater {
        template<typename U, typename V>
        inline bool operator()(const std::pair<U, V>& lhs, const std::pair<U, V>& rhs) const {
            return s_Greater(lhs.first, rhs.first);
        }

        template<typename U, typename V>
        inline bool operator()(const U& lhs, const std::pair<U, V>& rhs) const {
            return s_Greater(lhs, rhs.first);
        }

        template<typename U, typename V>
        inline bool operator()(const std::pair<U, V>& lhs, const U& rhs) const {
            return s_Greater(lhs.first, rhs);
        }

        SGreater s_Greater;
    };

    //! \brief Partial ordering of pairs based on smaller second element.
    //!
    //! \note That while this functionality can be implemented by boost
    //! bind, since it overloads the comparison operators, the resulting
    //! code is more than an order of magnitude slower than this version.
    struct SSecondLess {
        template<typename U, typename V>
        inline bool operator()(const std::pair<U, V>& lhs, const std::pair<U, V>& rhs) const {
            return s_Less(lhs.second, rhs.second);
        }

        template<typename U, typename V>
        inline bool operator()(const V& lhs, const std::pair<U, V>& rhs) const {
            return s_Less(lhs, rhs.second);
        }

        template<typename U, typename V>
        inline bool operator()(const std::pair<U, V>& lhs, const V& rhs) const {
            return s_Less(lhs.second, rhs);
        }

        SLess s_Less;
    };

    //! \brief Partial ordering of pairs based on larger second element.
    //!
    //! \note That while this functionality can be implemented by boost
    //! bind, since it overloads the comparison operators, the resulting
    //! code is more than an order of magnitude slower than this version.
    struct SSecondGreater {
        template<typename U, typename V>
        inline bool operator()(const std::pair<U, V>& lhs, const std::pair<U, V>& rhs) const {
            return s_Greater(lhs.second, rhs.second);
        }

        template<typename U, typename V>
        inline bool operator()(const V& lhs, const std::pair<U, V>& rhs) const {
            return s_Greater(lhs, rhs.second);
        }

        template<typename U, typename V>
        inline bool operator()(const std::pair<U, V>& lhs, const V& rhs) const {
            return s_Greater(lhs.second, rhs);
        }

        SGreater s_Greater;
    };

    //! \name Simultaneously Sort Multiple Vectors
    //!
    //! This simultaneously sorts a number of vectors based on ordering
    //! a collection of keys. For examples, the following code
    //! \code{cpp}
    //!   double someids[] = { 3.1, 2.2, 0.5, 1.5 };
    //!   std::string somenames[] =
    //!       {
    //!           std::string('a'),
    //!           std::string('b'),
    //!           std::string('c'),
    //!           std::string('d')
    //!       };
    //!   std::vector<double> ids(someids, someids + 4);
    //!   std::vector<std::string> names(somenames, somenames + 4);
    //!
    //!   maths::COrderings::simultaneousSort(ids, names);
    //!
    //!   for (std::size_t i = 0u; i < 4; ++i)
    //!   {
    //!       std::cout << ids[i] << ' ' << names[i] << std::endl;
    //!   }
    //! \endcode
    //!
    //! Will produce the following output:
    //! <pre>
    //! 0.5 c
    //! 1.5 d
    //! 2.2 b
    //! 3.1 a
    //! </pre>
    //!
    //! These support simultaneously sorting up to 4 additional containers
    //! to the keys.
    //!
    //! \note The complexity is O(N log(N)) where N is the length of the
    //! containers.
    //! \warning All containers must have the same length.
    //@{
private:
    //! Orders a set of indices into an array based using the default
    //! comparison operator of the corresponding key type.
    template<typename KEY_VECTOR, typename COMP = std::less<typename KEY_VECTOR::value_type>>
    class CIndexLess {
    public:
        CIndexLess(const KEY_VECTOR& keys, const COMP& comp = COMP())
            : m_Keys(&keys), m_Comp(comp) {}

        bool operator()(std::size_t lhs, std::size_t rhs) {
            return m_Comp((*m_Keys)[lhs], (*m_Keys)[rhs]);
        }

    private:
        const KEY_VECTOR* m_Keys;
        COMP m_Comp;
    };

public:
    // The logic in this function is rather subtle because we want to
    // sort the collections in place. In particular, we create a sorted
    // collection of indices where each index tells us where to get the
    // element from at that location and we want to re-order all the
    // collections by that ordering in place. If an index matches its
    // position then we can move to the next position. Otherwise, we
    // need to swap the items at the index in to its position. To work
    // in place we need to do something with the items which are displaced.
    // If these are the items required at the swapped in position then
    // we are done. Otherwise, we just repeat until we find this position.
    // It is easy to verify that this process finds a closed cycle with
    // at most N steps. Each time a swap is made at least one more item
    // is in its correct place, and we update the ordering accordingly.
    // So the containers are sorted in at most O(N) additional steps to
    // the N * log(N) taken to sort the indices.
#define SIMULTANEOUS_SORT_IMPL                                                                     \
    if (boost::algorithm::is_sorted(keys.begin(), keys.end(), comp)) {                             \
        return true;                                                                               \
    }                                                                                              \
    typedef std::vector<std::size_t> TSizeVec;                                                     \
    TSizeVec ordering;                                                                             \
    ordering.reserve(keys.size());                                                                 \
    for (std::size_t i = 0u; i < keys.size(); ++i) {                                               \
        ordering.push_back(i);                                                                     \
    }                                                                                              \
    std::stable_sort(ordering.begin(), ordering.end(), CIndexLess<KEY_VECTOR, COMP>(keys, comp));  \
    for (std::size_t i = 0u; i < ordering.size(); ++i) {                                           \
        std::size_t j_ = i;                                                                        \
        std::size_t j = ordering[j_];                                                              \
        while (i != j) {                                                                           \
            using std::swap;                                                                       \
            swap(keys[j_], keys[j]);                                                               \
            CUSTOM_SWAP_VALUES                                                                     \
            ordering[j_] = j_;                                                                     \
            j_ = j;                                                                                \
            j = ordering[j_];                                                                      \
        }                                                                                          \
        ordering[j_] = j_;                                                                         \
    }                                                                                              \
    return true;

#define CUSTOM_SWAP_VALUES swap(values[j_], values[j]);
    //! Simultaneously sort \p keys and \p values using the \p comp
    //! order of \p keys.
    template<typename KEY_VECTOR, typename VALUE_VECTOR, typename COMP>
    static bool simultaneousSort(KEY_VECTOR& keys, VALUE_VECTOR& values, const COMP& comp) {
        if (keys.size() != values.size()) {
            return false;
        }
        SIMULTANEOUS_SORT_IMPL
    }
#undef CUSTOM_SWAP_VALUES
    //! Overload for default operator< comparison.
    template<typename KEY_VECTOR, typename VALUE_VECTOR>
    static bool simultaneousSort(KEY_VECTOR& keys, VALUE_VECTOR& values) {
        return simultaneousSort(keys, values, std::less<typename KEY_VECTOR::value_type>());
    }
    //! Overload for default operator< comparison.
    template<typename KEY_VECTOR, typename VALUE_VECTOR>
    static bool simultaneousSort(core::CVectorRange<KEY_VECTOR>& keys,
                                 core::CVectorRange<VALUE_VECTOR>& values) {
        return simultaneousSort(keys, values, std::less<typename KEY_VECTOR::value_type>());
    }

#define CUSTOM_SWAP_VALUES                                                                         \
    swap(values1[j_], values1[j]);                                                                 \
    swap(values2[j_], values2[j]);
    //! Simultaneously sort \p keys, \p values1 and \p values2
    //! using the \p comp order of \p keys.
    template<typename KEY_VECTOR, typename VALUE1_VECTOR, typename VALUE2_VECTOR, typename COMP>
    static bool simultaneousSort(KEY_VECTOR& keys,
                                 VALUE1_VECTOR& values1,
                                 VALUE2_VECTOR& values2,
                                 const COMP& comp) {
        if (keys.size() != values1.size() || values1.size() != values2.size()) {
            return false;
        }
        SIMULTANEOUS_SORT_IMPL
    }
#undef CUSTOM_SWAP_VALUES
    //! Overload for default operator< comparison.
    template<typename KEY_VECTOR, typename VALUE1_VECTOR, typename VALUE2_VECTOR>
    static bool simultaneousSort(KEY_VECTOR& keys, VALUE1_VECTOR& values1, VALUE2_VECTOR& values2) {
        return simultaneousSort(keys,
                                values1,
                                values2,
                                std::less<typename KEY_VECTOR::value_type>());
    }
    //! Overload for default operator< comparison.
    template<typename KEY_VECTOR, typename VALUE1_VECTOR, typename VALUE2_VECTOR>
    static bool simultaneousSort(core::CVectorRange<KEY_VECTOR> keys,
                                 core::CVectorRange<VALUE1_VECTOR> values1,
                                 core::CVectorRange<VALUE2_VECTOR> values2) {
        return simultaneousSort(keys,
                                values1,
                                values2,
                                std::less<typename KEY_VECTOR::value_type>());
    }

#define CUSTOM_SWAP_VALUES                                                                         \
    swap(values1[j_], values1[j]);                                                                 \
    swap(values2[j_], values2[j]);                                                                 \
    swap(values3[j_], values3[j]);
    //! Simultaneously sort \p keys, \p values1, \p values2
    //! and \p values3 using the \p comp order of \p keys.
    template<typename KEY_VECTOR,
             typename VALUE1_VECTOR,
             typename VALUE2_VECTOR,
             typename VALUE3_VECTOR,
             typename COMP>
    static bool simultaneousSort(KEY_VECTOR& keys,
                                 VALUE1_VECTOR& values1,
                                 VALUE2_VECTOR& values2,
                                 VALUE3_VECTOR& values3,
                                 const COMP& comp) {
        if (keys.size() != values1.size() || values1.size() != values2.size() ||
            values2.size() != values3.size()) {
            return false;
        }
        SIMULTANEOUS_SORT_IMPL
    }
#undef CUSTOM_SWAP_VALUES
    //! Overload for default operator< comparison.
    template<typename KEY_VECTOR,
             typename VALUE1_VECTOR,
             typename VALUE2_VECTOR,
             typename VALUE3_VECTOR>
    static bool simultaneousSort(KEY_VECTOR& keys,
                                 VALUE1_VECTOR& values1,
                                 VALUE2_VECTOR& values2,
                                 VALUE3_VECTOR& values3) {
        return simultaneousSort(keys,
                                values1,
                                values2,
                                values3,
                                std::less<typename KEY_VECTOR::value_type>());
    }
    //! Overload for default operator< comparison.
    template<typename KEY_VECTOR,
             typename VALUE1_VECTOR,
             typename VALUE2_VECTOR,
             typename VALUE3_VECTOR>
    static bool simultaneousSort(core::CVectorRange<KEY_VECTOR> keys,
                                 core::CVectorRange<VALUE1_VECTOR> values1,
                                 core::CVectorRange<VALUE2_VECTOR> values2,
                                 core::CVectorRange<VALUE3_VECTOR> values3) {
        return simultaneousSort(keys,
                                values1,
                                values2,
                                values3,
                                std::less<typename KEY_VECTOR::value_type>());
    }

#define CUSTOM_SWAP_VALUES                                                                         \
    swap(values1[j_], values1[j]);                                                                 \
    swap(values2[j_], values2[j]);                                                                 \
    swap(values3[j_], values3[j]);                                                                 \
    swap(values4[j_], values4[j]);
    //! Simultaneously sort \p keys, \p values1, \p values2,
    //! \p values3 and \p values4 using the \p comp order of
    //! \p keys.
    template<typename KEY_VECTOR,
             typename VALUE1_VECTOR,
             typename VALUE2_VECTOR,
             typename VALUE3_VECTOR,
             typename VALUE4_VECTOR,
             typename COMP>
    static bool simultaneousSort(KEY_VECTOR& keys,
                                 VALUE1_VECTOR& values1,
                                 VALUE2_VECTOR& values2,
                                 VALUE3_VECTOR& values3,
                                 VALUE4_VECTOR& values4,
                                 const COMP& comp) {
        if (keys.size() != values1.size() || values1.size() != values2.size() ||
            values2.size() != values3.size() || values3.size() != values4.size()) {
            return false;
        }
        SIMULTANEOUS_SORT_IMPL
    }
#undef CUSTOM_SWAP_VALUES
    //! Overload for default operator< comparison.
    template<typename KEY_VECTOR,
             typename VALUE1_VECTOR,
             typename VALUE2_VECTOR,
             typename VALUE3_VECTOR,
             typename VALUE4_VECTOR>
    static bool simultaneousSort(KEY_VECTOR& keys,
                                 VALUE1_VECTOR& values1,
                                 VALUE2_VECTOR& values2,
                                 VALUE3_VECTOR& values3,
                                 VALUE4_VECTOR& values4) {
        return simultaneousSort(keys,
                                values1,
                                values2,
                                values3,
                                values4,
                                std::less<typename KEY_VECTOR::value_type>());
    }
    //! Overload for default operator< comparison.
    template<typename KEY_VECTOR,
             typename VALUE1_VECTOR,
             typename VALUE2_VECTOR,
             typename VALUE3_VECTOR,
             typename VALUE4_VECTOR>
    static bool simultaneousSort(core::CVectorRange<KEY_VECTOR> keys,
                                 core::CVectorRange<VALUE1_VECTOR> values1,
                                 core::CVectorRange<VALUE2_VECTOR> values2,
                                 core::CVectorRange<VALUE3_VECTOR> values3,
                                 core::CVectorRange<VALUE4_VECTOR> values4) {
        return simultaneousSort(keys,
                                values1,
                                values2,
                                values3,
                                values4,
                                std::less<typename KEY_VECTOR::value_type>());
    }

#undef SIMULTANEOUS_SORT_IMPL
    //@}
};
}
}

#endif // INCLUDED_ml_maths_COrderings_h
