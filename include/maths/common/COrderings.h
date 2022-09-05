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

#ifndef INCLUDED_ml_maths_common_COrderings_h
#define INCLUDED_ml_maths_common_COrderings_h

#include <core/CNonInstantiatable.h>
#include <core/CStoredStringPtr.h>
#include <core/UnwrapRef.h>

#include <maths/common/ImportExport.h>

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace ml {
namespace maths {
namespace common {
//! \brief A collection of useful functionality to order collections
//! of objects.
//!
//! DESCRIPTION:\n
//! This implements some generic commonly occurring ordering functionality.
//! In particular,
//!   -# Assorted comparison objects which handle derefencing optional and pointer
//!      types, and unwrapping reference wrapped types.
//!   -# Lexicographical comparison for small collections of objects with distinct
//!      types (std::lexicographical_compare only supports a single type).
//!   -# Efficiently, O(N log(N)), simultaneously sorting equal length collections
//!      supporting random access iteration using one the collections to to define
//!      the order for all.
class COrderings : private core::CNonInstantiatable {
public:
    //! \brief Orders two pointers or std::optional values such that non-null
    //! are less than null values and otherwise compares using std::less<>.
    struct MATHS_COMMON_EXPORT SDerefLess {
        template<typename T>
        inline bool operator()(const T* lhs, const T* rhs) const {
            return less(lhs, rhs);
        }

        template<typename T>
        inline bool operator()(const std::optional<T>& lhs, const std::optional<T>& rhs) const {
            return less(lhs, rhs);
        }

        template<typename T>
        static inline bool less(const T* lhs, const T* rhs) {
            bool lInitialized{lhs != nullptr};
            bool rInitialized{rhs != nullptr};
            return lInitialized && rInitialized
                       ? s_Less(core::unwrap_ref(*lhs), core::unwrap_ref(*rhs))
                       : rInitialized < lInitialized;
        }

        template<typename T>
        static inline bool less(const std::optional<T>& lhs, const std::optional<T>& rhs) {
            bool lInitialized{lhs != std::nullopt};
            bool rInitialized{rhs != std::nullopt};
            return lInitialized && rInitialized
                       ? s_Less(core::unwrap_ref(*lhs), core::unwrap_ref(*rhs))
                       : rInitialized < lInitialized;
        }

        static const std::less<> s_Less;
    };

    //! \brief Orders two pointers or std::optional types such that null are
    //! greater than non-null values and otherwise compares using std::greater<>.
    struct MATHS_COMMON_EXPORT SDerefGreater {
        template<typename T>
        inline bool operator()(const T* lhs, const T* rhs) const {
            return greater(lhs, rhs);
        }

        template<typename T>
        inline bool operator()(const std::optional<T>& lhs, const std::optional<T>& rhs) const {
            return greater(lhs, rhs);
        }

        template<typename T>
        static inline bool greater(const T* lhs, const T* rhs) {
            bool lInitialized{lhs != nullptr};
            bool rInitialized{rhs != nullptr};
            return lInitialized && rInitialized
                       ? s_Greater(core::unwrap_ref(*lhs), core::unwrap_ref(*rhs))
                       : rInitialized > lInitialized;
        }

        template<typename T>
        static inline bool
        greater(const std::optional<T>& lhs, const std::optional<T>& rhs) {
            bool lInitialized{lhs != std::nullopt};
            bool rInitialized{rhs != std::nullopt};
            return lInitialized && rInitialized
                       ? s_Greater(core::unwrap_ref(*lhs), core::unwrap_ref(*rhs))
                       : rInitialized > lInitialized;
        }

        static const std::greater<> s_Greater;
    };

    //! \brief Orders two reference wrapped objects which are
    //! comparable with std::less<>.
    struct MATHS_COMMON_EXPORT SReferenceLess {
        template<typename U, typename V>
        inline bool operator()(const U& lhs, const V& rhs) const {
            return less(lhs, rhs);
        }

        template<typename U, typename V>
        static inline bool less(const U& lhs, const V& rhs) {
            return s_Less(core::unwrap_ref(lhs), core::unwrap_ref(rhs));
        }

        static const std::less<> s_Less;
    };

    //! \brief Orders two reference wrapped objects which are
    //! comparable with std::greater<>.
    struct SReferenceGreater {
        template<typename U, typename V>
        inline bool operator()(const U& lhs, const V& rhs) const {
            return greater(lhs, rhs);
        }

        template<typename U, typename V>
        static inline bool greater(const U& lhs, const V& rhs) {
            return s_Greater(core::unwrap_ref(lhs), core::unwrap_ref(rhs));
        }

        static const std::greater<> s_Greater;
    };

    //! \brief Wrapper around various less than comparisons.
    struct MATHS_COMMON_EXPORT SLess {
        template<typename U, typename V, std::enable_if_t<!std::is_pointer_v<U> || !std::is_pointer_v<V>>* = nullptr>
        inline bool operator()(const U& lhs, const V& rhs) const {
            return SReferenceLess::less(lhs, rhs);
        }

        template<typename T>
        inline bool operator()(const T* lhs, const T* rhs) const {
            return SDerefLess::less(lhs, rhs);
        }

        template<typename T>
        inline bool operator()(const std::optional<T>& lhs, const std::optional<T>& rhs) const {
            return SDerefLess::less(lhs, rhs);
        }

        inline bool operator()(const core::CStoredStringPtr& lhs,
                               const core::CStoredStringPtr& rhs) const {
            return SDerefLess::less(lhs.get(), rhs.get());
        }

        template<typename T>
        inline bool operator()(const std::shared_ptr<T>& lhs,
                               const std::shared_ptr<T>& rhs) const {
            return SDerefLess::less(lhs.get(), rhs.get());
        }

        template<typename T>
        inline bool operator()(const std::unique_ptr<T>& lhs,
                               const std::unique_ptr<T>& rhs) const {
            return SDerefLess::less(lhs.get(), rhs.get());
        }

        template<typename U, typename V>
        inline bool operator()(const std::pair<U, V>& lhs, const std::pair<U, V>& rhs) const {
            return lexicographicalCompareWith(*this, // compare with SLess
                                              lhs.first, lhs.second, rhs.first, rhs.second);
        }
    };

    //! \brief Wrapper around various less than comparisons.
    struct MATHS_COMMON_EXPORT SGreater {
        template<typename U, typename V, std::enable_if_t<!std::is_pointer_v<U> || !std::is_pointer_v<V>>* = nullptr>
        inline bool operator()(const U& lhs, const V& rhs) const {
            return SReferenceGreater::greater(lhs, rhs);
        }

        template<typename T>
        inline bool operator()(const T* lhs, const T* rhs) const {
            return SDerefGreater::greater(lhs, rhs);
        }

        template<typename T>
        inline bool operator()(const std::optional<T>& lhs, const std::optional<T>& rhs) const {
            return SDerefGreater::greater(lhs, rhs);
        }

        inline bool operator()(const core::CStoredStringPtr& lhs,
                               const core::CStoredStringPtr& rhs) const {
            return SDerefGreater::greater(lhs.get(), rhs.get());
        }

        template<typename T>
        inline bool operator()(const std::shared_ptr<T>& lhs,
                               const std::shared_ptr<T>& rhs) const {
            return SDerefGreater::greater(lhs.get(), rhs.get());
        }

        template<typename T>
        inline bool operator()(const std::unique_ptr<T>& lhs,
                               const std::unique_ptr<T>& rhs) const {
            return SDerefGreater::greater(lhs.get(), rhs.get());
        }

        template<typename U, typename V>
        inline bool operator()(const std::pair<U, V>& lhs, const std::pair<U, V>& rhs) const {
            return lexicographicalCompareWith(*this, // Compare with SGreater
                                              lhs.first, lhs.second, rhs.first, rhs.second);
        }
    };

    //! \brief Partial ordering of std::pairs on smaller first element.
    //!
    //! \note That while this functionality can be implemented by boost
    //! bind, since it overloads the comparison operators, the resulting
    //! code is more than an order of magnitude slower than this version.
    struct MATHS_COMMON_EXPORT SFirstLess {
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

        SLess s_Less;
    };

    //! \brief Partial ordering of std::pairs based on larger first element.
    //!
    //! \note That while this functionality can be implemented by bind
    //! bind, since it overloads the comparison operators, the resulting
    //! code is more than an order of magnitude slower than this version.
    struct MATHS_COMMON_EXPORT SFirstGreater {
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
    struct MATHS_COMMON_EXPORT SSecondLess {
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
    struct MATHS_COMMON_EXPORT SSecondGreater {
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

private:
    template<std::size_t I, typename PRED, typename... ARGS>
    static bool lexicographicalCompareAt(const PRED& pred, const ARGS&... args) {
        static_assert(sizeof...(ARGS) % 2 == 0, "The number of values to compare must be equal");
        const auto& lhs = std::get<I>(std::forward_as_tuple(args...));
        const auto& rhs = std::get<I + sizeof...(ARGS) / 2>(std::forward_as_tuple(args...));
        if (pred(lhs, rhs)) {
            return true;
        }
        if constexpr (sizeof...(args) > 2 * (I + 1)) {
            return pred(rhs, lhs) == false &&
                   lexicographicalCompareAt<I + 1>(pred, args...);
        }
        return false;
    }

public:
    //! Equivalent to std::lexicographical_compare for mixed types.
    //!
    //! Uses \p pred for comparison which must be able to compare each type.
    template<typename PRED, typename... ARGS>
    static bool lexicographicalCompareWith(const PRED& pred, const ARGS&... args) {
        return lexicographicalCompareAt<0>(pred, args...);
    }

    //! Calls lexicographicalCompareWith supplying std::less<> for comparison.
    template<typename... ARGS>
    static bool lexicographicalCompare(const ARGS&... args) {
        return lexicographicalCompareWith(std::less<>{}, args...);
    }

    //! Simultaneously sort multiple vectors using \p pred order of \p keys.
    //!
    //! This simultaneously sorts a number of vectors based on ordering a
    //! collection of keys. For examples, the following code:
    //! \code{cpp}
    //! std::vector<double> ids{3.1, 2.2, 0.5, 1.5};
    //! std::vector<std::string> names{"a", "b", "c", "d"};
    //! maths::common::COrderings::simultaneousSort(ids, names);
    //! for (std::size_t i = 0; i < 4; ++i) {
    //!     std::cout << ids[i] << ' ' << names[i] << std::endl;
    //! }
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
    //! \note The complexity is O(N log(N)) where N is the length of the
    //! containers.
    //! \warning All containers must have the same length.
    template<typename PRED, typename K, typename... V>
    static bool simultaneousSortWith(const PRED& pred, K&& keys, V&&... values);
    //! Overload for default operator< comparison.
    template<typename K, typename... V>
    static bool simultaneousSort(K&& keys, V&&... values) {
        return simultaneousSortWith(std::less<>(), std::forward<K>(keys),
                                    std::forward<V>(values)...);
    }
};
}
}
}

#endif // INCLUDED_ml_maths_common_COrderings_h
