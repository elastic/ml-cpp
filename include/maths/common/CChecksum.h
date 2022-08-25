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

#ifndef INCLUDED_ml_maths_common_CChecksum_h
#define INCLUDED_ml_maths_common_CChecksum_h

#include <core/CHashing.h>
#include <core/UnwrapRef.h>

#include <maths/common/CLinearAlgebraFwd.h>
#include <maths/common/ImportExport.h>

#include <boost/unordered/unordered_map_fwd.hpp>
#include <boost/unordered/unordered_set_fwd.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace ml {
namespace core {
class CStoredStringPtr;
}
namespace maths {
namespace common {
namespace checksum_detail {

class BasicChecksum {};
class ContainerChecksum {};
class MemberChecksumWithSeed {};
class MemberChecksumWithoutSeed {};
class MemberHash {};

//! \name Class used to select appropriate checksum implementation
//! for containers.
//@{
template<typename T, typename ITR = void>
struct container_selector {
    using value = BasicChecksum;
};
template<typename T>
struct container_selector<T, std::void_t<typename T::const_iterator>> {
    using value = ContainerChecksum;
};
//@}

//! \name Class used to select appropriate checksum implementation.
//@{
// clang-format off
template<typename T, typename ENABLE = void>
struct selector {
    using value = typename container_selector<T>::value;
};
template<typename T>
struct selector<T, std::enable_if_t<
            std::is_same_v<decltype(&T::checksum), std::uint64_t (T::*)(std::uint64_t) const>>> {
    using value = MemberChecksumWithSeed;
};
template<typename T>
struct selector<T, std::enable_if_t<
            std::is_same_v<decltype(&T::checksum), std::uint64_t (T::*)() const>>> {
    using value = MemberChecksumWithoutSeed;
};
template<typename T>
struct selector<T, std::enable_if_t<
            std::is_same_v<decltype(&T::hash), std::size_t (T::*)() const>>> {
    using value = MemberHash;
};
// clang-format on
//@}

template<typename SELECTOR>
class CChecksumImpl {};

//! Convenience function to select implementation.
template<typename T>
std::uint64_t checksum(std::uint64_t seed, const T& target) {
    return CChecksumImpl<typename selector<T>::value>::dispatch(seed, target);
}

//! Basic checksum functionality implementation.
template<>
class MATHS_COMMON_EXPORT CChecksumImpl<BasicChecksum> {
public:
    //! Checksum types castable to std::uint64_t.
    template<typename T, std::enable_if_t<std::is_integral_v<T> || std::is_enum_v<T>>* = nullptr>
    static std::uint64_t dispatch(std::uint64_t seed, const T& target) {
        return core::CHashing::hashCombine(seed, static_cast<std::uint64_t>(target));
    }

    //! Checksum a double.
    static std::uint64_t dispatch(std::uint64_t seed, double target);

    //! Checksum a universal hash function.
    static std::uint64_t
    dispatch(std::uint64_t seed,
             const core::CHashing::CUniversalHash::CUInt32UnrestrictedHash& target);

    //! Checksum a string.
    static std::uint64_t dispatch(std::uint64_t seed, const std::string& target);

    //! Checksum a stored string pointer.
    static std::uint64_t dispatch(std::uint64_t seed, const core::CStoredStringPtr& target);

    //! Checksum a reference_wrapper.
    template<typename T>
    static std::uint64_t
    dispatch(std::uint64_t seed, const std::reference_wrapper<T>& target) {
        return checksum(seed, target.get());
    }

    //! Checksum an optional.
    template<typename T>
    static std::uint64_t dispatch(std::uint64_t seed, const std::optional<T>& target) {
        return target == std::nullopt ? seed : checksum(seed, *target);
    }

    //! Checksum a raw pointer.
    template<typename T, std::enable_if_t<std::is_pointer_v<T> || std::is_null_pointer_v<T>>* = nullptr>
    static std::uint64_t dispatch(std::uint64_t seed, const T& target) {
        return target == nullptr ? seed : checksum(seed, *target);
    }

    //! Checksum a shared pointer.
    template<typename T>
    static std::uint64_t dispatch(std::uint64_t seed, const std::shared_ptr<T>& target) {
        return target == nullptr ? seed : checksum(seed, *target);
    }

    //! Checksum a unique pointer.
    template<typename T>
    static std::uint64_t dispatch(std::uint64_t seed, const std::unique_ptr<T>& target) {
        return target == nullptr ? seed : checksum(seed, *target);
    }

    //! Checksum a pair.
    template<typename U, typename V>
    static std::uint64_t dispatch(std::uint64_t seed, const std::pair<U, V>& target) {
        seed = checksum(seed, target.first);
        return checksum(seed, target.second);
    }

    //! Checksum a tuple.
    template<typename... T>
    static std::uint64_t dispatch(std::uint64_t seed, const std::tuple<T...>& target) {
        std::apply(
            [&](const auto&... element) {
                ((seed = checksum(seed, element)), ...);
            },
            target);
        return seed;
    }

    //! Checksum an Eigen dense vector.
    template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS>
    static std::uint64_t
    dispatch(std::uint64_t seed,
             const Eigen::Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>& target) {
        std::ptrdiff_t dimension(target.size());
        if (dimension > 0) {
            for (std::ptrdiff_t i = 0; i + 1 < dimension; ++i) {
                seed = dispatch(seed, target(i));
            }
            return dispatch(seed, target(dimension - 1));
        }
        return seed;
    }

    //! Checksum an Eigen sparse vector.
    template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
    static std::uint64_t
    dispatch(std::uint64_t seed,
             const Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX>& target) {
        using TIterator = typename Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX>::InnerIterator;
        std::uint64_t result{seed};
        for (TIterator i(target, 0); i; ++i) {
            result = dispatch(seed, i.index());
            result = dispatch(result, i.value());
        }
        return result;
    }

    //! Checksum of an annotated vector.
    template<typename VECTOR, typename ANNOTATION>
    static std::uint64_t
    dispatch(std::uint64_t seed, const CAnnotatedVector<VECTOR, ANNOTATION>& target) {
        seed = checksum(seed, static_cast<const VECTOR&>(target));
        return checksum(seed, target.annotation());
    }
};

//! Type with checksum member function implementation.
template<>
class MATHS_COMMON_EXPORT CChecksumImpl<MemberChecksumWithSeed> {
public:
    //! Call member checksum.
    template<typename T>
    static std::uint64_t dispatch(std::uint64_t seed, const T& target) {
        return target.checksum(seed);
    }
};

//! Type with checksum member function implementation.
template<>
class MATHS_COMMON_EXPORT CChecksumImpl<MemberChecksumWithoutSeed> {
public:
    //! Call member checksum.
    template<typename T>
    static std::uint64_t dispatch(std::uint64_t seed, const T& target) {
        return core::CHashing::hashCombine(seed, target.checksum());
    }
};

//! Type with hash member function implementation.
template<>
class MATHS_COMMON_EXPORT CChecksumImpl<MemberHash> {
public:
    //! Call member checksum.
    template<typename T>
    static std::uint64_t dispatch(std::uint64_t seed, const T& target) {
        return core::CHashing::hashCombine(
            seed, static_cast<std::uint64_t>(target.hash()));
    }
};

//! Container checksum implementation.
template<>
class MATHS_COMMON_EXPORT CChecksumImpl<ContainerChecksum> {
public:
    //! Call on elements.
    template<typename T>
    static std::uint64_t dispatch(std::uint64_t seed, const T& target) {
        std::uint64_t result{seed};
        for (const auto& element : target) {
            result = checksum(result, element);
        }
        return result;
    }

    //! Hash of vector bool.
    //!
    //! \note The default implementation generates a compiler warning for
    //! std::vector<bool> because its operator[] doesn't return by reference.
    //! In any case, the std::hash specialisation is more efficient.
    static std::uint64_t dispatch(std::uint64_t seed, const std::vector<bool>& target);

    //! Stable hash of unordered set.
    template<typename T>
    static std::uint64_t dispatch(std::uint64_t seed, const boost::unordered_set<T>& target) {
        using TCRefVec = std::vector<std::reference_wrapper<const std::remove_cv_t<T>>>;

        TCRefVec ordered;
        ordered.reserve(target.size());
        for (const auto& element : target) {
            ordered.emplace_back(element);
        }

        std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
            return core::unwrap_ref(lhs) < core::unwrap_ref(rhs);
        });

        return dispatch(seed, ordered);
    }

    //! Stable hash of unordered map.
    template<typename U, typename V>
    static std::uint64_t
    dispatch(std::uint64_t seed, const boost::unordered_map<U, V>& target) {
        using TUCRef = std::reference_wrapper<const std::remove_cv_t<U>>;
        using TVCRef = std::reference_wrapper<const std::remove_cv_t<V>>;
        using TUCRefVCRefPrVec = std::vector<std::pair<TUCRef, TVCRef>>;

        TUCRefVCRefPrVec ordered;
        ordered.reserve(target.size());
        for (const auto& element : target) {
            ordered.emplace_back(TUCRef{element.first}, TVCRef{element.second});
        }

        std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
            return core::unwrap_ref(lhs.first) < core::unwrap_ref(rhs.first);
        });

        return dispatch(seed, ordered);
    }

    //! Handle std::string which resolves to a container.
    static std::uint64_t dispatch(std::uint64_t seed, const std::string& target);
};
} // checksum_detail::

//! \brief Implementation of utility functionality for creating
//! object checksums which are stable over model state persistence
//! and restoration.
class MATHS_COMMON_EXPORT CChecksum {
public:
    //! The basic checksum implementation.
    template<typename T>
    static std::uint64_t calculate(std::uint64_t seed, const T& target) {
        return checksum_detail::checksum(seed, target);
    }

    //! Overload for arrays which chains checksums.
    template<typename T, std::size_t SIZE>
    static std::uint64_t calculate(std::uint64_t seed, const T (&target)[SIZE]) {
        for (std::size_t i = 0; i + 1 < SIZE; ++i) {
            seed = calculate(seed, target[i]);
        }
        return calculate(seed, target[SIZE - 1]);
    }
};
}
}
}

#endif // INCLUDED_ml_maths_common_CChecksum_h
