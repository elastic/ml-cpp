/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CChecksum_h
#define INCLUDED_ml_maths_CChecksum_h

#include <core/CHashing.h>
#include <core/CStoredStringPtr.h>
#include <core/CStringUtils.h>

#include <maths/CLinearAlgebraFwd.h>
#include <maths/COrderings.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/optional/optional_fwd.hpp>
#include <boost/unordered/unordered_map_fwd.hpp>
#include <boost/unordered/unordered_set_fwd.hpp>

#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace ml {
namespace maths {
namespace checksum_detail {

class BasicChecksum {};
class ContainerChecksum {};
class MemberChecksumWithSeed {};
class MemberChecksumWithoutSeed {};
class MemberHash {};

//! Auxiliary type used by has_const_iterator to test for a nested
//! typedef.
template<typename T, typename R = void>
struct enable_if_type {
    using type = R;
};

//! Auxiliary type used by has_checksum_function to test for a nested
//! member function.
template<typename T, T, typename R = void>
struct enable_if_is_type {
    using type = R;
};

//! \name Class used to select appropriate checksum implementation
//! for containers.
//!
//! \note Partial specializations can't be nested classes.
//! \note Uses SFINAE to check for nested typedef.
//! \note Uses the enable_if trick to get around the restriction that
//! "A partially specialized non-type argument expression shall not
//! involve a template parameter of the partial specialization except
//! when the argument expression is a simple identifier" (see section
//! 14.5.4/9 of the standard).
//@{
template<typename T, typename ITR = void>
struct container_selector {
    using value = BasicChecksum;
};
template<typename T>
struct container_selector<T, typename enable_if_type<typename T::const_iterator>::type> {
    using value = ContainerChecksum;
};
//@}

//! \name Class used to select appropriate checksum implementation.
//!
//! Template and partial specialization which chooses between our
//! various checksum implementations.
//!
//! \note Partial specializations can't be nested classes.
//! \note Uses SFINAE to check for checksum or hash functions.
//! \note Uses the enable_if trick to get around the restriction that
//! "A partially specialized non-type argument expression shall not
//! involve a template parameter of the partial specialization except
//! when the argument expression is a simple identifier" (see section
//! 14.5.4/9 of the standard).
//@{
template<typename T, typename ENABLE = void>
struct selector {
    using value = typename container_selector<T>::value;
};
template<typename T>
struct selector<T, typename enable_if_is_type<uint64_t (T::*)(uint64_t) const, &T::checksum>::type> {
    using value = MemberChecksumWithSeed;
};
template<typename T>
struct selector<T, typename enable_if_is_type<uint64_t (T::*)() const, &T::checksum>::type> {
    using value = MemberChecksumWithoutSeed;
};
template<typename T>
struct selector<T, typename enable_if_is_type<std::size_t (T::*)() const, &T::hash>::type> {
    using value = MemberHash;
};
//@}

template<typename SELECTOR>
class CChecksumImpl {};

//! Basic checksum functionality implementation.
template<>
class CChecksumImpl<BasicChecksum> {
public:
    //! Checksum integral type.
    template<typename INTEGRAL>
    static uint64_t dispatch(uint64_t seed, INTEGRAL target) {
        return core::CHashing::hashCombine(seed, static_cast<uint64_t>(target));
    }

    //! Checksum of double.
    static uint64_t dispatch(uint64_t seed, double target) {
        return dispatch(seed, core::CStringUtils::typeToStringPrecise(
                                  target, core::CIEEE754::E_SinglePrecision));
    }

    //! Checksum of a universal hash function.
    static uint64_t
    dispatch(uint64_t seed,
             const core::CHashing::CUniversalHash::CUInt32UnrestrictedHash& target) {
        seed = core::CHashing::hashCombine(seed, static_cast<uint64_t>(target.a()));
        return core::CHashing::hashCombine(seed, static_cast<uint64_t>(target.b()));
    }

    //! Checksum of float storage.
    static uint64_t dispatch(uint64_t seed, CFloatStorage target) {
        return dispatch(seed, target.toString());
    }

    //! Checksum of string.
    static uint64_t dispatch(uint64_t seed, const std::string& target) {
        return core::CHashing::safeMurmurHash64(
            target.data(), static_cast<int>(target.size()), seed);
    }

    //! Checksum a stored string pointer.
    static uint64_t dispatch(uint64_t seed, const core::CStoredStringPtr& target) {
        return target == nullptr ? seed : dispatch(seed, *target);
    }

    //! Checksum of a reference_wrapper.
    template<typename T>
    static uint64_t dispatch(uint64_t seed, const std::reference_wrapper<T>& target) {
        return CChecksumImpl<typename selector<T>::value>::dispatch(seed, target.get());
    }

    //! Checksum of a optional.
    template<typename T>
    static uint64_t dispatch(uint64_t seed, const boost::optional<T>& target) {
        return !target ? seed
                       : CChecksumImpl<typename selector<T>::value>::dispatch(seed, *target);
    }

    //! Checksum a shared pointer.
    template<typename T>
    static uint64_t dispatch(uint64_t seed, const std::shared_ptr<T>& target) {
        return !target ? seed
                       : CChecksumImpl<typename selector<T>::value>::dispatch(seed, *target);
    }

    //! Checksum a unique pointer.
    template<typename T>
    static uint64_t dispatch(uint64_t seed, const std::unique_ptr<T>& target) {
        return !target ? seed
                       : CChecksumImpl<typename selector<T>::value>::dispatch(seed, *target);
    }

    //! Checksum a pair.
    template<typename U, typename V>
    static uint64_t dispatch(uint64_t seed, const std::pair<U, V>& target) {
        seed = CChecksumImpl<typename selector<U>::value>::dispatch(seed, target.first);
        return CChecksumImpl<typename selector<V>::value>::dispatch(seed, target.second);
    }

    //! Checksum an Eigen dense vector.
    template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS>
    static uint64_t
    dispatch(uint64_t seed,
             const Eigen::Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>& target) {
        std::ptrdiff_t dimension = target.size();
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
    static uint64_t
    dispatch(uint64_t seed, const Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX>& target) {
        using TIterator = typename Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX>::InnerIterator;
        uint64_t result = seed;
        for (TIterator i(target, 0); i; ++i) {
            result = dispatch(seed, i.index());
            result = dispatch(result, i.value());
        }
        return result;
    }

    //! Checksum of an annotated vector.
    template<typename VECTOR, typename ANNOTATION>
    static uint64_t
    dispatch(uint64_t seed, const CAnnotatedVector<VECTOR, ANNOTATION>& target) {
        seed = CChecksumImpl<typename selector<VECTOR>::value>::dispatch(
            seed, static_cast<const VECTOR&>(target));
        return CChecksumImpl<typename selector<ANNOTATION>::value>::dispatch(
            seed, target.annotation());
    }
};

//! Type with checksum member function implementation.
template<>
class CChecksumImpl<MemberChecksumWithSeed> {
public:
    //! Call member checksum.
    template<typename T>
    static uint64_t dispatch(uint64_t seed, const T& target) {
        return target.checksum(seed);
    }
};

//! Type with checksum member function implementation.
template<>
class CChecksumImpl<MemberChecksumWithoutSeed> {
public:
    //! Call member checksum.
    template<typename T>
    static uint64_t dispatch(uint64_t seed, const T& target) {
        return core::CHashing::hashCombine(seed, target.checksum());
    }
};

//! Type with hash member function implementation.
template<>
class CChecksumImpl<MemberHash> {
public:
    //! Call member checksum.
    template<typename T>
    static uint64_t dispatch(uint64_t seed, const T& target) {
        return core::CHashing::hashCombine(seed, static_cast<uint64_t>(target.hash()));
    }
};

//! Container checksum implementation.
template<>
class CChecksumImpl<ContainerChecksum> {
public:
    //! Call on elements.
    template<typename T>
    static uint64_t dispatch(uint64_t seed, const T& target) {
        uint64_t result = seed;
        for (const auto& element : target) {
            result = CChecksumImpl<typename selector<typename T::value_type>::value>::dispatch(
                result, element);
        }
        return result;
    }

    //! Hash of vector bool.
    //!
    //! \note The default implementation generates a compiler warning for
    //! std::vector<bool> because its operator[] doesn't return by reference.
    //! In any case, the std::hash specialisation is more efficient.
    static uint64_t dispatch(uint64_t seed, const std::vector<bool>& target) {
        return core::CHashing::hashCombine(seed, ms_VectorBoolHasher(target));
    }

    //! Stable hash of unordered set.
    template<typename T>
    static uint64_t dispatch(uint64_t seed, const boost::unordered_set<T>& target) {
        using TCRef = std::reference_wrapper<const T>;
        using TCRefVec = std::vector<TCRef>;

        TCRefVec ordered;
        ordered.reserve(target.size());
        for (const auto& element : target) {
            ordered.push_back(TCRef(element));
        }

        std::sort(ordered.begin(), ordered.end(), maths::COrderings::SReferenceLess());

        return dispatch(seed, ordered);
    }

    //! Stable hash of unordered map.
    template<typename U, typename V>
    static uint64_t dispatch(uint64_t seed, const boost::unordered_map<U, V>& target) {
        using TUCRef = std::reference_wrapper<const U>;
        using TVCRef = std::reference_wrapper<const V>;
        using TUCRefVCRefPr = std::pair<TUCRef, TVCRef>;
        using TUCRefVCRefPrVec = std::vector<TUCRefVCRefPr>;

        TUCRefVCRefPrVec ordered;
        ordered.reserve(target.size());
        for (const auto& element : target) {
            ordered.emplace_back(TUCRef(element.first), TVCRef(element.second));
        }

        std::sort(ordered.begin(), ordered.end(), maths::COrderings::SFirstLess());

        return dispatch(seed, ordered);
    }

    //! Handle std::string which resolves to a container.
    static uint64_t dispatch(uint64_t seed, const std::string& target) {
        return CChecksumImpl<BasicChecksum>::dispatch(seed, target);
    }

private:
    static const std::hash<std::vector<bool>> ms_VectorBoolHasher;
};

//! Convenience function to select implementation.
template<typename T>
uint64_t checksum(uint64_t seed, const T& target) {
    return CChecksumImpl<typename selector<T>::value>::dispatch(seed, target);
}

} // checksum_detail::

//! \brief Implementation of utility functionality for creating
//! object checksums which are stable over model state persistence
//! and restoration.
class MATHS_EXPORT CChecksum {
public:
    //! The basic checksum implementation.
    template<typename T>
    static uint64_t calculate(uint64_t seed, const T& target) {
        return checksum_detail::checksum(seed, target);
    }

    //! Overload for arrays which chains checksums.
    template<typename T, std::size_t SIZE>
    static uint64_t calculate(uint64_t seed, const T (&target)[SIZE]) {
        for (std::size_t i = 0u; i + 1 < SIZE; ++i) {
            seed = checksum_detail::checksum(seed, target[i]);
        }
        return checksum_detail::checksum(seed, target[SIZE - 1]);
    }

    //! Overload for nested arrays which chains checksums.
    template<typename T, std::size_t SIZE1, std::size_t SIZE2>
    static uint64_t calculate(uint64_t seed, const T (&target)[SIZE1][SIZE2]) {
        for (std::size_t i = 0u; i + 1 < SIZE1; ++i) {
            seed = calculate(seed, target[i]);
        }
        return calculate(seed, target[SIZE1 - 1]);
    }
};
}
}

#endif // INCLUDED_ml_maths_CChecksum_h
