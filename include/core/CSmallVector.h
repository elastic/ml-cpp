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

#ifndef INCLUDED_ml_core_CSmallVector_h
#define INCLUDED_ml_core_CSmallVector_h

#include <core/CSmallVectorFwd.h>

#include <boost/container/small_vector.hpp>

#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <vector>

namespace ml {
namespace core {
namespace small_vector_detail {

inline std::size_t min(std::size_t a, std::size_t b) {
    return a < b ? a : b;
}

template<typename T, typename = void>
struct SPlusAssign {
    static_assert(sizeof(T) < 0, "The contained type has no defined += operator");
};

template<typename T>
struct SPlusAssign<T, std::void_t<decltype(std::declval<T&>() += std::declval<T>())>> {
    static void compute(CSmallVectorBase<T>& lhs, const CSmallVectorBase<T>& rhs) {
        for (std::size_t i = 0, n = min(lhs.size(), rhs.size()); i < n; ++i) {
            lhs[i] += rhs[i];
        }
    }
};

template<typename T, typename = void>
struct SMinusAssign {
    static_assert(sizeof(T) < 0, "The contained type has no defined -= operator");
};

template<typename T>
struct SMinusAssign<T, std::void_t<decltype(std::declval<T&>() -= std::declval<T>())>> {
    static void compute(CSmallVectorBase<T>& lhs, const CSmallVectorBase<T>& rhs) {
        for (std::size_t i = 0, n = min(lhs.size(), rhs.size()); i < n; ++i) {
            lhs[i] -= rhs[i];
        }
    }
};
}

//! \brief This inherits from boost::container::small_vector.
//!
//! DESCRIPTION:\n
//! The reasons for this class are largely historical in that we didn't have
//! boost::container::small_vector in the version of boost we were using when
//! this was originally implemented. However, a lot of our code now uses this
//! class and it is convenient to provide some non-standard extensions.
//!
//! IMPLEMENTATION:\n
//! Inherits from boost::container::small_vector.
//!
//! \tparam T The element type.
//! \tparam N The maximum number of elements which are stored on the stack.
template<typename T, std::size_t N>
class CSmallVector : public boost::container::small_vector<T, N> {
private:
    using TBase = boost::container::small_vector<T, N>;

public:
    // Forward typedefs
    using value_type = typename TBase::value_type;
    using allocator_type = typename TBase::allocator_type;
    using reference = typename TBase::reference;
    using const_reference = typename TBase::const_reference;
    using pointer = typename TBase::pointer;
    using const_pointer = typename TBase::const_pointer;
    using difference_type = typename TBase::difference_type;
    using size_type = typename TBase::size_type;
    using iterator = typename TBase::iterator;
    using const_iterator = typename TBase::const_iterator;
    using reverse_iterator = typename TBase::reverse_iterator;
    using const_reverse_iterator = typename TBase::const_reverse_iterator;

public:
    //! \name Constructors
    //@{
    CSmallVector() {}
    CSmallVector(const CSmallVector& other) : TBase(other) {}
    CSmallVector(CSmallVector&& other) noexcept(std::is_nothrow_move_constructible<TBase>::value)
        : TBase(std::move(other.baseRef())) {}
    explicit CSmallVector(size_type n, const value_type& val = value_type())
        : TBase(n, val) {}
    CSmallVector(std::initializer_list<value_type> list)
        : TBase(list.begin(), list.end()) {}
    template<class ITR>
    CSmallVector(ITR first, ITR last) : TBase(first, last) {}
    template<typename U, std::size_t M>
    CSmallVector(const CSmallVector<U, M>& other)
        : TBase(other.begin(), other.end()) {}
    template<typename U>
    CSmallVector(std::initializer_list<U> list)
        : TBase(list.begin(), list.end()) {}
    // Extend to construct implicitly from a vector.
    template<typename U>
    CSmallVector(const std::vector<U>& other)
        : TBase(other.begin(), other.end()) {}

    CSmallVector&
    operator=(CSmallVector&& rhs) noexcept(std::is_nothrow_move_assignable<TBase>::value) {
        this->baseRef() = std::move(rhs.baseRef());
        return *this;
    }
    CSmallVector& operator=(const CSmallVector& rhs) {
        this->baseRef() = rhs.baseRef();
        return *this;
    }

    // Extend to convert implicitly to a vector.
    inline operator std::vector<T>() const {
        return std::vector<T>(this->begin(), this->end());
    }

    // Non-standard plus assign for the case that T has operator+=.
    const CSmallVector& operator+=(const CSmallVectorBase<T>& rhs) {
        small_vector_detail::SPlusAssign<T>::compute(*this, rhs);
        return *this;
    }

    // Non-standard minus assign for the case that T has operator-=.
    const CSmallVector& operator-=(const CSmallVectorBase<T>& rhs) {
        small_vector_detail::SMinusAssign<T>::compute(*this, rhs);
        return *this;
    }

private:
    TBase& baseRef() { return *this; }
    const TBase& baseRef() const { return *this; }
};
}
}

#endif // INCLUDED_ml_core_CSmallVector_h
