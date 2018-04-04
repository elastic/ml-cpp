/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLinearAlgebra_h
#define INCLUDED_ml_maths_CLinearAlgebra_h

#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CSmallVector.h>

#include <maths/CLinearAlgebraFwd.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/array.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/adapted/boost_array.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/operators.hpp>

#include <cmath>
#include <cstddef>

BOOST_GEOMETRY_REGISTER_BOOST_ARRAY_CS(cs::cartesian)

namespace ml {
namespace maths {

namespace linear_algebra_detail {

//! SFINAE check that \p N is at least 1.
struct CEmpty {};
template<std::size_t N>
struct CBoundsCheck {
    using InRange = CEmpty;
};
template<>
struct CBoundsCheck<0> {};

//! \brief Common vector functionality for variable storage type.
template<typename STORAGE>
struct SSymmetricMatrix {
    using Type = typename STORAGE::value_type;

    //! Get read only reference.
    inline const SSymmetricMatrix& base() const { return *this; }

    //! Get writable reference.
    inline SSymmetricMatrix& base() { return *this; }

    //! Set this vector equal to \p other.
    template<typename OTHER_STORAGE>
    void assign(const SSymmetricMatrix<OTHER_STORAGE>& other) {
        std::copy(other.m_LowerTriangle.begin(), other.m_LowerTriangle.end(), m_LowerTriangle.begin());
    }

    //! Create from a delimited string.
    bool fromDelimited(const std::string& str);

    //! Convert to a delimited string.
    std::string toDelimited() const;

    //! Get the i,j 'th component (no bounds checking).
    inline Type element(std::size_t i, std::size_t j) const {
        if (i < j) {
            std::swap(i, j);
        }
        return m_LowerTriangle[i * (i + 1) / 2 + j];
    }

    //! Get the i,j 'th component (no bounds checking).
    inline Type& element(std::size_t i, std::size_t j) {
        if (i < j) {
            std::swap(i, j);
        }
        return m_LowerTriangle[i * (i + 1) / 2 + j];
    }

    //! Component-wise negative.
    void negative() {
        for (std::size_t i = 0u; i < m_LowerTriangle.size(); ++i) {
            m_LowerTriangle[i] = -m_LowerTriangle[i];
        }
    }

    //! Matrix subtraction.
    void minusEquals(const SSymmetricMatrix& rhs) {
        for (std::size_t i = 0u; i < m_LowerTriangle.size(); ++i) {
            m_LowerTriangle[i] -= rhs.m_LowerTriangle[i];
        }
    }

    //! Matrix addition.
    void plusEquals(const SSymmetricMatrix& rhs) {
        for (std::size_t i = 0u; i < m_LowerTriangle.size(); ++i) {
            m_LowerTriangle[i] += rhs.m_LowerTriangle[i];
        }
    }

    //! Component-wise multiplication.
    void multiplyEquals(const SSymmetricMatrix& rhs) {
        for (std::size_t i = 0u; i < m_LowerTriangle.size(); ++i) {
            m_LowerTriangle[i] *= rhs.m_LowerTriangle[i];
        }
    }

    //! Scalar multiplication.
    void multiplyEquals(Type scale) {
        for (std::size_t i = 0u; i < m_LowerTriangle.size(); ++i) {
            m_LowerTriangle[i] *= scale;
        }
    }

    //! Scalar division.
    void divideEquals(Type scale) {
        for (std::size_t i = 0u; i < m_LowerTriangle.size(); ++i) {
            m_LowerTriangle[i] /= scale;
        }
    }

    //! Check if two matrices are identically equal.
    bool equal(const SSymmetricMatrix& other) const { return m_LowerTriangle == other.m_LowerTriangle; }

    //! Lexicographical total ordering.
    bool less(const SSymmetricMatrix& rhs) const { return m_LowerTriangle < rhs.m_LowerTriangle; }

    //! Check if this is zero.
    bool isZero() const {
        return std::find_if(m_LowerTriangle.begin(), m_LowerTriangle.end(), [](double ei) { return ei != 0.0; }) == m_LowerTriangle.end();
    }

    //! Get the matrix diagonal.
    template<typename VECTOR>
    VECTOR diagonal(std::size_t d) const {
        VECTOR result(d);
        for (std::size_t i = 0u; i < d; ++i) {
            result[i] = this->element(i, i);
        }
        return result;
    }

    //! Get the trace.
    Type trace(std::size_t d) const {
        Type result(0);
        for (std::size_t i = 0u; i < d; ++i) {
            result += this->element(i, i);
        }
        return result;
    }

    //! The Frobenius norm.
    double frobenius(std::size_t d) const {
        double result = 0.0;
        for (std::size_t i = 0u, i_ = 0u; i < d; ++i, ++i_) {
            for (std::size_t j = 0u; j < i; ++j, ++i_) {
                result += 2.0 * m_LowerTriangle[i_] * m_LowerTriangle[i_];
            }
            result += m_LowerTriangle[i_] * m_LowerTriangle[i_];
        }
        return std::sqrt(result);
    }

    //! Convert to the MATRIX representation.
    template<typename MATRIX>
    inline MATRIX& toType(std::size_t d, MATRIX& result) const {
        for (std::size_t i = 0u, i_ = 0u; i < d; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++i_) {
                result(i, j) = result(j, i) = m_LowerTriangle[i_];
            }
        }
        return result;
    }

    //! Get a checksum of the elements of this matrix.
    uint64_t checksum() const {
        uint64_t result = 0u;
        for (std::size_t i = 0u; i < m_LowerTriangle.size(); ++i) {
            result = core::CHashing::hashCombine(result, static_cast<uint64_t>(m_LowerTriangle[i]));
        }
        return result;
    }

    STORAGE m_LowerTriangle;
};

} // linear_algebra_detail::

// ************************ STACK SYMMETRIC MATRIX ************************

//! \brief A stack based lightweight dense symmetric matrix class.
//!
//! DESCRIPTION:\n
//! This implements a stack based mathematical symmetric matrix object.
//! The idea is to provide a few simple to implement utility functions,
//! however it is primarily intended for storage and is not an alternative
//! to a good linear analysis package implementation. In fact, all utilities
//! for doing any serious linear algebra should convert this to the Eigen
//! library self adjoint representation, an implicit conversion operator
//! for doing this has been supplied. Commonly used operations on matrices
//! for example computing the inverse quadratic product or determinant
//! should be added to this header.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This uses the best possible encoding for space, i.e. packed into a
//! N * (N+1) / 2 length array. This is not the best representation to use
//! for speed as it cuts down on vectorization opportunities. The Eigen
//! library does not support a packed representation for exactly this
//! reason. Our requirements are somewhat different, i.e. we potentially
//! want to store a lot of small matrices with lowest possible space
//! overhead.
//!
//! This also provides a convenience constructor to initialize to a
//! multiple of the ones matrix. Any bounds checking in matrix matrix and
//! matrix vector operations is compile time since the size is a template
//! parameter. The floating point type is templated so that one can use
//! float when space really at a premium.
//!
//! \tparam T The floating point type.
//! \tparam N The matrix dimension.
// clang-format off
template<typename T, std::size_t N>
class CSymmetricMatrixNxN : private boost::equality_comparable< CSymmetricMatrixNxN<T, N>,
                                    boost::partially_ordered< CSymmetricMatrixNxN<T, N>,
                                    boost::addable< CSymmetricMatrixNxN<T, N>,
                                    boost::subtractable< CSymmetricMatrixNxN<T, N>,
                                    boost::multipliable< CSymmetricMatrixNxN<T, N>,
                                    boost::multipliable2< CSymmetricMatrixNxN<T, N>, T,
                                    boost::dividable2< CSymmetricMatrixNxN<T, N>, T > > > > > > >,
                            private linear_algebra_detail::SSymmetricMatrix<boost::array<T, N * (N + 1) / 2> >,
                            private linear_algebra_detail::CBoundsCheck<N>::InRange {
    // clang-format on
private:
    using TBase = linear_algebra_detail::SSymmetricMatrix<boost::array<T, N*(N + 1) / 2>>;
    template<typename U, std::size_t>
    friend class CSymmetricMatrixNxN;

public:
    using TArray = T[N][N];
    using TVec = std::vector<T>;
    using TVecVec = std::vector<TVec>;
    using TConstIterator = typename boost::array<T, N*(N + 1) / 2>::const_iterator;

public:
    //! See core::CMemory.
    static bool dynamicSizeAlwaysZero() { return core::memory_detail::SDynamicSizeAlwaysZero<T>::value(); }

public:
    //! Set to multiple of ones matrix.
    explicit CSymmetricMatrixNxN(T v = T(0)) { std::fill_n(&TBase::m_LowerTriangle[0], N * (N + 1) / 2, v); }

    //! Construct from C-style array of arrays.
    explicit CSymmetricMatrixNxN(const TArray& m) {
        for (std::size_t i = 0u, i_ = 0u; i < N; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++i_) {
                TBase::m_LowerTriangle[i_] = m[i][j];
            }
        }
    }

    //! Construct from a vector of vectors.
    explicit CSymmetricMatrixNxN(const TVecVec& m) {
        for (std::size_t i = 0u, i_ = 0u; i < N; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++i_) {
                TBase::m_LowerTriangle[i_] = m[i][j];
            }
        }
    }

    //! Construct from a small vector of small vectors.
    template<std::size_t M>
    explicit CSymmetricMatrixNxN(const core::CSmallVectorBase<core::CSmallVector<T, M>>& m) {
        for (std::size_t i = 0u, i_ = 0u; i < N; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++i_) {
                TBase::m_LowerTriangle[i_] = m[i][j];
            }
        }
    }

    //! Construct from a forward iterator.
    //!
    //! \warning The user must ensure that the range iterated has
    //! at least N (N+1) / 2 items.
    template<typename ITR>
    CSymmetricMatrixNxN(ITR begin, ITR end) {
        for (std::size_t i = 0u; i < N * (N + 1) / 2 && begin != end; ++i, ++begin) {
            TBase::m_LowerTriangle[i] = static_cast<T>(*begin);
        }
    }

    explicit CSymmetricMatrixNxN(ESymmetricMatrixType type, const CVectorNx1<T, N>& x);

    //! Construct from a dense matrix.
    template<typename MATRIX>
    CSymmetricMatrixNxN(const CDenseMatrixInitializer<MATRIX>& m);

    //! Copy construction if the underlying type is implicitly
    //! convertible.
    template<typename U>
    CSymmetricMatrixNxN(const CSymmetricMatrixNxN<U, N>& other) {
        this->operator=(other);
    }

    //! Assignment if the underlying type is implicitly convertible.
    template<typename U>
    const CSymmetricMatrixNxN& operator=(const CSymmetricMatrixNxN<U, N>& other) {
        this->assign(other.base());
        return *this;
    }

    //! \name Persistence
    //@{
    //! Create from a delimited string.
    bool fromDelimited(const std::string& str) { return this->TBase::fromDelimited(str); }

    //! Convert to a delimited string.
    std::string toDelimited() const { return this->TBase::toDelimited(); }
    //@}

    //! Get the number of rows.
    std::size_t rows() const { return N; }

    //! Get the number of columns.
    std::size_t columns() const { return N; }

    //! Get the i,j 'th component (no bounds checking).
    inline T operator()(std::size_t i, std::size_t j) const { return this->element(i, j); }

    //! Get the i,j 'th component (no bounds checking).
    inline T& operator()(std::size_t i, std::size_t j) { return this->element(i, j); }

    //! Get an iterator over the elements.
    TConstIterator begin() const { return TBase::m_LowerTriangle.begin(); }

    //! Get an iterator to the end of the elements.
    TConstIterator end() const { return TBase::m_LowerTriangle.end(); }

    //! Component-wise negation.
    CSymmetricMatrixNxN operator-() const {
        CSymmetricMatrixNxN result(*this);
        result.negative();
        return result;
    }

    //! Matrix subtraction.
    const CSymmetricMatrixNxN& operator-=(const CSymmetricMatrixNxN& rhs) {
        this->minusEquals(rhs.base());
        return *this;
    }

    //! Matrix addition.
    const CSymmetricMatrixNxN& operator+=(const CSymmetricMatrixNxN& rhs) {
        this->plusEquals(rhs.base());
        return *this;
    }

    //! Component-wise multiplication.
    //!
    //! \note This is handy in some cases and since symmetric matrices
    //! are not closed under regular matrix multiplication we use
    //! multiplication operator for implementing the Hadamard product.
    const CSymmetricMatrixNxN& operator*=(const CSymmetricMatrixNxN& rhs) {
        this->multiplyEquals(rhs);
        return *this;
    }

    //! Scalar multiplication.
    const CSymmetricMatrixNxN& operator*=(T scale) {
        this->multiplyEquals(scale);
        return *this;
    }

    //! Scalar division.
    const CSymmetricMatrixNxN& operator/=(T scale) {
        this->divideEquals(scale);
        return *this;
    }

    // Matrix multiplication doesn't necessarily produce a symmetric
    // matrix because matrix multiplication is non-commutative.
    // Matrix division requires computing the inverse and is not
    // supported.

    //! Check if two matrices are identically equal.
    bool operator==(const CSymmetricMatrixNxN& other) const { return this->equal(other.base()); }

    //! Lexicographical total ordering.
    bool operator<(const CSymmetricMatrixNxN& rhs) const { return this->less(rhs.base()); }

    //! Check if this is zero.
    bool isZero() const { return this->TBase::isZero(); }

    //! Get the matrix diagonal.
    template<typename VECTOR>
    VECTOR diagonal() const {
        return this->TBase::template diagonal<VECTOR>(N);
    }

    //! Get the trace.
    T trace() const { return this->TBase::trace(N); }

    //! Get the Frobenius norm.
    double frobenius() const { return this->TBase::frobenius(N); }

    //! Convert to a vector of vectors.
    template<typename VECTOR_OF_VECTORS>
    inline VECTOR_OF_VECTORS toVectors() const {
        VECTOR_OF_VECTORS result(N);
        for (std::size_t i = 0u; i < N; ++i) {
            result[i].resize(N);
        }
        for (std::size_t i = 0u; i < N; ++i) {
            result[i][i] = this->operator()(i, i);
            for (std::size_t j = 0u; j < i; ++j) {
                result[i][j] = result[j][i] = this->operator()(i, j);
            }
        }
        return result;
    }

    //! Convert to the specified matrix representation.
    //!
    //! \note The copy should be avoided by RVO.
    template<typename MATRIX>
    inline MATRIX toType() const {
        MATRIX result(N, N);
        return this->TBase::toType(N, result);
    }

    //! Get a checksum for the matrix.
    uint64_t checksum() const { return this->TBase::checksum(); }
};

//! \brief Gets a zero symmetric matrix with specified dimension.
template<typename T, std::size_t N>
struct SZero<CSymmetricMatrixNxN<T, N>> {
    static CSymmetricMatrixNxN<T, N> get(std::size_t /*dimension*/) { return CSymmetricMatrixNxN<T, N>(T(0)); }
};

// ************************ HEAP SYMMETRIC MATRIX ************************

//! \brief A heap based lightweight dense symmetric matrix class.
//!
//! DESCRIPTION:\n
//! This implements a heap based mathematical symmetric matrix object.
//! The idea is to provide a few simple to implement utility functions,
//! however it is primarily intended for storage and is not an alternative
//! to a good linear analysis package implementation. In fact, all utilities
//! for doing any serious linear algebra should convert this to the Eigen
//! library self adjoint representation, an explicit conversion operator
//! for doing this has been supplied. Commonly used operations on matrices
//! for example computing the inverse quadratic product or determinant
//! should be added to this header.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This uses the best possible encoding for space, i.e. packed into a
//! D * (D+1) / 2 length vector where D is the dimension. This is not the
//! best representation to use for speed as it cuts down on vectorization
//! opportunities. The Eigen library does not support a packed representation
//! for exactly this reason. Our requirements are somewhat different, i.e.
//! we potentially want to store a lot of small(ish) matrices with lowest
//! possible space overhead.
//!
//! This also provides a convenience constructor to initialize to a
//! multiple of the ones matrix. There is no bounds checking in matrix
//! matrix and matrix vector operations for speed. The floating point
//! type is templated so that one can use float when space really at a
//! premium.
//!
//! \tparam T The floating point type.
// clang-format off
template<typename T>
class CSymmetricMatrix : private boost::equality_comparable< CSymmetricMatrix<T>,
                                 boost::partially_ordered< CSymmetricMatrix<T>,
                                 boost::addable< CSymmetricMatrix<T>,
                                 boost::subtractable< CSymmetricMatrix<T>,
                                 boost::multipliable< CSymmetricMatrix<T>,
                                 boost::multipliable2< CSymmetricMatrix<T>, T,
                                 boost::dividable2< CSymmetricMatrix<T>, T > > > > > > >,
                         private linear_algebra_detail::SSymmetricMatrix<std::vector<T> > {
    // clang-format on
private:
    using TBase = linear_algebra_detail::SSymmetricMatrix<std::vector<T>>;
    template<typename U>
    friend class CSymmetricMatrix;

public:
    using TArray = std::vector<std::vector<T>>;
    using TConstIterator = typename std::vector<T>::const_iterator;

public:
    //! Set to multiple of ones matrix.
    explicit CSymmetricMatrix(std::size_t d = 0u, T v = T(0)) : m_D(d) {
        if (d > 0) {
            TBase::m_LowerTriangle.resize(d * (d + 1) / 2, v);
        }
    }

    //! Construct from C-style array of arrays.
    explicit CSymmetricMatrix(const TArray& m) : m_D(m.size()) {
        TBase::m_LowerTriangle.resize(m_D * (m_D + 1) / 2);
        for (std::size_t i = 0u, i_ = 0u; i < m_D; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++i_) {
                TBase::m_LowerTriangle[i_] = m[i][j];
            }
        }
    }

    //! Construct from a small vector of small vectors.
    template<std::size_t M>
    explicit CSymmetricMatrix(const core::CSmallVectorBase<core::CSmallVector<T, M>>& m) : m_D(m.size()) {
        TBase::m_LowerTriangle.resize(m_D * (m_D + 1) / 2);
        for (std::size_t i = 0u, i_ = 0u; i < m_D; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++i_) {
                TBase::m_LowerTriangle[i_] = m[i][j];
            }
        }
    }

    //! Construct from a forward iterator.
    //!
    //! \warning The user must ensure that the range iterated has
    //! at least N (N+1) / 2 items.
    template<typename ITR>
    CSymmetricMatrix(ITR begin, ITR end) {
        m_D = this->dimension(std::distance(begin, end));
        TBase::m_LowerTriangle.resize(m_D * (m_D + 1) / 2);
        for (std::size_t i = 0u; i < m_D * (m_D + 1) / 2 && begin != end; ++i, ++begin) {
            TBase::m_LowerTriangle[i] = static_cast<T>(*begin);
        }
    }

    explicit CSymmetricMatrix(ESymmetricMatrixType type, const CVector<T>& x);

    //! Construct from a dense matrix.
    template<typename MATRIX>
    CSymmetricMatrix(const CDenseMatrixInitializer<MATRIX>& m);

    //! Copy construction if the underlying type is implicitly
    //! convertible.
    template<typename U>
    CSymmetricMatrix(const CSymmetricMatrix<U>& other) : m_D(other.m_D) {
        this->operator=(other);
    }

    //! Assignment if the underlying type is implicitly convertible.
    template<typename U>
    const CSymmetricMatrix& operator=(const CSymmetricMatrix<U>& other) {
        m_D = other.m_D;
        TBase::m_LowerTriangle.resize(m_D * (m_D + 1) / 2);
        this->assign(other.base());
        return *this;
    }

    //! Efficiently swap the contents of two matrices.
    void swap(CSymmetricMatrix& other) {
        std::swap(m_D, other.m_D);
        TBase::m_LowerTriangle.swap(other.TBase::m_LowerTriangle);
    }

    //! \name Persistence
    //@{
    //! Create from a delimited string.
    bool fromDelimited(const std::string& str) {
        if (this->TBase::fromDelimited(str)) {
            m_D = this->dimension(TBase::m_X.size());
            return true;
        }
        return false;
    }

    //! Convert to a delimited string.
    std::string toDelimited() const { return this->TBase::toDelimited(); }
    //@}

    //! Get the number of rows.
    std::size_t rows() const { return m_D; }

    //! Get the number of columns.
    std::size_t columns() const { return m_D; }

    //! Get the i,j 'th component (no bounds checking).
    inline T operator()(std::size_t i, std::size_t j) const { return this->element(i, j); }

    //! Get the i,j 'th component (no bounds checking).
    inline T& operator()(std::size_t i, std::size_t j) { return this->element(i, j); }

    //! Get an iterator over the elements.
    TConstIterator begin() const { return TBase::m_X.begin(); }

    //! Get an iterator to the end of the elements.
    TConstIterator end() const { return TBase::m_X.end(); }

    //! Component-wise negation.
    CSymmetricMatrix operator-() const {
        CSymmetricMatrix result(*this);
        result.negative();
        return result;
    }

    //! Matrix subtraction.
    const CSymmetricMatrix& operator-=(const CSymmetricMatrix& rhs) {
        this->minusEquals(rhs.base());
        return *this;
    }

    //! Matrix addition.
    const CSymmetricMatrix& operator+=(const CSymmetricMatrix& rhs) {
        this->plusEquals(rhs.base());
        return *this;
    }

    //! Component-wise multiplication.
    //!
    //! \note This is handy in some cases and since symmetric matrices
    //! are not closed under regular matrix multiplication we use
    //! multiplication operator for implementing the Hadamard product.
    const CSymmetricMatrix& operator*=(const CSymmetricMatrix& rhs) {
        this->multiplyEquals(rhs);
        return *this;
    }

    //! Scalar multiplication.
    const CSymmetricMatrix& operator*=(T scale) {
        this->multiplyEquals(scale);
        return *this;
    }

    //! Scalar division.
    const CSymmetricMatrix& operator/=(T scale) {
        this->divideEquals(scale);
        return *this;
    }

    // Matrix multiplication doesn't necessarily produce a symmetric
    // matrix because matrix multiplication is non-commutative.
    // Matrix division requires computing the inverse and is not
    // supported.

    //! Check if two matrices are identically equal.
    bool operator==(const CSymmetricMatrix& other) const { return this->equal(other.base()); }

    //! Lexicographical total ordering.
    bool operator<(const CSymmetricMatrix& rhs) const { return this->less(rhs.base()); }

    //! Check if this is zero.
    bool isZero() const { return this->TBase::isZero(); }

    //! Get the matrix diagonal.
    template<typename VECTOR>
    VECTOR diagonal() const {
        return this->TBase::template diagonal<VECTOR>(m_D);
    }

    //! Get the trace.
    T trace() const { return this->TBase::trace(m_D); }

    //! The Frobenius norm.
    double frobenius() const { return this->TBase::frobenius(m_D); }

    //! Convert to a vector of vectors.
    template<typename VECTOR_OF_VECTORS>
    inline VECTOR_OF_VECTORS toVectors() const {
        VECTOR_OF_VECTORS result(m_D);
        for (std::size_t i = 0u; i < m_D; ++i) {
            result[i].resize(m_D);
        }
        for (std::size_t i = 0u; i < m_D; ++i) {
            result[i][i] = this->operator()(i, i);
            for (std::size_t j = 0u; j < i; ++j) {
                result[i][j] = result[j][i] = this->operator()(i, j);
            }
        }
        return result;
    }

    //! Convert to the specified matrix representation.
    //!
    //! \note The copy should be avoided by RVO.
    template<typename MATRIX>
    inline MATRIX toType() const {
        MATRIX result(m_D, m_D);
        return this->TBase::toType(m_D, result);
    }

    //! Get a checksum for the matrix.
    uint64_t checksum() const { return core::CHashing::hashCombine(this->TBase::checksum(), static_cast<uint64_t>(m_D)); }

private:
    //! Compute the dimension from the number of elements.
    std::size_t dimension(std::size_t n) const {
        return static_cast<std::size_t>((std::sqrt(8.0 * static_cast<double>(n) + 1.0) - 1.0) / 2.0 + 0.5);
    }

private:
    //! The rows (and columns) of this matrix.
    std::size_t m_D;
};

//! \brief Gets a zero symmetric matrix with specified dimension.
template<typename T>
struct SZero<CSymmetricMatrix<T>> {
    static CSymmetricMatrix<T> get(std::size_t dimension) { return CSymmetricMatrix<T>(dimension, T(0)); }
};

namespace linear_algebra_detail {

//! \brief Common vector functionality for variable storage type.
template<typename STORAGE>
struct SVector {
    using Type = typename STORAGE::value_type;

    //! Get read only reference.
    inline const SVector& base() const { return *this; }

    //! Get writable reference.
    inline SVector& base() { return *this; }

    //! Set this vector equal to \p other.
    template<typename OTHER_STORAGE>
    void assign(const SVector<OTHER_STORAGE>& other) {
        std::copy(other.m_X.begin(), other.m_X.end(), m_X.begin());
    }

    //! Create from delimited values.
    bool fromDelimited(const std::string& str);

    //! Convert to a delimited string.
    std::string toDelimited() const;

    //! Component-wise negative.
    void negative() {
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            m_X[i] = -m_X[i];
        }
    }

    //! Vector subtraction.
    void minusEquals(const SVector& rhs) {
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            m_X[i] -= rhs.m_X[i];
        }
    }

    //! Vector addition.
    void plusEquals(const SVector& rhs) {
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            m_X[i] += rhs.m_X[i];
        }
    }

    //! Component-wise multiplication.
    void multiplyEquals(const SVector& scale) {
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            m_X[i] *= scale.m_X[i];
        }
    }

    //! Scalar multiplication.
    void multiplyEquals(Type scale) {
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            m_X[i] *= scale;
        }
    }

    //! Component-wise division.
    void divideEquals(const SVector& scale) {
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            m_X[i] /= scale.m_X[i];
        }
    }

    //! Scalar division.
    void divideEquals(Type scale) {
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            m_X[i] /= scale;
        }
    }

    //! Compare this and \p other for equality.
    bool equal(const SVector& other) const { return m_X == other.m_X; }

    //! Lexicographical total ordering.
    bool less(const SVector& rhs) const { return m_X < rhs.m_X; }

    //! Check if this is zero.
    bool isZero() const {
        return std::find_if(m_X.begin(), m_X.end(), [](double xi) { return xi != 0.0; }) == m_X.end();
    }

    //! Inner product.
    double inner(const SVector& covector) const {
        double result = 0.0;
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            result += m_X[i] * covector.m_X[i];
        }
        return result;
    }

    //! Inner product.
    template<typename VECTOR>
    double inner(const VECTOR& covector) const {
        double result = 0.0;
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            result += m_X[i] * covector(i);
        }
        return result;
    }

    //! The L1 norm of the vector.
    double L1() const {
        double result = 0.0;
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            result += std::fabs(static_cast<double>(m_X[i]));
        }
        return result;
    }

    //! Convert to the VECTOR representation.
    template<typename VECTOR>
    inline VECTOR& toType(VECTOR& result) const {
        for (std::size_t i = 0u; i < m_X.size(); ++i) {
            result(i) = m_X[i];
        }
        return result;
    }

    //! Get a checksum of the components of this vector.
    uint64_t checksum() const {
        uint64_t result = static_cast<uint64_t>(m_X[0]);
        for (std::size_t i = 1u; i < m_X.size(); ++i) {
            result = core::CHashing::hashCombine(result, static_cast<uint64_t>(m_X[i]));
        }
        return result;
    }

    //! The components
    STORAGE m_X;
};

} // linear_algebra_detail::

// ************************ STACK VECTOR ************************

//! \brief A stack based lightweight dense vector class.
//!
//! DESCRIPTION:\n
//! This implements a stack based mathematical vector object. The idea
//! is to provide utility functions and operators which mean that it
//! works with other ml::maths:: classes, such as the symmetric
//! matrix object and the sample (co)variance accumulators, and keep the
//! memory footprint as small as possible. This is not meant to be an
//! alternative to a good linear analysis package implementation. For
//! example, if you want to any serious linear algebra use the Eigen
//! library. An implicit conversion operator for doing this has been
//! supplied.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Operators follow the Matlab component-wise convention. This provides
//! a constructor to initialize to a multiple of the 1 vector. Bounds
//! checking for vector vector and matrix vector operations is compile
//! time since the size is a template parameter. The floating point type
//! is templated so that one can use float when space is really at a
//! premium.
//!
//! \tparam T The floating point type.
//! \tparam N The vector dimension.
// clang-format off
template<typename T, std::size_t N>
class CVectorNx1 : private boost::equality_comparable< CVectorNx1<T, N>,
                           boost::partially_ordered< CVectorNx1<T, N>,
                           boost::addable< CVectorNx1<T, N>,
                           boost::subtractable< CVectorNx1<T, N>,
                           boost::multipliable< CVectorNx1<T, N>,
                           boost::multipliable2< CVectorNx1<T, N>, T,
                           boost::dividable< CVectorNx1<T, N>,
                           boost::dividable2< CVectorNx1<T, N>, T > > > > > > > >,
                   private linear_algebra_detail::SVector<boost::array<T, N> >,
                   private linear_algebra_detail::CBoundsCheck<N>::InRange {
    // clang-format on
private:
    using TBase = linear_algebra_detail::SVector<boost::array<T, N>>;
    template<typename U, std::size_t>
    friend class CVectorNx1;

public:
    using TArray = T[N];
    using TVec = std::vector<T>;
    using TBoostArray = boost::array<T, N>;
    using TConstIterator = typename TBoostArray::const_iterator;

public:
    //! See core::CMemory.
    static bool dynamicSizeAlwaysZero() { return core::memory_detail::SDynamicSizeAlwaysZero<T>::value(); }

public:
    //! Set to multiple of ones vector.
    explicit CVectorNx1(T v = T(0)) { std::fill_n(&TBase::m_X[0], N, v); }

    //! Construct from a C-style array.
    explicit CVectorNx1(const TArray& v) {
        for (std::size_t i = 0u; i < N; ++i) {
            TBase::m_X[i] = v[i];
        }
    }

    //! Construct from a boost array.
    explicit CVectorNx1(const boost::array<T, N>& a) {
        for (std::size_t i = 0u; i < N; ++i) {
            TBase::m_X[i] = a[i];
        }
    }

    //! Construct from a vector.
    explicit CVectorNx1(const TVec& v) {
        for (std::size_t i = 0u; i < N; ++i) {
            TBase::m_X[i] = v[i];
        }
    }

    //! Construct from a vector.
    explicit CVectorNx1(const core::CSmallVectorBase<T>& v) {
        for (std::size_t i = 0u; i < N; ++i) {
            TBase::m_X[i] = v[i];
        }
    }

    //! Construct from a forward iterator.
    //!
    //! \warning The user must ensure that the range iterated has
    //! at least N items.
    template<typename ITR>
    CVectorNx1(ITR begin, ITR end) {
        if (std::distance(begin, end) != N) {
            LOG_ERROR("Bad range");
            return;
        }
        std::copy(begin, end, &TBase::m_X[0]);
    }

    //! Construct from a dense vector.
    template<typename VECTOR>
    CVectorNx1(const CDenseVectorInitializer<VECTOR>& v);

    //! Copy construction if the underlying type is implicitly
    //! convertible.
    template<typename U>
    CVectorNx1(const CVectorNx1<U, N>& other) {
        this->operator=(other);
    }

    //! Assignment if the underlying type is implicitly convertible.
    template<typename U>
    const CVectorNx1& operator=(const CVectorNx1<U, N>& other) {
        this->assign(other.base());
        return *this;
    }

    //! \name Persistence
    //@{
    //! Create from a delimited string.
    bool fromDelimited(const std::string& str) { return this->TBase::fromDelimited(str); }

    //! Convert to a delimited string.
    std::string toDelimited() const { return this->TBase::toDelimited(); }
    //@}

    //! Get the dimension.
    std::size_t dimension() const { return N; }

    //! Get the i'th component (no bounds checking).
    inline T operator()(std::size_t i) const { return TBase::m_X[i]; }

    //! Get the i'th component (no bounds checking).
    inline T& operator()(std::size_t i) { return TBase::m_X[i]; }

    //! Get an iterator over the elements.
    TConstIterator begin() const { return TBase::m_X.begin(); }

    //! Get an iterator to the end of the elements.
    TConstIterator end() const { return TBase::m_X.end(); }

    //! Component-wise negation.
    CVectorNx1 operator-() const {
        CVectorNx1 result(*this);
        result.negative();
        return result;
    }

    //! Vector subtraction.
    const CVectorNx1& operator-=(const CVectorNx1& lhs) {
        this->minusEquals(lhs.base());
        return *this;
    }

    //! Vector addition.
    const CVectorNx1& operator+=(const CVectorNx1& lhs) {
        this->plusEquals(lhs.base());
        return *this;
    }

    //! Component-wise multiplication.
    const CVectorNx1& operator*=(const CVectorNx1& scale) {
        this->multiplyEquals(scale.base());
        return *this;
    }

    //! Scalar multiplication.
    const CVectorNx1& operator*=(T scale) {
        this->multiplyEquals(scale);
        return *this;
    }

    //! Component-wise division.
    const CVectorNx1& operator/=(const CVectorNx1& scale) {
        this->divideEquals(scale.base());
        return *this;
    }

    //! Scalar division.
    const CVectorNx1& operator/=(T scale) {
        this->divideEquals(scale);
        return *this;
    }

    //! Check if two vectors are identically equal.
    bool operator==(const CVectorNx1& other) const { return this->equal(other.base()); }

    //! Lexicographical total ordering.
    bool operator<(const CVectorNx1& rhs) const { return this->less(rhs.base()); }

    //! Check if this is zero.
    bool isZero() const { return this->TBase::isZero(); }

    //! Inner product.
    double inner(const CVectorNx1& covector) const { return this->TBase::inner(covector.base()); }

    //! Inner product.
    template<typename VECTOR>
    double inner(const VECTOR& covector) const {
        return this->TBase::template inner<VECTOR>(covector);
    }

    //! Outer product.
    //!
    //! \note The copy should be avoided by RVO.
    CSymmetricMatrixNxN<T, N> outer() const { return CSymmetricMatrixNxN<T, N>(E_OuterProduct, *this); }

    //! A diagonal matrix.
    //!
    //! \note The copy should be avoided by RVO.
    CSymmetricMatrixNxN<T, N> diagonal() const { return CSymmetricMatrixNxN<T, N>(E_Diagonal, *this); }

    //! L1 norm.
    double L1() const { return this->TBase::L1(); }

    //! Euclidean norm.
    double euclidean() const { return std::sqrt(this->inner(*this)); }

    //! Convert to a vector on a different underlying type.
    template<typename U>
    inline CVectorNx1<U, N> to() const {
        return CVectorNx1<U, N>(*this);
    }

    //! Convert to a vector.
    template<typename VECTOR>
    inline VECTOR toVector() const {
        return VECTOR(this->begin(), this->end());
    }

    //! Convert to a boost array.
    inline TBoostArray toBoostArray() const { return TBase::m_X; }

    //! Convert to the specified vector representation.
    //!
    //! \note The copy should be avoided by RVO.
    template<typename VECTOR>
    inline VECTOR toType() const {
        VECTOR result(N);
        return this->TBase::toType(result);
    }

    //! Get a checksum of this vector's components.
    uint64_t checksum() const { return this->TBase::checksum(); }

    //! Get the smallest possible vector.
    static const CVectorNx1& smallest() {
        static const CVectorNx1 result(boost::numeric::bounds<T>::lowest());
        return result;
    }

    //! Get the largest possible vector.
    static const CVectorNx1& largest() {
        static const CVectorNx1 result(boost::numeric::bounds<T>::highest());
        return result;
    }
};

//! Construct from the outer product of a vector with itself.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N>::CSymmetricMatrixNxN(ESymmetricMatrixType type, const CVectorNx1<T, N>& x) {
    switch (type) {
    case E_OuterProduct:
        for (std::size_t i = 0u, i_ = 0u; i < N; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++i_) {
                TBase::m_LowerTriangle[i_] = x(i) * x(j);
            }
        }
        break;
    case E_Diagonal:
        for (std::size_t i = 0u, i_ = 0u; i < N; ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++i_) {
                TBase::m_LowerTriangle[i_] = i == j ? x(i) : T(0);
            }
        }
        break;
    }
}

//! \brief Gets a zero vector with specified dimension.
template<typename T, std::size_t N>
struct SZero<CVectorNx1<T, N>> {
    static CVectorNx1<T, N> get(std::size_t /*dimension*/) { return CVectorNx1<T, N>(T(0)); }
};

// ************************ HEAP VECTOR ************************

//! \brief A heap based lightweight dense vector class.
//!
//! DESCRIPTION:\n
//! This implements a heap based mathematical vector object. The idea
//! is to provide utility functions and operators which mean that it
//! works with other ml::maths:: classes, such as the symmetric
//! matrix object and the sample (co)variance accumulators, and keep the
//! memory footprint as small as possible. This is not meant to be an
//! alternative to a good linear analysis package implementation. For
//! example, if you want to any serious linear algebra use the Eigen
//! library. An implicit conversion operator for doing this has been
//! supplied.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Operators follow the Matlab component-wise convention. This provides
//! a constructor to initialize to a multiple of the 1 vector. There is
//! no bounds checking for efficiency. The floating point type is templated
//! so that one can use float when space is really at a premium.
//!
//! \tparam T The floating point type.
// clang-format off
template<typename T>
class CVector : private boost::equality_comparable< CVector<T>,
                        boost::partially_ordered< CVector<T>,
                        boost::addable< CVector<T>,
                        boost::subtractable< CVector<T>,
                        boost::multipliable< CVector<T>,
                        boost::multipliable2< CVector<T>, T,
                        boost::dividable< CVector<T>,
                        boost::dividable2< CVector<T>, T > > > > > > > >,
                private linear_algebra_detail::SVector<std::vector<T> > {
    // clang-format on
private:
    using TBase = linear_algebra_detail::SVector<std::vector<T>>;
    template<typename U>
    friend class CVector;

public:
    using TArray = std::vector<T>;
    using TConstIterator = typename TArray::const_iterator;

public:
    //! Set to multiple of ones vector.
    explicit CVector(std::size_t d = 0u, T v = T(0)) {
        if (d > 0) {
            TBase::m_X.resize(d, v);
        }
    }

    //! Construct from a boost array.
    template<std::size_t N>
    explicit CVector(const boost::array<T, N>& a) {
        for (std::size_t i = 0u; i < N; ++i) {
            TBase::m_X[i] = a[i];
        }
    }

    //! Construct from a vector.
    explicit CVector(const TArray& v) { TBase::m_X = v; }

    //! Construct from a vector.
    explicit CVector(const core::CSmallVectorBase<T>& v) { TBase::m_X.assign(v.begin(), v.end()); }

    //! Construct from the range [\p begin, \p end).
    template<typename ITR>
    CVector(ITR begin, ITR end) {
        TBase::m_X.assign(begin, end);
    }

    //! Construct from a dense vector.
    template<typename VECTOR>
    CVector(const CDenseVectorInitializer<VECTOR>& v);

    //! Copy construction if the underlying type is implicitly
    //! convertible.
    template<typename U>
    CVector(const CVector<U>& other) {
        this->operator=(other);
    }

    //! Assignment if the underlying type is implicitly convertible.
    template<typename U>
    const CVector& operator=(const CVector<U>& other) {
        TBase::m_X.resize(other.dimension());
        this->TBase::assign(other.base());
        return *this;
    }

    //! Efficiently swap the contents of two vectors.
    void swap(CVector& other) { TBase::m_X.swap(other.TBase::m_X); }

    //! Reserve enough memory to hold \p d components.
    void reserve(std::size_t d) { TBase::m_X.reserve(d); }

    //! Assign the components from the range [\p begin, \p end).
    template<typename ITR>
    void assign(ITR begin, ITR end) {
        TBase::m_X.assign(begin, end);
    }

    //! Extend the vector to dimension \p d adding components
    //! initialized to \p v.
    void extend(std::size_t d, T v = T(0)) { TBase::m_X.resize(this->dimension() + d, v); }

    //! Extend the vector adding components initialized to \p v.
    template<typename ITR>
    void extend(ITR begin, ITR end) {
        TBase::m_X.insert(TBase::m_X.end(), begin, end);
    }

    //! \name Persistence
    //@{
    //! Create from a delimited string.
    bool fromDelimited(const std::string& str) { return this->TBase::fromDelimited(str); }

    //! Persist state to delimited values.
    std::string toDelimited() const { return this->TBase::toDelimited(); }
    //@}

    //! Get the dimension.
    std::size_t dimension() const { return TBase::m_X.size(); }

    //! Get the i'th component (no bounds checking).
    inline T operator()(std::size_t i) const { return TBase::m_X[i]; }

    //! Get the i'th component (no bounds checking).
    inline T& operator()(std::size_t i) { return TBase::m_X[i]; }

    //! Get an iterator over the elements.
    TConstIterator begin() const { return TBase::m_X.begin(); }

    //! Get an iterator to the end of the elements.
    TConstIterator end() const { return TBase::m_X.end(); }

    //! Component-wise negation.
    CVector operator-() const {
        CVector result(*this);
        result.negative();
        return result;
    }

    //! Vector subtraction.
    const CVector& operator-=(const CVector& lhs) {
        this->minusEquals(lhs.base());
        return *this;
    }

    //! Vector addition.
    const CVector& operator+=(const CVector& lhs) {
        this->plusEquals(lhs.base());
        return *this;
    }

    //! Component-wise multiplication.
    const CVector& operator*=(const CVector& scale) {
        this->multiplyEquals(scale.base());
        return *this;
    }

    //! Scalar multiplication.
    const CVector& operator*=(T scale) {
        this->multiplyEquals(scale);
        return *this;
    }

    //! Component-wise division.
    const CVector& operator/=(const CVector& scale) {
        this->divideEquals(scale.base());
        return *this;
    }

    //! Scalar division.
    const CVector& operator/=(T scale) {
        this->divideEquals(scale);
        return *this;
    }

    //! Check if two vectors are identically equal.
    bool operator==(const CVector& other) const { return this->equal(other.base()); }

    //! Lexicographical total ordering.
    bool operator<(const CVector& rhs) const { return this->less(rhs.base()); }

    //! Check if this is zero.
    bool isZero() const { return this->TBase::isZero(); }

    //! Inner product.
    double inner(const CVector& covector) const { return this->TBase::inner(covector.base()); }

    //! Inner product.
    template<typename VECTOR>
    double inner(const VECTOR& covector) const {
        return this->TBase::template inner<VECTOR>(covector);
    }

    //! Outer product.
    //!
    //! \note The copy should be avoided by RVO.
    CSymmetricMatrix<T> outer() const { return CSymmetricMatrix<T>(E_OuterProduct, *this); }

    //! A diagonal matrix.
    //!
    //! \note The copy should be avoided by RVO.
    CSymmetricMatrix<T> diagonal() const { return CSymmetricMatrix<T>(E_Diagonal, *this); }

    //! L1 norm.
    double L1() const { return this->TBase::L1(); }

    //! Euclidean norm.
    double euclidean() const { return std::sqrt(this->inner(*this)); }

    //! Convert to a vector on a different underlying type.
    template<typename U>
    inline CVector<U> to() const {
        return CVector<U>(*this);
    }

    //! Convert to a vector.
    template<typename VECTOR>
    inline VECTOR toVector() const {
        return VECTOR(this->begin(), this->end());
    }

    //! Convert to the specified vector representation.
    //!
    //! \note The copy should be avoided by RVO.
    template<typename VECTOR>
    inline VECTOR toType() const {
        VECTOR result(this->dimension());
        return this->TBase::toType(result);
    }

    //! Get a checksum of this vector's components.
    uint64_t checksum() const { return this->TBase::checksum(); }

    //! Get the smallest possible vector.
    static const CVector& smallest(std::size_t d) {
        static const CVector result(d, boost::numeric::bounds<T>::lowest());
        return result;
    }

    //! Get the largest possible vector.
    static const CVector& largest(std::size_t d) {
        static const CVector result(d, boost::numeric::bounds<T>::highest());
        return result;
    }
};

//! Construct from the outer product of a vector with itself.
template<typename T>
CSymmetricMatrix<T>::CSymmetricMatrix(ESymmetricMatrixType type, const CVector<T>& x) {
    m_D = x.dimension();
    TBase::m_LowerTriangle.resize(m_D * (m_D + 1) / 2);
    switch (type) {
    case E_OuterProduct:
        for (std::size_t i = 0u, i_ = 0u; i < x.dimension(); ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++i_) {
                TBase::m_LowerTriangle[i_] = x(i) * x(j);
            }
        }
        break;
    case E_Diagonal:
        for (std::size_t i = 0u, i_ = 0u; i < x.dimension(); ++i) {
            for (std::size_t j = 0u; j <= i; ++j, ++i_) {
                TBase::m_LowerTriangle[i_] = i == j ? x(i) : T(0);
            }
        }
        break;
    }
}

//! \brief Gets a zero vector with specified dimension.
template<typename T>
struct SZero<CVector<T>> {
    static CVector<T> get(std::size_t dimension) { return CVector<T>(dimension, T(0)); }
};

// ************************ FREE FUNCTIONS ************************

//! Free swap picked up by std:: algorithms etc.
template<typename T>
void swap(CSymmetricMatrix<T>& lhs, CSymmetricMatrix<T>& rhs) {
    lhs.swap(rhs);
}

//! Free swap picked up by std:: algorithms etc.
template<typename T>
void swap(CVector<T>& lhs, CVector<T>& rhs) {
    lhs.swap(rhs);
}

//! Compute the matrix vector product
//! <pre class="fragment">
//!   \(M x\)
//! </pre>
//!
//! \param[in] m The matrix.
//! \param[in] x The vector.
template<typename T, std::size_t N>
CVectorNx1<T, N> operator*(const CSymmetricMatrixNxN<T, N>& m, const CVectorNx1<T, N>& x) {
    CVectorNx1<T, N> result;
    for (std::size_t i = 0u; i < N; ++i) {
        double component = 0.0;
        for (std::size_t j = 0u; j < N; ++j) {
            component += m(i, j) * x(j);
        }
        result(i) = component;
    }
    return result;
}

//! Compute the matrix vector product
//! <pre class="fragment">
//!   \(M x\)
//! </pre>
//!
//! \param[in] m The matrix.
//! \param[in] x The vector.
template<typename T>
CVector<T> operator*(const CSymmetricMatrix<T>& m, const CVector<T>& x) {
    CVector<T> result(x.dimension());
    for (std::size_t i = 0u; i < m.rows(); ++i) {
        double component = 0.0;
        for (std::size_t j = 0u; j < m.columns(); ++j) {
            component += m(i, j) * x(j);
        }
        result(i) = component;
    }
    return result;
}
}
}

#endif // INCLUDED_ml_maths_CLinearAlgebra_h
