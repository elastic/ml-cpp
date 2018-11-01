/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLinearAlgebraShims_h
#define INCLUDED_ml_maths_CLinearAlgebraShims_h

#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraFwd.h>
#include <maths/CLinearAlgebraMemoryMapped.h>
#include <maths/CTypeTraits.h>

#include <cmath>
#include <cstddef>

namespace ml {
namespace maths {
namespace las {

//! Get the dimension of our internal vectors.
template<typename VECTOR>
std::size_t dimension(const VECTOR& x) {
    return x.dimension();
}

//! Get the dimension of an Eigen dense vector.
template<typename SCALAR>
std::size_t dimension(const CDenseVector<SCALAR>& x) {
    return static_cast<std::size_t>(x.size());
}

//! Get the dimension of an Eigen memory mapped vector.
template<typename SCALAR>
std::size_t dimension(const CMemoryMappedDenseVector<SCALAR>& x) {
    return static_cast<std::size_t>(x.size());
}

//! Get the dimension of an annotated vector.
template<typename VECTOR, typename ANNOTATION>
std::size_t dimension(const CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return dimension(static_cast<const VECTOR&>(x));
}

//! Get the concomitant zero vector.
template<typename VECTOR>
typename SConstant<VECTOR>::Type zero(const VECTOR& x) {
    return SConstant<VECTOR>::get(dimension(x), 0);
}

//! Get the conformable zero initialized matrix for our internal stack vector.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N> conformableZeroMatrix(const CVectorNx1<T, N>& /*x*/) {
    return CSymmetricMatrixNxN<T, N>(0);
}

//! Get the conformable zero initialized matrix for our internal heap vector.
template<typename T>
CSymmetricMatrix<T> conformableZeroMatrix(const CVector<T>& x) {
    return CSymmetricMatrix<T>(x.dimension(), 0);
}

//! Get the conformable zero initialized matrix for the Eigen dense vector.
template<typename SCALAR>
CDenseMatrix<SCALAR> conformableZeroMatrix(const CDenseVector<SCALAR>& x) {
    return SConstant<CDenseMatrix<SCALAR>>::get(x.size(), 0);
}

//! Get the conformable zero initialized matrix for the Eigen memory mapped vector.
template<typename SCALAR>
CDenseMatrix<SCALAR> conformableZeroMatrix(const CMemoryMappedDenseVector<SCALAR>& x) {
    return SConstant<CMemoryMappedDenseMatrix<SCALAR>>::get(dimension(x), 0);
}

//! Get the conformable zero initialized matrix for the underlying vector.
template<typename VECTOR, typename ANNOTATION>
typename SConformableMatrix<VECTOR>::Type
conformableZeroMatrix(const CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return conformableZeroMatrix(static_cast<const VECTOR&>(x));
}

//! Check if a vector is the zero vector.
template<typename VECTOR>
bool isZero(const VECTOR& x) {
    for (std::size_t i = 0; i < dimension(x); ++i) {
        if (x(i) != 0) {
            return false;
        }
    }
    return true;
}

//! Get the concomitant ones vector.
template<typename VECTOR>
typename SConstant<VECTOR>::Type ones(const VECTOR& x) {
    return SConstant<VECTOR>::get(dimension(x), 1);
}

//! Get the concomitant constant \p c vector.
template<typename VECTOR>
typename SConstant<VECTOR>::Type constant(const VECTOR& x, typename SCoordinate<VECTOR>::Type constant) {
    return SConstant<VECTOR>::get(dimension(x), constant);
}

//! In-place minimum, writing to \p y.
template<typename VECTOR>
void min(const VECTOR& x, VECTOR& y) {
    for (std::size_t i = 0; i < dimension(x); ++i) {
        y(i) = std::min(x(i), y(i));
    }
}

//! In-place maximum, writing to \p y.
template<typename VECTOR>
void max(const VECTOR& x, VECTOR& y) {
    for (std::size_t i = 0; i < dimension(x); ++i) {
        y(i) = std::max(x(i), y(i));
    }
}

//! Expose componentwise operations for our internal vectors.
template<typename VECTOR>
typename SArrayView<VECTOR>::Type componentwise(VECTOR& x) {
    return x;
}

//! Expose componentwise operations for Eigen dense vectors.
template<typename SCALAR>
typename SArrayView<const CDenseVector<SCALAR>>::Type
componentwise(const CDenseVector<SCALAR>& x) {
    return x.array();
}
template<typename SCALAR>
typename SArrayView<CDenseVector<SCALAR>>::Type componentwise(CDenseVector<SCALAR>& x) {
    return x.array();
}

//! Expose componentwise operations for Eigen memory mapped vectors.
template<typename SCALAR>
typename SArrayView<const CMemoryMappedDenseVector<SCALAR>>::Type
componentwise(const CMemoryMappedDenseVector<SCALAR>& x) {
    return x.array();
}
template<typename SCALAR>
typename SArrayView<CMemoryMappedDenseVector<SCALAR>>::Type
componentwise(CMemoryMappedDenseVector<SCALAR>& x) {
    return x.array();
}

//! Expose componentwise operations for our annotated vectors.
template<typename VECTOR, typename ANNOTATION>
typename SArrayView<const VECTOR>::Type
componentwise(const CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return componentwise(static_cast<const VECTOR&>(x));
}
template<typename VECTOR, typename ANNOTATION>
typename SArrayView<VECTOR>::Type& componentwise(CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return componentwise(static_cast<VECTOR&>(x));
}

//! Euclidean distance implementation for our internal vectors.
template<typename VECTOR>
typename SCoordinate<VECTOR>::Type distance(const VECTOR& x, const VECTOR& y) {
    using TCoordinate = typename SPromoted<typename SCoordinate<VECTOR>::Type>::Type;
    TCoordinate result(0);
    for (std::size_t i = 0; i < x.dimension(); ++i) {
        TCoordinate x_(y(i) - x(i));
        result += x_ * x_;
    }
    return std::sqrt(result);
}

//! Euclidean distance implementation for the Eigen dense vector.
template<typename SCALAR>
SCALAR distance(const CDenseVector<SCALAR>& x, const CDenseVector<SCALAR>& y) {
    return (y - x).norm();
}

//! Euclidean distance implementation for our memory mapped vector.
template<typename SCALAR>
SCALAR distance(const CMemoryMappedDenseVector<SCALAR>& x, const CMemoryMappedDenseVector<SCALAR>& y) {
    return (y - x).norm();
}

//! Euclidean distance implementation for our annotated vector.
template<typename VECTOR, typename ANNOTATION>
typename SCoordinate<VECTOR>::Type
distance(const CAnnotatedVector<VECTOR, ANNOTATION>& x,
         const CAnnotatedVector<VECTOR, ANNOTATION>& y) {
    return distance(static_cast<const VECTOR&>(x), static_cast<const VECTOR&>(y));
}

//! Get the Euclidean norm of our internal vectors.
template<typename VECTOR>
typename SCoordinate<VECTOR>::Type norm(const VECTOR& x) {
    return x.euclidean();
}

//! Get the Euclidean norm of an Eigen dense vector.
template<typename SCALAR>
SCALAR norm(const CDenseVector<SCALAR>& x) {
    return x.norm();
}

//! Get the Euclidean norm of an Eigen memory mapped vector.
template<typename SCALAR>
SCALAR norm(const CMemoryMappedDenseVector<SCALAR>& x) {
    return x.norm();
}

//! Get the Euclidean norm of an annotated vector.
template<typename VECTOR, typename ANNOTATION>
typename SCoordinate<VECTOR>::Type norm(const CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return norm(static_cast<const VECTOR&>(x));
}

//! Get the Manhattan norm of our internal vector classes.
template<typename VECTOR>
typename SCoordinate<VECTOR>::Type L1(const VECTOR& x) {
    return x.L1();
}

//! Get the Manhattan norm of an Eigen dense vector.
template<typename SCALAR>
SCALAR L1(const CDenseVector<SCALAR>& x) {
    return x.template lpNorm<1>();
}

//! Get the Manhattan norm of an Eigen memory mapped vector.
template<typename SCALAR>
SCALAR L1(const CMemoryMappedDenseVector<SCALAR>& x) {
    return x.template lpNorm<1>();
}

//! Get the Manhattan norm of an annotated vector.
template<typename VECTOR, typename ANNOTATION>
typename SCoordinate<VECTOR>::Type L1(const CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return L1(static_cast<const VECTOR&>(x));
}

//! Get the Frobenius norm of one of our internal matrices.
template<typename MATRIX>
typename SCoordinate<MATRIX>::Type frobenius(const MATRIX& m) {
    return m.frobenius();
}

//! Get the Euclidean norm of an Eigen dense vector.
template<typename SCALAR>
SCALAR frobenius(const CDenseMatrix<SCALAR>& x) {
    return x.norm();
}

//! Get the Euclidean norm of an Eigen memory mapped matrix.
template<typename SCALAR>
SCALAR frobenius(const CMemoryMappedDenseMatrix<SCALAR>& x) {
    return x.norm();
}

//! Get the inner product of two of our internal vectors.
template<typename VECTOR>
typename SCoordinate<VECTOR>::Type inner(const VECTOR& x, const VECTOR& y) {
    return x.inner(y);
}

//! Get the inner product of two Eigen dense vectors.
template<typename SCALAR>
SCALAR inner(const CDenseVector<SCALAR>& x, const CDenseVector<SCALAR>& y) {
    return x.dot(y);
}

//! Get the inner product of two Eigen memory mapped vectors.
template<typename SCALAR>
SCALAR inner(const CMemoryMappedDenseVector<SCALAR>& x, const CMemoryMappedDenseVector<SCALAR>& y) {
    return x.dot(y);
}

//! Get the inner product of two annotated vectors.
template<typename VECTOR, typename ANNOTATION>
typename SCoordinate<VECTOR>::Type
inner(const CAnnotatedVector<VECTOR, ANNOTATION>& x,
      const CAnnotatedVector<VECTOR, ANNOTATION>& y) {
    return inner(static_cast<const VECTOR&>(x), static_cast<const VECTOR&>(y));
}

//! Get the outer product of our internal vector types.
template<typename VECTOR>
typename SConformableMatrix<VECTOR>::Type outer(const VECTOR& x) {
    return x.outer();
}

//! Get the outer product of an Eigen dense vector.
template<typename SCALAR>
CDenseMatrix<SCALAR> outer(const CDenseVector<SCALAR>& x) {
    return x * x.transpose();
}

//! Get the outer product of an Eigen memory mapped vector.
template<typename SCALAR>
CDenseMatrix<SCALAR> outer(const CMemoryMappedDenseVector<SCALAR>& x) {
    return outer(CDenseVector<SCALAR>(x));
}

//! Get the outer product of an annotated vector.
template<typename VECTOR, typename ANNOTATION>
typename SConformableMatrix<VECTOR>::Type
outer(const CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return outer(static_cast<const VECTOR&>(x));
}
}
}
}

#endif // ml_maths_CLinearAlgebraShims_h
