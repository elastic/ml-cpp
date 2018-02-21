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
#include <maths/CTypeTraits.h>

#include <cmath>
#include <cstddef>

namespace ml
{
namespace maths
{
namespace las
{

//! Get the dimension of one of our internal vectors.
template<typename VECTOR>
std::size_t dimension(const VECTOR &x)
{
    return x.dimension();
}

//! Get the dimension of an Eigen dense vector.
template<typename SCALAR>
std::size_t dimension(const CDenseVector<SCALAR> &x)
{
    return static_cast<std::size_t>(x.size());
}

//! Get the dimension of an annotated vector.
template<typename VECTOR, typename ANNOTATION>
std::size_t dimension(const CAnnotatedVector<VECTOR, ANNOTATION> &x)
{
    return dimension(static_cast<const VECTOR &>(x));
}

//! Get the concomitant zero vector.
template<typename VECTOR>
VECTOR zero(const VECTOR &x)
{
    return SConstant<VECTOR>::get(dimension(x), 0);
}

//! Get the concomitant zero annotated vector.
template<typename VECTOR, typename ANNOTATION>
CAnnotatedVector<VECTOR, ANNOTATION> zero(const CAnnotatedVector<VECTOR, ANNOTATION> &x)
{
    return zero(static_cast<const VECTOR &>(x));
}

//! Get the conformable zero initialized matrix for our internal stack vector.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N> conformableZeroMatrix(const CVectorNx1<T, N> &/*x*/)
{
    return CSymmetricMatrixNxN<T, N>(0);
}

//! Get the conformable zero initialized matrix for our internal heap vector.
template<typename T>
CSymmetricMatrix<T> conformableZeroMatrix(const CVector<T> &x)
{
    return CSymmetricMatrix<T>(x.dimension(), 0);
}

//! Get the conformable zero initialized matrix for the Eigen dense vector.
template<typename SCALAR>
CDenseMatrix<SCALAR> conformableZeroMatrix(const CDenseVector<SCALAR> &x)
{
    return CDenseMatrix<SCALAR>::Zero(x.size(), x.size());
}

//! Get the conformable zero initialized matrix for the underlying vector.
template<typename VECTOR, typename ANNOTATION>
typename SConformableMatrix<VECTOR>::Type conformableZeroMatrix(const CAnnotatedVector<VECTOR, ANNOTATION> &x)
{
    return conformableZeroMatrix(static_cast<const VECTOR &>(x));
}

//! Check if a vector is the zero vector.
template<typename VECTOR>
bool isZero(const VECTOR &x)
{
    for (std::size_t i = 0u; i < dimension(x); ++i)
    {
        if (x(i) != 0)
        {
            return false;
        }
    }
    return true;
}

//! Get the concomitant ones vector.
template<typename VECTOR>
VECTOR ones(const VECTOR &x)
{
    return SConstant<VECTOR>::get(dimension(x), 1);
}

//! Get the concomitant ones annotated vector.
template<typename VECTOR, typename ANNOTATION>
CAnnotatedVector<VECTOR, ANNOTATION> ones(const CAnnotatedVector<VECTOR, ANNOTATION> &x)
{
    return ones(static_cast<const VECTOR &>(x));
}

//! Get the concomitant constant \p c vector.
template<typename VECTOR>
VECTOR constant(const VECTOR &x, typename SCoordinate<VECTOR>::Type c)
{
    return SConstant<VECTOR>::get(dimension(x), c);
}

//! Get the concomitant constant \p c annotated vector.
template<typename VECTOR, typename ANNOTATION>
CAnnotatedVector<VECTOR, ANNOTATION> constant(const CAnnotatedVector<VECTOR, ANNOTATION> &x,
                                              typename SCoordinate<VECTOR>::Type c)
{
    return constant(static_cast<const VECTOR &>(x), c);
}

//! In-place minimum, writing to \p y, for our internal vectors.
template<typename VECTOR>
void min(const VECTOR &x, VECTOR &y)
{
    for (std::size_t i = 0u; i < x.dimension(); ++i)
    {
        y(i) = std::min(x(i), y(i));
    }
}

//! In-place minimum, writing to \p y, for an Eigen dense vector.
template<typename SCALAR>
void min(const CDenseVector<SCALAR> &x, CDenseVector<SCALAR> &y)
{
    for (typename CDenseVector<SCALAR>::Index i = 0; i < x.size(); ++i)
    {
        y(i) = std::min(x(i), y(i));
    }
}

//! In-place minimum, writing to \p y, for our annotated vector.
template<typename VECTOR, typename ANNOTATION>
void min(const CAnnotatedVector<VECTOR, ANNOTATION> &x, CAnnotatedVector<VECTOR, ANNOTATION> &y)
{
    return min(static_cast<const VECTOR &>(x), static_cast<VECTOR &>(y));
}

//! In-place maximum, writing to \p y, for our internal vectors.
template<typename VECTOR>
void max(const VECTOR &x, VECTOR &y)
{
    for (std::size_t i = 0u; i < x.dimension(); ++i)
    {
        y(i) = std::max(x(i), y(i));
    }
}

//! In-place maximum, writing to \p y, for the Eigen dense vector.
template<typename SCALAR>
void max(const CDenseVector<SCALAR> &x, CDenseVector<SCALAR> &y)
{
    for (typename CDenseVector<SCALAR>::Index i = 0; i < x.size(); ++i)
    {
        y(i) = std::max(x(i), y(i));
    }
}

//! In-place maximum, writing to \p y, for our annotated vector.
template<typename VECTOR, typename ANNOTATION>
void max(const CAnnotatedVector<VECTOR, ANNOTATION> &x, CAnnotatedVector<VECTOR, ANNOTATION> &y)
{
    return max(static_cast<const VECTOR &>(x), static_cast<VECTOR &>(y));
}

//! Expose componentwise operations for our internal vectors.
template<typename VECTOR>
typename SArrayView<VECTOR>::Type componentwise(VECTOR &x)
{
    return x;
}

//! Expose componentwise operations for Eigen dense vectors.
//!
//! \note This is the array "view".
template<typename SCALAR>
typename SArrayView<const CDenseVector<SCALAR>>::Type componentwise(const CDenseVector<SCALAR> &x)
{
    return x.array();
}
//! Expose componentwise operations for Eigen dense vectors.
//!
//! \note This is the array "view".
template<typename SCALAR>
typename SArrayView<CDenseVector<SCALAR>>::Type componentwise(CDenseVector<SCALAR> &x)
{
    return x.array();
}

//! Expose componentwise operations for our annotated vectors.
template<typename VECTOR, typename ANNOTATION>
typename SArrayView<const VECTOR>::Type componentwise(const CAnnotatedVector<VECTOR, ANNOTATION> &x)
{
    return componentwise(static_cast<const VECTOR &>(x));
}
template<typename VECTOR, typename ANNOTATION>
typename SArrayView<VECTOR>::Type &componentwise(CAnnotatedVector<VECTOR, ANNOTATION> &x)
{
    return componentwise(static_cast<VECTOR &>(x));
}

//! Euclidean distance implementation for our internal vectors.
template<typename VECTOR>
typename SCoordinate<VECTOR>::Type distance(const VECTOR &x, const VECTOR &y)
{
    using TCoordinate = typename SPromoted<typename SCoordinate<VECTOR>::Type>::Type;
    TCoordinate result(0);
    for (std::size_t i = 0u; i < x.dimension(); ++i)
    {
        result += ::pow(y(i) - x(i), TCoordinate(2));
    }
    return ::sqrt(result);
}

//! Euclidean distance implementation for the Eigen dense vector.
template<typename SCALAR>
SCALAR distance(const CDenseVector<SCALAR> &x, const CDenseVector<SCALAR> &y)
{
    return (y - x).norm();
}

//! Euclidean distance implementation for our annotated vector.
template<typename VECTOR, typename ANNOTATION>
typename SCoordinate<VECTOR>::Type distance(const CAnnotatedVector<VECTOR, ANNOTATION> &x,
                                            const CAnnotatedVector<VECTOR, ANNOTATION> &y)
{
    return distance(static_cast<const VECTOR &>(x), static_cast<const VECTOR &>(y));
}

//! Get the Euclidean norm of one of our internal vectors.
template<typename VECTOR>
typename SCoordinate<VECTOR>::Type norm(const VECTOR &x)
{
    return x.euclidean();
}

//! Get the Euclidean norm of an Eigen dense vector.
template<typename SCALAR>
SCALAR norm(const CDenseVector<SCALAR> &x)
{
    return x.norm();
}

//! Get the Euclidean norm of an annotated vector.
template<typename VECTOR, typename ANNOTATION>
typename SCoordinate<VECTOR>::Type norm(const CAnnotatedVector<VECTOR, ANNOTATION> &x)
{
    return norm(static_cast<const VECTOR &>(x));
}

//! Get the Manhattan of one of our internal vector classes.
template<typename VECTOR>
typename SCoordinate<VECTOR>::Type L1(const VECTOR &x)
{
    return x.L1();
}

//! Get the Manhattan of an Eigen dense vector.
template<typename SCALAR>
SCALAR L1(const CDenseVector<SCALAR> &x)
{
    return x.template lpNorm<1>();
}

//! Get the Manhattan norm of an annotated vector.
template<typename VECTOR, typename ANNOTATION>
typename SCoordinate<VECTOR>::Type L1(const CAnnotatedVector<VECTOR, ANNOTATION> &x)
{
    return L1(static_cast<const VECTOR &>(x));
}

//! Get the Frobenius norm of one of our internal matrices.
template<typename MATRIX>
typename SCoordinate<MATRIX>::Type frobenius(const MATRIX &m)
{
    return m.frobenius();
}

//! Get the Euclidean norm of an Eigen dense vector.
template<typename SCALAR>
SCALAR frobenius(const CDenseMatrix<SCALAR> &x)
{
    return x.norm();
}

//! Get the inner product of two of our internal vectors.
template<typename VECTOR>
typename SCoordinate<VECTOR>::Type inner(const VECTOR &x, const VECTOR &y)
{
    return x.inner(y);
}

//! Get the inner product of two Eigen dense vectors.
template<typename SCALAR>
SCALAR inner(const CDenseVector<SCALAR> &x, const CDenseVector<SCALAR> &y)
{
    return x.dot(y);
}

//! Get the inner product of two annotated vectors.
template<typename VECTOR, typename ANNOTATION>
typename SCoordinate<VECTOR>::Type inner(const CAnnotatedVector<VECTOR, ANNOTATION> &x,
                                         const CAnnotatedVector<VECTOR, ANNOTATION> &y)
{
    return inner(static_cast<const VECTOR &>(x), static_cast<const VECTOR &>(y));
}

//! Get the outer product of one of our internal vector types.
template<typename VECTOR>
typename SConformableMatrix<VECTOR>::Type outer(const VECTOR &x)
{
    return x.outer();
}

//! Get the outer product of two Eigen dense vectors.
template<typename SCALAR>
CDenseMatrix<SCALAR> outer(const CDenseVector<SCALAR> &x)
{
    return x * x.transpose();
}

//! Get the outer product of two annotated vectors.
template<typename VECTOR, typename ANNOTATION>
typename SConformableMatrix<VECTOR>::Type outer(const CAnnotatedVector<VECTOR, ANNOTATION> &x)
{
    return outer(static_cast<const VECTOR &>(x));
}

}
}
}

#endif // ml_maths_CLinearAlgebraShims_h
