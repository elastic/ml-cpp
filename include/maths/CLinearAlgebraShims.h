/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLinearAlgebraShims_h
#define INCLUDED_ml_maths_CLinearAlgebraShims_h

#include <maths/CLinearAlgebraFwd.h>
#include <maths/CTypeTraits.h>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace ml {
namespace maths {
namespace las {

//! Get the dimension of one of our internal vectors.
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
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
std::size_t dimension(const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& x) {
    return static_cast<std::size_t>(x.size());
}

//! Get the dimension of an annotated vector.
template<typename VECTOR, typename ANNOTATION>
std::size_t dimension(const CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return dimension(static_cast<const VECTOR&>(x));
}

//! Get the concomitant zero vector.
template<typename VECTOR>
auto zero(const VECTOR& x) -> decltype(SConstant<VECTOR>::get(dimension(x), 0)) {
    return SConstant<VECTOR>::get(dimension(x), 0);
}

//! Zero all the components of \p x.
template<typename VECTOR>
void setZero(VECTOR& x) {
    for (std::size_t i = 0; i < dimension(x); ++i) {
        x(i) = 0.0;
    }
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
    return SConstant<CDenseMatrix<SCALAR>>::get(dimension(x), 0);
}

//! Get the conformable zero initialized matrix for the Eigen memory mapped vector.
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
CDenseMatrix<SCALAR>
conformableZeroMatrix(const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& x) {
    return SConstant<CMemoryMappedDenseMatrix<SCALAR, ALIGNMENT>>::get(dimension(x), 0);
}

//! Get the conformable zero initialized matrix for the underlying vector.
template<typename VECTOR, typename ANNOTATION>
typename SConformableMatrix<VECTOR>::Type
conformableZeroMatrix(const CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return conformableZeroMatrix(static_cast<const VECTOR&>(x));
}

//! Check if a vector is a zero vector.
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
auto ones(const VECTOR& x) -> decltype(SConstant<VECTOR>::get(dimension(x), 1)) {
    return SConstant<VECTOR>::get(dimension(x), 1);
}

//! Get the concomitant constant \p constant vector.
template<typename VECTOR>
auto constant(const VECTOR& x, typename SCoordinate<VECTOR>::Type constant)
    -> decltype(SConstant<VECTOR>::get(dimension(x), constant)) {
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
VECTOR& componentwise(VECTOR& x) {
    return x;
}

//! Expose componentwise operations for Eigen dense vectors.
template<typename SCALAR>
auto componentwise(const CDenseVector<SCALAR>& x) -> decltype(x.array()) {
    return x.array();
}
template<typename SCALAR>
auto componentwise(CDenseVector<SCALAR>& x) -> decltype(x.array()) {
    return x.array();
}

//! Expose componentwise operations for Eigen memory mapped vectors.
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
auto componentwise(const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& x)
    -> decltype(x.array()) {
    return x.array();
}
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
auto componentwise(CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& x)
    -> decltype(x.array()) {
    return x.array();
}

//! Expose componentwise operations for our annotated vectors.
template<typename VECTOR, typename ANNOTATION>
auto componentwise(const CAnnotatedVector<VECTOR, ANNOTATION>& x)
    -> decltype(componentwise(static_cast<const VECTOR&>(x))) {
    return componentwise(static_cast<const VECTOR&>(x));
}
template<typename VECTOR, typename ANNOTATION>
auto componentwise(CAnnotatedVector<VECTOR, ANNOTATION>& x)
    -> decltype(componentwise(static_cast<VECTOR&>(x))) {
    return componentwise(static_cast<VECTOR&>(x));
}

//! Euclidean distance implementation for one of our internal vectors.
template<typename VECTOR>
typename SCoordinate<VECTOR>::Type distance(const VECTOR& x, const VECTOR& y) {
    using TCoordinate = typename SPromoted<typename SCoordinate<VECTOR>::Type>::Type;
    TCoordinate result(0);
    for (std::size_t i = 0; i < dimension(x); ++i) {
        TCoordinate x_(y(i) - x(i));
        result += x_ * x_;
    }
    return std::sqrt(result);
}

//! Euclidean distance implementation for an Eigen dense vector.
template<typename SCALAR>
SCALAR distance(const CDenseVector<SCALAR>& x, const CDenseVector<SCALAR>& y) {
    return (y - x).norm();
}

//! Euclidean distance implementation for an Eigen memory mapped vector.
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
SCALAR distance(const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& x,
                const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& y) {
    return (y - x).norm();
}

//! Euclidean distance implementation for an annotated vector.
template<typename VECTOR, typename ANNOTATION>
typename SCoordinate<VECTOR>::Type
distance(const CAnnotatedVector<VECTOR, ANNOTATION>& x,
         const CAnnotatedVector<VECTOR, ANNOTATION>& y) {
    return distance(static_cast<const VECTOR&>(x), static_cast<const VECTOR&>(y));
}

//! Get the Euclidean norm of one of our internal vectors.
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
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
SCALAR norm(const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& x) {
    return x.norm();
}

//! Get the Euclidean norm of an annotated vector.
template<typename VECTOR, typename ANNOTATION>
typename SCoordinate<VECTOR>::Type norm(const CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return norm(static_cast<const VECTOR&>(x));
}

//! Get the Manhattan norm of one of our internal vector classes.
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
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
SCALAR L1(const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& x) {
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
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
SCALAR frobenius(const CMemoryMappedDenseMatrix<SCALAR, ALIGNMENT>& x) {
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
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
SCALAR inner(const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& x,
             const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& y) {
    return x.dot(y);
}
//! Get the inner product of Eigen dense and memory mapped vectors.
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
SCALAR inner(const CDenseVector<SCALAR>& x,
             const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& y) {
    return x.dot(y);
}
//! Get the inner product of Eigen dense and memory mapped vectors.
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
SCALAR inner(const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& x,
             const CDenseVector<SCALAR>& y) {
    return x.dot(y);
}

//! Get the inner product of two annotated vectors.
template<typename V1, typename V2, typename ANNOTATION>
typename std::common_type<typename SCoordinate<V1>::Type, typename SCoordinate<V2>::Type>::type
inner(const CAnnotatedVector<V1, ANNOTATION>& x, const CAnnotatedVector<V2, ANNOTATION>& y) {
    return inner(static_cast<const V1&>(x), static_cast<const V2&>(y));
}
//! Get the inner product of an annotated and some other vector.
template<typename V1, typename V2, typename ANNOTATION>
typename std::common_type<typename SCoordinate<V1>::Type, typename SCoordinate<V2>::Type>::type
inner(const V1& x, const CAnnotatedVector<V2, ANNOTATION>& y) {
    return inner(x, static_cast<const V2&>(y));
}
//! Get the inner product of an annotated and some other vector.
template<typename V1, typename V2, typename ANNOTATION>
typename std::common_type<typename SCoordinate<V1>::Type, typename SCoordinate<V2>::Type>::type
inner(const CAnnotatedVector<V1, ANNOTATION>& x, const V2& y) {
    return inner(static_cast<const V1&>(x), y);
}

//! Get the outer product of one of our internal vector types.
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
template<typename SCALAR, Eigen::AlignmentType ALIGNMENT>
CDenseMatrix<SCALAR> outer(const CMemoryMappedDenseVector<SCALAR, ALIGNMENT>& x) {
    return outer(CDenseVector<SCALAR>(x));
}

//! Get the outer product of an annotated vector.
template<typename VECTOR, typename ANNOTATION>
typename SConformableMatrix<VECTOR>::Type
outer(const CAnnotatedVector<VECTOR, ANNOTATION>& x) {
    return outer(static_cast<const VECTOR&>(x));
}

namespace las_detail {
template<typename VECTOR>
struct SEstimateVectorMemoryUsage {
    static std::size_t value(std::size_t) { return 0; }
};
template<typename T>
struct SEstimateVectorMemoryUsage<CVector<T>> {
    static std::size_t value(std::size_t dimension) {
        return dimension * sizeof(T);
    }
};
template<typename SCALAR>
struct SEstimateVectorMemoryUsage<CDenseVector<SCALAR>> {
    static std::size_t value(std::size_t dimension) {
        // Ignore pad for alignment.
        return dimension * sizeof(SCALAR);
    }
};
template<typename VECTOR, typename ANNOTATION>
struct SEstimateVectorMemoryUsage<CAnnotatedVector<VECTOR, ANNOTATION>> {
    static std::size_t value(std::size_t dimension) {
        // Ignore any dynamic memory used by the annotation: we don't know how to
        // compute this here. It will be up to the calling code to estimate this
        // correctly.
        return SEstimateVectorMemoryUsage<VECTOR>::value(dimension);
    }
};

template<typename MATRIX>
struct SEstimateMatrixMemoryUsage {
    static std::size_t value(std::size_t, std::size_t) { return 0; }
};
template<typename T>
struct SEstimateMatrixMemoryUsage<CSymmetricMatrix<T>> {
    static std::size_t value(std::size_t rows, std::size_t) {
        return sizeof(T) * rows * (rows + 1) / 2;
    }
};
template<typename SCALAR>
struct SEstimateMatrixMemoryUsage<CDenseMatrix<SCALAR>> {
    static std::size_t value(std::size_t rows, std::size_t columns) {
        // Ignore pad for alignment.
        return sizeof(SCALAR) * rows * columns;
    }
};
}

//! Estimate the amount of memory a vector of type VECTOR and size \p dimension
//! will use.
template<typename VECTOR>
std::size_t estimateMemoryUsage(std::size_t dimension) {
    return las_detail::SEstimateVectorMemoryUsage<VECTOR>::value(dimension);
}

//! Estimate the amount of memory a matrix of type MATRIX and size \p rows by
//! \p columns will use.
template<typename MATRIX>
std::size_t estimateMemoryUsage(std::size_t rows, std::size_t columns) {
    return las_detail::SEstimateMatrixMemoryUsage<MATRIX>::value(rows, columns);
}
}
}
}

#endif // ml_maths_CLinearAlgebraShims_h
