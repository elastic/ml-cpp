/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTypeTraits_h
#define INCLUDED_ml_maths_CTypeTraits_h

#include <maths/CLinearAlgebraFwd.h>
#include <maths/MathsTypes.h>

#include <type_traits>

namespace ml {
namespace maths {

//! \brief Defines the promoted type.
template<typename T>
struct SPromoted {
    using Type = T;
};

//! \brief Defines the promoted type for float.
template<>
struct SPromoted<float> {
    using Type = double;
};

//! \brief Defines the promoted type for CFloatStorage.
template<>
struct SPromoted<CFloatStorage> {
    using Type = double;
};

//! \brief Defines the promoted type for a CVectorNx1.
template<typename T, std::size_t N>
struct SPromoted<CVectorNx1<T, N>> {
    using Type = CVectorNx1<typename SPromoted<T>::Type, N>;
};

//! \brief Defines the promoted type for a CVector.
template<typename T>
struct SPromoted<CVector<T>> {
    using Type = CVector<typename SPromoted<T>::Type>;
};

//! \brief Defines the promoted type for a CSymmetricMatrixNxN.
template<typename T, std::size_t N>
struct SPromoted<CSymmetricMatrixNxN<T, N>> {
    using Type = CSymmetricMatrixNxN<typename SPromoted<T>::Type, N>;
};

//! \brief Defines the promoted type for a CSymmetricMatrix.
template<typename T>
struct SPromoted<CSymmetricMatrix<T>> {
    using Type = CSymmetricMatrix<typename SPromoted<T>::Type>;
};

//! \brief Defines the promoted type for an Eigen dense matrix.
template<typename SCALAR>
struct SPromoted<CDenseMatrix<SCALAR>> {
    using Type = CDenseMatrix<typename SPromoted<SCALAR>::Type>;
};

//! \brief Defines the promoted type for an Eigen dense vector.
template<typename SCALAR>
struct SPromoted<CDenseVector<SCALAR>> {
    using Type = CDenseVector<typename SPromoted<SCALAR>::Type>;
};

//! \brief Defines the promoted type for an Eigen memory mapped matrix.
template<typename SCALAR>
struct SPromoted<CMemoryMappedDenseMatrix<SCALAR>> {
    using Type = CDenseMatrix<typename SPromoted<SCALAR>::Type>;
};

//! \brief Defines the promoted type for an Eigen memory mapped vector.
template<typename SCALAR>
struct SPromoted<CMemoryMappedDenseVector<SCALAR>> {
    using Type = CDenseVector<typename SPromoted<SCALAR>::Type>;
};

//! \brief Defines the promoted type for an Eigen sparse matrix.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
struct SPromoted<Eigen::SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX>> {
    using Type = Eigen::SparseMatrix<typename SPromoted<SCALAR>::Type, FLAGS, STORAGE_INDEX>;
};

//! \brief Defines the promoted type for an Eigen sparse vector.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
struct SPromoted<Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX>> {
    using Type = Eigen::SparseVector<typename SPromoted<SCALAR>::Type, FLAGS, STORAGE_INDEX>;
};

//! \brief Defines the promoted type for a CAnnotatedVector.
template<typename VECTOR, typename ANNOTATION>
struct SPromoted<CAnnotatedVector<VECTOR, ANNOTATION>> {
    using Type = CAnnotatedVector<typename SPromoted<VECTOR>::Type, ANNOTATION>;
};

//! \brief Defines a suitable floating point type.
template<typename T, typename U>
struct SFloatingPoint {
    using Type = typename std::conditional<std::is_floating_point<T>::value, T, U>::type;
};

//! \brief Defines CVectorNx1 on a suitable floating point type.
template<typename T, std::size_t N, typename U>
struct SFloatingPoint<CVectorNx1<T, N>, U> {
    using Type = CVectorNx1<typename SFloatingPoint<T, U>::Type, N>;
};

//! \brief Defines CVector on a suitable floating point type.
template<typename T, typename U>
struct SFloatingPoint<CVector<T>, U> {
    using Type = CVector<typename SFloatingPoint<T, U>::Type>;
};

//! \brief Defines CSymmetricMatrixNxN on a suitable floating point type.
template<typename T, std::size_t N, typename U>
struct SFloatingPoint<CSymmetricMatrixNxN<T, N>, U> {
    using Type = CSymmetricMatrixNxN<typename SFloatingPoint<T, U>::Type, N>;
};

//! \brief Defines CSymmetricMatrix on a suitable floating point type.
template<typename T, typename U>
struct SFloatingPoint<CSymmetricMatrix<T>, U> {
    using Type = CSymmetricMatrix<typename SFloatingPoint<T, U>::Type>;
};

//! \brief Defines an Eigen dense matrix on a suitable floating point type.
template<typename SCALAR, typename U>
struct SFloatingPoint<CDenseMatrix<SCALAR>, U> {
    using Type = CDenseMatrix<typename SFloatingPoint<SCALAR, U>::Type>;
};

//! \brief Defines an Eigen dense vector on a suitable floating point type.
template<typename SCALAR, typename U>
struct SFloatingPoint<CDenseVector<SCALAR>, U> {
    using Type = CDenseVector<typename SFloatingPoint<SCALAR, U>::Type>;
};

//! \brief Defines an Eigen dense matrix on a suitable floating point type.
template<typename SCALAR, typename U>
struct SFloatingPoint<CMemoryMappedDenseMatrix<SCALAR>, U> {
    using Type = CDenseMatrix<typename SFloatingPoint<SCALAR, U>::Type>;
};

//! \brief Defines an Eigen dense vector on a suitable floating point type.
template<typename SCALAR, typename U>
struct SFloatingPoint<CMemoryMappedDenseVector<SCALAR>, U> {
    using Type = CDenseVector<typename SFloatingPoint<SCALAR, U>::Type>;
};

//! \brief Defines an Eigen sparse matrix on a suitable floating point type.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX, typename U>
struct SFloatingPoint<Eigen::SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX>, U> {
    using Type = Eigen::SparseMatrix<typename SFloatingPoint<SCALAR, U>::Type, FLAGS, STORAGE_INDEX>;
};

//! \brief Defines an Eigen sparse vector on a suitable floating point type.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX, typename U>
struct SFloatingPoint<Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX>, U> {
    using Type = Eigen::SparseVector<typename SFloatingPoint<SCALAR, U>::Type, FLAGS, STORAGE_INDEX>;
};

//! \brief Defines CAnnotatedVector on a suitable floating point type.
template<typename VECTOR, typename ANNOTATION, typename U>
struct SFloatingPoint<CAnnotatedVector<VECTOR, ANNOTATION>, U> {
    using Type = CAnnotatedVector<typename SFloatingPoint<VECTOR, U>::Type, ANNOTATION>;
};

//! \brief Extracts the coordinate type for a vector or matrix.
template<typename T>
struct SCoordinate {
    using Type = T;
};

//! \brief Extracts the coordinate type for CVectorNx1.
template<typename T, std::size_t N>
struct SCoordinate<CVectorNx1<T, N>> {
    using Type = T;
};

//! \brief Extracts the coordinate type for CVector.
template<typename T>
struct SCoordinate<CVector<T>> {
    using Type = T;
};

//! \brief Extracts the coordinate type for CSymmetricMatrixNxN.
template<typename T, std::size_t N>
struct SCoordinate<CSymmetricMatrixNxN<T, N>> {
    using Type = T;
};

//! \brief Extracts the coordinate type for CSymmetricMatrix.
template<typename T>
struct SCoordinate<CSymmetricMatrix<T>> {
    using Type = T;
};

//! \brief Extracts the coordinate type for an Eigen dense matrix.
template<typename SCALAR>
struct SCoordinate<CDenseMatrix<SCALAR>> {
    using Type = SCALAR;
};

//! \brief Extracts the coordinate type for an Eigen dense vector.
template<typename SCALAR>
struct SCoordinate<CDenseVector<SCALAR>> {
    using Type = SCALAR;
};

//! \brief Extracts the coordinate type for an Eigen memory mapped matrix.
template<typename SCALAR>
struct SCoordinate<CMemoryMappedDenseMatrix<SCALAR>> {
    using Type = SCALAR;
};

//! \brief Extracts the coordinate type for an Eigen memory mapped vector.
template<typename SCALAR>
struct SCoordinate<CMemoryMappedDenseVector<SCALAR>> {
    using Type = SCALAR;
};

//! \brief Extracts the coordinate type for an Eigen sparse matrix.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
struct SCoordinate<Eigen::SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX>> {
    using Type = SCALAR;
};

//! \brief Extracts the coordinate type for an Eigen sparse vector.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
struct SCoordinate<Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX>> {
    using Type = SCALAR;
};

//! \brief Extracts the coordinate type for the underlying vector type.
template<typename VECTOR, typename ANNOTATION>
struct SCoordinate<CAnnotatedVector<VECTOR, ANNOTATION>> {
    using Type = typename SCoordinate<VECTOR>::Type;
};

//! \brief Extracts the conformable matrix type for a vector.
template<typename VECTOR>
struct SConformableMatrix {
    static_assert(sizeof(VECTOR) < 0, "No conformable matrix type defined");
};

//! \brief Extracts the conformable matrix type for a CVectorNx1.
template<typename T, std::size_t N>
struct SConformableMatrix<CVectorNx1<T, N>> {
    using Type = CSymmetricMatrixNxN<T, N>;
};

//! \brief Extracts the conformable matrix type for a CVector.
template<typename T>
struct SConformableMatrix<CVector<T>> {
    using Type = CSymmetricMatrix<T>;
};

//! \brief Extracts the conformable matrix type for an Eigen dense vector.
template<typename SCALAR>
struct SConformableMatrix<CDenseVector<SCALAR>> {
    using Type = CDenseMatrix<SCALAR>;
};

//! \brief Extracts the conformable matrix type for an Eigen memory mapped vector.
template<typename SCALAR>
struct SConformableMatrix<CMemoryMappedDenseVector<SCALAR>> {
    using Type = CMemoryMappedDenseMatrix<SCALAR>;
};

//! \brief Extracts the conformable matrix type for an Eigen sparse vector.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
struct SConformableMatrix<Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX>> {
    using Type = Eigen::SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX>;
};

//! \brief Extracts the conformable matrix type for the underlying vector type.
template<typename VECTOR, typename ANNOTATION>
struct SConformableMatrix<CAnnotatedVector<VECTOR, ANNOTATION>> {
    using Type = typename SConformableMatrix<VECTOR>::Type;
};

//! \brief Defines the array view for componentwise operations on our internal
//! vectors and matrices.
template<typename VECTOR>
struct SArrayView {
    using Type = VECTOR&;
};

//! \brief Defines the array view for componentwise operations on a Eigen dense matrix.
template<typename SCALAR>
struct SArrayView<const CDenseMatrix<SCALAR>> {
    using Type =
        Eigen::ArrayWrapper<const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, Eigen::Dynamic>>;
};
template<typename SCALAR>
struct SArrayView<CDenseMatrix<SCALAR>> {
    using Type =
        Eigen::ArrayWrapper<Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, Eigen::Dynamic>>;
};

//! \brief Defines the array view for componentwise operations on an Eigen dense matrix.
template<typename SCALAR>
struct SArrayView<const CDenseVector<SCALAR>> {
    using Type =
        Eigen::ArrayWrapper<const Eigen::Matrix<SCALAR, Eigen::Dynamic, 1, 0, Eigen::Dynamic, 1>>;
};
template<typename SCALAR>
struct SArrayView<CDenseVector<SCALAR>> {
    using Type =
        Eigen::ArrayWrapper<Eigen::Matrix<SCALAR, Eigen::Dynamic, 1, 0, Eigen::Dynamic, 1>>;
};

//! \brief Defines the array view for componentwise operations on an Eigen memory mapped matrix.
template<typename SCALAR>
struct SArrayView<const CMemoryMappedDenseMatrix<SCALAR>> {
    using Type = Eigen::ArrayWrapper<
        const Eigen::Map<Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<0, 0>>>;
};
template<typename SCALAR>
struct SArrayView<CMemoryMappedDenseMatrix<SCALAR>> {
    using Type = Eigen::ArrayWrapper<
        Eigen::Map<Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<0, 0>>>;
};

//! \brief Defines the array view for componentwise operations on an Eigen memory mapped vector.
template<typename SCALAR>
struct SArrayView<const CMemoryMappedDenseVector<SCALAR>> {
    using Type = Eigen::ArrayWrapper<
        const Eigen::Map<Eigen::Matrix<SCALAR, Eigen::Dynamic, 1, 0, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>>>;
};
template<typename SCALAR>
struct SArrayView<CMemoryMappedDenseVector<SCALAR>> {
    using Type = Eigen::ArrayWrapper<
        Eigen::Map<Eigen::Matrix<SCALAR, Eigen::Dynamic, 1, 0, Eigen::Dynamic, 1>, 0, Eigen::Stride<0, 0>>>;
};

//! \brief Defines the array view for componentwise operations on Eigen dense
//! vectors and matrices.
template<typename VECTOR, typename ANNOTATION>
struct SArrayView<CAnnotatedVector<VECTOR, ANNOTATION>> {
    using Type = typename SArrayView<VECTOR>::Type;
};

//! \brief Defines the type of a singular value decomposition of a matrix.
template<typename MATRIX>
struct SJacobiSvd {
    static_assert(sizeof(MATRIX) < 0, "SVD not supported for type");
};

//! \brief Defines the type of a singular value decomposition of our decorator
//! of an Eigen dense matrix.
template<typename SCALAR>
struct SJacobiSvd<CDenseMatrix<SCALAR>> {
    using Type = Eigen::JacobiSVD<Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, Eigen::Dynamic>, 
                                  Eigen::ColPivHouseholderQRPreconditioner>;
};

//! \brief Defines the type of a singular value decomposition an Eigen dense matrix.
template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS>
struct SJacobiSvd<Eigen::Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>> {
    using Type = Eigen::JacobiSVD<Eigen::Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>, 
                                  Eigen::ColPivHouseholderQRPreconditioner>;
};

//! \brief Defines a type which strips off any annotation from a vector.
//! This is the raw vector type by default.
template<typename VECTOR>
struct SStripped {
    using Type = VECTOR;
};

//! \brief Specialisation for annotated vectors. This is the underlying
//! vector type.
template<typename VECTOR, typename ANNOTATION>
struct SStripped<CAnnotatedVector<VECTOR, ANNOTATION>> {
    using Type = VECTOR;
};
}
}

#endif // INCLUDED_ml_maths_CTypeTraits_h
