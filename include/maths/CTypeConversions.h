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

#ifndef INCLUDED_ml_maths_CTypeConversions_h
#define INCLUDED_ml_maths_CTypeConversions_h

#include <maths/CLinearAlgebraFwd.h>
#include <maths/MathsTypes.h>

#include <boost/type_traits/is_floating_point.hpp>

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
struct SPromoted<CVectorNx1<T, N> > {
    using Type = CVectorNx1<typename SPromoted<T>::Type, N>;
};

//! \brief Defines the promoted type for a CVector.
template<typename T>
struct SPromoted<CVector<T> > {
    using Type = CVector<typename SPromoted<T>::Type>;
};

//! \brief Defines the promoted type for a CSymmetricMatrixNxN.
template<typename T, std::size_t N>
struct SPromoted<CSymmetricMatrixNxN<T, N> > {
    using Type = CSymmetricMatrixNxN<typename SPromoted<T>::Type, N>;
};

//! \brief Defines the promoted type for a CSymmetricMatrix.
template<typename T>
struct SPromoted<CSymmetricMatrix<T> > {
    using Type = CSymmetricMatrix<typename SPromoted<T>::Type>;
};

//! \brief Defines the promoted type for an Eigen sparse matrix.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
struct SPromoted<Eigen::SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX> > {
    using Type = Eigen::SparseMatrix<typename SPromoted<SCALAR>::Type,
                                     FLAGS, STORAGE_INDEX>;
};

//! \brief Defines the promoted type for an Eigen sparse vector.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
struct SPromoted<Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX> > {
    using Type = Eigen::SparseVector<typename SPromoted<SCALAR>::Type,
                                     FLAGS, STORAGE_INDEX>;
};

//! \brief Defines the promoted type for an Eigen dense matrix.
template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS>
struct SPromoted<Eigen::Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS> > {
    using Type = Eigen::Matrix<typename SPromoted<SCALAR>::Type,
                               ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>;
};

//! \brief Defines the promoted type for a CAnnotatedVector.
template<typename VECTOR, typename ANNOTATION>
struct SPromoted<CAnnotatedVector<VECTOR, ANNOTATION> > {
    using Type = CAnnotatedVector<typename SPromoted<VECTOR>::Type, ANNOTATION>;
};


namespace type_conversion_detail {

//! \brief Chooses between T and U based on the checks for
//! integral and floating point types.
template<typename T, typename U, bool FLOATING_POINT>
struct SSelector {
    using Type = U;
};
template<typename T, typename U>
struct SSelector<T, U, true> {
    using Type = T;
};

} // type_conversion_detail::

//! \brief Defines a suitable floating point type.
template<typename T, typename U>
struct SFloatingPoint {
    using Type = typename type_conversion_detail::SSelector<
        T, U, boost::is_floating_point<T>::value>::Type;
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

//! \brief Defines an Eigen sparse matrix on a suitable floating point type.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX, typename U>
struct SFloatingPoint<Eigen::SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX>, U> {
    using Type = Eigen::SparseMatrix<typename SFloatingPoint<SCALAR, U>::Type,
                                     FLAGS, STORAGE_INDEX>;
};

//! \brief Defines an Eigen sparse vector on a suitable floating point type.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX, typename U>
struct SFloatingPoint<Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX>, U> {
    using Type = Eigen::SparseVector<typename SFloatingPoint<SCALAR, U>::Type,
                                     FLAGS, STORAGE_INDEX>;
};

//! \brief Defines an Eigen dense matrix on a suitable floating point type.
template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS, typename U>
struct SFloatingPoint<Eigen::Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>, U> {
    using Type = Eigen::Matrix<typename SFloatingPoint<SCALAR, U>::Type,
                               ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>;
};

//! \brief Defines CAnnotatedVector on a suitable floating point type.
template<typename VECTOR, typename ANNOTATION, typename U>
struct SFloatingPoint<CAnnotatedVector<VECTOR, ANNOTATION>, U> {
    using Type = CAnnotatedVector<typename SFloatingPoint<VECTOR, U>::Type, ANNOTATION>;
};


//! \brief Extracts the coordinate type for a point.
template<typename T>
struct SCoordinate {
    using Type = T;
};

//! \brief Extracts the coordinate type for CVectorNx1.
template<typename T, std::size_t N>
struct SCoordinate<CVectorNx1<T, N> > {
    using Type = T;
};

//! \brief Extracts the coordinate type for CVector.
template<typename T>
struct SCoordinate<CVector<T> > {
    using Type = T;
};

//! \brief Extracts the coordinate type for CSymmetricMatrixNxN.
template<typename T, std::size_t N>
struct SCoordinate<CSymmetricMatrixNxN<T, N> > {
    using Type = T;
};

//! \brief Extracts the coordinate type for CSymmetricMatrix.
template<typename T>
struct SCoordinate<CSymmetricMatrix<T> > {
    using Type = T;
};

//! \brief Extracts the coordinate type for an Eigen sparse matrix.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
struct SCoordinate<Eigen::SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX> > {
    using Type = SCALAR;
};

//! \brief Extracts the coordinate type for an Eigen sparse vector.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
struct SCoordinate<Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX> > {
    using Type = SCALAR;
};

//! \brief Extracts the coordinate type for an Eigen dense matrix.
template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS>
struct SCoordinate<Eigen::Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS> > {
    using Type = SCALAR;
};

//! \brief Extracts the coordinate type for the underlying vector type.
template<typename VECTOR, typename ANNOTATION>
struct SCoordinate<CAnnotatedVector<VECTOR, ANNOTATION> > {
    using Type = typename SCoordinate<VECTOR>::Type;
};


//! \brief Extracts the conformable matrix type for a point.
template<typename POINT>
struct SConformableMatrix {
    using Type = POINT;
};

//! \brief Extracts the conformable matrix type for a CVectorNx1.
template<typename T, std::size_t N>
struct SConformableMatrix<CVectorNx1<T, N> > {
    using Type = CSymmetricMatrixNxN<T, N>;
};

//! \brief Extracts the conformable matrix type for a CVector.
template<typename T>
struct SConformableMatrix<CVector<T> > {
    using Type = CSymmetricMatrix<T>;
};

//! \brief Extracts the conformable matrix type for an Eigen sparse vector.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
struct SConformableMatrix<Eigen::SparseVector<SCALAR, FLAGS, STORAGE_INDEX> > {
    using Type = Eigen::SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX>;
};

//! \brief Extracts the conformable matrix type for an Eigen dense vector.
template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS>
struct SConformableMatrix<Eigen::Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS> > {
    using Type = Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic, OPTIONS, MAX_ROWS, MAX_COLS>;
};

//! \brief Extracts the conformable matrix type for the underlying vector type.
template<typename VECTOR, typename ANNOTATION>
struct SConformableMatrix<CAnnotatedVector<VECTOR, ANNOTATION> > {
    using Type = typename SConformableMatrix<VECTOR>::Type;
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
struct SStripped<CAnnotatedVector<VECTOR, ANNOTATION> > {
    using Type = VECTOR;
};

}
}

#endif // INCLUDED_ml_maths_CTypeConversions_h
