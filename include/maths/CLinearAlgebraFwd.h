/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLinearAlgebraFwd_h
#define INCLUDED_ml_maths_CLinearAlgebraFwd_h

#include <maths/ImportExport.h>

#include <Eigen/Core>

#include <cstddef>

// Unfortunately, Eigen headers seem to be super fragile to
// include directly so we just forward declare here ourselves.
namespace Eigen {
template<typename, int, int, int, int, int>
class Matrix;
template<typename, int, typename>
class SparseMatrix;
template<typename, int, typename>
class SparseVector;
template<typename>
class ArrayWrapper;
template<typename, int, typename>
class Map;
template<typename, int>
class JacobiSVD;
template<int, int>
class Stride;
}

namespace ml {
namespace maths {

//! Types of symmetric matrices constructed with a vector.
enum ESymmetricMatrixType { E_OuterProduct, E_Diagonal };

//! \brief Common types used by the vector and matrix classes.
class MATHS_EXPORT CLinearAlgebra {
public:
    static const char DELIMITER = ',';
};

//! \brief Get a constant initialized version of \p TYPE.
//!
//! Each of our vector and matrix types provides a specialization
//! of this class and define a static get method which takes the
//! dimension(s) and the constant value.
template<typename TYPE>
struct SConstant {
    static_assert(sizeof(TYPE) < 0, "Missing specialisation of SConstant");
};

template<typename T, std::size_t N>
class CVectorNx1;
template<typename T, std::size_t N>
class CSymmetricMatrixNxN;
template<typename T>
class CVector;
template<typename T>
class CSymmetricMatrix;
template<typename VECTOR, typename ANNOTATION>
class CAnnotatedVector;
template<typename SCALAR>
class CDenseVector;
template<typename SCALAR>
class CDenseMatrix;
template<typename VECTOR>
class CDenseVectorInitializer;
template<typename MATRIX>
class CDenseMatrixInitializer;
template<typename SCALAR>
class CMemoryMappedDenseVector;
template<typename SCALAR>
class CMemoryMappedDenseMatrix;
}
}

#endif // INCLUDED_ml_maths_CLinearAlgebraFwd_h
