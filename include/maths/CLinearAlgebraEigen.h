/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_ml_maths_CLinearAlgebraEigen_h
#define INCLUDED_ml_maths_CLinearAlgebraEigen_h

#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraFwd.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/SparseCore>

#include <iterator>

namespace Eigen {
#define LESS_OR_GREATER(l, r) if (l < r) { return true; } else if (r > l) { return false; }

//! Less than on Eigen sparse matrix.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
bool operator<(const SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX> &lhs,
               const SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX> &rhs) {
    LESS_OR_GREATER(lhs.rows(), rhs.rows())
    LESS_OR_GREATER(lhs.cols(), rhs.cols())
    for (STORAGE_INDEX i = 0; i < lhs.rows(); ++i) {
        for (STORAGE_INDEX j = 0; j < lhs.cols(); ++j) {
            LESS_OR_GREATER(lhs.coeff(i, j), rhs.coeff(i, j))
        }
    }
    return false;
}

//! Less than on Eigen sparse vector.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
bool operator<(const SparseVector<SCALAR, FLAGS, STORAGE_INDEX> &lhs,
               const SparseVector<SCALAR, FLAGS, STORAGE_INDEX> &rhs) {
    LESS_OR_GREATER(lhs.size(), rhs.size())
    for (STORAGE_INDEX i = 0; i < lhs.size(); ++i) {
        LESS_OR_GREATER(lhs.coeff(i), rhs(i))
    }
    return false;
}

//! Less than on Eigen dense matrix.
template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS>
bool operator<(const Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS> &lhs,
               const Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS> &rhs) {
    using TIndex = typename Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>::Index;
    LESS_OR_GREATER(lhs.rows(), rhs.rows())
    LESS_OR_GREATER(lhs.cols(), rhs.cols())
    for (TIndex i = 0; i < lhs.rows(); ++i) {
        for (TIndex j = 0; j < lhs.cols(); ++j) {
            LESS_OR_GREATER(lhs.coeff(i, j), rhs.coeff(i, j))
        }
    }
    return false;
}

//! Free swap picked up by std:: algorithms etc.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
void swap(SparseVector<SCALAR, FLAGS, STORAGE_INDEX> &lhs,
          SparseVector<SCALAR, FLAGS, STORAGE_INDEX> &rhs) {
    lhs.swap(rhs);
}

//! Free swap picked up by std:: algorithms etc.
template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS>
void swap(Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS> &lhs,
          Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS> &rhs) {
    lhs.swap(rhs);
}

#undef LESS_OR_GREATER
}

namespace ml {
namespace maths {

//! Rename to follow our conventions and add to ml::maths.
template<typename SCALAR, int FLAGS = 0>
using CSparseMatrix = Eigen::SparseMatrix<SCALAR, FLAGS, std::ptrdiff_t>;

//! \brief Gets a zero sparse matrix with specified dimensions.
template<typename SCALAR, int FLAGS>
struct SZero<CSparseMatrix<SCALAR, FLAGS>> {
    static CSparseMatrix<SCALAR, FLAGS> get(std::ptrdiff_t rows, std::ptrdiff_t cols) {
        return CSparseMatrix<SCALAR, FLAGS>(rows, cols);
    }
};

//! The type of an element of a sparse matrix in coordinate form.
template<typename SCALAR>
using CSparseMatrixElement = Eigen::Triplet<SCALAR>;

//! Rename to follow our conventions and add to ml::maths.
template<typename SCALAR, int FLAGS = Eigen::RowMajorBit>
using CSparseVector = Eigen::SparseVector<SCALAR, FLAGS, std::ptrdiff_t>;

//! \brief Gets a zero sparse vector with specified dimension.
template<typename SCALAR, int FLAGS>
struct SZero<CSparseVector<SCALAR, FLAGS>> {
    static CSparseVector<SCALAR, FLAGS> get(std::ptrdiff_t dimension) {
        return CSparseVector<SCALAR, FLAGS>(dimension);
    }
};

//! The type of an element of a sparse vector in coordinate form.
template<typename SCALAR>
using CSparseVectorCoordinate = Eigen::Triplet<SCALAR>;

//! Create a tuple with which to initialize a sparse matrix.
template<typename SCALAR>
inline CSparseMatrixElement<SCALAR> matrixElement(std::ptrdiff_t row, std::ptrdiff_t column, SCALAR value) {
    return CSparseMatrixElement<SCALAR>(row, column, value);
}

//! Create a tuple with which to initialize a sparse column vector.
template<typename SCALAR>
inline CSparseVectorCoordinate<SCALAR> vectorCoordinate(std::ptrdiff_t row, SCALAR value) {
    return CSparseVectorCoordinate<SCALAR>(row, 0, value);
}

//! \brief Adapts Eigen::SparseVector::InnerIterator for use with STL.
template<typename SCALAR, int FLAGS = Eigen::RowMajorBit>
class CSparseVectorIndexIterator : public std::iterator<std::input_iterator_tag, std::ptrdiff_t> {
        CSparseVectorIndexIterator(const CSparseVector<SCALAR, FLAGS> &vector,
                                   std::size_t index) :
            m_Vector(&vector), m_Base(vector, index)
        {}

        bool operator==(const CSparseVectorIndexIterator &rhs) const {
            return   m_Vector == rhs.m_Vector
                     && m_Base.row() == rhs.m_Base.row()
                     && m_Base.col() == rhs.m_Base.col();
        }
        bool operator!=(const CSparseVectorIndexIterator &rhs) const {
            return !(*this == rhs);
        }

        std::ptrdiff_t operator*(void) const {
            return std::max(m_Base.row(), m_Base.col());
        }

        CSparseVectorIndexIterator &operator++(void) {
            ++m_Base;
            return *this;
        }
        CSparseVectorIndexIterator operator++(int) {
            CSparseVectorIndexIterator result(*this);
            ++m_Base;
            return result;
        }

    private:
        using TIterator = typename CSparseVector<SCALAR, FLAGS>::InnerIterator;

    private:
        CSparseVector<SCALAR, FLAGS> *m_Vector;
        TIterator m_Base;
};

//! Get an iterator over the indices of \p vector.
template<typename SCALAR, int FLAGS>
CSparseVectorIndexIterator<SCALAR, FLAGS>
beginIndices(const CSparseVector<SCALAR, FLAGS> &vector) {
    return CSparseVectorIndexIterator<SCALAR, FLAGS>(vector, 0);
}

//! Get the end iterator of the indices of \p vector.
template<typename SCALAR, int FLAGS>
CSparseVectorIndexIterator<SCALAR, FLAGS>
endIndices(const CSparseVector<SCALAR, FLAGS> &vector) {
    return CSparseVectorIndexIterator<SCALAR, FLAGS>(vector, vector.data().size());
}

//! Rename to follow our conventions and add to ml::maths.
template<typename SCALAR>
using CDenseMatrix = Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>;

//! \brief Gets a zero dense vector with specified dimension.
template<typename SCALAR>
struct SZero<CDenseMatrix<SCALAR>> {
    static CDenseMatrix<SCALAR> get(std::ptrdiff_t rows, std::ptrdiff_t cols) {
        return CDenseMatrix<SCALAR>::Zero(rows, cols);
    }
};

//! Rename to follow our conventions and add to ml::maths.
template<typename SCALAR>
using CDenseVector = Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>;

//! \brief Gets a zero dense vector with specified dimension.
template<typename SCALAR>
struct SZero<CDenseVector<SCALAR>> {
    static CDenseVector<SCALAR> get(std::ptrdiff_t dimension) {
        return CDenseVector<SCALAR>::Zero(dimension);
    }
};

//! \brief Eigen matrix typedef.
//!
//! DESCRIPTION:\n
//! Instantiating many different sizes of Eigen::Matrix really hurts
//! our compile times and executable sizes with debug symbols. The
//! idea of this class is to limit the maximum size of N for which
//! we instantiate different versions. Also, Eigen matrices are always
//! used for calculation for which we want to use double precision.
template<typename MATRIX>
struct SDenseMatrix {
    using Type = CDenseMatrix<double>;
};
//! \brief Use stack matrix for size 2.
template<typename T>
struct SDenseMatrix<CSymmetricMatrixNxN<T, 2>> {
    using Type = Eigen::Matrix<double, 2, 2>;
};
//! \brief Use stack matrix for size 3.
template<typename T>
struct SDenseMatrix<CSymmetricMatrixNxN<T, 3>> {
    using Type = Eigen::Matrix<double, 3, 3>;
};
//! \brief Use stack matrix for size 4.
template<typename T>
struct SDenseMatrix<CSymmetricMatrixNxN<T, 4>> {
    using Type = Eigen::Matrix<double, 4, 4>;
};

//! Get the Eigen matrix for \p matrix.
template<typename MATRIX>
typename SDenseMatrix<MATRIX>::Type toDenseMatrix(const MATRIX &matrix) {
    return matrix.template toType<typename SDenseMatrix<MATRIX>::Type>();
}

//! Get the dynamic Eigen matrix for \p matrix.
template<typename MATRIX>
CDenseMatrix<double> toDynamicDenseMatrix(const MATRIX &matrix) {
    return matrix.template toType<CDenseMatrix<double>>();
}

//! \brief Eigen vector typedef.
//!
//! DESCRIPTION:\n
//! See SDenseMatrix.
template<typename VECTOR>
struct SDenseVector {
    using Type = CDenseVector<double>;
};
//! \brief Use stack vector for size 2.
template<typename T>
struct SDenseVector<CVectorNx1<T, 2>> {
    using Type = Eigen::Matrix<double, 2, 1>;
};
//! \brief Use stack vector for size 3.
template<typename T>
struct SDenseVector<CVectorNx1<T, 3>> {
    using Type = Eigen::Matrix<double, 3, 1>;
};
//! \brief Use stack vector for size 4.
template<typename T>
struct SDenseVector<CVectorNx1<T, 4>> {
    using Type = Eigen::Matrix<double, 4, 1>;
};

//! Get the Eigen vector for \p vector.
template<typename VECTOR>
typename SDenseMatrix<VECTOR>::Type toDenseVector(const VECTOR &vector) {
    return vector.template toType<typename SDenseVector<VECTOR>::Type>();
}

//! Get the dynamic Eigen vector for \p vector.
template<typename VECTOR>
CDenseVector<double> toDynamicDenseVector(const VECTOR &vector) {
    return vector.template toType<CDenseVector<double>>();
}

//! \brief The default type for converting Eigen matrices to our
//! internal symmetric matrices.
//!
//! IMPLEMENTATION:\n
//! This type is needed to get Eigen GEMM expressions to play nicely
//! with our symmetric matrix type constructors. Also, I think it is
//! useful to flag explicitly when a conversion is taking place, the
//! fromDenseMatrix function plays this role in code where we want a
//! conversion.
template<typename MATRIX>
class CDenseMatrixInitializer {
    public:
        explicit CDenseMatrixInitializer(const MATRIX &type) : m_Type(&type) {}

        std::size_t rows(void) const {
            return m_Type->rows();
        }

        double get(std::size_t i, std::size_t j) const {
            return (m_Type->template selfadjointView<Eigen::Lower>())(i, j);
        }

    private:
        const MATRIX *m_Type;
};

//! Convert an Eigen matrix to a form which can initialize one of our
//! symmetric matrix objects.
template<typename MATRIX>
CDenseMatrixInitializer<MATRIX> fromDenseMatrix(const MATRIX &type) {
    return CDenseMatrixInitializer<MATRIX>(type);
}

//! \brief The default type for converting Eigen vectors to our
//! internal vectors.
//!
//! IMPLEMENTATION:\n
//! This type is needed to get Eigen GEMM expressions to play nicely
//! with our vector type constructors. Also, I think it is useful to
//! flag explicitly when a conversion is taking place, the fromDenseVector
//! function plays this role in code where we want a conversion.
template<typename VECTOR>
class CDenseVectorInitializer {
    public:
        explicit CDenseVectorInitializer(const VECTOR &type) : m_Type(&type) {}

        std::size_t dimension(void) const {
            return m_Type->size();
        }

        double get(std::size_t i) const {
            return (*m_Type)(i);
        }

    private:
        const VECTOR *m_Type;
};

//! Convert an Eigen vector to a form which can initialize one of our
//! vector objects.
template<typename VECTOR>
CDenseVectorInitializer<VECTOR> fromDenseVector(const VECTOR &type) {
    return CDenseVectorInitializer<VECTOR>(type);
}

template<typename T, std::size_t N>
template<typename MATRIX>
CSymmetricMatrixNxN<T, N>::CSymmetricMatrixNxN(const CDenseMatrixInitializer<MATRIX> &m) {
    for (std::size_t i = 0u, i_ = 0u; i < N; ++i) {
        for (std::size_t j = 0u; j <= i; ++j, ++i_) {
            TBase::m_LowerTriangle[i_] = m.get(i, j);
        }
    }
}

template<typename T>
template<typename MATRIX>
CSymmetricMatrix<T>::CSymmetricMatrix(const CDenseMatrixInitializer<MATRIX> &m) {
    m_D = m.rows();
    TBase::m_LowerTriangle.resize(m_D * (m_D + 1) / 2);
    for (std::size_t i = 0u, i_ = 0u; i < m_D; ++i) {
        for (std::size_t j = 0u; j <= i; ++j, ++i_) {
            TBase::m_LowerTriangle[i_] = m.get(i, j);
        }
    }
}

template<typename T, std::size_t N>
template<typename VECTOR>
CVectorNx1<T, N>::CVectorNx1(const CDenseVectorInitializer<VECTOR> &v) {
    for (std::size_t i = 0u; i < N; ++i) {
        TBase::m_X[i] = v.get(i);
    }
}

template<typename T>
template<typename VECTOR>
CVector<T>::CVector(const CDenseVectorInitializer<VECTOR> &v) {
    TBase::m_X.resize(v.dimension());
    for (std::size_t i = 0u; i < TBase::m_X.size(); ++i) {
        TBase::m_X[i] = v.get(i);
    }
}

}
}

#endif // INCLUDED_ml_maths_CLinearAlgebraEigen_h
