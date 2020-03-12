/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLinearAlgebraEigen_h
#define INCLUDED_ml_maths_CLinearAlgebraEigen_h

#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CAnnotatedVector.h>
#include <maths/CChecksum.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraFwd.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/SparseCore>

#include <algorithm>
#include <iterator>

namespace Eigen {
#define LESS_OR_GREATER(l, r)                                                  \
    if (l < r) {                                                               \
        return true;                                                           \
    } else if (r > l) {                                                        \
        return false;                                                          \
    }

//! Less than on Eigen sparse matrix.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
bool operator<(const SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX>& lhs,
               const SparseMatrix<SCALAR, FLAGS, STORAGE_INDEX>& rhs) {
    LESS_OR_GREATER(lhs.rows(), rhs.rows())
    LESS_OR_GREATER(lhs.cols(), rhs.cols())
    for (STORAGE_INDEX i = 0; i < lhs.rows(); ++i) {
        for (STORAGE_INDEX j = 0; j < lhs.cols(); ++j) {
            LESS_OR_GREATER(lhs(i, j), rhs(i, j))
        }
    }
    return false;
}

//! Less than on Eigen sparse vector.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
bool operator<(const SparseVector<SCALAR, FLAGS, STORAGE_INDEX>& lhs,
               const SparseVector<SCALAR, FLAGS, STORAGE_INDEX>& rhs) {
    LESS_OR_GREATER(lhs.size(), rhs.size())
    for (STORAGE_INDEX i = 0; i < lhs.size(); ++i) {
        LESS_OR_GREATER(lhs(i), rhs(i))
    }
    return false;
}

//! Less than on Eigen dense matrix.
template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS>
bool operator<(const Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>& lhs,
               const Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>& rhs) {
    LESS_OR_GREATER(lhs.rows(), rhs.rows())
    LESS_OR_GREATER(lhs.cols(), rhs.cols())
    return std::lexicographical_compare(lhs.data(), lhs.data() + lhs.size(),
                                        rhs.data(), rhs.data() + rhs.size());
}

//! Less than on an Eigen memory mapped matrix.
template<typename PLAIN_OBJECT_TYPE, int OPTIONS, typename STRIDE_TYPE>
bool operator<(const Map<PLAIN_OBJECT_TYPE, OPTIONS, STRIDE_TYPE>& lhs,
               const Map<PLAIN_OBJECT_TYPE, OPTIONS, STRIDE_TYPE>& rhs) {
    LESS_OR_GREATER(lhs.rows(), rhs.rows())
    LESS_OR_GREATER(lhs.cols(), rhs.cols())
    return std::lexicographical_compare(lhs.data(), lhs.data() + lhs.size(),
                                        rhs.data(), rhs.data() + rhs.size());
}

#undef LESS_OR_GREATER

//! Free swap picked up by std:: algorithms etc.
template<typename SCALAR, int FLAGS, typename STORAGE_INDEX>
void swap(SparseVector<SCALAR, FLAGS, STORAGE_INDEX>& lhs,
          SparseVector<SCALAR, FLAGS, STORAGE_INDEX>& rhs) {
    lhs.swap(rhs);
}

//! Free swap picked up by std:: algorithms etc.
template<typename SCALAR, int ROWS, int COLS, int OPTIONS, int MAX_ROWS, int MAX_COLS>
void swap(Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>& lhs,
          Matrix<SCALAR, ROWS, COLS, OPTIONS, MAX_ROWS, MAX_COLS>& rhs) {
    lhs.swap(rhs);
}
}

namespace ml {
namespace maths {

//! Rename to follow our conventions and add to ml::maths.
template<typename SCALAR, int FLAGS = 0>
using CSparseMatrix = Eigen::SparseMatrix<SCALAR, FLAGS, std::ptrdiff_t>;

//! The type of an element of a sparse matrix in coordinate form.
template<typename SCALAR>
using CSparseMatrixElement = Eigen::Triplet<SCALAR>;

//! Rename to follow our conventions and add to ml::maths.
template<typename SCALAR, int FLAGS = Eigen::RowMajorBit>
using CSparseVector = Eigen::SparseVector<SCALAR, FLAGS, std::ptrdiff_t>;

//! The type of an element of a sparse vector in coordinate form.
template<typename SCALAR>
using CSparseVectorCoordinate = Eigen::Triplet<SCALAR>;

//! Create a tuple with which to initialize a sparse matrix.
template<typename SCALAR>
inline CSparseMatrixElement<SCALAR>
matrixElement(std::ptrdiff_t row, std::ptrdiff_t column, SCALAR value) {
    return CSparseMatrixElement<SCALAR>(row, column, value);
}

//! Create a tuple with which to initialize a sparse column vector.
template<typename SCALAR>
inline CSparseVectorCoordinate<SCALAR> vectorCoordinate(std::ptrdiff_t row, SCALAR value) {
    return CSparseVectorCoordinate<SCALAR>(row, 0, value);
}

//! \brief Adapts Eigen::SparseVector::InnerIterator for use with STL.
template<typename SCALAR, int FLAGS = Eigen::RowMajorBit>
class CSparseVectorIndexIterator
    : public std::iterator<std::input_iterator_tag, std::ptrdiff_t> {
    CSparseVectorIndexIterator(const CSparseVector<SCALAR, FLAGS>& vector, std::size_t index)
        : m_Vector(&vector), m_Base(vector, index) {}

    bool operator==(const CSparseVectorIndexIterator& rhs) const {
        return m_Vector == rhs.m_Vector && m_Base.row() == rhs.m_Base.row() &&
               m_Base.col() == rhs.m_Base.col();
    }
    bool operator!=(const CSparseVectorIndexIterator& rhs) const {
        return !(*this == rhs);
    }

    std::ptrdiff_t operator*() const {
        return std::max(m_Base.row(), m_Base.col());
    }

    CSparseVectorIndexIterator& operator++() {
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
    CSparseVector<SCALAR, FLAGS>* m_Vector;
    TIterator m_Base;
};

//! Get an iterator over the indices of \p vector.
template<typename SCALAR, int FLAGS>
CSparseVectorIndexIterator<SCALAR, FLAGS>
beginIndices(const CSparseVector<SCALAR, FLAGS>& vector) {
    return CSparseVectorIndexIterator<SCALAR, FLAGS>(vector, 0);
}

//! Get the end iterator of the indices of \p vector.
template<typename SCALAR, int FLAGS>
CSparseVectorIndexIterator<SCALAR, FLAGS>
endIndices(const CSparseVector<SCALAR, FLAGS>& vector) {
    return CSparseVectorIndexIterator<SCALAR, FLAGS>(vector, vector.data().size());
}

//! \brief Decorates an Eigen matrix with some useful methods.
template<typename SCALAR>
class CDenseMatrix : public Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> {
public:
    using TBase = Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>;

public:
    //! Forwarding constructor.
    template<typename... ARGS>
    CDenseMatrix(ARGS&&... args) : TBase(std::forward<ARGS>(args)...) {}

    //! \name Copy and Move Semantics
    //@{
    CDenseMatrix(const CDenseMatrix& other) = default;
    CDenseMatrix(CDenseMatrix&& other) = default;
    CDenseMatrix& operator=(const CDenseMatrix& other) = default;
    CDenseMatrix& operator=(CDenseMatrix&& other) = default;
    // @}

    //! Debug the memory usage of this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CDenseMatrix");
        mem->addItem("components", this->memoryUsage());
    }
    //! Get the memory used by this object.
    std::size_t memoryUsage() const { return sizeof(SCALAR) * this->size(); }

    //! Get a checksum of this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const {
        for (std::ptrdiff_t i = 0; i < this->size(); ++i) {
            seed = CChecksum::calculate(seed, this->coeff(i));
        }
        return seed;
    }
};

//! \brief Gets a constant dense square matrix with specified dimension or with
//! specified numbers of rows and columns.
template<typename SCALAR>
struct SConstant<CDenseMatrix<SCALAR>> {
    static CDenseMatrix<SCALAR> get(std::ptrdiff_t dimension, SCALAR constant) {
        return get(dimension, dimension, constant);
    }
    static CDenseMatrix<SCALAR> get(std::ptrdiff_t rows, std::ptrdiff_t cols, SCALAR constant) {
        return CDenseMatrix<SCALAR>::Constant(rows, cols, constant);
    }
};

//! \brief Decorates an Eigen column vector with some useful methods.
template<typename SCALAR>
class CDenseVector : public Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> {
public:
    using TBase = Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>;

public:
    static const std::string DENSE_VECTOR_TAG;

public:
    //! Forwarding constructor.
    template<typename... ARGS>
    CDenseVector(ARGS&&... args) : TBase(std::forward<ARGS>(args)...) {}

    //! \name Copy and Move Semantics
    //@{
    CDenseVector(const CDenseVector& other) = default;
    CDenseVector(CDenseVector&& other) = default;
    CDenseVector& operator=(const CDenseVector& other) = default;
    CDenseVector& operator=(CDenseVector&& other) = default;
    // @}

    //! Debug the memory usage of this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CDenseVector");
        mem->addItem("components", this->memoryUsage());
    }
    //! Get the memory used by this object.
    std::size_t memoryUsage() const { return sizeof(SCALAR) * this->size(); }

    //! Get a checksum of this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const {
        for (std::ptrdiff_t i = 0; i < this->size(); ++i) {
            seed = CChecksum::calculate(seed, this->coeff(i));
        }
        return seed;
    }

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        inserter.insertValue(DENSE_VECTOR_TAG, core::CPersistUtils::toString(
                                                   this->to<std::vector<SCALAR>>()));
    }

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        std::vector<SCALAR> tempVector;
        if (core::CPersistUtils::restore(DENSE_VECTOR_TAG, tempVector, traverser) == false) {
            LOG_ERROR(<< "Failed to restore " << DENSE_VECTOR_TAG << ", got "
                      << traverser.value());
            return false;
        }
        *this = fromStdVector(tempVector);
        return true;
    }

    //! Convert to a std::vector.
    //!
    //! It is assumed that COLLECTION supports reserve and push_back.
    template<typename COLLECTION>
    COLLECTION to() const {
        COLLECTION result;
        result.reserve(this->size());
        for (int i = 0; i < this->size(); ++i) {
            result.push_back(this->coeff(i));
        }
        return result;
    }

    //! Convert from a std::vector.
    static CDenseVector<SCALAR> fromStdVector(const std::vector<SCALAR>& vector) {
        CDenseVector<SCALAR> result(vector.size());
        for (std::size_t i = 0; i < vector.size(); ++i) {
            result(i) = vector[i];
        }
        return result;
    }
};

template<typename SCALAR>
const std::string CDenseVector<SCALAR>::DENSE_VECTOR_TAG{"dense_vector"};

//! \brief Gets a constant dense vector with specified dimension.
template<typename SCALAR>
struct SConstant<CDenseVector<SCALAR>> {
    static CDenseVector<SCALAR> get(std::ptrdiff_t dimension, SCALAR constant) {
        return CDenseVector<SCALAR>::Constant(dimension, constant);
    }
};

//! \brief Decorates an Eigen::Map of a dense matrix with some useful methods
//! and changes default copy semantics to shallow copy.
//!
//! IMPLEMENTATION:\n
//! This effectively acts like a std::reference_wrapper of an Eigen::Map for
//! an Eigen matrix. In particular, all copying is shallow unlike Eigen::Map
//! that acts directly on the referenced memory. This is to match the behaviour
//! of CMemoryMappedDenseVector.
//!
//! \sa CMemoryMappedDenseVector for more information.
template<typename SCALAR>
class CMemoryMappedDenseMatrix
    : public Eigen::Map<typename CDenseMatrix<SCALAR>::TBase> {
public:
    using TBase = Eigen::Map<typename CDenseMatrix<SCALAR>::TBase>;

    //! See core::CMemory.
    static bool dynamicSizeAlwaysZero() { return true; }

public:
    //! Forwarding constructor.
    template<typename... ARGS>
    CMemoryMappedDenseMatrix(ARGS&&... args)
        : TBase{std::forward<ARGS>(args)...} {}

    //! \name Copy and Move Semantics
    //@{
    CMemoryMappedDenseMatrix(const CMemoryMappedDenseMatrix& other)
        : TBase{nullptr, 1, 1} {
        this->reseat(other);
    }
    CMemoryMappedDenseMatrix(CMemoryMappedDenseMatrix&& other)
        : TBase{nullptr, 1, 1} {
        this->reseat(other);
    }
    CMemoryMappedDenseMatrix& operator=(const CMemoryMappedDenseMatrix& other) {
        if (this != &other) {
            this->reseat(other);
        }
        return *this;
    }
    CMemoryMappedDenseMatrix& operator=(CMemoryMappedDenseMatrix&& other) {
        if (this != &other) {
            this->reseat(other);
        }
        return *this;
    }
    //@}

    //! Assignment from a dense matrix.
    template<typename OTHER_SCALAR>
    CMemoryMappedDenseMatrix& operator=(const CDenseMatrix<OTHER_SCALAR>& rhs) {
        static_cast<TBase&>(*this) = rhs.template cast<SCALAR>();
        return *this;
    }

    //! Get a checksum of this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const {
        for (std::ptrdiff_t i = 0; i < this->rows(); ++i) {
            for (std::ptrdiff_t j = 0; j < this->rows(); ++j) {
                seed = CChecksum::calculate(seed, (*this)(i, j));
            }
        }
        return seed;
    }

private:
    void reseat(const CMemoryMappedDenseMatrix& other) {
        TBase* base{static_cast<TBase*>(this)};
        new (base) TBase{const_cast<SCALAR*>(other.data()), other.rows(), other.cols()};
    }
};

//! \brief Gets a constant square dense matrix with specified dimension or with
//! specified numbers of rows and columns.
template<typename SCALAR>
struct SConstant<CMemoryMappedDenseMatrix<SCALAR>> {
    static auto get(std::ptrdiff_t dimension, SCALAR constant)
        -> decltype(SConstant<CDenseMatrix<SCALAR>>::get(dimension, 1)) {
        return SConstant<CDenseMatrix<SCALAR>>::get(dimension, constant);
    }
    static auto get(std::ptrdiff_t rows, std::ptrdiff_t cols, SCALAR constant)
        -> decltype(SConstant<CDenseMatrix<SCALAR>>::get(rows, cols, constant)) {
        return SConstant<CDenseMatrix<SCALAR>>::get(rows, cols, constant);
    }
};

//! \brief Decorates an Eigen::Map of a dense vector with some useful methods
//! and changes default copy semantics to shallow.
//!
//! IMPLEMENTATION:\n
//! This effectively acts like a std::reference_wrapper of an Eigen::Map for
//! an Eigen vector. In particular, all copying is shallow unlike Eigen::Map
//! that acts directly on the referenced memory, i.e.
//! \code{.cpp}
//! double values1[]{1.0, 1.0};
//! double values2[]{2.0, 2.0};
//!
//! CMemoryMappedDenseVector<double> mm1{values1, 2};
//! CMemoryMappedDenseVector<double> mm2{values2, 2};
//!
//! mm1 = mm2;
//! std::cout << mm1(0) << "," << mm1(1) << "," << values1[0] << "," << values1[1] << std::endl;
//!
//! Eigen::Map<Eigen::VectorXd> map1{values1, 2};
//! Eigen::Map<Eigen::VectorXd> map2{values2, 2};
//!
//! map1 = map2;
//! std::cout << map1(0) << "," << map1(1) << "," << values1[0] << "," << values1[1] << std::endl;
//! \endcode
//!
//! Outputs:\n
//! 2,2,1,1\n
//! 2,2,2,2
//!
//! This better fits our needs with data frames where we want to reference the
//! memory stored in the data frame rows, but never modify it directly through
//! this vector type.
template<typename SCALAR>
class CMemoryMappedDenseVector
    : public Eigen::Map<typename CDenseVector<SCALAR>::TBase> {
public:
    using TDenseVector = CDenseVector<SCALAR>;
    using TBase = Eigen::Map<typename TDenseVector::TBase>;

    //! See core::CMemory.
    static bool dynamicSizeAlwaysZero() { return true; }

public:
    //! Forwarding constructor.
    template<typename... ARGS>
    CMemoryMappedDenseVector(ARGS&&... args)
        : TBase{std::forward<ARGS>(args)...} {}

    //! Added because the forwarding constructor above doesn't work with
    //! annotated vector arguments with Visual Studio 2019 in C++17 mode.
    template<typename ANNOTATION>
    CMemoryMappedDenseVector(CAnnotatedVector<TDenseVector, ANNOTATION>&& annotatedDense)
        : TBase{annotatedDense.data(), annotatedDense.size()} {}

    //! \name Copy and Move Semantics
    //@{
    CMemoryMappedDenseVector(const CMemoryMappedDenseVector& other)
        : TBase{nullptr, 1} {
        this->reseat(other);
    }
    CMemoryMappedDenseVector(CMemoryMappedDenseVector&& other)
        : TBase{nullptr, 1} {
        this->reseat(other);
    }
    CMemoryMappedDenseVector& operator=(const CMemoryMappedDenseVector& other) {
        if (this != &other) {
            this->reseat(other);
        }
        return *this;
    }
    CMemoryMappedDenseVector& operator=(CMemoryMappedDenseVector&& other) {
        if (this != &other) {
            this->reseat(other);
        }
        return *this;
    }
    //@}

    //! Assignment from a dense vector.
    template<typename OTHER_SCALAR>
    CMemoryMappedDenseVector& operator=(const CDenseVector<OTHER_SCALAR>& rhs) {
        static_cast<TBase&>(*this) = rhs.template cast<SCALAR>();
        return *this;
    }

    //! Get a checksum of this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const {
        for (std::ptrdiff_t i = 0; i < this->size(); ++i) {
            seed = CChecksum::calculate(seed, this->coeff(i));
        }
        return seed;
    }

private:
    void reseat(const CMemoryMappedDenseVector& other) {
        TBase* base{static_cast<TBase*>(this)};
        new (base) TBase{const_cast<SCALAR*>(other.data()), other.size()};
    }
};

//! \brief Gets a constant dense vector with specified dimension.
template<typename SCALAR>
struct SConstant<CMemoryMappedDenseVector<SCALAR>> {
    static auto get(std::ptrdiff_t dimension, SCALAR constant)
        -> decltype(SConstant<CDenseVector<SCALAR>>::get(dimension, constant)) {
        return SConstant<CDenseVector<SCALAR>>::get(dimension, constant);
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
typename SDenseMatrix<MATRIX>::Type toDenseMatrix(const MATRIX& matrix) {
    return matrix.template toType<typename SDenseMatrix<MATRIX>::Type>();
}

//! Get the dynamic Eigen matrix for \p matrix.
template<typename MATRIX>
CDenseMatrix<double> toDynamicDenseMatrix(const MATRIX& matrix) {
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
typename SDenseVector<VECTOR>::Type toDenseVector(const VECTOR& vector) {
    return vector.template toType<typename SDenseVector<VECTOR>::Type>();
}

//! Get the dynamic Eigen vector for \p vector.
template<typename VECTOR>
CDenseVector<double> toDynamicDenseVector(const VECTOR& vector) {
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
    explicit CDenseMatrixInitializer(const MATRIX& type) : m_Type(&type) {}

    std::size_t rows() const { return m_Type->rows(); }

    double get(std::size_t i, std::size_t j) const {
        return (m_Type->template selfadjointView<Eigen::Lower>())(i, j);
    }

private:
    const MATRIX* m_Type;
};

//! Convert an Eigen matrix to a form which can initialize one of our
//! symmetric matrix objects.
template<typename MATRIX>
CDenseMatrixInitializer<MATRIX> fromDenseMatrix(const MATRIX& type) {
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
    explicit CDenseVectorInitializer(const VECTOR& type) : m_Type(&type) {}

    std::size_t dimension() const { return m_Type->size(); }

    double get(std::size_t i) const { return (*m_Type)(i); }

private:
    const VECTOR* m_Type;
};

//! Convert an Eigen vector to a form which can initialize one of our
//! vector objects.
template<typename VECTOR>
CDenseVectorInitializer<VECTOR> fromDenseVector(const VECTOR& type) {
    return CDenseVectorInitializer<VECTOR>(type);
}

template<typename T, std::size_t N>
template<typename MATRIX>
CSymmetricMatrixNxN<T, N>::CSymmetricMatrixNxN(const CDenseMatrixInitializer<MATRIX>& m) {
    for (std::size_t i = 0u, i_ = 0u; i < N; ++i) {
        for (std::size_t j = 0u; j <= i; ++j, ++i_) {
            TBase::m_LowerTriangle[i_] = m.get(i, j);
        }
    }
}

template<typename T>
template<typename MATRIX>
CSymmetricMatrix<T>::CSymmetricMatrix(const CDenseMatrixInitializer<MATRIX>& m) {
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
CVectorNx1<T, N>::CVectorNx1(const CDenseVectorInitializer<VECTOR>& v) {
    for (std::size_t i = 0u; i < N; ++i) {
        TBase::m_X[i] = v.get(i);
    }
}

template<typename T>
template<typename VECTOR>
CVector<T>::CVector(const CDenseVectorInitializer<VECTOR>& v) {
    TBase::m_X.resize(v.dimension());
    for (std::size_t i = 0u; i < TBase::m_X.size(); ++i) {
        TBase::m_X[i] = v.get(i);
    }
}
}
}

namespace Eigen {
template<typename BIN_OP>
struct ScalarBinaryOpTraits<ml::core::CFloatStorage, double, BIN_OP> {
    using ReturnType = double;
};

template<typename BIN_OP>
struct ScalarBinaryOpTraits<double, ml::core::CFloatStorage, BIN_OP> {
    using ReturnType = double;
};
}

#endif // INCLUDED_ml_maths_CLinearAlgebraEigen_h
