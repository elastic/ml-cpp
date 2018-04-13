/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLinearAlgebraTools_h
#define INCLUDED_ml_maths_CLinearAlgebraTools_h

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CTools.h>
#include <maths/ImportExport.h>

#include <cstddef>
#include <limits>
#include <ostream>
#include <vector>

namespace ml
{
namespace maths
{
namespace linear_algebra_tools_detail
{

struct VectorTag;
struct MatrixTag;
struct VectorVectorTag;
struct MatrixMatrixTag;
struct VectorScalarTag;
struct MatrixScalarTag;
struct ScalarVectorTag;
struct ScalarMatrixTag;

template<typename TAG> struct SSqrt {};
//! Component-wise sqrt for a vector.
template<>
struct SSqrt<VectorTag>
{
    template<typename VECTOR>
    static void calculate(std::size_t d, VECTOR &result)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            result(i) = std::sqrt(result(i));
        }
    }
};
//! Element-wise sqrt for a symmetric matrix.
template<>
struct SSqrt<MatrixTag>
{
    template<typename MATRIX>
    static void calculate(std::size_t d, MATRIX &result)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j)
            {
                result(i, j) = std::sqrt(result(i, j));
            }
        }
    }
};

template<typename TAG> struct SMin {};
//! Component-wise minimum for a vector.
template<>
struct SMin<VectorVectorTag>
{
    template<typename VECTOR>
    static void calculate(std::size_t d, const VECTOR &lhs, VECTOR &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            rhs(i) = std::min(lhs(i), rhs(i));
        }
    }
};
//! Component-wise minimum for a vector.
template<>
struct SMin<VectorScalarTag>
{
    template<typename VECTOR, typename T>
    static void calculate(std::size_t d, VECTOR &lhs, const T &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            lhs(i) = std::min(lhs(i), rhs);
        }
    }
};
//! Component-wise minimum for a vector.
template<>
struct SMin<ScalarVectorTag>
{
    template<typename T, typename VECTOR>
    static void calculate(std::size_t d, const T &lhs, VECTOR &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            rhs(i) = std::min(rhs(i), lhs);
        }
    }
};
//! Element-wise minimum for a symmetric matrix.
template<>
struct SMin<MatrixMatrixTag>
{
    template<typename MATRIX>
    static void calculate(std::size_t d, const MATRIX &lhs, MATRIX &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j)
            {
                rhs(i, j) = std::min(lhs(i, j), rhs(i, j));
            }
        }
    }
};
//! Element-wise minimum for a symmetric matrix.
template<>
struct SMin<MatrixScalarTag>
{
    template<typename MATRIX, typename T>
    static void calculate(std::size_t d, MATRIX &lhs, const T &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j)
            {
                lhs(i, j) = std::min(lhs(i, j), rhs);
            }
        }
    }
};
//! Element-wise minimum for a symmetric matrix.
template<>
struct SMin<ScalarMatrixTag>
{
    template<typename T, typename MATRIX>
    static void calculate(std::size_t d, const T &lhs, MATRIX &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j)
            {
                rhs(i, j) = std::min(lhs, rhs(i, j));
            }
        }
    }
};

template<typename TAG> struct SMax {};
//! Component-wise maximum for a vector.
template<>
struct SMax<VectorVectorTag>
{
    template<typename VECTOR>
    static void calculate(std::size_t d, const VECTOR &lhs, VECTOR &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            rhs(i) = std::max(lhs(i), rhs(i));
        }
    }
};
//! Component-wise maximum for a vector.
template<>
struct SMax<VectorScalarTag>
{
    template<typename VECTOR, typename T>
    static void calculate(std::size_t d, VECTOR &lhs, const T &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            lhs(i) = std::max(lhs(i), rhs);
        }
    }
};
//! Component-wise maximum for a vector.
template<>
struct SMax<ScalarVectorTag>
{
    template<typename T, typename VECTOR>
    static void calculate(std::size_t d, const T &lhs, VECTOR &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            rhs(i) = std::max(rhs(i), lhs);
        }
    }
};
//! Element-wise maximum for a symmetric matrix.
template<>
struct SMax<MatrixMatrixTag>
{
    template<typename MATRIX>
    static void calculate(std::size_t d, const MATRIX &lhs, MATRIX &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j)
            {
                rhs(i, j) = std::max(lhs(i, j), rhs(i, j));
            }
        }
    }
};
//! Element-wise maximum for a symmetric matrix.
template<>
struct SMax<MatrixScalarTag>
{
    template<typename MATRIX, typename T>
    static void calculate(std::size_t d, MATRIX &lhs, const T &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j)
            {
                lhs(i, j) = std::max(lhs(i, j), rhs);
            }
        }
    }
};
//! Element-wise maximum for a symmetric matrix.
template<>
struct SMax<ScalarMatrixTag>
{
    template<typename T, typename MATRIX>
    static void calculate(std::size_t d, const T &lhs, MATRIX &rhs)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j)
            {
                rhs(i, j) = std::max(lhs, rhs(i, j));
            }
        }
    }
};

template<typename TAG> struct SFabs {};
//! Component-wise fabs for a vector.
template<>
struct SFabs<VectorTag>
{
    template<typename VECTOR>
    static void calculate(std::size_t d, VECTOR &result)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            result(i) = std::fabs(result(i));
        }
    }
};
//! Element-wise fabs for a symmetric matrix.
template<>
struct SFabs<MatrixTag>
{
    template<typename MATRIX>
    static void calculate(std::size_t d, MATRIX &result)
    {
        for (std::size_t i = 0u; i < d; ++i)
        {
            for (std::size_t j = 0u; j <= i; ++j)
            {
                result(i, j) = std::fabs(result(i, j));
            }
        }
    }
};

#define INVERSE_QUADRATIC_PRODUCT(T, N)                                                                 \
MATHS_EXPORT                                                                                            \
maths_t::EFloatingPointErrorStatus inverseQuadraticProduct(std::size_t d,                               \
                                                           const CSymmetricMatrixNxN<T, N> &covariance, \
                                                           const CVectorNx1<T, N> &residual,            \
                                                           double &result,                              \
                                                           bool ignoreSingularSubspace)
INVERSE_QUADRATIC_PRODUCT(CFloatStorage, 2);
INVERSE_QUADRATIC_PRODUCT(CFloatStorage, 3);
INVERSE_QUADRATIC_PRODUCT(CFloatStorage, 4);
INVERSE_QUADRATIC_PRODUCT(CFloatStorage, 5);
INVERSE_QUADRATIC_PRODUCT(double, 2);
INVERSE_QUADRATIC_PRODUCT(double, 3);
INVERSE_QUADRATIC_PRODUCT(double, 4);
INVERSE_QUADRATIC_PRODUCT(double, 5);
#undef INVERSE_QUADRATIC_PRODUCT
MATHS_EXPORT
maths_t::EFloatingPointErrorStatus inverseQuadraticProduct(std::size_t d,
                                                           const CSymmetricMatrix<CFloatStorage> &covariance,
                                                           const CVector<CFloatStorage> &residual,
                                                           double &result,
                                                           bool ignoreSingularSubspace);
MATHS_EXPORT
maths_t::EFloatingPointErrorStatus inverseQuadraticProduct(std::size_t d,
                                                           const CSymmetricMatrix<double> &covariance,
                                                           const CVector<double> &residual,
                                                           double &result,
                                                           bool ignoreSingularSubspace);


#define GAUSSIAN_LOG_LIKELIHOOD(T, N)                                                                 \
MATHS_EXPORT                                                                                          \
maths_t::EFloatingPointErrorStatus gaussianLogLikelihood(std::size_t d,                               \
                                                         const CSymmetricMatrixNxN<T, N> &covariance, \
                                                         const CVectorNx1<T, N> &residual,            \
                                                         double &result,                              \
                                                         bool ignoreSingularSubspace)
GAUSSIAN_LOG_LIKELIHOOD(CFloatStorage, 2);
GAUSSIAN_LOG_LIKELIHOOD(CFloatStorage, 3);
GAUSSIAN_LOG_LIKELIHOOD(CFloatStorage, 4);
GAUSSIAN_LOG_LIKELIHOOD(CFloatStorage, 5);
GAUSSIAN_LOG_LIKELIHOOD(double, 2);
GAUSSIAN_LOG_LIKELIHOOD(double, 3);
GAUSSIAN_LOG_LIKELIHOOD(double, 4);
GAUSSIAN_LOG_LIKELIHOOD(double, 5);
#undef GAUSSIAN_LOG_LIKELIHOOD
MATHS_EXPORT
maths_t::EFloatingPointErrorStatus gaussianLogLikelihood(std::size_t d,
                                                         const CSymmetricMatrix<CFloatStorage> &covariance,
                                                         const CVector<CFloatStorage> &residual,
                                                         double &result,
                                                         bool ignoreSingularSubspace);
MATHS_EXPORT
maths_t::EFloatingPointErrorStatus gaussianLogLikelihood(std::size_t d,
                                                         const CSymmetricMatrix<double> &covariance,
                                                         const CVector<double> &residual,
                                                         double &result,
                                                         bool ignoreSingularSubspace);

//! Shared implementation of Gaussian sampling.
#define SAMPLE_GAUSSIAN(T, N)                                    \
MATHS_EXPORT                                                     \
void sampleGaussian(std::size_t n,                               \
                    const CVectorNx1<T, N> &mean,                \
                    const CSymmetricMatrixNxN<T, N> &covariance, \
                    std::vector<CVectorNx1<double, N> > &result)
SAMPLE_GAUSSIAN(CFloatStorage, 2);
SAMPLE_GAUSSIAN(CFloatStorage, 3);
SAMPLE_GAUSSIAN(CFloatStorage, 4);
SAMPLE_GAUSSIAN(CFloatStorage, 5);
SAMPLE_GAUSSIAN(double, 2);
SAMPLE_GAUSSIAN(double, 3);
SAMPLE_GAUSSIAN(double, 4);
SAMPLE_GAUSSIAN(double, 5);
#undef SAMPLE_GAUSSIAN
MATHS_EXPORT
void sampleGaussian(std::size_t n,
                    const CVector<CFloatStorage> &mean,
                    const CSymmetricMatrix<CFloatStorage> &covariance,
                    std::vector<CVector<double> > &result);
MATHS_EXPORT
void sampleGaussian(std::size_t n,
                    const CVector<double> &mean,
                    const CSymmetricMatrix<double> &covariance,
                    std::vector<CVector<double> > &result);

//! Shared implementation of the log-determinant function.
#define LOG_DETERMINANT(T, N)                                                              \
MATHS_EXPORT                                                                               \
maths_t::EFloatingPointErrorStatus logDeterminant(std::size_t d,                           \
                                                  const CSymmetricMatrixNxN<T, N> &matrix, \
                                                  double &result,                          \
                                                  bool ignoreSingularSubspace)
LOG_DETERMINANT(CFloatStorage, 2);
LOG_DETERMINANT(CFloatStorage, 3);
LOG_DETERMINANT(CFloatStorage, 4);
LOG_DETERMINANT(CFloatStorage, 5);
LOG_DETERMINANT(double, 2);
LOG_DETERMINANT(double, 3);
LOG_DETERMINANT(double, 4);
LOG_DETERMINANT(double, 5);
#undef LOG_DETERMINANT
MATHS_EXPORT
maths_t::EFloatingPointErrorStatus logDeterminant(std::size_t d,
                                                  const CSymmetricMatrix<CFloatStorage> &matrix,
                                                  double &result,
                                                  bool ignoreSingularSubspace);
MATHS_EXPORT
maths_t::EFloatingPointErrorStatus logDeterminant(std::size_t d,
                                                  const CSymmetricMatrix<double> &matrix,
                                                  double &result,
                                                  bool ignoreSingularSubspace);

}

//! Output for debug.
template<typename T>
std::ostream &operator<<(std::ostream &o, const CSymmetricMatrix<T> &m)
{
    for (std::size_t i = 0u; i < m.rows(); ++i)
    {
        o << "\n    ";
        for (std::size_t j = 0u; j < m.columns(); ++j)
        {
            std::string element = core::CStringUtils::typeToStringPretty(m(i, j));
            o << element << std::string(15 - element.size(), ' ');
        }
    }
    return o;
}

//! Output for debug.
template<typename T, std::size_t N>
std::ostream &operator<<(std::ostream &o, const CSymmetricMatrixNxN<T, N> &m)
{
    for (std::size_t i = 0u; i < N; ++i)
    {
        o << "\n    ";
        for (std::size_t j = 0u; j < N; ++j)
        {
            std::string element = core::CStringUtils::typeToStringPretty(m(i, j));
            o << element << std::string(15 - element.size(), ' ');
        }
    }
    return o;
}

//! Output for debug.
template<typename T, std::size_t N>
std::ostream &operator<<(std::ostream &o, const CVectorNx1<T, N> &v)
{
    o << "[";
    for (std::size_t i = 0u; i+1 < N; ++i)
    {
        o << core::CStringUtils::typeToStringPretty(v(i)) << ' ';
    }
    o << core::CStringUtils::typeToStringPretty(v(N-1)) << ']';
    return o;
}

//! Output for debug.
template<typename T>
std::ostream &operator<<(std::ostream &o, const CVector<T> &v)
{
    if (v.dimension() == 0)
    {
        return o << "[]";
    }
    o << "[";
    for (std::size_t i = 0u; i+1 < v.dimension(); ++i)
    {
        o << core::CStringUtils::typeToStringPretty(v(i)) << ' ';
    }
    o << core::CStringUtils::typeToStringPretty(v(v.dimension()-1)) << ']';
    return o;
}

//! Overload sqrt for CVectorNx1.
template<typename T, std::size_t N>
CVectorNx1<T, N> sqrt(const CVectorNx1<T, N> &v)
{
    CVectorNx1<T, N> result(v);
    linear_algebra_tools_detail::SSqrt<linear_algebra_tools_detail::VectorTag>::calculate(N, result);
    return result;
}
//! Overload sqrt for CSymmetricMatrixNxN.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N> sqrt(const CSymmetricMatrixNxN<T, N> &m)
{
    CSymmetricMatrixNxN<T, N> result(m);
    linear_algebra_tools_detail::SSqrt<linear_algebra_tools_detail::MatrixTag>::calculate(N, result);
    return result;
}

//! Overload minimum for CVectorNx1.
template<typename T, std::size_t N>
CVectorNx1<T, N> min(const CVectorNx1<T, N> &lhs,
                     const CVectorNx1<T, N> &rhs)
{
    CVectorNx1<T, N> result(rhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::VectorVectorTag>::calculate(N, lhs, result);
    return result;
}
//! Overload minimum for CVectorNx1.
template<typename T, std::size_t N>
CVectorNx1<T, N> min(const CVectorNx1<T, N> &lhs, const T &rhs)
{
    CVectorNx1<T, N> result(lhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::VectorScalarTag>::calculate(N, result, rhs);
    return result;
}
//! Overload minimum for CVectorNx1.
template<typename T, std::size_t N>
CVectorNx1<T, N> min(const T &lhs, const CVectorNx1<T, N> &rhs)
{
    CVectorNx1<T, N> result(rhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::ScalarVectorTag>::calculate(N, lhs, result);
    return result;
}
//! Overload minimum for CSymmetricMatrixNxN.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N> min(const CSymmetricMatrixNxN<T, N> &lhs,
                              const CSymmetricMatrixNxN<T, N> &rhs)
{
    CSymmetricMatrixNxN<T, N> result(rhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::MatrixMatrixTag>::calculate(N, lhs, result);
    return result;
}
//! Overload minimum for CSymmetricMatrixNxN.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N> min(const CSymmetricMatrixNxN<T, N> &lhs,
                              const T &rhs)
{
    CSymmetricMatrixNxN<T, N> result(lhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::MatrixScalarTag>::calculate(N, result, rhs);
    return result;
}
//! Overload minimum for CSymmetricMatrixNxN.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N> min(const T &lhs,
                              const CSymmetricMatrixNxN<T, N> &rhs)
{
    CSymmetricMatrixNxN<T, N> result(rhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::ScalarMatrixTag>::calculate(N, lhs, result);
    return result;
}

//! Overload maximum for CVectorNx1.
template<typename T, std::size_t N>
CVectorNx1<T, N> max(const CVectorNx1<T, N> &lhs,
                     const CVectorNx1<T, N> &rhs)
{
    CVectorNx1<T, N> result(rhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::VectorVectorTag>::calculate(N, lhs, result);
    return result;
}
//! Overload maximum for CVectorNx1.
template<typename T, std::size_t N>
CVectorNx1<T, N> max(const CVectorNx1<T, N> &lhs, const T &rhs)
{
    CVectorNx1<T, N> result(lhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::VectorScalarTag>::calculate(N, result, rhs);
    return result;
}
//! Overload maximum for CVectorNx1.
template<typename T, std::size_t N>
CVectorNx1<T, N> max(const T &lhs, const CVectorNx1<T, N> &rhs)
{
    CVectorNx1<T, N> result(rhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::ScalarVectorTag>::calculate(N, lhs, result);
    return result;
}
//! Overload maximum for CSymmetricMatrixNxN.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N> max(const CSymmetricMatrixNxN<T, N> &lhs,
                              const CSymmetricMatrixNxN<T, N> &rhs)
{
    CSymmetricMatrixNxN<T, N> result(rhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::MatrixMatrixTag>::calculate(N, lhs, result);
    return result;
}
//! Overload maximum for CSymmetricMatrixNxN.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N> max(const CSymmetricMatrixNxN<T, N> &lhs,
                              const T &rhs)
{
    CSymmetricMatrixNxN<T, N> result(lhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::MatrixScalarTag>::calculate(N, result, rhs);
    return result;
}
//! Overload maximum for CSymmetricMatrixNxN.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N> max(const T &lhs,
                              const CSymmetricMatrixNxN<T, N> &rhs)
{
    CSymmetricMatrixNxN<T, N> result(rhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::ScalarMatrixTag>::calculate(N, lhs, result);
    return result;
}

//! Overload ::fabs for CVectorNx1.
template<typename T, std::size_t N>
CVectorNx1<T, N> fabs(const CVectorNx1<T, N> &v)
{
    CVectorNx1<T, N> result(v);
    linear_algebra_tools_detail::SFabs<linear_algebra_tools_detail::VectorTag>::calculate(N, result);
    return result;
}
//! Overload ::fabs for CSymmetricMatrixNxN.
template<typename T, std::size_t N>
CSymmetricMatrixNxN<T, N> fabs(const CSymmetricMatrixNxN<T, N> &m)
{
    CSymmetricMatrixNxN<T, N> result(m);
    linear_algebra_tools_detail::SFabs<linear_algebra_tools_detail::MatrixTag>::calculate(N, result);
    return result;
}

//! Overload sqrt for CVector.
template<typename T>
CVector<T> sqrt(const CVector<T> &v)
{
    CVector<T> result(v);
    linear_algebra_tools_detail::SSqrt<linear_algebra_tools_detail::VectorTag>::calculate(result.dimension(), result);
    return result;
}
//! Overload sqrt for CSymmetricMatrix.
template<typename T>
CSymmetricMatrix<T> sqrt(const CSymmetricMatrix<T> &m)
{
    CSymmetricMatrix<T> result(m);
    linear_algebra_tools_detail::SSqrt<linear_algebra_tools_detail::MatrixTag>::calculate(result.rows(), result);
    return result;
}

//! Overload minimum for CVector.
template<typename T>
CVector<T> min(const CVector<T> &lhs, const CVector<T> &rhs)
{
    CVector<T> result(rhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::VectorVectorTag>::calculate(result.dimension(), lhs, result);
    return result;
}
//! Overload minimum for CVector.
template<typename T>
CVector<T> min(const CVector<T> &lhs, const T &rhs)
{
    CVector<T> result(lhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::VectorScalarTag>::calculate(result.dimension(), result, rhs);
    return result;
}
//! Overload minimum for CVector.
template<typename T>
CVector<T> min(const T &lhs, const CVector<T> &rhs)
{
    CVector<T> result(rhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::ScalarVectorTag>::calculate(result.dimension(), lhs, result);
    return result;
}
//! Overload minimum for CSymmetricMatrix.
template<typename T>
CSymmetricMatrix<T> min(const CSymmetricMatrix<T> &lhs, const CSymmetricMatrix<T> &rhs)
{
    CSymmetricMatrix<T> result(rhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::MatrixMatrixTag>::calculate(result.rows(), lhs, result);
    return result;
}
//! Overload minimum for CSymmetricMatrix.
template<typename T>
CSymmetricMatrix<T> min(const CSymmetricMatrix<T> &lhs, const T &rhs)
{
    CSymmetricMatrix<T> result(lhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::MatrixScalarTag>::calculate(result.rows(), result, rhs);
    return result;
}
//! Overload minimum for CSymmetricMatrix.
template<typename T>
CSymmetricMatrix<T> min(const T &lhs, const CSymmetricMatrix<T> &rhs)
{
    CSymmetricMatrix<T> result(rhs);
    linear_algebra_tools_detail::SMin<linear_algebra_tools_detail::ScalarMatrixTag>::calculate(result.rows(), lhs, result);
    return result;
}

//! Overload maximum for CVector.
template<typename T>
CVector<T> max(const CVector<T> &lhs, const CVector<T> &rhs)
{
    CVector<T> result(rhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::VectorVectorTag>::calculate(result.dimension(), lhs, result);
    return result;
}
//! Overload maximum for CVector.
template<typename T>
CVector<T> max(const CVector<T> &lhs, const T &rhs)
{
    CVector<T> result(lhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::VectorScalarTag>::calculate(result.dimension(), result, rhs);
    return result;
}
//! Overload maximum for CVector.
template<typename T>
CVector<T> max(const T &lhs, const CVector<T> &rhs)
{
    CVector<T> result(rhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::ScalarVectorTag>::calculate(result.dimension(), lhs, result);
    return result;
}
//! Overload maximum for CSymmetricMatrix.
template<typename T>
CSymmetricMatrix<T> max(const CSymmetricMatrix<T> &lhs, const CSymmetricMatrix<T> &rhs)
{
    CSymmetricMatrix<T> result(rhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::MatrixMatrixTag>::calculate(result.rows(), lhs, result);
    return result;
}
//! Overload maximum for CSymmetricMatrix.
template<typename T>
CSymmetricMatrix<T> max(const CSymmetricMatrix<T> &lhs, const T &rhs)
{
    CSymmetricMatrix<T> result(lhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::MatrixScalarTag>::calculate(result.rows(), result, rhs);
    return result;
}
//! Overload maximum for CSymmetricMatrix.
template<typename T>
CSymmetricMatrix<T> max(const T &lhs, const CSymmetricMatrix<T> &rhs)
{
    CSymmetricMatrix<T> result(rhs);
    linear_algebra_tools_detail::SMax<linear_algebra_tools_detail::ScalarMatrixTag>::calculate(result.rows(), lhs, result);
    return result;
}

//! Overload ::fabs for CVector.
template<typename T>
CVector<T> fabs(const CVector<T> &v)
{
    CVector<T> result(v);
    linear_algebra_tools_detail::SFabs<linear_algebra_tools_detail::VectorTag>::calculate(result.dimension(), result);
    return result;
}
//! Overload ::fabs for CSymmetricMatrix.
template<typename T>
CSymmetricMatrix<T> fabs(const CSymmetricMatrix<T> &m)
{
    CSymmetricMatrix<T> result(m);
    linear_algebra_tools_detail::SFabs<linear_algebra_tools_detail::MatrixTag>::calculate(result.dimension(), result);
    return result;
}

//! Efficiently scale the \p i'th row and column by \p scale.
template<typename T, std::size_t N>
void scaleCovariances(std::size_t i,
                      T scale,
                      CSymmetricMatrixNxN<T, N> &m)
{
    scale = std::sqrt(scale);
    for (std::size_t j = 0u; j < m.columns(); ++j)
    {
        if (i == j)
        {
            m(i, j) *= scale;
        }
        m(i, j) *= scale;
    }
}

//! Efficiently scale the rows and columns by \p scale.
template<typename T, std::size_t N>
void scaleCovariances(const CVectorNx1<T, N> &scale,
                      CSymmetricMatrixNxN<T, N> &m)
{
    for (std::size_t i = 0u; i < scale.dimension(); ++i)
    {
        scaleCovariances(i, scale(i), m);
    }
}

//! Efficiently scale the \p i'th row and column by \p scale.
template<typename T>
void scaleCovariances(std::size_t i,
                      T scale,
                      CSymmetricMatrix<T> &m)
{
    scale = std::sqrt(scale);
    for (std::size_t j = 0u; j < m.columns(); ++j)
    {
        if (i == j)
        {
            m(i, j) = scale;
        }
        m(i, j) = scale;
    }
}

//! Efficiently scale the rows and columns by \p scale.
template<typename T>
void scaleCovariances(const CVector<T> &scale,
                      CSymmetricMatrix<T> &m)
{
    for (std::size_t i = 0u; i < scale.dimension(); ++i)
    {
        scaleRowAndColumn(i, scale(i), m);
    }
}

//! Compute the inverse quadratic form \f$x^tC^{-1}x\f$.
//!
//! \param[in] covariance The matrix.
//! \param[in] residual The vector.
//! \param[out] result Filled in with the log likelihood.
//! \param[in] ignoreSingularSubspace If true then we ignore the
//! residual on a singular subspace of m. Otherwise the result is
//! minus infinity in this case.
template<typename T, std::size_t N>
maths_t::EFloatingPointErrorStatus inverseQuadraticForm(const CSymmetricMatrixNxN<T, N> &covariance,
                                                        const CVectorNx1<T, N> &residual,
                                                        double &result,
                                                        bool ignoreSingularSubspace = true)
{
    return linear_algebra_tools_detail::inverseQuadraticProduct(N, covariance, residual,
                                                                result, ignoreSingularSubspace);
}

//! Compute the log-likelihood for the residual \p x and covariance
//! matrix \p m.
//!
//! \param[in] covariance The matrix.
//! \param[in] residual The vector.
//! \param[out] result Filled in with the log likelihood.
//! \param[in] ignoreSingularSubspace If true then we ignore the
//! residual on a singular subspace of m. Otherwise the result is
//! minus infinity in this case.
template<typename T, std::size_t N>
maths_t::EFloatingPointErrorStatus gaussianLogLikelihood(const CSymmetricMatrixNxN<T, N> &covariance,
                                                         const CVectorNx1<T, N> &residual,
                                                         double &result,
                                                         bool ignoreSingularSubspace = true)
{
    return linear_algebra_tools_detail::gaussianLogLikelihood(N, covariance, residual,
                                                              result, ignoreSingularSubspace);
}

//! Sample from a Gaussian with \p mean and \p covariance in such
//! a way as to preserve the mean, covariance matrix and some of
//! the quantiles of the generalised c.d.f.
//!
//! \param[in] n The desired number of samples.
//! \param[in] mean The mean of the Gaussian.
//! \param[in] covariance The covariance matrix of the Gaussian.
//! \param[out] result Filled in with the samples.
template<typename T, typename U, std::size_t N>
void sampleGaussian(std::size_t n,
                    const CVectorNx1<T, N> &mean,
                    const CSymmetricMatrixNxN<T, N> &covariance,
                    std::vector<CVectorNx1<U, N> > &result)
{
    return linear_algebra_tools_detail::sampleGaussian(n, mean, covariance, result);
}

//! Compute the log-determinant of the symmetric matrix \p m.
//!
//! \param[in] matrix The matrix.
//! \param[in] ignoreSingularSubspace If true then we ignore any
//! singular subspace of m. Otherwise, the result is minus infinity.
template<typename T, std::size_t N>
maths_t::EFloatingPointErrorStatus logDeterminant(const CSymmetricMatrixNxN<T, N> &matrix,
                                                  double &result,
                                                  bool ignoreSingularSubspace = true)
{
    return linear_algebra_tools_detail::logDeterminant(N, matrix, result, ignoreSingularSubspace);
}


//! Compute the inverse quadratic form \f$x^tC^{-1}x\f$.
//!
//! \param[in] covariance The matrix.
//! \param[in] residual The vector.
//! \param[out] result Filled in with the log likelihood.
//! \param[in] ignoreSingularSubspace If true then we ignore the
//! residual on a singular subspace of m. Otherwise the result is
//! minus infinity in this case.
template<typename T>
maths_t::EFloatingPointErrorStatus inverseQuadraticForm(const CSymmetricMatrix<T> &covariance,
                                                        const CVector<T> &residual,
                                                        double &result,
                                                        bool ignoreSingularSubspace = true)
{
    return linear_algebra_tools_detail::inverseQuadraticProduct(covariance.rows(),
                                                                covariance, residual,
                                                                result, ignoreSingularSubspace);
}

//! Compute the log-likelihood for the residual \p x and covariance
//! matrix \p m.
//!
//! \param[in] covariance The covariance matrix.
//! \param[in] residual The residual, i.e. x - mean.
//! \param[out] result Filled in with the log likelihood.
//! \param[in] ignoreSingularSubspace If true then we ignore the
//! residual on a singular subspace of m. Otherwise the result is
//! minus infinity in this case.
template<typename T>
maths_t::EFloatingPointErrorStatus gaussianLogLikelihood(const CSymmetricMatrix<T> &covariance,
                                                         const CVector<T> &residual,
                                                         double &result,
                                                         bool ignoreSingularSubspace = true)
{
    return linear_algebra_tools_detail::gaussianLogLikelihood(covariance.rows(),
                                                              covariance, residual,
                                                              result, ignoreSingularSubspace);
}

//! Sample from a Gaussian with \p mean and \p covariance in such
//! a way as to preserve the mean, covariance matrix and some of
//! the quantiles of the generalised c.d.f.
//!
//! \param[in] n The desired number of samples.
//! \param[in] mean The mean of the Gaussian.
//! \param[in] covariance The covariance matrix of the Gaussian.
//! \param[out] result Filled in with the samples.
template<typename T, typename U>
void sampleGaussian(std::size_t n,
                    const CVector<T> &mean,
                    const CSymmetricMatrix<T> &covariance,
                    std::vector<CVector<U> > &result)
{
    return linear_algebra_tools_detail::sampleGaussian(n, mean, covariance, result);
}

//! Compute the log-determinant of the symmetric matrix \p m.
//!
//! \param[in] matrix The matrix.
//! \param[in] ignoreSingularSubspace If true then we ignore any
//! singular subspace of m. Otherwise, the result is minus infinity.
template<typename T>
maths_t::EFloatingPointErrorStatus logDeterminant(const CSymmetricMatrix<T> &matrix,
                                                  double &result,
                                                  bool ignoreSingularSubspace = true)
{
    return linear_algebra_tools_detail::logDeterminant(matrix.rows(), matrix, result, ignoreSingularSubspace);
}

//! Project the matrix on to \p subspace.
template<typename MATRIX>
inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
    projectedMatrix(const std::vector<std::size_t> &subspace, const MATRIX &matrix)
{
    std::size_t d = subspace.size();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> result(d, d);
    for (std::size_t i = 0u; i < d; ++i)
    {
        for (std::size_t j = 0u; j < d; ++j)
        {
            result(i,j) = matrix(subspace[i], subspace[j]);
        }
    }
    return result;
}

//! Project the vector on to \p subspace.
template<typename VECTOR>
inline Eigen::Matrix<double, Eigen::Dynamic, 1>
    projectedVector(const std::vector<std::size_t> &subspace, const VECTOR &vector)
{
    std::size_t d = subspace.size();
    Eigen::Matrix<double, Eigen::Dynamic, 1> result(d);
    for (std::size_t i = 0u; i < d; ++i)
    {
        result(i) = vector(subspace[i]);
    }
    return result;
}

}
}

#endif // INCLUDED_ml_maths_CLinearAlgebraTools_h
