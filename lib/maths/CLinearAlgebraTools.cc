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

#include <maths/CLinearAlgebraTools.h>

#include <core/Constants.h>

#include <boost/math/distributions/normal.hpp>

namespace ml {
namespace maths {
namespace linear_algebra_tools_detail {

namespace {

//! \brief Shared implementation of the inverse quadratic product.
template<typename EIGENMATRIX, typename EIGENVECTOR>
class CInverseQuadraticProduct {
public:
    template<typename MATRIX, typename VECTOR>
    static maths_t::EFloatingPointErrorStatus
    compute(std::size_t d, const MATRIX& covariance_, const VECTOR& residual, double& result, bool ignoreSingularSubspace) {
        if (residual.isZero()) {
            result = 0.0;
            return maths_t::E_FpNoErrors;
        }

        result = core::constants::LOG_MAX_DOUBLE + 1.0;

        switch (d) {
        case 1:
            if (covariance_(0, 0) == 0.0) {
                return maths_t::E_FpOverflowed;
            }
            result = residual(0) * residual(0) / covariance_(0, 0);
            return maths_t::E_FpNoErrors;

        default: {
            // Note we use Jacobi SVD here so that we handle the case
            // that m is singular to working precision.
            Eigen::JacobiSVD<EIGENMATRIX> covariance(toDenseMatrix(covariance_), Eigen::ComputeFullU | Eigen::ComputeFullV);
            EIGENVECTOR y(toDenseVector(residual));

            // Check the residual is zero on the singular subspace.
            std::size_t rank = static_cast<std::size_t>(covariance.rank());
            if (!ignoreSingularSubspace && rank < d) {
                double normC = (y.transpose() * covariance.matrixU().leftCols(rank)).norm();
                double normS = (y.transpose() * covariance.matrixU().rightCols(d - rank)).norm();
                if (normS > std::numeric_limits<double>::epsilon() * normC) {
                    return maths_t::E_FpOverflowed;
                }
            }
            y = covariance.solve(y);
            result = residual.inner(y);
            return maths_t::E_FpNoErrors;
        }
        }
    }
};

//! \brief Shared implementation of the log-likelihood function.
template<typename EIGENMATRIX, typename EIGENVECTOR>
class CGaussianLogLikelihood {
public:
    template<typename MATRIX, typename VECTOR>
    static maths_t::EFloatingPointErrorStatus
    compute(std::size_t d, const MATRIX& covariance_, const VECTOR& residual, double& result, bool ignoreSingularSubspace) {
        result = core::constants::LOG_MIN_DOUBLE - 1.0;

        switch (d) {
        case 1:
            if (covariance_(0, 0) == 0.0) {
                return maths_t::E_FpOverflowed;
            }
            result = -0.5 * (residual(0) * residual(0) / covariance_(0, 0) + core::constants::LOG_TWO_PI + std::log(covariance_(0, 0)));
            return maths_t::E_FpNoErrors;

        default: {
            // Note we use Jacobi SVD here so that we handle the case
            // that m is singular to working precision.
            Eigen::JacobiSVD<EIGENMATRIX> covariance(toDenseMatrix(covariance_), Eigen::ComputeFullU | Eigen::ComputeFullV);
            EIGENVECTOR y(toDenseVector(residual));

            // Check the residual is zero on the singular subspace.
            std::size_t rank = static_cast<std::size_t>(covariance.rank());
            if (!ignoreSingularSubspace && rank < d) {
                double normC = (y.transpose() * covariance.matrixU().leftCols(rank)).norm();
                double normS = (y.transpose() * covariance.matrixU().rightCols(d - rank)).norm();
                result = normS > std::numeric_limits<double>::epsilon() * normC ? core::constants::LOG_MIN_DOUBLE - 1.0
                                                                                : core::constants::LOG_MAX_DOUBLE + 1.0;
                return maths_t::E_FpOverflowed;
            }
            y = covariance.solve(y);
            double logDeterminant = 0.0;
            for (std::size_t i = 0u; i < rank; ++i) {
                logDeterminant += std::log(covariance.singularValues()(i));
            }
            result = -0.5 * (residual.inner(y) + static_cast<double>(rank) * core::constants::LOG_TWO_PI + logDeterminant);
            return maths_t::E_FpNoErrors;
        }
        }
    }
};

//! \brief Shared implementation of Gaussian sampling.
template<typename EIGENMATRIX>
class CSampleGaussian {
public:
    template<typename MATRIX, typename VECTOR, typename VECTOR_PRECISE>
    static void generate(std::size_t n, const VECTOR& mean_, const MATRIX& covariance_, std::vector<VECTOR_PRECISE>& result) {
        result.clear();
        if (n == 0) {
            return;
        }

        // We sample at the points:
        //  { m + (E_{X_i}[ x I{[x_q, x_{q+1}]} ] * u_i },
        //
        // where m is the mean, X_i is the normal associated with the i'th
        // eigenvector of the covariance matrix, x_q denotes the x value
        // corresponding to the quantile q, q ranges over { k*rank/n } for
        // k in {0, 1, ..., n/rank-1} and u_i are the eigenvectors of the
        // covariance matrix. See the discussion in CNormalMeanPrecConjugate
        // for more discussion on this sampling strategy.

        VECTOR_PRECISE mean(mean_);
        Eigen::JacobiSVD<EIGENMATRIX> covariance(toDenseMatrix(covariance_), Eigen::ComputeFullU | Eigen::ComputeFullV);
        std::size_t rank = static_cast<std::size_t>(covariance.rank());

        std::size_t numberIntervals = n / rank;
        if (numberIntervals == 0) {
            result.push_back(mean);
        } else {
            LOG_TRACE(<< "# intervals = " << numberIntervals);
            result.reserve(rank * numberIntervals);
            double scale = std::sqrt(static_cast<double>(rank));
            LOG_TRACE(<< "scale = " << scale)

            for (std::size_t i = 0u; i < rank; ++i) {
                VECTOR_PRECISE u(fromDenseVector(covariance.matrixU().col(i)));
                try {
                    double variance = covariance.singularValues()(i);
                    boost::math::normal_distribution<> normal(0.0, std::sqrt(variance));
                    LOG_TRACE(<< "[U]_{.i} = " << covariance.matrixU().col(i).transpose())
                    LOG_TRACE(<< "variance = " << variance);
                    LOG_TRACE(<< "u = " << u);

                    double lastPartialExpectation = 0.0;
                    for (std::size_t j = 1u; j < numberIntervals; ++j) {
                        double q = static_cast<double>(j) / static_cast<double>(numberIntervals);
                        double xq = boost::math::quantile(normal, q);
                        double partialExpectation = -variance * CTools::safePdf(normal, xq);
                        double dx = scale * static_cast<double>(numberIntervals) * (partialExpectation - lastPartialExpectation);
                        lastPartialExpectation = partialExpectation;
                        LOG_TRACE(<< "dx = " << dx);
                        result.push_back(mean + dx * u);
                    }
                    double dx = -scale * static_cast<double>(numberIntervals) * lastPartialExpectation;
                    LOG_TRACE(<< "dx = " << dx);
                    result.push_back(mean + dx * u);
                } catch (const std::exception& e) { LOG_ERROR(<< "Failed to sample eigenvector " << u << ": " << e.what()); }
            }
        }
    }
};

//! \brief Shared implementation of the log-determinant function.
template<typename EIGENMATRIX>
class CLogDeterminant {
public:
    template<typename MATRIX>
    static maths_t::EFloatingPointErrorStatus compute(std::size_t d, const MATRIX& m_, double& result, bool ignoreSingularSubspace) {
        result = core::constants::LOG_MIN_DOUBLE - 1.0;

        switch (d) {
        case 1:
            if (m_(0, 0) == 0.0) {
                return maths_t::E_FpOverflowed;
            }
            result = std::log(m_(0, 0));
            return maths_t::E_FpNoErrors;

        default: {
            // Note we use Jacobi SVD here so that we handle the case
            // that m is singular to working precision.
            Eigen::JacobiSVD<EIGENMATRIX> svd(toDenseMatrix(m_));

            // Check the residual is zero on the singular subspace.
            std::size_t rank = static_cast<std::size_t>(svd.rank());
            if (!ignoreSingularSubspace && rank < d) {
                result = static_cast<double>(d - rank) * std::log(svd.threshold() * svd.singularValues()(0));
                return maths_t::E_FpOverflowed;
            }
            result = 0.0;
            for (std::size_t i = 0u; i < rank; ++i) {
                result += std::log(svd.singularValues()(i));
            }
            return maths_t::E_FpNoErrors;
        }
        }
    }
};
}

#define INVERSE_QUADRATIC_PRODUCT(T, N)                                                                                                    \
    maths_t::EFloatingPointErrorStatus inverseQuadraticProduct(std::size_t d,                                                              \
                                                               const CSymmetricMatrixNxN<T, N>& covariance,                                \
                                                               const CVectorNx1<T, N>& residual,                                           \
                                                               double& result,                                                             \
                                                               bool ignoreSingularSubspace) {                                              \
        return CInverseQuadraticProduct<SDenseMatrix<CSymmetricMatrixNxN<T, N>>::Type, SDenseVector<CVectorNx1<T, N>>::Type>::compute(     \
            d, covariance, residual, result, ignoreSingularSubspace);                                                                      \
    }
INVERSE_QUADRATIC_PRODUCT(CFloatStorage, 2)
INVERSE_QUADRATIC_PRODUCT(CFloatStorage, 3)
INVERSE_QUADRATIC_PRODUCT(CFloatStorage, 4)
INVERSE_QUADRATIC_PRODUCT(CFloatStorage, 5)
INVERSE_QUADRATIC_PRODUCT(double, 2)
INVERSE_QUADRATIC_PRODUCT(double, 3)
INVERSE_QUADRATIC_PRODUCT(double, 4)
INVERSE_QUADRATIC_PRODUCT(double, 5)
#undef INVERSE_QUADRATIC_PRODUCT
maths_t::EFloatingPointErrorStatus inverseQuadraticProduct(std::size_t d,
                                                           const CSymmetricMatrix<CFloatStorage>& covariance,
                                                           const CVector<CFloatStorage>& residual,
                                                           double& result,
                                                           bool ignoreSingularSubspace) {
    return CInverseQuadraticProduct<SDenseMatrix<CSymmetricMatrix<CFloatStorage>>::Type,
                                    SDenseVector<CVector<CFloatStorage>>::Type>::compute(d,
                                                                                         covariance,
                                                                                         residual,
                                                                                         result,
                                                                                         ignoreSingularSubspace);
}
maths_t::EFloatingPointErrorStatus inverseQuadraticProduct(std::size_t d,
                                                           const CSymmetricMatrix<double>& covariance,
                                                           const CVector<double>& residual,
                                                           double& result,
                                                           bool ignoreSingularSubspace) {
    return CInverseQuadraticProduct<SDenseMatrix<CSymmetricMatrix<double>>::Type, SDenseVector<CVector<double>>::Type>::compute(
        d, covariance, residual, result, ignoreSingularSubspace);
}

#define GAUSSIAN_LOG_LIKELIHOOD(T, N)                                                                                                      \
    maths_t::EFloatingPointErrorStatus gaussianLogLikelihood(std::size_t d,                                                                \
                                                             const CSymmetricMatrixNxN<T, N>& covariance,                                  \
                                                             const CVectorNx1<T, N>& residual,                                             \
                                                             double& result,                                                               \
                                                             bool ignoreSingularSubspace) {                                                \
        return CGaussianLogLikelihood<SDenseMatrix<CSymmetricMatrixNxN<T, N>>::Type, SDenseVector<CVector<CFloatStorage>>::Type>::compute( \
            d, covariance, residual, result, ignoreSingularSubspace);                                                                      \
    }
GAUSSIAN_LOG_LIKELIHOOD(CFloatStorage, 2)
GAUSSIAN_LOG_LIKELIHOOD(CFloatStorage, 3)
GAUSSIAN_LOG_LIKELIHOOD(CFloatStorage, 4)
GAUSSIAN_LOG_LIKELIHOOD(CFloatStorage, 5)
GAUSSIAN_LOG_LIKELIHOOD(double, 2)
GAUSSIAN_LOG_LIKELIHOOD(double, 3)
GAUSSIAN_LOG_LIKELIHOOD(double, 4)
GAUSSIAN_LOG_LIKELIHOOD(double, 5)
#undef GAUSSIAN_LOG_LIKELIHOOD
maths_t::EFloatingPointErrorStatus gaussianLogLikelihood(std::size_t d,
                                                         const CSymmetricMatrix<CFloatStorage>& covariance,
                                                         const CVector<CFloatStorage>& residual,
                                                         double& result,
                                                         bool ignoreSingularSubspace) {
    return CGaussianLogLikelihood<SDenseMatrix<CSymmetricMatrix<CFloatStorage>>::Type, SDenseVector<CVector<CFloatStorage>>::Type>::compute(
        d, covariance, residual, result, ignoreSingularSubspace);
}
maths_t::EFloatingPointErrorStatus gaussianLogLikelihood(std::size_t d,
                                                         const CSymmetricMatrix<double>& covariance,
                                                         const CVector<double>& residual,
                                                         double& result,
                                                         bool ignoreSingularSubspace) {
    return CGaussianLogLikelihood<SDenseMatrix<CSymmetricMatrix<double>>::Type, SDenseVector<CVector<double>>::Type>::compute(
        d, covariance, residual, result, ignoreSingularSubspace);
}

#define SAMPLE_GAUSSIAN(T, N)                                                                                                              \
    void sampleGaussian(std::size_t d,                                                                                                     \
                        const CVectorNx1<T, N>& mean,                                                                                      \
                        const CSymmetricMatrixNxN<T, N>& covariance,                                                                       \
                        std::vector<CVectorNx1<double, N>>& result) {                                                                      \
        CSampleGaussian<SDenseMatrix<CSymmetricMatrixNxN<T, N>>::Type>::generate(d, mean, covariance, result);                             \
    }
SAMPLE_GAUSSIAN(CFloatStorage, 2)
SAMPLE_GAUSSIAN(CFloatStorage, 3)
SAMPLE_GAUSSIAN(CFloatStorage, 4)
SAMPLE_GAUSSIAN(CFloatStorage, 5)
SAMPLE_GAUSSIAN(double, 2)
SAMPLE_GAUSSIAN(double, 3)
SAMPLE_GAUSSIAN(double, 4)
SAMPLE_GAUSSIAN(double, 5)
#undef SAMPLE_GAUSSIAN
void sampleGaussian(std::size_t d,
                    const CVector<CFloatStorage>& mean,
                    const CSymmetricMatrix<CFloatStorage>& covariance,
                    std::vector<CVector<double>>& result) {
    return CSampleGaussian<SDenseMatrix<CSymmetricMatrix<CFloatStorage>>::Type>::generate(d, mean, covariance, result);
}
void sampleGaussian(std::size_t d,
                    const CVector<double>& mean,
                    const CSymmetricMatrix<double>& covariance,
                    std::vector<CVector<double>>& result) {
    return CSampleGaussian<SDenseMatrix<CSymmetricMatrix<double>>::Type>::generate(d, mean, covariance, result);
}

#define LOG_DETERMINANT(T, N)                                                                                                              \
    maths_t::EFloatingPointErrorStatus logDeterminant(                                                                                     \
        std::size_t d, const CSymmetricMatrixNxN<T, N>& matrix, double& result, bool ignoreSingularSubspace) {                             \
        return CLogDeterminant<SDenseMatrix<CSymmetricMatrixNxN<T, N>>::Type>::compute(d, matrix, result, ignoreSingularSubspace);         \
    }
LOG_DETERMINANT(CFloatStorage, 2)
LOG_DETERMINANT(CFloatStorage, 3)
LOG_DETERMINANT(CFloatStorage, 4)
LOG_DETERMINANT(CFloatStorage, 5)
LOG_DETERMINANT(double, 2)
LOG_DETERMINANT(double, 3)
LOG_DETERMINANT(double, 4)
LOG_DETERMINANT(double, 5)
#undef LOG_DETERMINANT
maths_t::EFloatingPointErrorStatus
logDeterminant(std::size_t d, const CSymmetricMatrix<CFloatStorage>& matrix, double& result, bool ignoreSingularSubspace) {
    return CLogDeterminant<SDenseMatrix<CSymmetricMatrix<CFloatStorage>>::Type>::compute(d, matrix, result, ignoreSingularSubspace);
}
maths_t::EFloatingPointErrorStatus
logDeterminant(std::size_t d, const CSymmetricMatrix<double>& matrix, double& result, bool ignoreSingularSubspace) {
    return CLogDeterminant<SDenseMatrix<CSymmetricMatrix<double>>::Type>::compute(d, matrix, result, ignoreSingularSubspace);
}
}
}
}
