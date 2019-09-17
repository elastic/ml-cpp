/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLeastSquaresOnlineRegressionDetail_h
#define INCLUDED_ml_maths_CLeastSquaresOnlineRegressionDetail_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsCovariances.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CTypeTraits.h>

#include <sstream>

namespace ml {
namespace maths {

template<std::size_t N, typename T>
bool CLeastSquaresOnlineRegression<N, T>::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(STATISTIC_TAG, m_S.fromDelimited(traverser.value()))
    } while (traverser.next());

    return true;
}

template<std::size_t N, typename T>
void CLeastSquaresOnlineRegression<N, T>::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(STATISTIC_TAG, m_S.toDelimited());
}

template<std::size_t N, typename T>
void CLeastSquaresOnlineRegression<N, T>::shiftAbscissa(double dx) {
    if (CBasicStatistics::count(m_S) == 0.0) {
        return;
    }

    // The update scheme is as follows:
    //
    // 1/n sum_i{ (t(i) + dx)^i }
    //   -> 1/n * (  sum_i{ t(i)^i }
    //             + sum_j{ (i j) * dx^(i - j) * sum_i{ t(i)^j } } )
    //
    // 1/n sum_i{ (t(i) + dx)^i * y(i) }
    //   -> 1/n * (  sum_i{ t(i)^i * y(i) }
    //             + sum_j{ (i j) * dx^(i - j) * sum_i{ t(i)^j y(i) } } )

    double d[2 * N - 2] = {dx};
    for (std::size_t i = 1u; i < 2 * N - 2; ++i) {
        d[i] = d[i - 1] * dx;
    }
    LOG_TRACE(<< "d = " << core::CContainerPrinter::print(d));

    LOG_TRACE(<< "S(before) " << CBasicStatistics::mean(m_S));
    for (std::size_t i = 2 * N - 2; i > 0; --i) {
        LOG_TRACE(<< "i = " << i);
        for (std::size_t j = 0; j < i; ++j) {
            double bij = CCategoricalTools::binomialCoefficient(i, j) * d[i - j - 1];
            LOG_TRACE(<< "bij = " << bij);
            CBasicStatistics::moment<0>(m_S)(i) += bij * CBasicStatistics::mean(m_S)(j);
            if (i >= N) {
                continue;
            }
            std::size_t yi = i + 2 * N - 1;
            std::size_t yj = j + 2 * N - 1;
            LOG_TRACE(<< "yi = " << yi << ", yj = " << yj);
            CBasicStatistics::moment<0>(m_S)(yi) += bij * CBasicStatistics::mean(m_S)(yj);
        }
    }
    LOG_TRACE(<< "S(after) = " << CBasicStatistics::mean(m_S));
}

template<std::size_t N, typename T>
bool CLeastSquaresOnlineRegression<N, T>::parameters(TArray& result, double maxCondition) const {
    return this->parameters(N, result, maxCondition);
}

template<std::size_t N, typename T>
bool CLeastSquaresOnlineRegression<N, T>::parameters(std::size_t n,
                                                     TArray& result,
                                                     double maxCondition) const {
    result.fill(0.0);

    // Search for non-singular solution.
    n += 1;
    while (--n > 0) {
        switch (n) {
        case 1: {
            result[0] = CBasicStatistics::mean(m_S)(2 * N - 1);
            return true;
        }
        case N: {
            constexpr int N_{static_cast<int>(N)};
            Eigen::Matrix<double, N_, N_> x;
            Eigen::Matrix<double, N_, 1> y;
            if (this->parameters(N, x, y, maxCondition, result)) {
                return true;
            }
            break;
        }
        default: {
            CDenseMatrix<double> x(n, n);
            CDenseVector<double> y(n);
            if (this->parameters(n, x, y, maxCondition, result)) {
                return true;
            }
            break;
        }
        }
    }
    return false;
}

template<std::size_t N, typename T>
bool CLeastSquaresOnlineRegression<N, T>::covariances(double variance,
                                                      TMatrix& result,
                                                      double maxCondition) const {
    return this->covariances(N, variance, result, maxCondition);
}

template<std::size_t N, typename T>
bool CLeastSquaresOnlineRegression<N, T>::covariances(std::size_t n,
                                                      double variance,
                                                      TMatrix& result,
                                                      double maxCondition) const {
    result = TMatrix(0.0);

    // Search for the covariance matrix of a non-singular subproblem.
    n += 1;
    while (--n > 0) {
        switch (n) {
        case 1: {
            result(0, 0) = variance / CBasicStatistics::count(m_S);
            return true;
        }
        case N: {
            constexpr int N_{static_cast<int>(N)};
            Eigen::Matrix<double, N_, N_> x;
            if (!this->covariances(N, x, variance, maxCondition, result)) {
                continue;
            }
            break;
        }
        default: {
            CDenseMatrix<double> x(n, n);
            if (!this->covariances(n, x, variance, maxCondition, result)) {
                continue;
            }
            break;
        }
        }
        return true;
    }
    return false;
}

template<std::size_t N, typename T>
std::string CLeastSquaresOnlineRegression<N, T>::print() const {
    TArray params;
    if (this->parameters(params)) {
        std::string result;
        for (std::size_t i = params.size() - 1; i > 0; --i) {
            result += core::CStringUtils::typeToStringPretty(params[i]) +
                      " x^" + core::CStringUtils::typeToStringPretty(i) + " + ";
        }
        result += core::CStringUtils::typeToStringPretty(params[0]);
        return result;
    }
    return std::string("bad");
}

template<std::size_t N, typename T>
template<typename MATRIX, typename VECTOR>
bool CLeastSquaresOnlineRegression<N, T>::parameters(std::size_t n,
                                                     MATRIX& x,
                                                     VECTOR& y,
                                                     double maxCondition,
                                                     TArray& result) const {
    if (n == 1) {
        result[0] = CBasicStatistics::mean(m_S)(2 * N - 1);
        return true;
    }

    this->gramian(n, x);
    for (std::size_t i = 0; i < n; ++i) {
        y(i) = CBasicStatistics::mean(m_S)(i + 2 * N - 1);
    }
    LOG_TRACE(<< "S = " << CBasicStatistics::mean(m_S));
    LOG_TRACE(<< "x =\n" << x);
    LOG_TRACE(<< "y =\n" << y);

    typename SJacobiSvd<MATRIX>::Type svd(x.template selfadjointView<Eigen::Upper>(),
                                          Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.singularValues()(0) > maxCondition * svd.singularValues()(n - 1)) {
        LOG_TRACE(<< "singular values = " << svd.singularValues());
        return false;
    }

    // Don't bother checking the solution since we check the matrix
    // condition above.
    VECTOR r = svd.solve(y);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = r(i);
    }

    return true;
}

template<std::size_t N, typename T>
template<typename MATRIX>
bool CLeastSquaresOnlineRegression<N, T>::covariances(std::size_t n,
                                                      MATRIX& x,
                                                      double variance,
                                                      double maxCondition,
                                                      TMatrix& result) const {
    if (n == 1) {
        x(0) = variance / CBasicStatistics::count(m_S);
        return true;
    }

    this->gramian(n, x);
    typename SJacobiSvd<MATRIX>::Type svd(x.template selfadjointView<Eigen::Upper>(),
                                          Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.singularValues()(0) > maxCondition * svd.singularValues()(n - 1)) {
        LOG_TRACE(<< "singular values = " << svd.singularValues());
        return false;
    }

    // Don't bother checking for division by zero since we check
    // the matrix condition above. Also, we zero initialize result
    // in the calling code so any values we don't fill in the
    // following loop are zero (as required).
    x = (svd.matrixV() * svd.singularValues().cwiseInverse().asDiagonal() *
         svd.matrixU().transpose()) *
        variance / CBasicStatistics::count(m_S);
    for (std::size_t i = 0; i < n; ++i) {
        result(i, i) = x(i, i);
        for (std::size_t j = 0; j < i; ++j) {
            result(i, j) = x(i, j);
        }
    }

    return true;
}
}
}

#endif // INCLUDED_ml_maths_CLeastSquaresOnlineRegressionDetail_h
