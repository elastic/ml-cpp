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

#include <maths/CInformationCriteria.h>

#include <maths/Constants.h>

#include <boost/math/distributions/chi_squared.hpp>

namespace ml
{
namespace maths
{
namespace information_criteria_detail
{
namespace
{

//! The implementation of log determinant used for the Gaussian
//! information criterion.
template<typename MATRIX>
double logDeterminant_(const MATRIX &covariance, double upper)
{
    Eigen::JacobiSVD<MATRIX> svd(covariance);
    double result = 0.0;
    double epsilon = svd.threshold() * svd.singularValues()(0);
    for (int i = 0u; i < svd.singularValues().size(); ++i)
    {
        result += ::log(std::max(upper * svd.singularValues()(i), epsilon));
    }
    return result;
}

const double VARIANCE_CONFIDENCE = 0.99;

}

double confidence(double df)
{
    boost::math::chi_squared_distribution<> chi(df);
    return boost::math::quantile(chi, VARIANCE_CONFIDENCE) / df;
}

#define LOG_DETERMINANT(N)                                                   \
double logDeterminant(const CSymmetricMatrixNxN<double, N> &c, double upper) \
{                                                                            \
    return logDeterminant_(toDenseMatrix(c), upper);                         \
}
LOG_DETERMINANT(2)
LOG_DETERMINANT(3)
LOG_DETERMINANT(4)
LOG_DETERMINANT(5)
#undef LOG_DETERMINANT

double logDeterminant(const CSymmetricMatrix<double> &c, double upper)
{
    return logDeterminant_(toDenseMatrix(c), upper);
}

double logDeterminant(const CDenseMatrix<double> &c, double upper)
{
    return logDeterminant_(c, upper);
}

}
}
}
