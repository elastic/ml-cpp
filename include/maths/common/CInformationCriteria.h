/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_maths_common_CInformationCriteria_h
#define INCLUDED_ml_maths_common_CInformationCriteria_h

#include <core/Constants.h>

#include <maths/common/CBasicStatisticsCovariances.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CLinearAlgebraShims.h>
#include <maths/common/CSphericalCluster.h>
#include <maths/common/ImportExport.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ml {
namespace maths {
namespace common {
namespace information_criteria_detail {

//! The confidence interval we use when computing the singular values
//! of the covariance matrix. This is high to stop x-means creating
//! clusters with small numbers of points where there is chance that
//! the evidence from the covariance matrix is a fluke.
MATHS_COMMON_EXPORT
double confidence(double df);

#define LOG_DETERMINANT(N)                                                     \
    MATHS_COMMON_EXPORT                                                        \
    double logDeterminant(const CSymmetricMatrixNxN<double, N>& c, double upper)
LOG_DETERMINANT(2);
LOG_DETERMINANT(3);
LOG_DETERMINANT(4);
LOG_DETERMINANT(5);
#undef LOG_DETERMINANT

//! The log determinant of our internal heap symmetric matrix.
MATHS_COMMON_EXPORT
double logDeterminant(const CSymmetricMatrix<double>& c, double upper);

//! The log determinant of an Eigen matrix.
MATHS_COMMON_EXPORT
double logDeterminant(const CDenseMatrix<double>& c, double upper);

} // information_criteria_detail::

//! Enumeration of different types of information criterion supported.
enum EInfoCriterionType { E_AICc, E_BIC };

//! \brief Computes the information content of a collection of point
//! clouds under the assumption that they are distributed as a weighted
//! sum of spherically symmetric Gaussians.
//!
//! DESCRIPTION:\n
//! This can calculate two types of information criterion values: Akaike
//! and Bayes. The difference is in the treatment of the penalty on the
//! maximum log likelihood due to the number of parameters. In particular,
//! BIC is defined as:
//! <pre class="fragment">
//!   \f$BIC = -2 log(L(x)) - k log(n)\f$
//! </pre>
//!
//! Here, \f$L(x)\f$ is the maximum likelihood of the data, \f$k\f$ is
//! the number of free parameters in the distribution and \f$n\f$ is the
//! number of data points. The AICc is defined as:
//! <pre class="fragment">
//!   \f$AIC_c = -2 log(L(x)) + 2 k + \frac{k(k+1)}{n - k - 1}\f$
//! </pre>
//!
//! Here the finite sample correction applies to data that are normally
//! distributed. The maximum likelihood as a function of the input data
//! can be found by maximizing the log likelihood w.r.t. the free
//! parameters, in this case the mean vector and the single variance
//! parameter (from the spherical covariance assumption). Specifically,
//! the maximum likelihood mean is
//! <pre class="fragment">
//!   \f$m = \frac{1}{n}\sum_{i=1}{n}{x_i}\f$
//! </pre>
//! and the maximum likelihood variance is
//! <pre class="fragment">
//!   \f$v = \sum_{i=1}{n}{ (x_i - m)^t (x_i - m) }\f$
//! </pre>
//!
//! See also http://en.wikipedia.org/wiki/Bayesian_information_criterion
//! and http://en.wikipedia.org/wiki/Akaike_information_criterion.
template<typename POINT, EInfoCriterionType TYPE>
class CSphericalGaussianInfoCriterion {
public:
    using TPointVec = std::vector<POINT>;
    using TPointVecVec = std::vector<TPointVec>;
    using TBarePoint = typename SUnannotated<POINT>::Type;
    using TBarePointPrecise = typename SFloatingPoint<TBarePoint, double>::Type;
    using TCoordinate = typename SCoordinate<TBarePointPrecise>::Type;
    using TMeanVarAccumulator =
        typename CBasicStatistics::SSampleMeanVar<TBarePointPrecise>::TAccumulator;

public:
    CSphericalGaussianInfoCriterion() = default;
    explicit CSphericalGaussianInfoCriterion(const TPointVecVec& x) {
        this->add(x);
    }
    explicit CSphericalGaussianInfoCriterion(const TPointVec& x) {
        this->add(x);
    }

    //! Update the sufficient statistics for computing info content.
    void add(const TPointVecVec& x) {
        for (const auto& xi : x) {
            this->add(xi);
        }
    }

    //! Update the sufficient statistics for computing info content.
    void add(const TPointVec& x) {
        if (x.size() > 0) {
            TMeanVarAccumulator moments(las::zero(x[0]));
            moments.add(x);
            this->add(moments);
        }
    }

    //! Update the sufficient statistics for computing info content.
    void add(const TMeanVarAccumulator& moments) {
        double ni = CBasicStatistics::count(moments);
        const TBarePointPrecise& mean{CBasicStatistics::mean(moments)};
        const TBarePointPrecise& covarianceDiag{
            CBasicStatistics::maximumLikelihoodVariance(moments)};
        std::size_t d{las::dimension(covarianceDiag)};
        double vi{0.0};
        for (std::size_t i = 0; i < d; ++i) {
            vi += covarianceDiag(i);
        }
        vi = std::max(vi, EPS * las::norm(mean));

        m_D = static_cast<double>(d);
        m_K += 1.0;
        m_N += ni;
        if (ni > 1.0) {
            double upper = information_criteria_detail::confidence(ni - 1.0);
            m_Likelihood += ni * log(ni) - 0.5 * m_D * ni *
                                               (1.0 + core::constants::LOG_TWO_PI +
                                                std::log(upper * vi / m_D));
        } else {
            m_Likelihood += ni * log(ni) - 0.5 * m_D * ni *
                                               (1.0 + core::constants::LOG_TWO_PI +
                                                core::constants::LOG_MAX_DOUBLE);
        }
    }

    //! Calculate the information content of the clusters added so far.
    double calculate(double p = 0.0) const {
        if (m_N != 0.0) {
            double logN{std::log(m_N)};
            p += m_D * m_K + 2.0 * m_K - 1.0;
            switch (TYPE) {
            case E_BIC:
                return -2.0 * (m_Likelihood - m_N * logN) + p * logN;
            case E_AICc:
                return -2.0 * (m_Likelihood - m_N * logN) + 2.0 * p +
                       p * (p + 1.0) / (m_N - p - 1.0);
            }
        }
        return 0.0;
    }

private:
    static constexpr double EPS{10.0 * std::numeric_limits<TCoordinate>::epsilon()};

private:
    //! The point dimension.
    double m_D = 0.0;
    //! The number of clusters.
    double m_K = 0.0;
    //! The number of points.
    double m_N = 0.0;
    //! The data likelihood for the k spherically symmetric Gaussians.
    double m_Likelihood = 0.0;
};

//! \brief Computes the information content of a collection of point
//! clouds under the assumption that they are distributed as a weighted
//! sum of Gaussians.
//!
//! DESCRIPTION:\n
//! This places no restriction on the covariance matrix in particular
//! it is assumed to have \f$frac{D(D+1)}{2}\f$ parameters. For more
//! details on the information criteria see CSphericalGaussianInfoCriterion.
template<typename POINT, EInfoCriterionType TYPE>
class CGaussianInfoCriterion {
public:
    using TPointVec = std::vector<POINT>;
    using TPointVecVec = std::vector<TPointVec>;
    using TBarePoint = typename SUnannotated<POINT>::Type;
    using TBarePointPrecise = typename SFloatingPoint<TBarePoint, double>::Type;
    using TCoordinate = typename SCoordinate<TBarePointPrecise>::Type;
    using TCovariances = CBasicStatistics::SSampleCovariances<TBarePointPrecise>;

public:
    CGaussianInfoCriterion() = default;
    explicit CGaussianInfoCriterion(const TPointVecVec& x) { this->add(x); }
    explicit CGaussianInfoCriterion(const TPointVec& x) { this->add(x); }

    //! Update the sufficient statistics for computing info content.
    void add(const TPointVecVec& x) {
        for (const auto& xi : x) {
            this->add(xi);
        }
    }

    //! Update the sufficient statistics for computing info content.
    void add(const TPointVec& x) {
        if (x.size() > 0) {
            TCovariances covariances(las::dimension(x[0]));
            covariances.add(x);
            this->add(covariances);
        }
    }

    //! Update the sufficient statistics for computing info content.
    void add(const TCovariances& covariance) {
        double ni{CBasicStatistics::count(covariance)};
        m_D = static_cast<double>(las::dimension(CBasicStatistics::mean(covariance)));
        m_K += 1.0;
        m_N += ni;
        m_Likelihood += ni * log(ni) -
                        0.5 * ni *
                            (m_D + m_D * core::constants::LOG_TWO_PI +
                             (ni <= m_D + 1.0 ? core::constants::LOG_MAX_DOUBLE
                                              : this->logDeterminant(covariance)));
    }

    //! Calculate the information content of the clusters added so far.
    double calculate(double p = 0.0) const {
        if (m_N != 0.0) {
            double logN{std::log(m_N)};
            p += m_D * (1.0 + 0.5 * (m_D + 1.0)) * m_K + m_K - 1.0;
            switch (TYPE) {
            case E_BIC:
                return -2.0 * (m_Likelihood - m_N * logN) + p * logN;
            case E_AICc:
                return -2.0 * (m_Likelihood - m_N * logN) + 2.0 * p +
                       p * (p + 1.0) / (m_N - p - 1.0);
            }
        }
        return 0.0;
    }

private:
    //! Compute the log of the determinant of \p covariance.
    double logDeterminant(const TCovariances& covariance) const {
        double n{CBasicStatistics::count(covariance)};
        const auto& c = CBasicStatistics::maximumLikelihoodCovariances(covariance);
        double upper{information_criteria_detail::confidence(n - m_D - 1.0)};
        return information_criteria_detail::logDeterminant(c, upper);
    }

private:
    //! The point dimension.
    double m_D = 0.0;
    //! The number of clusters.
    double m_K = 0.0;
    //! The number of points.
    double m_N = 0.0;
    //! The data likelihood for the k Gaussians.
    double m_Likelihood = 0.0;
};
}
}
}

#endif // INCLUDED_ml_maths_common_CInformationCriteria_h
