/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CInformationCriteria_h
#define INCLUDED_ml_maths_CInformationCriteria_h

#include <core/Constants.h>

#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CSphericalCluster.h>
#include <maths/ImportExport.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ml
{
namespace maths
{

namespace information_criteria_detail
{

//! \brief Defines the sample covariance accumulator.
template<typename T>
struct SSampleCovariances
{
};

//! \brief Defines the sample covariance accumulator for a CVectorNx1.
template<typename T, std::size_t N>
struct SSampleCovariances<CVectorNx1<T, N>>
{
    using Type = CBasicStatistics::SSampleCovariances<T, N>;
};

//! The confidence interval we use when computing the singular values
//! of the covariance matrix. This is high to stop x-means creating
//! clusters with small numbers of points where there is chance that
//! the evidence from the covariance matrix is a fluke.
MATHS_EXPORT
double confidence(double df);

#define LOG_DETERMINANT(N)                                                    \
MATHS_EXPORT                                                                  \
double logDeterminant(const CSymmetricMatrixNxN<double, N> &c,  double upper)
LOG_DETERMINANT(2);
LOG_DETERMINANT(3);
LOG_DETERMINANT(4);
LOG_DETERMINANT(5);
#undef LOG_DETERMINANT

//! The log determinant of our internal heap symmetric matrix.
double logDeterminant(const CSymmetricMatrix<double> &c, double upper);

//! The log determinant of an Eigen matrix.
double logDeterminant(const CDenseMatrix<double> &c, double upper);

} // information_criteria_detail::

//! Enumeration of different types of information criterion supported.
enum EInfoCriterionType
{
    E_AICc,
    E_BIC
};

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
class CSphericalGaussianInfoCriterion
{
    public:
        using TPointVec = std::vector<POINT>;
        using TPointVecVec = std::vector<TPointVec>;
        using TBarePoint = typename SStripped<POINT>::Type;
        using TBarePointPrecise = typename SFloatingPoint<TBarePoint, double>::Type;
        using TCoordinate = typename SCoordinate<TBarePointPrecise>::Type;
        using TMeanVarAccumulator = typename CBasicStatistics::SSampleMeanVar<TBarePointPrecise>::TAccumulator;

    public:
        CSphericalGaussianInfoCriterion(void) :
            m_D(0.0),
            m_K(0.0),
            m_N(0.0),
            m_Likelihood(0.0)
        {}
        explicit CSphericalGaussianInfoCriterion(const TPointVecVec &x) :
            m_D(0.0),
            m_K(0.0),
            m_N(0.0),
            m_Likelihood(0.0)
        {
            this->add(x);
        }
        explicit CSphericalGaussianInfoCriterion(const TPointVec &x) :
            m_D(0.0),
            m_K(0.0),
            m_N(0.0),
            m_Likelihood(0.0)
        {
            this->add(x);
        }

        //! Update the sufficient statistics for computing info content.
        void add(const TPointVecVec &x)
        {
            for (std::size_t i = 0u; i < x.size(); ++i)
            {
                this->add(x[i]);
            }
        }

        //! Update the sufficient statistics for computing info content.
        void add(const TPointVec &x)
        {
            if (x.empty())
            {
                return;
            }

            TMeanVarAccumulator moments;
            moments.add(x);
            this->add(moments);
        }

        //! Update the sufficient statistics for computing info content.
        void add(const TMeanVarAccumulator &moments)
        {
            double ni = CBasicStatistics::count(moments);
            const TBarePointPrecise &m = CBasicStatistics::mean(moments);
            const TBarePointPrecise &c = CBasicStatistics::maximumLikelihoodVariance(moments);
            std::size_t d = c.dimension();
            double vi = 0.0;
            for (std::size_t i = 0u; i < d; ++i)
            {
                vi += c(i);
            }
            vi = std::max(vi, 10.0 * std::numeric_limits<TCoordinate>::epsilon()
                                   * m.euclidean());

            m_D = static_cast<double>(c.dimension());
            m_K += 1.0;
            m_N += ni;
            if (ni > 1.0)
            {
                double upper = information_criteria_detail::confidence(ni - 1.0);
                m_Likelihood +=   ni * log(ni)
                               - 0.5 * m_D * ni * (  1.0
                                                   + core::constants::LOG_TWO_PI
                                                   + std::log(upper * vi / m_D));
            }
            else
            {
                m_Likelihood +=   ni * log(ni)
                               - 0.5 * m_D * ni * (  1.0
                                                   + core::constants::LOG_TWO_PI
                                                   + core::constants::LOG_MAX_DOUBLE);
            }
        }

        //! Calculate the information content of the clusters added so far.
        double calculate(void) const
        {
            if (m_N == 0.0)
            {
                return 0.0;
            }

            double logN = std::log(m_N);
            double p = (m_D * m_K + 2.0 * m_K - 1.0);
            switch (TYPE)
            {
            case E_BIC:
                return -2.0 * (m_Likelihood - m_N * logN) + p * logN;
            case E_AICc:
                return -2.0 * (m_Likelihood - m_N * logN)
                      + 2.0 * p + p * (p + 1.0) / (m_N - p - 1.0);
            }
            return 0.0;
        }

    private:
        //! The point dimension.
        double m_D;
        //! The number of clusters.
        double m_K;
        //! The number of points.
        double m_N;
        //! The data likelihood for the k spherically symmetric Gaussians.
        double m_Likelihood;
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
class CGaussianInfoCriterion
{
    public:
        using TPointVec = std::vector<POINT>;
        using TPointVecVec = std::vector<TPointVec>;
        using TBarePoint = typename SStripped<POINT>::Type;
        using TBarePointPrecise = typename SFloatingPoint<TBarePoint, double>::Type;
        using TCoordinate = typename SCoordinate<TBarePointPrecise>::Type;
        using TCovariances = typename information_criteria_detail::SSampleCovariances<TBarePointPrecise>::Type;
        using TMatrix = typename SConformableMatrix<TBarePointPrecise>::Type;

    public:
        CGaussianInfoCriterion(void) :
            m_D(0.0),
            m_K(0.0),
            m_N(0.0),
            m_Likelihood(0.0)
        {}
        explicit CGaussianInfoCriterion(const TPointVecVec &x) :
            m_D(0.0),
            m_K(0.0),
            m_N(0.0),
            m_Likelihood(0.0)
        {
            this->add(x);
        }
        explicit CGaussianInfoCriterion(const TPointVec &x) :
            m_D(0.0),
            m_K(0.0),
            m_N(0.0),
            m_Likelihood(0.0)
        {
            this->add(x);
        }

        //! Update the sufficient statistics for computing info content.
        void add(const TPointVecVec &x)
        {
            for (std::size_t i = 0u; i < x.size(); ++i)
            {
                this->add(x[i]);
            }
        }

        //! Update the sufficient statistics for computing info content.
        void add(const TPointVec &x)
        {
            if (x.empty())
            {
                return;
            }

            TCovariances covariances;
            covariances.add(x);
            this->add(covariances);
        }

        //! Update the sufficient statistics for computing info content.
        void add(const TCovariances &covariance)
        {
            double ni = CBasicStatistics::count(covariance);
            m_D = static_cast<double>(CBasicStatistics::mean(covariance).dimension());
            m_K += 1.0;
            m_N += ni;
            m_Likelihood +=   ni * log(ni)
                           - 0.5 * ni * (  m_D
                                         + m_D * core::constants::LOG_TWO_PI
                                         + (ni <= m_D + 1.0 ? core::constants::LOG_MAX_DOUBLE :
                                                              this->logDeterminant(covariance)));
        }

        //! Calculate the information content of the clusters added so far.
        double calculate(void) const
        {
            if (m_N == 0.0)
            {
                return 0.0;
            }

            double logN = std::log(m_N);
            double p = (m_D * (1.0 + 0.5 * (m_D + 1.0)) * m_K + m_K - 1.0);
            switch (TYPE)
            {
            case E_BIC:
                return -2.0 * (m_Likelihood - m_N * logN) + p * logN;
            case E_AICc:
                return -2.0 * (m_Likelihood - m_N * logN)
                      + 2.0 * p + p * (p + 1.0) / (m_N - p - 1.0);
            }
            return 0.0;
        }

    private:
        //! Compute the log of the determinant of \p covariance.
        double logDeterminant(const TCovariances &covariance) const
        {
            double n = CBasicStatistics::count(covariance);
            const TMatrix &c = CBasicStatistics::maximumLikelihoodCovariances(covariance);
            double upper = information_criteria_detail::confidence(n - m_D - 1.0);
            return information_criteria_detail::logDeterminant(c, upper);
        }

    private:
        //! The point dimension.
        double m_D;
        //! The number of clusters.
        double m_K;
        //! The number of points.
        double m_N;
        //! The data likelihood for the k Gaussians.
        double m_Likelihood;
};

}
}

#endif // INCLUDED_ml_maths_CInformationCriteria_h
