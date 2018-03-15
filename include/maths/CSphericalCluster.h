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

#ifndef INCLUDED_ml_maths_CSphericalCluster_h
#define INCLUDED_ml_maths_CSphericalCluster_h

#include <maths/CAnnotatedVector.h>
#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebra.h>
#include <maths/COrderings.h>
#include <maths/ImportExport.h>

#include <boost/functional/hash.hpp>

#include <math.h>

namespace ml {
namespace maths {

//! \brief A cluster's count and variance.
struct MATHS_EXPORT SCountAndVariance {
    SCountAndVariance(double count = 0.0,
                      double variance = 0.0) :
        s_Count(count),
        s_Variance(variance) {}

    //! The count of point in the cluster.
    double s_Count;

    //! The spherically symmetric variance of the cluster.
    double s_Variance;
};

//! \brief Defines the type of a spherical cluster using the specified
//! point type.
//!
//! This annotates the POINT vector corresponding to the centre of the
//! cluster with the count of points in the cluster and the spherically
//! symmetric variance of those points.
template<typename POINT>
class CSphericalCluster {
    public:
        typedef CAnnotatedVector<POINT, SCountAndVariance> Type;

        class CHash {
            public:
                std::size_t operator()(const Type &o) const {
                    std::size_t seed = boost::hash_combine(m_PointHash(o), o.annotation().s_Count);
                    return boost::hash_combine(seed, o.annotation().s_Variance);
                }

            private:
                typename POINT::CHash m_PointHash;
        };

        class CEqual {
            public:
                std::size_t operator()(const Type &lhs, const Type &rhs) const {
                    return static_cast<const POINT &>(lhs) == static_cast<const POINT &>(rhs) &&
                           lhs.annotation().s_Count == rhs.annotation().s_Count &&
                           lhs.annotation().s_Variance == rhs.annotation().s_Variance;
                }
        };

        struct SLess {
            bool operator()(const Type &lhs, const Type &rhs) const {
                return COrderings::lexicographical_compare(static_cast<const POINT &>(lhs),
                                                           lhs.annotation().s_Count,
                                                           lhs.annotation().s_Variance,
                                                           static_cast<const POINT &>(rhs),
                                                           rhs.annotation().s_Count,
                                                           rhs.annotation().s_Variance);
            }
        };
};

namespace basic_statistics_detail {

//! \brief Specialization for the implementation of the spherical
//! cluster to the sample mean and variance estimator.
template<typename U, std::size_t N>
struct SCentralMomentsCustomAdd<CAnnotatedVector<CVectorNx1<U, N>, SCountAndVariance> > {
    template<typename T>
    static inline void add(const CAnnotatedVector<CVectorNx1<U, N>, SCountAndVariance> &x,
                           typename SCoordinate<T>::Type n,
                           CBasicStatistics::SSampleCentralMoments<T, 1> &moments) {
        typedef typename SCoordinate<T>::Type TCoordinate;
        moments.add(x, TCoordinate(x.annotation().s_Count) * n, 0);
    }

    template<typename T>
    static inline void add(const CAnnotatedVector<CVectorNx1<U, N>, SCountAndVariance> &x,
                           typename SCoordinate<T>::Type n,
                           CBasicStatistics::SSampleCentralMoments<T, 2> &moments) {
        typedef typename SCoordinate<T>::Type TCoordinate;
        moments += CBasicStatistics::accumulator(TCoordinate(x.annotation().s_Count) * n,
                                                 T(x),
                                                 T(x.annotation().s_Variance));
    }
};

//! \brief Specialization for the implementation of add spherical
//! cluster to the covariances estimator.
template<typename T, std::size_t N>
struct SCovariancesCustomAdd<CAnnotatedVector<CVectorNx1<T, N>, SCountAndVariance> > {
    template<typename U>
    static inline void add(const CAnnotatedVector<CVectorNx1<T, N>, SCountAndVariance> &x,
                           const CAnnotatedVector<CVectorNx1<T, N>, SCountAndVariance> &n,
                           CBasicStatistics::SSampleCovariances<U, N> &covariances) {
        CSymmetricMatrixNxN<U, N> m(0);
        for (std::size_t i = 0u; i < N; ++i) {
            m(i, i) = x.annotation().s_Variance;
        }
        covariances += CBasicStatistics::SSampleCovariances<U, N>(T(x.annotation().s_Count) * n, x, m);
    }
};

//! \brief Specialization of the implementation of a covariance
//! matrix shrinkage estimator for spherical clusters.
//!
//! DESCRIPTION:\n
//! This uses a scaled identity shrinkage which is estimated using
//! Ledoit and Wolf's approach.
//!
//! See http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf
//! for the details.
template<typename T, std::size_t N>
struct SCovariancesLedoitWolf<CAnnotatedVector<CVectorNx1<T, N>, SCountAndVariance> > {
    template<typename U>
    static void estimate(const std::vector<CAnnotatedVector<CVectorNx1<T, N>, SCountAndVariance> > &points,
                         CBasicStatistics::SSampleCovariances<U, N> &covariances) {
        U d = static_cast<U>(N);

        U                               n = CBasicStatistics::count(covariances);
        const CVectorNx1<U, N>          &         m = CBasicStatistics::mean(covariances);
        const CSymmetricMatrixNxN<U, N> &s = CBasicStatistics::maximumLikelihoodCovariances(covariances);

        double mn = s.trace() / d;
        double dn = pow2((s - CVectorNx1<U, N>(mn).diagonal()).frobenius()) / d;
        double bn = 0.0;
        double z = n * n;
        for (std::size_t i = 0u; i < points.size(); ++i) {
            CVectorNx1<U, N> ci(points[i]);
            U                ni = static_cast<U>(points[i].annotation().s_Count);
            U                vi = static_cast<U>(points[i].annotation().s_Variance);
            bn += ni * pow2(((ci - m).outer() + CVectorNx1<U, N>(vi).diagonal() - s).frobenius()) / d / z;
        }
        bn = std::min(bn, dn);
        LOG_TRACE("m = " << mn << ", d = " << dn << ", b = " << bn);

        covariances.s_Covariances =  CVectorNx1<U, N>(bn / dn * mn).diagonal()
                                    + (U(1) - bn / dn) * covariances.s_Covariances;
    }

    template<typename U> static U pow2(U x) {
        return x * x;
    }
};

}

//! Write a description of \p cluster for debugging.
template<typename POINT>
std::ostream &operator<<(std::ostream &o,
                         const CAnnotatedVector<POINT, SCountAndVariance> &cluster) {
    return o << static_cast<const POINT&>(cluster)
             << " (" << cluster.annotation().s_Count
             << "," << ::sqrt(cluster.annotation().s_Variance) << ")";
}

}
}

#endif // INCLUDED_ml_maths_CSphericalCluster_h
