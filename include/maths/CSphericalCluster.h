/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CSphericalCluster_h
#define INCLUDED_ml_maths_CSphericalCluster_h

#include <maths/CAnnotatedVector.h>
#include <maths/CBasicStatisticsCovariances.h>
#include <maths/CLinearAlgebra.h>
#include <maths/COrderings.h>
#include <maths/ImportExport.h>

#include <boost/functional/hash.hpp>

#include <cmath>

namespace ml
{
namespace maths
{

//! \brief A cluster's count and variance.
struct MATHS_EXPORT SCountAndVariance
{
    SCountAndVariance(double count = 0.0,
                      double variance = 0.0) :
            s_Count(count),
            s_Variance(variance)
    {}

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
class CSphericalCluster
{
    public:
        using Type = CAnnotatedVector<POINT, SCountAndVariance>;

        //! \brief Hashes a spherical cluster.
        class CHash
        {
            public:
                std::size_t operator()(const Type &o) const
                {
                    std::size_t seed = boost::hash_combine(m_PointHash(o), o.annotation().s_Count);
                    return boost::hash_combine(seed, o.annotation().s_Variance);
                }

            private:
                typename POINT::CHash m_PointHash;
        };

        //! \brief Compares two spherical clusters for equality.
        class CEqual
        {
            public:
                std::size_t operator()(const Type &lhs, const Type &rhs) const
                {
                    return   static_cast<const POINT &>(lhs) == static_cast<const POINT &>(rhs)
                          && lhs.annotation().s_Count == rhs.annotation().s_Count
                          && lhs.annotation().s_Variance == rhs.annotation().s_Variance;
                }
        };

        //! \brief A total ordering of spherical clusters.
        struct SLess
        {
            bool operator()(const Type &lhs, const Type &rhs) const
            {
                return COrderings::lexicographical_compare(static_cast<const POINT &>(lhs),
                                                           lhs.annotation().s_Count,
                                                           lhs.annotation().s_Variance,
                                                           static_cast<const POINT &>(rhs),
                                                           rhs.annotation().s_Count,
                                                           rhs.annotation().s_Variance);
            }
        };
};

namespace basic_statistics_detail
{

//! \brief Specialization for the implementation of the spherical
//! cluster to the sample mean and variance estimator.
template<typename POINT>
struct SCentralMomentsCustomAdd<CAnnotatedVector<POINT, SCountAndVariance>>
{
    template<typename OTHER_POINT>
    static inline void add(const CAnnotatedVector<POINT, SCountAndVariance> &x,
                           typename SCoordinate<OTHER_POINT>::Type n,
                           CBasicStatistics::SSampleCentralMoments<OTHER_POINT, 1> &moments)
    {
        using TCoordinate = typename SCoordinate<OTHER_POINT>::Type;
        moments.add(x, TCoordinate(x.annotation().s_Count) * n, 0);
    }

    template<typename OTHER_POINT>
    static inline void add(const CAnnotatedVector<POINT, SCountAndVariance> &x,
                           typename SCoordinate<OTHER_POINT>::Type n,
                           CBasicStatistics::SSampleCentralMoments<OTHER_POINT, 2> &moments)
    {
        using TCoordinate = typename SCoordinate<OTHER_POINT>::Type;
        n *= TCoordinate(x.annotation().s_Count);
        OTHER_POINT m(x);
        TCoordinate v(x.annotation().s_Variance);
        moments += CBasicStatistics::momentsAccumulator(n, m, las::constant(m, v));
    }
};

//! \brief Specialization for the implementation of add spherical
//! cluster to the covariances estimator.
template<typename POINT>
struct SCovariancesCustomAdd<CAnnotatedVector<POINT, SCountAndVariance>>
{
    template<typename OTHER_POINT>
    static inline void add(const CAnnotatedVector<POINT, SCountAndVariance> &x,
                           CAnnotatedVector<POINT, SCountAndVariance> n,
                           CBasicStatistics::SSampleCovariances<OTHER_POINT> &covariances)
    {
        using TCoordinate = typename SCoordinate<OTHER_POINT>::Type;
        n *= TCoordinate(x.annotation().s_Count);
        OTHER_POINT diag{las::constant(x, x.annotation().s_Variance)};
        covariances += CBasicStatistics::SSampleCovariances<OTHER_POINT>(n, x, diag.asDiagonal());
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
template<typename POINT>
struct SCovariancesLedoitWolf<CAnnotatedVector<POINT, SCountAndVariance>>
{
    template<typename OTHER_POINT>
    static void estimate(const std::vector<CAnnotatedVector<POINT, SCountAndVariance>> &points,
                         CBasicStatistics::SSampleCovariances<OTHER_POINT> &covariances)
    {
        if (points.empty())
        {
            return;
        }

        using TCoordinate = typename SCoordinate<OTHER_POINT>::Type;
        using TMatrix = typename SConformableMatrix<OTHER_POINT>::Type;

        std::size_t dimension{las::dimension(points[0])};
        TCoordinate d{static_cast<TCoordinate>(dimension)};

        TCoordinate n{CBasicStatistics::count(covariances)};
        const OTHER_POINT &m{CBasicStatistics::mean(covariances)};
        const TMatrix &s{CBasicStatistics::maximumLikelihoodCovariances(covariances)};

        TCoordinate mn{s.trace() / d};
        TCoordinate norm(0);
        for (std::size_t i = 0u; i < dimension; ++i)
        {
            norm += pow2(s(i,i) - mn);
            for (std::size_t j = 0u; j < i; ++j)
            {
                norm += TCoordinate(2) * pow2(s(i,j));
            }
        }
        TCoordinate dn{norm / d};
        TCoordinate bn(0);
        TCoordinate z{n * n};
        for (const auto &point : points)
        {
            TCoordinate ni{static_cast<TCoordinate>(point.annotation().s_Count)};
            TCoordinate vi{static_cast<TCoordinate>(point.annotation().s_Variance)};
            norm = TCoordinate(0);
            for (std::size_t i = 0u; i < dimension; ++i)
            {
                norm += pow2(pow2(TCoordinate(point(i)) - m(i)) + vi - s(i,i));
                for (std::size_t j = 0u; j < i; ++j)
                {
                    norm += TCoordinate(2) * pow2(  (TCoordinate(point(i)) - m(i))
                                                  * (TCoordinate(point(j)) - m(j)) - s(i,j));
                }
            }
            bn += ni * norm / d / z;
        }
        bn = std::min(bn, dn);
        LOG_TRACE("m = " << mn << ", d = " << dn << ", b = " << bn);

        covariances.s_Covariances *= std::max((TCoordinate(1) - bn / dn), 0.0);
        for (std::size_t i = 0u; i < dimension; ++i)
        {
            covariances.s_Covariances(i,i) += bn / dn * mn;
        }
    }

    template<typename U> static U pow2(U x) { return x * x; }
};

}

//! Write a description of \p cluster for debugging.
template<typename POINT>
std::ostream &operator<<(std::ostream &o,
                         const CAnnotatedVector<POINT, SCountAndVariance> &cluster)
{
    return o << static_cast<const POINT&>(cluster)
             << " (" << cluster.annotation().s_Count
             << "," << std::sqrt(cluster.annotation().s_Variance) << ")";
}

}
}

#endif // INCLUDED_ml_maths_CSphericalCluster_h
