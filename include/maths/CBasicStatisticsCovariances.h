/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBasicStatisticsDetail_h
#define INCLUDED_ml_maths_CBasicStatisticsDetail_h

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/CLinearAlgebraTools.h>

#include <cstddef>

namespace ml {
namespace maths {
namespace basic_statistics_detail {
//! \brief Default custom add function for points to the covariance
//! estimator.
template<typename POINT>
struct SCovariancesCustomAdd {
    template<typename OTHER_POINT>
    static inline void add(const POINT& x,
                           const POINT& n,
                           CBasicStatistics::SSampleCovariances<OTHER_POINT>& covariances) {
        covariances.add(x, n, 0);
    }
};

//! \brief Default implementation of a covariance matrix shrinkage estimator.
//!
//! DESCRIPTION:\n
//! This uses a scaled identity shrinkage which is estimated using
//! Ledoit and Wolf's approach.
//!
//! See http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf
//! for the details.
template<typename POINT>
struct SCovariancesLedoitWolf {
    template<typename OTHER_POINT>
    static void estimate(const std::vector<POINT>& points,
                         CBasicStatistics::SSampleCovariances<OTHER_POINT>& covariances) {
        if (points.empty()) {
            return;
        }

        using TCoordinate = typename SCoordinate<OTHER_POINT>::Type;
        using TMatrix = typename SConformableMatrix<OTHER_POINT>::Type;

        std::size_t dimension{las::dimension(points[0])};
        TCoordinate d{static_cast<TCoordinate>(dimension)};

        TCoordinate n{CBasicStatistics::count(covariances)};
        const OTHER_POINT& m{CBasicStatistics::mean(covariances)};
        const TMatrix& s{CBasicStatistics::maximumLikelihoodCovariances(covariances)};

        TCoordinate mn{s.trace() / d};
        TCoordinate norm(0);
        for (std::size_t i = 0u; i < dimension; ++i) {
            norm += pow2(s(i, i) - mn);
            for (std::size_t j = 0u; j < i; ++j) {
                norm += TCoordinate(2) * pow2(s(i, j));
            }
        }
        TCoordinate dn{norm / d};
        TCoordinate bn(0);
        TCoordinate z{n * n};
        for (const auto& point : points) {
            norm = TCoordinate(0);
            for (std::size_t i = 0u; i < dimension; ++i) {
                norm += pow2(pow2(TCoordinate(point(i)) - m(i)) - s(i, i));
                for (std::size_t j = 0u; j < i; ++j) {
                    norm += TCoordinate(2) * pow2((TCoordinate(point(i)) - m(i)) *
                                                      (TCoordinate(point(j)) - m(j)) -
                                                  s(i, j));
                }
            }
            bn += norm / d / z;
        }
        bn = std::min(bn, dn);
        LOG_TRACE("m = " << mn << ", d = " << dn << ", b = " << bn);

        covariances.s_Covariances *= std::max((TCoordinate(1) - bn / dn), 0.0);
        for (std::size_t i = 0u; i < dimension; ++i) {
            covariances.s_Covariances(i, i) += bn / dn * mn;
        }
    }

    template<typename MATRIX, typename T>
    static MATRIX minusDiagonal(std::size_t dimension, MATRIX m, T diagonal) {
        for (std::size_t i = 0u; i < dimension; ++i) {
            m(i, i) -= diagonal;
        }
        return m;
    }

    template<typename T>
    static T pow2(T x) {
        return x * x;
    }
};
}

template<typename POINT>
CBasicStatistics::SSampleCovariances<POINT>::SSampleCovariances(std::size_t dimension)
    : s_Count(SConstant<TVector>::get(dimension, 0)),
      s_Mean(SConstant<TVector>::get(dimension, 0)),
      s_Covariances(SConstant<TMatrix>::get(dimension, 0)) {
}

template<typename POINT>
template<typename OTHER_POINT>
void CBasicStatistics::SSampleCovariances<POINT>::add(const OTHER_POINT& x) {
    this->add(x, las::ones(x));
}

template<typename POINT>
template<typename OTHER_POINT>
void CBasicStatistics::SSampleCovariances<POINT>::add(const OTHER_POINT& x,
                                                      const OTHER_POINT& n) {
    basic_statistics_detail::SCovariancesCustomAdd<OTHER_POINT>::add(x, n, *this);
}

template<typename POINT>
void CBasicStatistics::SSampleCovariances<POINT>::add(const TVector& x, const TVector& n, int) {
    if (!las::isZero(n)) {
        s_Count += n;

        // Note we don't trap the case alpha is less than epsilon,
        // because then we'd have to compute epsilon and it is very
        // unlikely the count will get big enough.
        TVector alpha{las::componentwise(n) / las::componentwise(s_Count)};
        TVector beta{las::ones(alpha) - alpha};

        TVector mean{s_Mean};
        s_Mean = las::componentwise(beta) * las::componentwise(mean) +
                 las::componentwise(alpha) * las::componentwise(x);

        TVector r{x - s_Mean};
        TMatrix r2{las::outer(r)};
        TVector dMean{mean - s_Mean};
        TMatrix dMean2{las::outer(dMean)};

        s_Covariances += dMean2;
        scaleCovariances(beta, s_Covariances);
        scaleCovariances(alpha, r2);
        s_Covariances += r2;
    }
}

//! Combine two moments. This is equivalent to running
//! a single accumulator on the entire collection.
template<typename POINT>
template<typename OTHER_POINT>
const CBasicStatistics::SSampleCovariances<POINT>&
CBasicStatistics::SSampleCovariances<POINT>::
operator+=(const SSampleCovariances<OTHER_POINT>& rhs) {
    s_Count = s_Count + rhs.s_Count;
    if (!las::isZero(s_Count)) {
        // Note we don't trap the case alpha is less than epsilon,
        // because then we'd have to compute epsilon and it is very
        // unlikely the count will get big enough.
        TVector alpha{las::componentwise(rhs.s_Count) / las::componentwise(s_Count)};
        TVector beta{las::ones(alpha) - alpha};

        TVector meanLhs{s_Mean};

        s_Mean = las::componentwise(beta) * las::componentwise(meanLhs) +
                 las::componentwise(alpha) * las::componentwise(rhs.s_Mean);

        TVector dMeanLhs{meanLhs - s_Mean};
        TMatrix dMean2Lhs{las::outer(dMeanLhs)};
        TVector dMeanRhs{rhs.s_Mean - s_Mean};
        TMatrix dMean2Rhs{las::outer(dMeanRhs)};

        s_Covariances += dMean2Lhs;
        scaleCovariances(beta, s_Covariances);
        dMean2Rhs += rhs.s_Covariances;
        scaleCovariances(alpha, dMean2Rhs);
        s_Covariances += dMean2Rhs;
    }

    return *this;
}

template<typename POINT>
template<typename OTHER_POINT>
const CBasicStatistics::SSampleCovariances<POINT>&
CBasicStatistics::SSampleCovariances<POINT>::
operator-=(const SSampleCovariances<OTHER_POINT>& rhs) {
    using TCoordinate = typename SCoordinate<POINT>::Type;

    s_Count = las::max(s_Count - rhs.s_Count, las::zero(s_Count));
    if (!las::isZero(s_Count)) {
        // Note we don't trap the case alpha is less than epsilon,
        // because then we'd have to compute epsilon and it is very
        // unlikely the count will get big enough.
        TVector alpha{las::componentwise(rhs.s_Count) / las::componentwise(s_Count)};
        TVector beta{las::ones(alpha) + alpha};

        TVector meanLhs(s_Mean);

        s_Mean = las::componentwise(beta) * las::componentwise(meanLhs) -
                 las::componentwise(alpha) * las::componentwise(rhs.s_Mean);

        TVector dMeanLhs{s_Mean - meanLhs};
        TMatrix dMean2Lhs{las::outer(dMeanLhs)};
        TVector dMeanRhs{rhs.s_Mean - meanLhs};
        TMatrix dMean2Rhs{las::outer(dMeanRhs)};

        s_Covariances = s_Covariances - dMean2Lhs;
        scaleCovariances(beta, s_Covariances);
        dMean2Rhs += rhs.s_Covariances - dMean2Lhs;
        scaleCovariances(alpha, dMean2Rhs);
        s_Covariances -= dMean2Rhs;

        // If any of the diagonal elements are negative round them
        // up to zero and zero the corresponding row and column.
        for (std::size_t i = 0u, dimension = las::dimension(s_Mean); i < dimension; ++i) {
            if (s_Covariances(i, i) < TCoordinate{0}) {
                for (std::size_t j = 0u; j < dimension; ++j) {
                    s_Covariances(i, j) = s_Covariances(j, i) = TCoordinate{0};
                }
            }
        }
    } else {
        s_Mean = las::zero(s_Mean);
        s_Covariances = las::zero(s_Covariances);
    }

    return *this;
}

template<typename POINT>
typename SCoordinate<POINT>::Type
CBasicStatistics::count(const SSampleCovariances<POINT>& accumulator) {
    using TCoordinate = typename SCoordinate<POINT>::Type;
    return las::L1(accumulator.s_Count) /
           static_cast<TCoordinate>(las::dimension(accumulator.s_Count));
}

template<typename POINT>
typename SConformableMatrix<POINT>::Type
CBasicStatistics::covariances(const SSampleCovariances<POINT>& accumulator) {
    using TCoordinate = typename SCoordinate<POINT>::Type;
    using TMatrix = typename SConformableMatrix<POINT>::Type;

    POINT bias(accumulator.s_Count);
    for (std::size_t i = 0u; i < las::dimension(bias); ++i) {
        if (bias(i) > TCoordinate{1}) {
            bias(i) /= bias(i) - TCoordinate(1);
        } else {
            bias(i) = TCoordinate{0};
        }
    }

    TMatrix result{accumulator.s_Covariances};
    scaleCovariances(bias, result);

    return result;
}

template<typename POINT, typename OTHER_POINT>
void CBasicStatistics::covariancesLedoitWolf(const std::vector<POINT>& points,
                                             SSampleCovariances<OTHER_POINT>& result) {
    result.add(points);
    basic_statistics_detail::SCovariancesLedoitWolf<POINT>::estimate(points, result);
}
}
}

#endif // INCLUDED_ml_maths_CBasicStatisticsDetail_h
