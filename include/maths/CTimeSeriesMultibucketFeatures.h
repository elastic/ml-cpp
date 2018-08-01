/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesMultibucketFeatures_h
#define INCLUDED_ml_maths_CTimeSeriesMultibucketFeatures_h

#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CTypeConversions.h>
#include <maths/MathsTypes.h>

#include <numeric>
#include <utility>

namespace ml {
namespace maths {

//! \brief Defines features on collections of time series values.
//!
//! DESCRIPTION:\n
//! The intention of these is to provide useful features for performing
//! anomaly detection. Specifically, unusual values of certain properties
//! of extended time ranges are often the most interesting events in a
//! time series from a user's perspective.
class CTimeSeriesMultibucketFeatures {
public:
    template<typename T>
    using TT1VecTWeightAry1VecPr =
        std::pair<core::CSmallVector<T, 1>, core::CSmallVector<maths_t::TWeightsAry<T>, 1>>;

    //! Get the mean of a collection of time series values.
    //!
    //! \tparam VECTOR This can be a floating point type or vector type.
    //! \tparam ITR It is assumed that the value type of ITR is a pair
    //! whose second type is a CBasicStatistics::SSampleCentralMoments
    //! object.
    template<typename VECTOR, typename ITR>
    static TT1VecTWeightAry1VecPr<VECTOR> mean(ITR begin, ITR end) {
        if (begin != end) {
            VECTOR value(toVector<VECTOR>(rangeMean(begin, end, 0.9)));
            VECTOR weight(toVector<VECTOR>(rangeCount(begin, end, 0.9)));
            return {{value}, {maths_t::countWeight(weight)}};
        }
        return {{}, {}};
    }

private:
    using TDoubleDoublePr = std::pair<double, double>;

    //! \name Traits for the mean and count calculation.
    template<typename ITR>
    struct STraits {
        using TValueType = typename std::iterator_traits<ITR>::value_type;
        using TSecondType = typename SSecondType<TValueType>::Type;
        using TPrecise = typename SPromoted<typename TSecondType::TValue>::Type;
        using TMeanAccumulator = typename CBasicStatistics::SSampleMean<TPrecise>::TAccumulator;
    };

    //! Univariate implementation returns \p value.
    template<typename T>
    static double conformable(const T& /*x*/, double value) {
        return value;
    }
    //! Multivariate implementation returns the \p value scalar multiple
    //! of the one vector which is conformable with \p x.
    template<typename T>
    static CVector<double> conformable(const CVector<T>& x, double value) {
        return CVector<double>(x.dimension(), value);
    }

    //! Univariate implementation returns 1.
    template<typename T>
    static std::size_t dimension(const T& /*x*/) {
        return 1;
    }
    //! Multivariate implementation returns the dimension of \p x.
    template<typename T>
    static std::size_t dimension(const CVector<T>& x) {
        return x.dimension();
    }

    //! Univariate implementation returns \p x.
    template<typename T>
    static double toVector(const T& x) {
        return x;
    }
    //! Multivariate implementation returns \p x as the VECTOR type.
    template<typename VECTOR, typename T>
    static VECTOR toVector(const CVector<T>& x) {
        return x.template toVector<VECTOR>();
    }

    //! Get mean count of the values in [\p begin, \p end).
    template<typename ITR>
    static typename STraits<ITR>::TPrecise
    rangeCount(ITR begin, ITR end, double factor = 1.0) {
        double latest, earliest;
        std::tie(earliest, latest) = range(begin, end);
        double n{static_cast<double>(std::distance(begin, end))};
        double scale{(n - 1.0) * (latest == earliest ? 1.0 : 1.0 / (latest - earliest))};

        auto zero = conformable(CBasicStatistics::mean(begin->second), 0.0);
        typename STraits<ITR>::TMeanAccumulator count{zero};
        for (double last{earliest}; begin != end; ++begin) {
            double dt{static_cast<double>(begin->first) - last};
            last = static_cast<double>(begin->first);
            count.age(std::pow(factor, scale * dt));
            count.add(conformable(CBasicStatistics::mean(begin->second),
                                  CBasicStatistics::count(begin->second)),
                      CBasicStatistics::count(begin->second));
        }

        return CBasicStatistics::mean(count);
    }

    //! Get mean of the values in [\p begin, \p end).
    template<typename ITR>
    static typename STraits<ITR>::TPrecise
    rangeMean(ITR begin, ITR end, double factor = 1.0) {
        double latest, earliest;
        std::tie(earliest, latest) = range(begin, end);
        double n{static_cast<double>(std::distance(begin, end))};
        double scale{(n - 1.0) * (latest == earliest ? 1.0 : 1.0 / (latest - earliest))};

        auto zero = conformable(CBasicStatistics::mean(begin->second), 0.0);
        typename STraits<ITR>::TMeanAccumulator mean{zero};
        for (double last{earliest}; begin != end; ++begin) {
            double dt{static_cast<double>(begin->first) - last};
            last = static_cast<double>(begin->first);
            mean.age(std::pow(factor, scale * dt));
            mean.add(CBasicStatistics::mean(begin->second),
                     CBasicStatistics::count(begin->second));
        }

        return CBasicStatistics::mean(mean);
    }

    //! Compute the time range of [\p begin, \p end).
    template<typename ITR>
    static TDoubleDoublePr range(ITR begin, ITR end) {
        auto range =
            std::accumulate(begin, end, CBasicStatistics::CMinMax<double>(),
                            [](CBasicStatistics::CMinMax<double> partial,
                               const typename STraits<ITR>::TValueType& value) {
                                partial.add(static_cast<double>(value.first));
                                return partial;
                            });
        return {range.min(), range.max()};
    }
};
}
}

#endif // INCLUDED_ml_maths_CTimeSeriesMultibucketFeatures_h
