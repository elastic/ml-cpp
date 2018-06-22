/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesBulkFeatures_h
#define INCLUDED_ml_maths_CTimeSeriesBulkFeatures_h

#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CTypeConversions.h>
#include <maths/MathsTypes.h>

#include <boost/bind.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <numeric>
#include <utility>

namespace ml {
namespace maths {

//! \brief Defines some of bulk properties of a collection of time
//! series values.
//!
//! DESCRIPTION:\n
//! The intention of these is to provide useful features for performing
//! anomaly detection. Specifically, unusual values of bulk properties
//! are expected to be indicative of interesting events in time series.
class CTimeSeriesBulkFeatures {
public:
    //! The mean of a collection of time series values.
    //!
    //! \tparam VECTOR This can be a floating point type or vector type.
    //! \tparam ITR It is assumed that the value type of ITR is a pair
    //! whose second type is a CBasicStatistics::SSampleCentralMoments
    //! object.
    template<typename VECTOR, typename ITR>
    static std::pair<core::CSmallVector<VECTOR, 1>, core::CSmallVector<maths_t::TWeightsAry<VECTOR>, 1>>
    mean(ITR begin, ITR end) {
        if (begin == end) {
            using TValueType = typename std::iterator_traits<ITR>::value_type;
            using TSecondType = typename SSecondType<TValueType>::Type;
            using TPrecise = typename SPromoted<typename TSecondType::TValue>::Type;
            using TMeanAccumulator = typename CBasicStatistics::SSampleMean<TPrecise>::TAccumulator;

            auto count = [](TMeanAccumulator partial, const TValueType& value) {
                partial.add(conformable(CBasicStatistics::mean(value.second),
                                        CBasicStatistics::count(value.second)));
                partial.age(0.9);
                return partial;
            };
            auto mean = [](TMeanAccumulator partial, const TValueType& value) {
                partial.add(CBasicStatistics::mean(value.second));
                partial.age(0.9);
                return partial;
            };

            auto zero = conformable(CBasicStatistics::mean(begin->second), 0.0);
            auto value = toVector<VECTOR>(CBasicStatistics::mean(
                std::accumulate(begin, end, TMeanAccumulator{zero}, mean)));
            auto weight = toVector<VECTOR>(CBasicStatistics::mean(
                std::accumulate(begin, end, TMeanAccumulator{zero}, count)));

            return {{value}, {maths_t::countWeight(weight)}};
        }
        return {{}, {}};
    }

    //! The contrast between two sets in a binary partition of a collection
    //! of univariate time series values.
    //!
    //! This is a signed quantity which is the minimum difference between
    //! approximately equal partitions, \f$L\f$ and \f$R\f$, of the range
    //! [\p begin, \p end). Specifically,
    //!   \f$max(min_{x\in R}{x}-max_{x\in L}{x}, 0) + min(max_{x\in R}{x}-min_{x\in L}{x}, 0)\f$
    //!
    //! \note It is assumed that the value type of ITR is a pair whose
    //! second type is a CBasicStatistics::SSampleCentralMoments object.
    //!
    //! \tparam ITR It is assumed that the value type of ITR is a pair
    //! whose second type is a CBasicStatistics::SSampleCentralMoments
    //! object.
    template<typename ITR>
    static std::pair<core::CSmallVector<double, 1>, maths_t::TDoubleWeightsAry1Vec>
    contrast(ITR begin, ITR end) {
        std::ptrdiff_t N{std::distance(begin, end)};
        std::ptrdiff_t m{N / 2};
        if (m >= 5) {
            using TDoubleDoublePr = std::pair<double, double>;
            using TMinAccumulator = CBasicStatistics::SMin<TDoubleDoublePr>::TAccumulator;
            using TMaxAccumulator = CBasicStatistics::SMax<TDoubleDoublePr>::TAccumulator;
            using TDoublePtrDiffPr = std::pair<double, std::ptrdiff_t>;
            using TMinMaxAccumulator = CBasicStatistics::CMinMax<TDoublePtrDiffPr>;

            auto minmax = [begin](TMinMaxAccumulator partial, std::ptrdiff_t index) {
                partial.add({CBasicStatistics::mean((begin + index)->second), index});
                return partial;
            };

            TMinAccumulator minContrast;
            TMaxAccumulator maxContrast;
            std::ptrdiff_t zero{0};
            TMinMaxAccumulator lseed{std::accumulate(
                boost::make_counting_iterator(zero),
                boost::make_counting_iterator(m - 4), TMinMaxAccumulator{}, minmax)};
            TMinMaxAccumulator rseed{std::accumulate(
                boost::make_counting_iterator(m + 4),
                boost::make_counting_iterator(N), TMinMaxAccumulator{}, minmax)};
            for (std::ptrdiff_t split : {m - 3, m - 2, m - 1, m, m + 1, m + 2, m + 3}) {
                auto l = std::accumulate(boost::make_counting_iterator(m - 4),
                                         boost::make_counting_iterator(split - 1),
                                         lseed, minmax);
                auto r = std::accumulate(boost::make_counting_iterator(split + 1),
                                         boost::make_counting_iterator(m + 4),
                                         rseed, minmax);
                if (r.max().first < l.min().first) {
                    double weight{std::sqrt(
                        CBasicStatistics::count((begin + l.min().second)->second) *
                        CBasicStatistics::count((begin + r.max().second)->second))};
                    minContrast.add({r.max().first - l.min().first, -weight / 2.0});
                    maxContrast.add({r.max().first - l.min().first, +weight / 2.0});
                } else if (r.min().first > l.max().first) {
                    double weight{std::sqrt(
                        CBasicStatistics::count((begin + l.max().second)->second) *
                        CBasicStatistics::count((begin + r.min().second)->second))};
                    minContrast.add({r.min().first - l.max().first, -weight / 2.0});
                    maxContrast.add({r.min().first - l.max().first, +weight / 2.0});
                }
            }

            if (minContrast.count() > 0 && maxContrast.count() > 0) {
                if (-minContrast[0].first > maxContrast[0].first) {
                    double weight{-minContrast[0].second};
                    return {{minContrast[0].first}, {maths_t::countWeight(weight)}};
                } else {
                    double weight{maxContrast[0].second};
                    return {{maxContrast[0].first}, {maths_t::countWeight(weight)}};
                }
            }
        }
        return {{}, {}};
    }

private:
    //! Univariate implementation returns zero.
    template<typename T>
    static double conformable(const T& /*x*/, double value) {
        return value;
    }
    //! Multivariate implementation returns zero vector conformable with \p x.
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

    //! Univariate implementation returns a vector containing \p x.
    template<typename T>
    static double toVector(const T& x) {
        return x;
    }
    //! Multivariate implementation returns a vector containing \p x.
    template<typename VECTOR, typename T>
    static VECTOR toVector(const CVector<T>& x) {
        return x.template toVector<VECTOR>();
    }
};
}
}

#endif // INCLUDED_ml_maths_CTimeSeriesBulkFeatures_h
