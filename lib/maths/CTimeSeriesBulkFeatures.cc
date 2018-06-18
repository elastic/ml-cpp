/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesBulkFeatures.h>

#include <maths/CBasicStatistics.h>

#include <boost/bind.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <numeric>

namespace ml {
namespace maths {

CTimeSeriesBulkFeatures::TDouble1VecDoubleWeightsArray1VecPr
CTimeSeriesBulkFeatures::mean(TTimeFloatMeanAccumulatorPrCBufCItr begin,
                              TTimeFloatMeanAccumulatorPrCBufCItr end) {
    if (begin == end) {
        return {{}, {}};
    }

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    auto count = [](TMeanAccumulator partial, const TTimeFloatMeanAccumulatorPr& value) {
        partial.add(CBasicStatistics::count(value.second));
        return partial;
    };
    auto mean = [](TMeanAccumulator partial, const TTimeFloatMeanAccumulatorPr& value) {
        partial.add(CBasicStatistics::mean(value.second));
        return partial;
    };

    return {{CBasicStatistics::mean(std::accumulate(begin, end, TMeanAccumulator{}, mean))},
            {maths_t::countWeight(CBasicStatistics::mean(
                std::accumulate(begin, end, TMeanAccumulator{}, count)))}};
}

CTimeSeriesBulkFeatures::TDouble1VecDoubleWeightsArray1VecPr
CTimeSeriesBulkFeatures::contrast(TTimeFloatMeanAccumulatorPrCBufCItr begin,
                                  TTimeFloatMeanAccumulatorPrCBufCItr end) {
    std::ptrdiff_t zero{0};
    std::ptrdiff_t N{std::distance(begin, end)};
    std::ptrdiff_t m{N / 2};
    if (m < 5) {
        return {{}, {}};
    }

    using TDoublePtrDiffPr = std::pair<double, std::ptrdiff_t>;
    using TMinMaxAccumulator = CBasicStatistics::CMinMax<TDoublePtrDiffPr>;

    auto minmax = [begin](TMinMaxAccumulator partial, std::ptrdiff_t index) {
        partial.add({CBasicStatistics::mean((begin + index)->second), index});
        return partial;
    };

    TMinMaxAccumulator support;
    TMinMaxAccumulator lseed{std::accumulate(boost::make_counting_iterator(zero),
                                             boost::make_counting_iterator(m - 4),
                                             TMinMaxAccumulator{}, minmax)};
    TMinMaxAccumulator rseed{std::accumulate(boost::make_counting_iterator(m + 4),
                                             boost::make_counting_iterator(N),
                                             TMinMaxAccumulator{}, minmax)};
    for (std::ptrdiff_t split : {m - 3, m - 2, m - 1, m, m + 1, m + 2, m + 3}) {
        auto l = std::accumulate(boost::make_counting_iterator(m - 4),
                                 boost::make_counting_iterator(split - 1), lseed, minmax);
        auto r = std::accumulate(boost::make_counting_iterator(split + 1),
                                 boost::make_counting_iterator(m + 4), rseed, minmax);
        if (r.max().first < l.min().first) {
            double lweight{CBasicStatistics::count((begin + l.min().second)->second)};
            double rweight{CBasicStatistics::count((begin + r.max().second)->second)};
            std::ptrdiff_t index{lweight > rweight ? l.min().second : r.max().second};
            support.add({r.max().first - l.min().first, index});
        } else if (r.min().first > l.max().first) {
            double lweight{CBasicStatistics::count((begin + l.max().second)->second)};
            double rweight{CBasicStatistics::count((begin + r.min().second)->second)};
            std::ptrdiff_t index{lweight > rweight ? l.max().second : r.min().second};
            support.add({r.min().first - l.max().first, index});
        }
    }

    if (support.initialized()) {
        if (-support.min().first > support.max().first) {
            double weight{CBasicStatistics::count((begin + support.min().second)->second)};
            return {{support.min().first}, {maths_t::countWeight(weight)}};
        } else {
            double weight{CBasicStatistics::count((begin + support.max().second)->second)};
            return {{support.max().first}, {maths_t::countWeight(weight)}};
        }
    }
    return {{}, {}};
}
}
}
