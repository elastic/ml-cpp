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
#include <maths/MathsTypes.h>

#include <boost/circular_buffer.hpp>

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
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TDouble1VecDoubleWeightsArray1VecPr =
        std::pair<TDouble1Vec, maths_t::TDoubleWeightsAry1Vec>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TTimeFloatMeanAccumulatorPr = std::pair<core_t::TTime, TFloatMeanAccumulator>;
    using TTimeFloatMeanAccumulatorPrCBuf = boost::circular_buffer<TTimeFloatMeanAccumulatorPr>;
    using TTimeFloatMeanAccumulatorPrCBufCItr = TTimeFloatMeanAccumulatorPrCBuf::const_iterator;

public:
    //! The mean of a collection of time series values.
    static TDouble1VecDoubleWeightsArray1VecPr
    mean(TTimeFloatMeanAccumulatorPrCBufCItr begin, TTimeFloatMeanAccumulatorPrCBufCItr end);

    //! The contrast between two sets in a binary partition of a collection
    //! of time series values.
    static TDouble1VecDoubleWeightsArray1VecPr
    contrast(TTimeFloatMeanAccumulatorPrCBufCItr begin,
             TTimeFloatMeanAccumulatorPrCBufCItr end);
};
}
}

#endif // INCLUDED_ml_maths_CTimeSeriesBulkFeatures_h
