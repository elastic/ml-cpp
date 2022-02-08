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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesMultibucketFeaturesFwd_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesMultibucketFeaturesFwd_h

#include <core/CoreTypes.h>

#include <maths/common/CLinearAlgebraFwd.h>
#include <maths/common/MathsTypes.h>

#include <functional>

namespace ml {
namespace core {
template<typename, std::size_t>
class CSmallVector;
}
namespace maths {
namespace time_series {
template<typename>
class CTimeSeriesMultibucketFeature;
using CTimeSeriesMultibucketScalarFeature = CTimeSeriesMultibucketFeature<double>;
using CTimeSeriesMultibucketVectorFeature =
    CTimeSeriesMultibucketFeature<core::CSmallVector<double, 10>>;
}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesMultibucketFeaturesFwd_h
