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

#include <maths/time_series/CTimeSeriesSmoothing.h>

namespace ml {
namespace maths {
namespace time_series {

CTimeSeriesSmoothing::CTimeSeriesSmoothing(core_t::TTime smoothingInterval)
    : m_SmoothingInterval(smoothingInterval) {
}

const core_t::TTime& CTimeSeriesSmoothing::smoothingInterval() const {
    return m_SmoothingInterval;
}

const core_t::TTime CTimeSeriesSmoothing::SMOOTHING_INTERVAL{14400};

}
}
}

