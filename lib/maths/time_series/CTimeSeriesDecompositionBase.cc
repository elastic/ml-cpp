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

#include <maths/time_series/CTimeSeriesDecompositionBase.h>

#include <maths/common/CRestoreParams.h>

namespace ml {
namespace maths {
namespace time_series {

CTimeSeriesDecompositionBase::CTimeSeriesDecompositionBase(double /*decayRate*/,
                                                           core_t::TTime bucketLength)
    : m_BucketLength{bucketLength} {
}

CTimeSeriesDecompositionBase::CTimeSeriesDecompositionBase(
    const common::STimeSeriesDecompositionRestoreParams& /*params*/,
    core::CStateRestoreTraverser& /*traverser*/)
    : m_BucketLength{0} {
    // Note: This is just the base class constructor.
    // Derived classes will handle the actual restoration.
}

core_t::TTime CTimeSeriesDecompositionBase::bucketLength() const {
    return m_BucketLength;
}

void CTimeSeriesDecompositionBase::bucketLength(core_t::TTime bucketLength) {
    m_BucketLength = bucketLength;
}

}
}
}
