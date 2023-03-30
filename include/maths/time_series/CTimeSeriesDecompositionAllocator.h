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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionAllocator_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionAllocator_h

#include <maths/time_series/ImportExport.h>

namespace ml {
namespace maths {
namespace time_series {

class MATHS_TIME_SERIES_EXPORT CTimeSeriesDecompositionAllocator {
public:
    virtual ~CTimeSeriesDecompositionAllocator() = default;

    //! Check if we can still allocate any components.
    virtual bool areAllocationsAllowed() const = 0;
};

class MATHS_TIME_SERIES_EXPORT CTimeSeriesDecompositionAllocatorStub
    : public CTimeSeriesDecompositionAllocator {
public:
    bool areAllocationsAllowed() const override { return true; }
};
}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionAllocator_h
