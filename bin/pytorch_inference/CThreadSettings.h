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

#ifndef INCLUDED_ml_torch_CThreadSettings_h
#define INCLUDED_ml_torch_CThreadSettings_h

#include <cstdint>

namespace ml {
namespace torch {

//! \brief
//! Holder for the thread settings used in inference.
//!
class CThreadSettings {
public:
public:
    explicit CThreadSettings(std::int32_t maxThreads,
                             std::int32_t numThreadsPerAllocation,
                             std::int32_t numAllocations);

    std::int32_t numThreadsPerAllocation() const;
    std::int32_t numAllocations() const;

    //! Set the number of allocations, but constraining the value given the
    //! current values of max threads and threads per allocation. The way the
    //! number of allocations is constrained is different to how it's done in
    //! validateThreadingParameters(), because we cannot change threads per
    //! allocation after it's initially set.
    void numAllocations(std::int32_t numAllocations);

    //! Validate the max threads, number of threads per allocation and number
    //! of allocations to ensure that CPU is not overcommitted, as this leads
    //! to worse performance than running with fewer threads. This method
    //! prefers to constrain threads per allocation instead of number of
    //! allocations.
    static void validateThreadingParameters(std::int32_t& maxThreads,
                                            std::int32_t& numThreadsPerAllocation,
                                            std::int32_t& numAllocations);

private:
    std::int32_t m_MaxThreads;
    std::int32_t m_NumThreadsPerAllocation;
    std::int32_t m_NumAllocations;
};
}
}

#endif // INCLUDED_ml_torch_CThreadSettings_h
