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
#include "CThreadSettings.h"

#include <core/CLogger.h>

#include <algorithm>
#include <thread>

namespace ml {
namespace torch {

CThreadSettings::CThreadSettings(std::int32_t maxThreads,
                                 std::int32_t numThreadsPerAllocation,
                                 std::int32_t numAllocations)
    : m_MaxThreads{maxThreads}, m_NumThreadsPerAllocation{numThreadsPerAllocation},
      m_NumAllocations{numAllocations} {
    validateThreadingParameters(m_MaxThreads, m_NumThreadsPerAllocation, m_NumAllocations);
}

std::int32_t CThreadSettings::numThreadsPerAllocation() const {
    return m_NumThreadsPerAllocation;
}

std::int32_t CThreadSettings::numAllocations() const {
    return m_NumAllocations;
}

void CThreadSettings::numAllocations(std::int32_t numAllocations) {
    std::int32_t maxAllocations{(m_MaxThreads + m_NumThreadsPerAllocation - 1) /
                                m_NumThreadsPerAllocation};
    if (numAllocations < 1) {
        LOG_WARN(<< "Setting number of allocations to minimum value of 1; value was "
                 << numAllocations);
        numAllocations = 1;
    } else if (numAllocations > maxAllocations) {
        LOG_WARN(<< "Setting number of allocations to maximum value of "
                 << maxAllocations << " (given " << m_NumThreadsPerAllocation
                 << " threads per allocation); value was " << numAllocations);
        numAllocations = maxAllocations;
    }
    m_NumAllocations = numAllocations;
}

void CThreadSettings::validateThreadingParameters(std::int32_t& maxThreads,
                                                  std::int32_t& numThreadsPerAllocation,
                                                  std::int32_t& numAllocations) {
    if (maxThreads < 1) {
        LOG_WARN(<< "Could not determine hardware concurrency; setting max threads to 1");
        maxThreads = 1;
    }

    if (numAllocations < 1) {
        LOG_WARN(<< "Setting number of allocations to minimum value of 1; value was "
                 << numAllocations);
        numAllocations = 1;
    } else if (numAllocations > maxThreads) {
        LOG_WARN(<< "Setting number of allocations to maximum value of "
                 << maxThreads << "; value was " << numAllocations);
        numAllocations = maxThreads;
    }

    // Max threads per allocation would ideally fit within the available
    // concurrency when multiplied by the number of allocations, but we allow
    // rounding up if there isn't a perfect fit.
    std::int32_t maxThreadsPerAllocation{(maxThreads + numAllocations - 1) / numAllocations};
    if (numThreadsPerAllocation < 1) {
        LOG_WARN(<< "Setting number of threads per allocation to minimum value of 1; value was "
                 << numThreadsPerAllocation);
        numThreadsPerAllocation = 1;
    } else if (numThreadsPerAllocation > maxThreadsPerAllocation) {
        LOG_WARN(<< "Setting number of threads per allocation to maximum value of "
                 << maxThreadsPerAllocation << " (given number of allocations "
                 << numAllocations << "); value was " << numThreadsPerAllocation);
        numThreadsPerAllocation = maxThreadsPerAllocation;
    }
}
}
}
