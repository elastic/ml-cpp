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

CThreadSettings::CThreadSettings(std::int32_t inferenceThreads, std::int32_t numAllocations)
    : m_InferenceThreads(inferenceThreads), m_NumAllocations(numAllocations) {
}

std::int32_t CThreadSettings::inferenceThreads() const {
    return m_InferenceThreads;
}

std::int32_t CThreadSettings::numAllocations() const {
    return m_NumAllocations;
}

void CThreadSettings::numAllocations(std::int32_t numAllocations) {
    m_NumAllocations = numAllocations;
}

void CThreadSettings::validateThreadingParameters(std::int32_t maxThreads,
                                                  std::int32_t& inferenceThreads,
                                                  std::int32_t& numAllocations) {
    if (maxThreads == 0) {
        LOG_WARN(<< "Could not determine hardware concurrency; setting max threads to 1");
        maxThreads = 1;
    }
    if (inferenceThreads < 1) {
        LOG_WARN(<< "Setting inference threads to minimum value of 1; value was "
                 << inferenceThreads);
        inferenceThreads = 1;
    } else if (inferenceThreads >= maxThreads) {
        // leave one allocation thread
        LOG_WARN(<< "Setting inference threads to maximum value of "
                 << std::max(1, maxThreads - 1) << "; value was " << inferenceThreads);
        inferenceThreads = std::max(1, maxThreads - 1);
    }
    if (numAllocations < 1) {
        LOG_WARN(<< "Setting number of allocations to minimum value of 1; value was "
                 << numAllocations);
        numAllocations = 1;
    } else if (numAllocations >= maxThreads) {
        // leave one thread for inference
        LOG_WARN(<< "Setting number of allocations to maximum value of "
                 << std::max(1, maxThreads - 1) << "; value was " << numAllocations);
        numAllocations = std::max(1, maxThreads - 1);
    }

    if (numAllocations + inferenceThreads > maxThreads) {
        std::int32_t oldInferenceThreadCount{inferenceThreads};
        inferenceThreads = std::max(1, maxThreads - numAllocations);
        LOG_WARN(<< "Sum of allocation cound [" << numAllocations << "] and inference threads ["
                 << oldInferenceThreadCount << "] is greater than max threads ["
                 << maxThreads << "]. Setting number of allocations to " << numAllocations
                 << " and inference threads to " << inferenceThreads);
    }
}
}
}
