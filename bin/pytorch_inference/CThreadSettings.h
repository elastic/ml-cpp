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
    explicit CThreadSettings(std::int32_t inferenceThreads, std::int32_t numAllocations);

    std::int32_t inferenceThreads() const;
    std::int32_t numAllocations() const;
    void numAllocations(std::int32_t numAllocations);

    static void validateThreadingParameters(std::int32_t maxThreads,
                                            std::int32_t& inferenceThreads,
                                            std::int32_t& numAllocations);

private:
    std::int32_t m_InferenceThreads;
    std::int32_t m_NumAllocations;
};
}
}

#endif // INCLUDED_ml_torch_CThreadSettings_h
