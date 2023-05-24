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

#ifndef INCLUDED_ml_core_CMemoryCircuitBreaker_h
#define INCLUDED_ml_core_CMemoryCircuitBreaker_h

#include <core/ImportExport.h>

namespace ml {
namespace core {
//! \brief The allocator interface.
//!
//! DESCRIPTION:\n
//! The allocator interface is used to control the ability
//! of a model to allocate new components. This is used to
//! comply with the memory constraints of the system.
class CORE_EXPORT CMemoryCircuitBreaker {
public:
    virtual ~CMemoryCircuitBreaker() = default;

    //! Check if we can still allocate any components.
    virtual bool areAllocationsAllowed() const = 0;
};

//! \brief The allocator stub.
//!
//! DESCRIPTION:\n
//! The allocator stub is used to allow all allocations.
class CORE_EXPORT CMemoryCircuitBreakerStub : public core::CMemoryCircuitBreaker {
public:
    static const CMemoryCircuitBreakerStub& instance() {
        static CMemoryCircuitBreakerStub instance;
        return instance;
    }
    bool areAllocationsAllowed() const override { return true; }
    CMemoryCircuitBreakerStub(const CMemoryCircuitBreakerStub&) = delete;
    CMemoryCircuitBreakerStub& operator=(const CMemoryCircuitBreakerStub&) = delete;

private:
    CMemoryCircuitBreakerStub() = default;
};
}
}

#endif // INCLUDED_ml_core_CMemoryCircuitBreaker_h
