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

#include <core/CMemoryCircuitBreaker.h>

namespace ml {
namespace core {

// To ensure the singleton is constructed before multiple threads may require it
// call instance() during the static initialisation phase of the program.  Of
// course, the instance may already be constructed before this if another static
// object has used it.
const ml::core::CMemoryCircuitBreakerStub& DO_NOT_USE_THIS_VARIABLE =
    ml::core::CMemoryCircuitBreakerStub::instance();

CMemoryCircuitBreakerStub& CMemoryCircuitBreakerStub::instance() {
    static CMemoryCircuitBreakerStub instance;
    return instance;
}
}
}
