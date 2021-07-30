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
#include <core/CCTimeR.h>

namespace ml {
namespace core {

char* CCTimeR::cTimeR(const time_t* clock, char* result) {
    // This is effectively bypassing the security feature of the Windows
    // ctime_s() call, but the wrapper function has the arguments of the
    // vulnerable Unix ctime_r() function, so we don't know the real buffer
    // size, and must assume it's big enough
    static const size_t MIN_BUF_SIZE(26);

    ::ctime_s(result, MIN_BUF_SIZE, clock);

    return result;
}
}
}
