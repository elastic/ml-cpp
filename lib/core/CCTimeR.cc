/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CCTimeR.h>

namespace ml {
namespace core {

char* CCTimeR::cTimeR(const time_t* clock, char* result) {
    return ::ctime_r(clock, result);
}
}
}
