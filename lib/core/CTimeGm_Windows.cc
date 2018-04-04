/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CTimeGm.h>

namespace ml {
namespace core {

time_t CTimeGm::timeGm(struct tm* ts) {
    return ::_mkgmtime(ts);
}
}
}
