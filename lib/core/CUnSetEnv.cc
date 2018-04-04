/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CUnSetEnv.h>

#include <stdlib.h>

namespace ml {
namespace core {

int CUnSetEnv::unSetEnv(const char* name) {
    return ::unsetenv(name);
}
}
}
