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
    // The Microsoft C runtime library treats a request to set an environment
    // variable to an empty string as a request to delete it
    return ::_putenv_s(name, "");
}
}
}
