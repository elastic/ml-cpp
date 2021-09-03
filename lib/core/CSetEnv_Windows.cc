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
#include <core/CSetEnv.h>

#include <stdlib.h>

namespace ml {
namespace core {

int CSetEnv::setEnv(const char* name, const char* value, int overwrite) {
    if (overwrite == 0 && ::getenv(name) != 0) {
        return 0;
    }

    return ::_putenv_s(name, value);
}
}
}
