/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CSetEnv.h>

#include <stdlib.h>


namespace ml
{
namespace core
{


int CSetEnv::setEnv(const char *name,
                    const char *value,
                    int overwrite)
{
    return ::setenv(name, value, overwrite);
}


}
}

