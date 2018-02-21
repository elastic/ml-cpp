/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CPOpen.h>


namespace ml
{
namespace core
{


FILE *CPOpen::pOpen(const char *command,
                    const char *mode)
{
    return ::popen(command, mode);
}

int CPOpen::pClose(FILE *stream)
{
    return ::pclose(stream);
}


}
}

