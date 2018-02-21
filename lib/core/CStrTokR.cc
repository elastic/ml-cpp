/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CStrTokR.h>

#include <string.h>


namespace ml
{
namespace core
{


char *CStrTokR::strTokR(char *str, const char *sep, char **lasts)
{
    return ::strtok_r(str, sep, lasts);
}


}
}

