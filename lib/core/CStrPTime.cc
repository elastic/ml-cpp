/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CStrPTime.h>


namespace ml
{
namespace core
{


char *CStrPTime::strPTime(const char *buf,
                          const char *format,
                          struct tm *tm)

{
    return ::strptime(buf, format, tm);
}


}
}

