/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CStrFTime.h>


namespace ml
{
namespace core
{


size_t CStrFTime::strFTime(char *buf,
                           size_t maxSize,
                           const char *format,
                           struct tm *tm)
{
    return ::strftime(buf, maxSize, format, tm);
}


}
}

