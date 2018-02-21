/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CGmTimeR.h>


namespace ml
{
namespace core
{


struct tm *CGmTimeR::gmTimeR(const time_t *clock,
                             struct tm *result)
{
    ::gmtime_s(result, clock);

    return result;
}


}
}

