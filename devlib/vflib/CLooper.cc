/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <vflib/CLooper.h>

#include <vflib/CIncrementer.h>


namespace ml
{
namespace vflib
{


size_t CLooper::inlinedLibraryCallLoop(CIncrementer &incrementer,
                                       size_t count,
                                       size_t val)
{
    for (size_t i = 0; i < count; ++i)
    {
        val = incrementer.inlinedIncrement(val);
    }
    return val;
}

size_t CLooper::nonVirtualLibraryCallLoop(CIncrementer &incrementer,
                                          size_t count,
                                          size_t val)
{
    for (size_t i = 0; i < count; ++i)
    {
        val = incrementer.nonVirtualIncrement(val);
    }
    return val;
}

size_t CLooper::virtualLibraryCallLoop(CIncrementer &incrementer,
                                       size_t count,
                                       size_t val)
{
    for (size_t i = 0; i < count; ++i)
    {
        val = incrementer.virtualIncrement(val);
    }
    return val;
}


}
}

