/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CScopedFastLock.h>

#include <core/CFastMutex.h>


namespace ml
{
namespace core
{


CScopedFastLock::CScopedFastLock(CFastMutex &mutex)
    : m_Mutex(mutex)
{
    m_Mutex.lock();
}

CScopedFastLock::~CScopedFastLock(void)
{
    m_Mutex.unlock();
}


}
}

