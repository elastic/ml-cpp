/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CReadWriteLock.h>

#include <core/CLogger.h>

#include <errno.h>
#include <string.h>


namespace ml
{
namespace core
{


CReadWriteLock::CReadWriteLock(void)
{
    // Valgrind can complain if this is not initialised
    memset(&m_ReadWriteLock, 0x00, sizeof(m_ReadWriteLock));
    int ret(pthread_rwlock_init(&m_ReadWriteLock, 0));
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }
}

CReadWriteLock::~CReadWriteLock(void)
{
    int ret(pthread_rwlock_destroy(&m_ReadWriteLock));
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }
}

void CReadWriteLock::readLock(void)
{
    int ret(pthread_rwlock_rdlock(&m_ReadWriteLock));
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }
}

void CReadWriteLock::readUnlock(void)
{
    int ret(pthread_rwlock_unlock(&m_ReadWriteLock));
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }
}

void CReadWriteLock::writeLock(void)
{
    int ret(pthread_rwlock_wrlock(&m_ReadWriteLock));
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }
}

void CReadWriteLock::writeUnlock(void)
{
    int ret(pthread_rwlock_unlock(&m_ReadWriteLock));
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }
}


}
}

