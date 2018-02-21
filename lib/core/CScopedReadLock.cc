/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CScopedReadLock.h>

#include <core/CReadWriteLock.h>


namespace ml
{
namespace core
{


CScopedReadLock::CScopedReadLock(CReadWriteLock &readWriteLock)
    : m_ReadWriteLock(readWriteLock)
{
    m_ReadWriteLock.readLock();
}

CScopedReadLock::~CScopedReadLock(void)
{
    m_ReadWriteLock.readUnlock();
}


}
}

