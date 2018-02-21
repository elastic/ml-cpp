/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CReadWriteLock.h>


namespace ml
{
namespace core
{


CReadWriteLock::CReadWriteLock(void)
{
    InitializeSRWLock(&m_ReadWriteLock);
}

CReadWriteLock::~CReadWriteLock(void)
{
    // There is no function to destroy the read/write lock on Windows
}

void CReadWriteLock::readLock(void)
{
    AcquireSRWLockShared(&m_ReadWriteLock);
}

void CReadWriteLock::readUnlock(void)
{
    ReleaseSRWLockShared(&m_ReadWriteLock);
}

void CReadWriteLock::writeLock(void)
{
    AcquireSRWLockExclusive(&m_ReadWriteLock);
}

void CReadWriteLock::writeUnlock(void)
{
    ReleaseSRWLockExclusive(&m_ReadWriteLock);
}


}
}

