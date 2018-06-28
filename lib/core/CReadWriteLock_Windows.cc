/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CReadWriteLock.h>

namespace ml {
namespace core {

CReadWriteLock::CReadWriteLock() {
    InitializeSRWLock(&m_ReadWriteLock);
}

CReadWriteLock::~CReadWriteLock() {
    // There is no function to destroy the read/write lock on Windows
}

void CReadWriteLock::readLock() {
    AcquireSRWLockShared(&m_ReadWriteLock);
}

void CReadWriteLock::readUnlock() {
    ReleaseSRWLockShared(&m_ReadWriteLock);
}

void CReadWriteLock::writeLock() {
    AcquireSRWLockExclusive(&m_ReadWriteLock);
}

void CReadWriteLock::writeUnlock() {
    ReleaseSRWLockExclusive(&m_ReadWriteLock);
}
}
}
