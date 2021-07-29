/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
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
