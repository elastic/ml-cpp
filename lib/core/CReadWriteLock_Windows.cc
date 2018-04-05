/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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
