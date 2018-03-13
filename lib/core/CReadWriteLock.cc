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

#include <core/CLogger.h>

#include <errno.h>
#include <string.h>

namespace ml {
namespace core {

CReadWriteLock::CReadWriteLock(void) {
    // Valgrind can complain if this is not initialised
    memset(&m_ReadWriteLock, 0x00, sizeof(m_ReadWriteLock));
    int ret(pthread_rwlock_init(&m_ReadWriteLock, 0));
    if (ret != 0) {
        LOG_WARN(::strerror(ret));
    }
}

CReadWriteLock::~CReadWriteLock(void) {
    int ret(pthread_rwlock_destroy(&m_ReadWriteLock));
    if (ret != 0) {
        LOG_WARN(::strerror(ret));
    }
}

void CReadWriteLock::readLock(void) {
    int ret(pthread_rwlock_rdlock(&m_ReadWriteLock));
    if (ret != 0) {
        LOG_WARN(::strerror(ret));
    }
}

void CReadWriteLock::readUnlock(void) {
    int ret(pthread_rwlock_unlock(&m_ReadWriteLock));
    if (ret != 0) {
        LOG_WARN(::strerror(ret));
    }
}

void CReadWriteLock::writeLock(void) {
    int ret(pthread_rwlock_wrlock(&m_ReadWriteLock));
    if (ret != 0) {
        LOG_WARN(::strerror(ret));
    }
}

void CReadWriteLock::writeUnlock(void) {
    int ret(pthread_rwlock_unlock(&m_ReadWriteLock));
    if (ret != 0) {
        LOG_WARN(::strerror(ret));
    }
}
}
}
