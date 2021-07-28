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

#include <core/CLogger.h>

#include <errno.h>
#include <string.h>

namespace ml {
namespace core {

CReadWriteLock::CReadWriteLock() {
    // Valgrind can complain if this is not initialised
    memset(&m_ReadWriteLock, 0x00, sizeof(m_ReadWriteLock));
    int ret(pthread_rwlock_init(&m_ReadWriteLock, nullptr));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

CReadWriteLock::~CReadWriteLock() {
    int ret(pthread_rwlock_destroy(&m_ReadWriteLock));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

void CReadWriteLock::readLock() {
    int ret(pthread_rwlock_rdlock(&m_ReadWriteLock));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

void CReadWriteLock::readUnlock() {
    int ret(pthread_rwlock_unlock(&m_ReadWriteLock));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

void CReadWriteLock::writeLock() {
    int ret(pthread_rwlock_wrlock(&m_ReadWriteLock));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

void CReadWriteLock::writeUnlock() {
    int ret(pthread_rwlock_unlock(&m_ReadWriteLock));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}
}
}
