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
#include <core/CMutex.h>

#include <core/CLogger.h>

#include <errno.h>
#include <string.h>

namespace ml {
namespace core {

CMutex::CMutex() {
    pthread_mutexattr_t attr;

    int ret(pthread_mutexattr_init(&attr));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }

    ret = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }

    ret = pthread_mutex_init(&m_Mutex, &attr);
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }

    ret = pthread_mutexattr_destroy(&attr);
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

CMutex::~CMutex() {
    int ret(pthread_mutex_destroy(&m_Mutex));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

void CMutex::lock() {
    int ret(pthread_mutex_lock(&m_Mutex));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

void CMutex::unlock() {
    int ret(pthread_mutex_unlock(&m_Mutex));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}
}
}
