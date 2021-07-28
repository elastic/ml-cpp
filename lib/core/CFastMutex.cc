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
#include <core/CFastMutex.h>

#include <core/CLogger.h>

#include <errno.h>
#include <string.h>

namespace ml {
namespace core {

CFastMutex::CFastMutex() {
    int ret(pthread_mutex_init(&m_Mutex, 0));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

CFastMutex::~CFastMutex() {
    int ret(pthread_mutex_destroy(&m_Mutex));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

void CFastMutex::lock() {
    int ret(pthread_mutex_lock(&m_Mutex));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

void CFastMutex::unlock() {
    int ret(pthread_mutex_unlock(&m_Mutex));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}
}
}
