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
#include <core/CFastMutex.h>

#include <core/CLogger.h>

#include <errno.h>
#include <string.h>

namespace ml {
namespace core {

CFastMutex::CFastMutex(void) {
    int ret(pthread_mutex_init(&m_Mutex, 0));
    if (ret != 0) {
        LOG_WARN(::strerror(ret));
    }
}

CFastMutex::~CFastMutex(void) {
    int ret(pthread_mutex_destroy(&m_Mutex));
    if (ret != 0) {
        LOG_WARN(::strerror(ret));
    }
}

void CFastMutex::lock(void) {
    int ret(pthread_mutex_lock(&m_Mutex));
    if (ret != 0) {
        LOG_WARN(::strerror(ret));
    }
}

void CFastMutex::unlock(void) {
    int ret(pthread_mutex_unlock(&m_Mutex));
    if (ret != 0) {
        LOG_WARN(::strerror(ret));
    }
}
}
}
