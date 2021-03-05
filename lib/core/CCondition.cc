/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CCondition.h>

#include <core/CLogger.h>
#include <core/CMutex.h>

#include <errno.h>
#include <string.h>
#include <sys/time.h>

namespace ml {
namespace core {

CCondition::CCondition(CMutex& mutex) : m_Mutex(mutex) {
    int ret(::pthread_cond_init(&m_Condition, nullptr));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

CCondition::~CCondition() {
    int ret(::pthread_cond_destroy(&m_Condition));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

bool CCondition::wait() {
    // Note: pthread_cond_wait() returns 0 if interrupted by a signal, so the
    // caller must check a condition that will detect spurious wakeups
    int ret(::pthread_cond_wait(&m_Condition, &m_Mutex.m_Mutex));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(errno));
        return false;
    }

    return true;
}

bool CCondition::wait(uint32_t t) {
    timespec tm;

    if (CCondition::convert(t, tm) == false) {
        return false;
    }

    // Note: pthread_cond_timedwait() returns 0 if interrupted by a signal, so
    // the caller must check a condition that will detect spurious wakeups
    int ret(::pthread_cond_timedwait(&m_Condition, &m_Mutex.m_Mutex, &tm));
    if (ret != 0) {
        if (ret != ETIMEDOUT) {
            LOG_WARN(<< t << ' ' << ::strerror(errno));
            return false;
        }
    }

    return true;
}

void CCondition::signal() {
    int ret(::pthread_cond_signal(&m_Condition));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

void CCondition::broadcast() {
    int ret(::pthread_cond_broadcast(&m_Condition));
    if (ret != 0) {
        LOG_WARN(<< ::strerror(ret));
    }
}

bool CCondition::convert(uint32_t t, timespec& tm) {
    timeval now;
    if (::gettimeofday(&now, nullptr) < 0) {
        LOG_WARN(<< ::strerror(errno));
        return false;
    }

    // We can't just add 't' to now as we need to deal
    // with overflows + convert timeval to timespec
    tm.tv_sec = now.tv_sec + (t / 1000);

    uint32_t remainder(static_cast<uint32_t>(t % 1000));
    if (remainder == 0) {
        tm.tv_nsec = now.tv_usec * 1000;
    } else {
        // s is in microseconds
        uint32_t s((remainder * 1000U) + static_cast<uint32_t>(now.tv_usec));

        tm.tv_sec = tm.tv_sec + (s / 1000000U);
        tm.tv_nsec = (s % 1000000U) * 1000;
    }

    return true;
}
}
}
