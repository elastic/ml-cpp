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
#include <core/CCondition.h>

#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CWindowsError.h>

namespace ml {
namespace core {

CCondition::CCondition(CMutex& mutex) : m_Mutex(mutex) {
    InitializeConditionVariable(&m_Condition);
}

CCondition::~CCondition() {
    // There's no need to explicitly destroy a Windows condition variable
}

bool CCondition::wait() {
    BOOL success(SleepConditionVariableCS(&m_Condition, &m_Mutex.m_Mutex, INFINITE));
    if (success == FALSE) {
        LOG_WARN(<< "Condition wait failed : " << CWindowsError());
        return false;
    }

    return true;
}

bool CCondition::wait(uint32_t t) {
    BOOL success(SleepConditionVariableCS(&m_Condition, &m_Mutex.m_Mutex, t));
    if (success == FALSE) {
        DWORD errorCode(GetLastError());
        if (errorCode != WAIT_TIMEOUT && errorCode != ERROR_TIMEOUT) {
            LOG_WARN(<< "Condition wait failed : " << CWindowsError(errorCode));
            return false;
        }
    }

    return true;
}

void CCondition::signal() {
    WakeConditionVariable(&m_Condition);
}

void CCondition::broadcast() {
    WakeAllConditionVariable(&m_Condition);
}
}
}
