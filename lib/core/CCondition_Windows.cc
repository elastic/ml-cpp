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
#include <core/CCondition.h>

#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CWindowsError.h>


namespace ml
{
namespace core
{


CCondition::CCondition(CMutex &mutex)
    : m_Mutex(mutex)
{
    InitializeConditionVariable(&m_Condition);
}

CCondition::~CCondition(void)
{
    // There's no need to explicitly destroy a Windows condition variable
}

bool CCondition::wait(void)
{
    BOOL success(SleepConditionVariableCS(&m_Condition,
                                          &m_Mutex.m_Mutex,
                                          INFINITE));
    if (success == FALSE)
    {
        LOG_WARN("Condition wait failed : " << CWindowsError());
        return false;
    }

    return true;
}

bool CCondition::wait(uint32_t t)
{
    BOOL success(SleepConditionVariableCS(&m_Condition,
                                          &m_Mutex.m_Mutex,
                                          t));
    if (success == FALSE)
    {
        DWORD errorCode(GetLastError());
        if (errorCode != WAIT_TIMEOUT && errorCode != ERROR_TIMEOUT)
        {
            LOG_WARN("Condition wait failed : " << CWindowsError(errorCode));
            return false;
        }
    }

    return true;
}

void CCondition::signal(void)
{
    WakeConditionVariable(&m_Condition);
}

void CCondition::broadcast(void)
{
    WakeAllConditionVariable(&m_Condition);
}


}
}

