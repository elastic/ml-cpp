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
#include <core/CMutex.h>

#include <core/CLogger.h>

#include <errno.h>
#include <string.h>


namespace ml
{
namespace core
{


CMutex::CMutex()
{
    pthread_mutexattr_t attr;

    int ret(pthread_mutexattr_init(&attr));
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }

    ret = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }

    ret = pthread_mutex_init(&m_Mutex, &attr);
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }

    ret = pthread_mutexattr_destroy(&attr);
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }
}

CMutex::~CMutex()
{
    int ret(pthread_mutex_destroy(&m_Mutex));
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }
}

void CMutex::lock()
{
    int ret(pthread_mutex_lock(&m_Mutex));
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }
}

void CMutex::unlock()
{
    int ret(pthread_mutex_unlock(&m_Mutex));
    if (ret != 0)
    {
        LOG_WARN(::strerror(ret));
    }
}


}
}

