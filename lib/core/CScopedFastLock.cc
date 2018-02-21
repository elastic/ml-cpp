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
#include <core/CScopedFastLock.h>

#include <core/CFastMutex.h>


namespace ml
{
namespace core
{


CScopedFastLock::CScopedFastLock(CFastMutex &mutex)
    : m_Mutex(mutex)
{
    m_Mutex.lock();
}

CScopedFastLock::~CScopedFastLock(void)
{
    m_Mutex.unlock();
}


}
}

