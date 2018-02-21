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


namespace ml
{
namespace core
{


CFastMutex::CFastMutex(void)
{
    InitializeSRWLock(&m_Mutex);
}

CFastMutex::~CFastMutex(void)
{
    // There is no function to destroy the read/write lock on Windows
}

void CFastMutex::lock(void)
{
    AcquireSRWLockExclusive(&m_Mutex);
}

void CFastMutex::unlock(void)
{
    ReleaseSRWLockExclusive(&m_Mutex);
}


}
}

