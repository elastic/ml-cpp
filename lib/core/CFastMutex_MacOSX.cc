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

namespace ml {
namespace core {

CFastMutex::CFastMutex()
    // The OSSpinLock type is just an integer, and zero means unlocked.  See
    // "man spinlock" for details.
    : m_Mutex(0) {
}

CFastMutex::~CFastMutex() {
}

void CFastMutex::lock() {
    OSSpinLockLock(&m_Mutex);
}

void CFastMutex::unlock() {
    OSSpinLockUnlock(&m_Mutex);
}
}
}
