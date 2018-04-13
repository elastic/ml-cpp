/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CFastMutex.h>

namespace ml {
namespace core {

CFastMutex::CFastMutex() {
    InitializeSRWLock(&m_Mutex);
}

CFastMutex::~CFastMutex() {
    // There is no function to destroy the read/write lock on Windows
}

void CFastMutex::lock() {
    AcquireSRWLockExclusive(&m_Mutex);
}

void CFastMutex::unlock() {
    ReleaseSRWLockExclusive(&m_Mutex);
}
}
}
