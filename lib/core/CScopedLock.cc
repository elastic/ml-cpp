/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CScopedLock.h>

#include <core/CMutex.h>

namespace ml {
namespace core {

CScopedLock::CScopedLock(CMutex& mutex) : m_Mutex(mutex) {
    m_Mutex.lock();
}

CScopedLock::~CScopedLock() {
    m_Mutex.unlock();
}
}
}
