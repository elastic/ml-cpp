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
#include <core/CFastMutex.h>

namespace ml {
namespace core {

CFastMutex::CFastMutex() : m_Mutex(OS_UNFAIR_LOCK_INIT) {
}

CFastMutex::~CFastMutex() {
}

void CFastMutex::lock() {
    os_unfair_lock_lock(&m_Mutex);
}

void CFastMutex::unlock() {
    os_unfair_lock_unlock(&m_Mutex);
}
}
}
