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
#include <core/CMutex.h>

namespace {
// 4000 is a value that Microsoft uses in some of their code, so it's
// hopefully a reasonably sensible setting
const DWORD SPIN_COUNT(4000);
}

namespace ml {
namespace core {

CMutex::CMutex() {
    // On Windows Vista and above this function cannot fail, hence no need to
    // check the return code
    InitializeCriticalSectionAndSpinCount(&m_Mutex, SPIN_COUNT);
}

CMutex::~CMutex() {
    DeleteCriticalSection(&m_Mutex);
}

void CMutex::lock() {
    EnterCriticalSection(&m_Mutex);
}

void CMutex::unlock() {
    LeaveCriticalSection(&m_Mutex);
}
}
}
