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


namespace {
// 4000 is a value that Microsoft uses in some of their code, so it's
// hopefully a reasonably sensible setting
static const DWORD SPIN_COUNT(4000);
}

namespace ml {
namespace core {


CMutex::CMutex(void) {
    // On Windows Vista and above this function cannot fail, hence no need to
    // check the return code
    InitializeCriticalSectionAndSpinCount(&m_Mutex, SPIN_COUNT);
}

CMutex::~CMutex(void) {
    DeleteCriticalSection(&m_Mutex);
}

void CMutex::lock(void) {
    EnterCriticalSection(&m_Mutex);
}

void CMutex::unlock(void) {
    LeaveCriticalSection(&m_Mutex);
}


}
}

