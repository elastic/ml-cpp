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
#include <core/CScopedWriteLock.h>

#include <core/CReadWriteLock.h>

namespace ml {
namespace core {

CScopedWriteLock::CScopedWriteLock(CReadWriteLock &readWriteLock) : m_ReadWriteLock(readWriteLock) {
    m_ReadWriteLock.writeLock();
}

CScopedWriteLock::~CScopedWriteLock(void) { m_ReadWriteLock.writeUnlock(); }
}
}
