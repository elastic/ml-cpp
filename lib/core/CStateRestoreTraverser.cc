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
#include <core/CStateRestoreTraverser.h>

#include <core/CLogger.h>


namespace ml {
namespace core {


CStateRestoreTraverser::CStateRestoreTraverser(void)
    : m_BadState(false) {}

CStateRestoreTraverser::~CStateRestoreTraverser(void) {}

bool CStateRestoreTraverser::haveBadState(void) const {
    return m_BadState;
}

void CStateRestoreTraverser::setBadState(void) {
    m_BadState = true;
}

CStateRestoreTraverser::CAutoLevel::CAutoLevel(CStateRestoreTraverser &traverser)
    : m_Traverser(traverser),
      m_Descended(traverser.descend()),
      m_BadState(false) {}

void CStateRestoreTraverser::CAutoLevel::setBadState(void) {
    m_BadState = true;
}

CStateRestoreTraverser::CAutoLevel::~CAutoLevel(void) {
    if (m_Descended && !m_BadState) {
        if (m_Traverser.ascend() == false) {
            LOG_ERROR("Inconsistency - could not ascend following previous descend");
            m_Traverser.setBadState();
        }
    }
}


}
}

