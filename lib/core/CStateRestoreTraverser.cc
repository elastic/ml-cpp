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
#include <core/CStateRestoreTraverser.h>

#include <core/CLogger.h>

namespace ml {
namespace core {

CStateRestoreTraverser::CStateRestoreTraverser() : m_BadState(false) {
}

CStateRestoreTraverser::~CStateRestoreTraverser() {
}

bool CStateRestoreTraverser::haveBadState() const {
    return m_BadState;
}

void CStateRestoreTraverser::setBadState() {
    m_BadState = true;
}

CStateRestoreTraverser::CAutoLevel::CAutoLevel(CStateRestoreTraverser& traverser)
    : m_Traverser{traverser}, m_Descended{traverser.descend()} {
}

CStateRestoreTraverser::CAutoLevel::~CAutoLevel() {
    if (m_Descended && m_Traverser.haveBadState() == false) {
        if (m_Traverser.ascend() == false) {
            LOG_ERROR(<< "Inconsistency - could not ascend following previous descend");
            m_Traverser.setBadState();
        }
    }
}
}
}
