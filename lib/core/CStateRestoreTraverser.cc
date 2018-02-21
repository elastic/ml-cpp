/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CStateRestoreTraverser.h>

#include <core/CLogger.h>


namespace ml
{
namespace core
{


CStateRestoreTraverser::CStateRestoreTraverser(void)
    : m_BadState(false)
{
}

CStateRestoreTraverser::~CStateRestoreTraverser(void)
{
}

bool CStateRestoreTraverser::haveBadState(void) const
{
    return m_BadState;
}

void CStateRestoreTraverser::setBadState(void)
{
    m_BadState = true;
}

CStateRestoreTraverser::CAutoLevel::CAutoLevel(CStateRestoreTraverser &traverser)
    : m_Traverser(traverser),
      m_Descended(traverser.descend()),
      m_BadState(false)
{
}

void CStateRestoreTraverser::CAutoLevel::setBadState(void)
{
    m_BadState = true;
}

CStateRestoreTraverser::CAutoLevel::~CAutoLevel(void)
{
    if (m_Descended && !m_BadState)
    {
        if (m_Traverser.ascend() == false)
        {
            LOG_ERROR("Inconsistency - could not ascend following previous descend");
            m_Traverser.setBadState();
        }
    }
}


}
}

