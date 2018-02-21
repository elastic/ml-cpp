/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CWindowsError.h>

#include <ostream>


namespace ml
{
namespace core
{


CWindowsError::CWindowsError(void)
    : m_ErrorCode(0)
{
}

CWindowsError::CWindowsError(uint32_t /* errorCode */)
    : m_ErrorCode(0)
{
}

uint32_t CWindowsError::errorCode(void) const
{
    return m_ErrorCode;
}

std::string CWindowsError::errorString(void) const
{
    return "Asking for Windows error message on Unix!";
}

std::ostream &operator<<(std::ostream &os,
                         const CWindowsError & /* windowsError */)
{
    os << "Asking for Windows error message on Unix!";
    return os;
}


}
}

