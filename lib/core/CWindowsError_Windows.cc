/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CWindowsError.h>

#include <core/CStringUtils.h>
#include <core/WindowsSafe.h>

#include <cmath>

namespace
{
static const size_t BUFFER_SIZE(1024);

// This is a workaround for a bug in the Visual Studio 2013 C runtime library.
// See http://connect.microsoft.com/VisualStudio/feedback/details/811093 for
// more details.  It should be fixed in the next major release of Visual Studio
// so this code could be removed then.
#ifdef _M_X64
int fmaRes(::_set_FMA3_enable(0));
#endif
}


namespace ml
{
namespace core
{


CWindowsError::CWindowsError()
    : m_ErrorCode(GetLastError())
{
}

CWindowsError::CWindowsError(uint32_t errorCode)
    : m_ErrorCode(errorCode)
{
}

uint32_t CWindowsError::errorCode() const
{
    return m_ErrorCode;
}

std::string CWindowsError::errorString() const
{
    char message[BUFFER_SIZE] = { '\0' };

    DWORD msgLen(FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK,
                               0,
                               m_ErrorCode,
                               0,
                               message,
                               BUFFER_SIZE,
                               0));
    if (msgLen == 0)
    {
        return "unknown error code (" + CStringUtils::typeToString(m_ErrorCode) + ')';
    }

    return message;
}

std::ostream &operator<<(std::ostream &os,
                         const CWindowsError &windowsError)
{
    char message[BUFFER_SIZE] = { '\0' };

    DWORD msgLen(FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK,
                               0,
                               windowsError.m_ErrorCode,
                               0,
                               message,
                               BUFFER_SIZE,
                               0));
    if (msgLen == 0)
    {
        os << "unknown error code (" << windowsError.m_ErrorCode << ')';
    }
    else
    {
        os << message;
    }

    return os;
}


}
}

