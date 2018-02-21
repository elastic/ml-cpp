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

