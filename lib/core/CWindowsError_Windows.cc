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

#include <core/CStringUtils.h>
#include <core/WindowsSafe.h>

#include <math.h>

namespace {
static const size_t BUFFER_SIZE(1024);
}

namespace ml {
namespace core {

CWindowsError::CWindowsError(void) : m_ErrorCode(GetLastError()) {
}

CWindowsError::CWindowsError(uint32_t errorCode) : m_ErrorCode(errorCode) {
}

uint32_t CWindowsError::errorCode(void) const {
    return m_ErrorCode;
}

std::string CWindowsError::errorString(void) const {
    char message[BUFFER_SIZE] = {'\0'};

    DWORD msgLen(
        FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK,
                      0,
                      m_ErrorCode,
                      0,
                      message,
                      BUFFER_SIZE,
                      0));
    if (msgLen == 0) {
        return "unknown error code (" + CStringUtils::typeToString(m_ErrorCode) + ')';
    }

    return message;
}

std::ostream& operator<<(std::ostream& os, const CWindowsError& windowsError) {
    char message[BUFFER_SIZE] = {'\0'};

    DWORD msgLen(
        FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK,
                      0,
                      windowsError.m_ErrorCode,
                      0,
                      message,
                      BUFFER_SIZE,
                      0));
    if (msgLen == 0) {
        os << "unknown error code (" << windowsError.m_ErrorCode << ')';
    } else {
        os << message;
    }

    return os;
}
}
}
