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
#include <core/CWindowsError.h>

#include <ostream>

namespace ml {
namespace core {

CWindowsError::CWindowsError() : m_ErrorCode(0) {
}

CWindowsError::CWindowsError(std::uint32_t /* errorCode */) : m_ErrorCode(0) {
}

uint32_t CWindowsError::errorCode() const {
    return m_ErrorCode;
}

std::string CWindowsError::errorString() const {
    return "Asking for Windows error message on Unix!";
}

std::ostream& operator<<(std::ostream& os, const CWindowsError& /* windowsError */) {
    os << "Asking for Windows error message on Unix!";
    return os;
}
}
}
