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

#include <core/CRapidJsonUnbufferedIStreamWrapper.h>

namespace ml {
namespace core {

CRapidJsonUnbufferedIStreamWrapper::CRapidJsonUnbufferedIStreamWrapper(std::istream& strm)
    : m_Stream{strm} {};

char CRapidJsonUnbufferedIStreamWrapper::Take() {
    int c{m_Stream.get()};
    if (RAPIDJSON_UNLIKELY(c == std::istream::traits_type::eof())) {
        return '\0';
    }
    ++m_Count;
    return static_cast<char>(c);
}
}
}
