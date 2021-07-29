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

#include "CBufferedIStreamAdapter.h"

#include <core/CLogger.h>

// For ntohl
#ifdef Windows
#include <WinSock2.h>
#else
#include <netinet/in.h>
#endif

namespace ml {
namespace torch {

CBufferedIStreamAdapter::CBufferedIStreamAdapter(std::istream& inputStream)
    : m_InputStream(inputStream) {
}

bool CBufferedIStreamAdapter::init() {
    if (parseSizeFromStream(m_Size) == false) {
        LOG_ERROR(<< "Failed to read model size");
        return false;
    }

    LOG_DEBUG(<< "Loading model of size: " << m_Size);

    m_Buffer = std::make_unique<char[]>(m_Size);
    m_InputStream.read(m_Buffer.get(), m_Size);

    // gcount is the number of bytes read in the last read operation,
    // not the total. If the model is read in chunks this test will not pass
    if (m_Size != static_cast<std::size_t>(m_InputStream.gcount())) {
        LOG_ERROR(<< "Input size [" << m_InputStream.gcount()
                  << "] did not match expected input size [" << m_Size << "]");
        return false;
    }

    return true;
}

std::size_t CBufferedIStreamAdapter::size() const {
    return m_Size;
}

std::size_t
CBufferedIStreamAdapter::read(std::uint64_t pos, void* buf, std::size_t n, const char* what) const {
    if (pos > m_Size) {
        LOG_ERROR(<< "cannot read when position [" << pos
                  << "] is > buffer size [" << m_Size << "]: " << what);
        return 0;
    }

    if (pos + n > m_Size) {
        LOG_DEBUG(<< "read size [" << n << "] + position [" << pos
                  << "] is > buffer size [" << m_Size << "]: " << what);
        n = m_Size - pos;
    }

    std::memcpy(buf, m_Buffer.get() + pos, n);
    return n;
}

bool CBufferedIStreamAdapter::parseSizeFromStream(std::size_t& num) {
    if (m_InputStream.eof()) {
        LOG_ERROR(<< "Unexpected end of stream reading model size");
        return false;
    }

    std::uint32_t netNum{0};
    m_InputStream.read(reinterpret_cast<char*>(&netNum), sizeof(std::uint32_t));

    // Integers are encoded in network byte order, so convert to host byte order
    // before interpreting
    num = ntohl(netNum);
    return m_InputStream.good();
}
}
}
