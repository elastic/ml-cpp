/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include "CBufferedIStreamAdapter.h"

// For ntohl
#ifdef Windows
#include <WinSock2.h>
#else
#include <netinet/in.h>
#endif


namespace ml {
namespace torch {

CBufferedIStreamAdapter::CBufferedIStreamAdapter(core::CNamedPipeFactory::TIStreamP inputStream)
{
	if (parseSizeFromStream(m_Size, inputStream) == false) {
		LOG_ERROR(<< "Failed to read model size");
	}

	LOG_INFO(<< "Loading model of size: " << m_Size);


	m_Buffer = std::make_unique<char[]>(m_Size);
	LOG_INFO(<< "reading stream");
	inputStream->read(m_Buffer.get(), m_Size);

	LOG_INFO(<<  "read " << inputStream->gcount());

	if (inputStream->eof()) {
		LOG_INFO(<< "end of stream");
	}

	if (inputStream->good() == false) {
		LOG_INFO(<< "stream not good");
	}
}

size_t CBufferedIStreamAdapter::size() const {
	return m_Size;
}

char* CBufferedIStreamAdapter::buffer() const {
	return m_Buffer.get();
}

size_t CBufferedIStreamAdapter::read(uint64_t pos, void* buf, size_t n, const char* what) const {
	if (pos > m_Size) {
		LOG_ERROR(<< "cannot read when pos is > size: " << what);
		// error with what
		return 0;

	}

	if (pos + n > m_Size) {
		LOG_ERROR(<< "trimming n to size");
		n = m_Size - pos;
	} 

	std::memcpy(buf, m_Buffer.get() + pos, n);
	return n;
}

// TODO: This reads a 4 byte int even though sizeof num is 8 bytes
bool CBufferedIStreamAdapter::parseSizeFromStream(std::size_t& num, core::CNamedPipeFactory::TIStreamP inputStream) {
    if (inputStream->eof()) {
        LOG_ERROR(<< "Unexpected end of stream reading model size");
        return false;
    }

    std::uint32_t netNum{0};
    inputStream->read(reinterpret_cast<char*>(&netNum), sizeof(std::uint32_t));

    // Integers are encoded in network byte order, so convert to host byte order
    // before interpreting
    num = ntohl(netNum);
    LOG_INFO(<< "sizeb " << num);

    return inputStream->good();    
}


}
}
