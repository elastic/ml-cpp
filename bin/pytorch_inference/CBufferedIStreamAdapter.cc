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

	LOG_DEBUG(<< "Loading model of size: " << m_Size);


	m_Buffer = std::make_unique<char[]>(m_Size);
	inputStream->read(m_Buffer.get(), m_Size);

	// gcount is the number of bytes read in the last read operation,
	// not the total. If the model is read in chunks this test will not pass
	if (m_Size != static_cast<std::size_t>(inputStream->gcount())) {
		LOG_ERROR(<< "Input size [" << inputStream->gcount() << 
			"] did not match expected input size [" << m_Size << "]");
	}
}

size_t CBufferedIStreamAdapter::size() const {
	return m_Size;
}

size_t CBufferedIStreamAdapter::read(uint64_t pos, void* buf, size_t n, const char* what) const {
	if (pos > m_Size) {
		LOG_ERROR(<< "cannot read when pos is > size: " << what);
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
    return inputStream->good();    
}


}
}
