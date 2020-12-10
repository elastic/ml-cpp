/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include "CBufferedIStreamAdapter.h"


namespace ml {
namespace torch {

CBufferedIStreamAdapter::CBufferedIStreamAdapter(size_t size, std::istream& inputStream) :
	m_Size(size)
{
	m_Buffer = std::make_unique<char[]>(size);
	LOG_INFO(<< "reading stream");
	inputStream.read(m_Buffer.get(), size);

	if (inputStream.eof()) {
		LOG_INFO(<< "end of stream");
	}

	if (inputStream.good() == false) {
		LOG_INFO(<< "stream not good");
	}
}

size_t CBufferedIStreamAdapter::size() const {
	return m_Size;
}

size_t CBufferedIStreamAdapter::read(uint64_t pos, void* buf, size_t n, const char* what) const {
	if (pos > m_Size) {
		LOG_ERROR(<< "cannot read when pos is > size");
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


}
}
