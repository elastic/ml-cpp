/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CRapidJsonConcurrentLineWriter.h>

namespace ml {
namespace core {

CRapidJsonConcurrentLineWriter::CRapidJsonConcurrentLineWriter(CJsonOutputStreamWrapper& outStream)
    : m_OutputStreamWrapper(outStream) {
    m_OutputStreamWrapper.acquireBuffer(*this, m_StringBuffer);
}

CRapidJsonConcurrentLineWriter::~CRapidJsonConcurrentLineWriter() {
    m_OutputStreamWrapper.releaseBuffer(*this, m_StringBuffer);
}

void CRapidJsonConcurrentLineWriter::flush() {
    TRapidJsonLineWriterBase::Flush();

    m_OutputStreamWrapper.flush();
}

bool CRapidJsonConcurrentLineWriter::EndObject(rapidjson::SizeType memberCount) {
    bool baseReturnCode = TRapidJsonLineWriterBase::EndObject(memberCount);

    if (TRapidJsonLineWriterBase::IsComplete()) {
        m_OutputStreamWrapper.flushBuffer(*this, m_StringBuffer);
    }

    return baseReturnCode;
}

void CRapidJsonConcurrentLineWriter::debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CRapidJsonConcurrentLineWriter", sizeof(*this));
    m_OutputStreamWrapper.debugMemoryUsage(mem->addChild());
}

std::size_t CRapidJsonConcurrentLineWriter::memoryUsage() const {
    return m_OutputStreamWrapper.memoryUsage();
}
}
}
