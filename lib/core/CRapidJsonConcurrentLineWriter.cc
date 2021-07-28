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

#include <core/CRapidJsonConcurrentLineWriter.h>

namespace ml {
namespace core {

CRapidJsonConcurrentLineWriter::CRapidJsonConcurrentLineWriter(CJsonOutputStreamWrapper& outStream)
    : m_OutputStreamWrapper(outStream) {
    m_OutputStreamWrapper.acquireBuffer(*this, m_StringBuffer);
}

CRapidJsonConcurrentLineWriter::~CRapidJsonConcurrentLineWriter() {
    this->flush();
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

void CRapidJsonConcurrentLineWriter::debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CRapidJsonConcurrentLineWriter", sizeof(*this));
    m_OutputStreamWrapper.debugMemoryUsage(mem->addChild());
}

std::size_t CRapidJsonConcurrentLineWriter::memoryUsage() const {
    return m_OutputStreamWrapper.memoryUsage();
}
}
}
