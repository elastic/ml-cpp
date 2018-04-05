/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#include <core/CRapidJsonConcurrentLineWriter.h>

namespace ml {
namespace core {

CRapidJsonConcurrentLineWriter::CRapidJsonConcurrentLineWriter(CJsonOutputStreamWrapper& outStream) : m_OutputStreamWrapper(outStream) {
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
