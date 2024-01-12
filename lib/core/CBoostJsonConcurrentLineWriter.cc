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

#include <core/CBoostJsonConcurrentLineWriter.h>

namespace ml {
namespace core {

CBoostJsonConcurrentLineWriter::CBoostJsonConcurrentLineWriter(CJsonOutputStreamWrapper& outStream)
    :  m_OutputStreamWrapper(outStream) {
        m_OutputStreamWrapper.acquireBuffer(*this, m_StringBuffer);
}

CBoostJsonConcurrentLineWriter::~CBoostJsonConcurrentLineWriter() {
    this->flush();
    m_OutputStreamWrapper.releaseBuffer(*this, m_StringBuffer);
}

void CBoostJsonConcurrentLineWriter::flush() {
//    TBoostJsonLineWriterBase::flush();

    m_OutputStreamWrapper.flush();
}

bool CBoostJsonConcurrentLineWriter::EndObject(std::size_t memberCount){
    bool baseReturnCode = TBoostJsonLineWriterBase::EndObject(memberCount);

    if (this->topLevel()) {
        m_OutputStreamWrapper.flushBuffer(*this, m_StringBuffer);
    }

    return baseReturnCode;
}

void CBoostJsonConcurrentLineWriter::debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CBoostJsonConcurrentLineWriter", sizeof(*this));
    m_OutputStreamWrapper.debugMemoryUsage(mem->addChild());
}

std::size_t CBoostJsonConcurrentLineWriter::memoryUsage() const {
    return m_OutputStreamWrapper.memoryUsage();
}
}
}
