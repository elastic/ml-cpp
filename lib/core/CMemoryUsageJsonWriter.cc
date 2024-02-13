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

#include <core/CMemoryUsageJsonWriter.h>

namespace {
const std::string MEMORY("memory");
const std::string UNUSED("unused");
}

namespace ml {
namespace core {

CMemoryUsageJsonWriter::CMemoryUsageJsonWriter(std::ostream& outStream)
    : m_WriteStream(outStream), m_Writer(m_WriteStream), m_Finalised(false) {
}

CMemoryUsageJsonWriter::~CMemoryUsageJsonWriter() {
    this->finalise();
}

void CMemoryUsageJsonWriter::startObject() {
    m_Writer.onObjectBegin();
}

void CMemoryUsageJsonWriter::endObject() {
    m_Writer.onObjectEnd();
}

void CMemoryUsageJsonWriter::startArray(const std::string& description) {
    m_Writer.onKey(description);
    m_Writer.onArrayBegin();
}

void CMemoryUsageJsonWriter::endArray() {
    m_Writer.onArrayEnd();
}

void CMemoryUsageJsonWriter::addItem(const CMemoryUsage::SMemoryUsage& item) {
    m_Writer.onKey(item.s_Name);
    m_Writer.onObjectBegin();

    m_Writer.onKey(MEMORY);
    m_Writer.onInt64(item.s_Memory);
    if (item.s_Unused) {
        m_Writer.onKey(UNUSED);
        m_Writer.onUint64(item.s_Unused);
    }
    m_Writer.onObjectEnd();
}

void CMemoryUsageJsonWriter::finalise() {
    if (m_Finalised) {
        return;
    }
    m_WriteStream.flush();
    m_Finalised = true;
}

} // core
} // ml
