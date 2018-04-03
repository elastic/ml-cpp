/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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

#include <core/CMemoryUsageJsonWriter.h>

namespace {
const std::string MEMORY("memory");
const std::string UNUSED("unused");
}

namespace ml {
namespace core {

CMemoryUsageJsonWriter::CMemoryUsageJsonWriter(std::ostream &outStream) :
    m_WriteStream(outStream), m_Writer(m_WriteStream), m_Finalised(false)
{}

CMemoryUsageJsonWriter::~CMemoryUsageJsonWriter() {
    this->finalise();
}

void CMemoryUsageJsonWriter::startObject() {
    m_Writer.StartObject();
}

void CMemoryUsageJsonWriter::endObject() {
    m_Writer.EndObject();
}

void CMemoryUsageJsonWriter::startArray(const std::string &description) {
    m_Writer.String(description);
    m_Writer.StartArray();
}

void CMemoryUsageJsonWriter::endArray() {
    m_Writer.EndArray();
}

void CMemoryUsageJsonWriter::addItem(const CMemoryUsage::SMemoryUsage &item) {
    m_Writer.String(item.s_Name);
    m_Writer.StartObject();

    m_Writer.String(MEMORY);
    m_Writer.Int64(item.s_Memory);
    if (item.s_Unused) {
        m_Writer.String(UNUSED);
        m_Writer.Uint64(item.s_Unused);
    }
    m_Writer.EndObject();
}

void CMemoryUsageJsonWriter::finalise() {
    if (m_Finalised) {
        return;
    }
    m_WriteStream.Flush();
    m_Finalised = true;
}


} // core
} // ml
