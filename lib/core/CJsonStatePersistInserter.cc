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
#include <core/CJsonStatePersistInserter.h>

#include <ostream>

namespace ml {
namespace core {

CJsonStatePersistInserter::CJsonStatePersistInserter(std::ostream& outputStream)
    : m_WriteStream(outputStream), m_Writer(m_WriteStream) {
    m_Writer.StartObject();
}

CJsonStatePersistInserter::~CJsonStatePersistInserter(void) {
    m_Writer.EndObject();
    m_WriteStream.Flush();
}

void CJsonStatePersistInserter::insertValue(const std::string& name, const std::string& value) {
    m_Writer.String(name);
    m_Writer.String(value);
}

void CJsonStatePersistInserter::insertInteger(const std::string& name, size_t value) {
    m_Writer.String(name);
    m_Writer.Uint64(value);
}

void CJsonStatePersistInserter::flush(void) {
    m_WriteStream.Flush();
}

void CJsonStatePersistInserter::newLevel(const std::string& name) {
    m_Writer.String(name);
    m_Writer.StartObject();
}

void CJsonStatePersistInserter::endLevel(void) {
    m_Writer.EndObject();
}
}
}
