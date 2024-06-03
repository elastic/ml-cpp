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
#include <core/CJsonStatePersistInserter.h>

#include <ostream>

namespace ml {
namespace core {

CJsonStatePersistInserter::CJsonStatePersistInserter(std::ostream& outputStream)
    : m_WriteStream(outputStream), m_Writer(m_WriteStream) {
    m_Writer.onObjectBegin();
}

CJsonStatePersistInserter::~CJsonStatePersistInserter() {
    m_Writer.onObjectEnd();
    m_WriteStream.flush();
}

void CJsonStatePersistInserter::insertValue(const std::string& name, const std::string& value) {
    m_Writer.onKey(name);
    m_Writer.onString(value);
}

void CJsonStatePersistInserter::insertInteger(const std::string& name, size_t value) {
    m_Writer.onKey(name);
    m_Writer.onUint64(value);
}

void CJsonStatePersistInserter::flush() {
    m_WriteStream.flush();
}

void CJsonStatePersistInserter::newLevel(const std::string& name) {
    m_Writer.onKey(name);
    m_Writer.onObjectBegin();
}

void CJsonStatePersistInserter::endLevel() {
    m_Writer.onObjectEnd();
}
}
}
