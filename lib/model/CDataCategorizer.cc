/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CDataCategorizer.h>

namespace ml {
namespace model {

// Initialise statics
const CDataCategorizer::TStrStrUMap CDataCategorizer::EMPTY_FIELDS;

CDataCategorizer::CDataCategorizer(const std::string& fieldName)
    : m_FieldName(fieldName), m_LastPersistTime(0) {
}

CDataCategorizer::~CDataCategorizer() {
}

int CDataCategorizer::computeCategory(bool isDryRun, const std::string& str, size_t rawStringLen) {
    return this->computeCategory(isDryRun, EMPTY_FIELDS, str, rawStringLen);
}

const std::string& CDataCategorizer::fieldName() const {
    return m_FieldName;
}

core_t::TTime CDataCategorizer::lastPersistTime() const {
    return m_LastPersistTime;
}

void CDataCategorizer::lastPersistTime(core_t::TTime lastPersistTime) {
    m_LastPersistTime = lastPersistTime;
}
}
}
