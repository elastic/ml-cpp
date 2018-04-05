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
#include <api/CDataTyper.h>

namespace ml {
namespace api {

// Initialise statics
const CDataTyper::TStrStrUMap CDataTyper::EMPTY_FIELDS;

CDataTyper::CDataTyper(const std::string& fieldName) : m_FieldName(fieldName), m_LastPersistTime(0) {
}

CDataTyper::~CDataTyper() {
}

int CDataTyper::computeType(bool isDryRun, const std::string& str, size_t rawStringLen) {
    return this->computeType(isDryRun, EMPTY_FIELDS, str, rawStringLen);
}

const std::string& CDataTyper::fieldName() const {
    return m_FieldName;
}

core_t::TTime CDataTyper::lastPersistTime() const {
    return m_LastPersistTime;
}

void CDataTyper::lastPersistTime(core_t::TTime lastPersistTime) {
    m_LastPersistTime = lastPersistTime;
}
}
}
