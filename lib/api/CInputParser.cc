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
#include <api/CInputParser.h>

namespace ml {
namespace api {

CInputParser::CInputParser() : m_GotFieldNames(false), m_GotData(false) {
}

CInputParser::~CInputParser() {
}

bool CInputParser::gotFieldNames() const {
    return m_GotFieldNames;
}

bool CInputParser::gotData() const {
    return m_GotData;
}

const CInputParser::TStrVec& CInputParser::fieldNames() const {
    return m_FieldNames;
}

void CInputParser::gotFieldNames(bool gotFieldNames) {
    m_GotFieldNames = gotFieldNames;
}

void CInputParser::gotData(bool gotData) {
    m_GotData = gotData;
}

CInputParser::TStrVec& CInputParser::fieldNames() {
    return m_FieldNames;
}
}
}
