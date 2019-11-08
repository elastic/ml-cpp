/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CTokenListReverseSearchCreatorIntf.h>

namespace ml {
namespace model {

CTokenListReverseSearchCreatorIntf::CTokenListReverseSearchCreatorIntf(const std::string& fieldName)
    : m_FieldName(fieldName) {
}

CTokenListReverseSearchCreatorIntf::~CTokenListReverseSearchCreatorIntf() {
}

void CTokenListReverseSearchCreatorIntf::closeStandardSearch(std::string& /*part1*/,
                                                             std::string& /*part2*/) const {
    // Default is to do nothing
}

const std::string& CTokenListReverseSearchCreatorIntf::fieldName() const {
    return m_FieldName;
}
}
}
