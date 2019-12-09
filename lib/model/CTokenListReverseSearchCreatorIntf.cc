/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CTokenListReverseSearchCreatorIntf.h>

#include <core/CMemory.h>

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

void CTokenListReverseSearchCreatorIntf::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CTokenListReverseSearchCreatorIntf");
    core::CMemoryDebug::dynamicSize("m_FieldName", m_FieldName, mem);
}

std::size_t CTokenListReverseSearchCreatorIntf::memoryUsage() const {
    std::size_t mem = 0;
    mem += core::CMemory::dynamicSize(m_FieldName);
    return mem;
}
}
}
