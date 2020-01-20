/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CTokenListReverseSearchCreator.h>

#include <core/CMemory.h>
#include <core/CRegex.h>

namespace ml {
namespace model {

CTokenListReverseSearchCreator::CTokenListReverseSearchCreator(const std::string& fieldName)
    : m_FieldName(fieldName) {
}

size_t CTokenListReverseSearchCreator::availableCost() const {
    // This is pretty arbitrary, but MUST be less than the maximum length of a
    // field in ES (currently 32766 bytes), and ideally should be quite a lot
    // less as a huge reverse search is pretty unwieldy
    return 10000;
}

size_t CTokenListReverseSearchCreator::costOfToken(const std::string& token,
                                                   size_t numOccurrences) const {
    size_t tokenLength = token.length();
    return (1 + tokenLength + // length of what we add to the terms (part 1)
            3 + tokenLength   // length of what we add to the regex (part 2)
            ) *
           numOccurrences;
}

bool CTokenListReverseSearchCreator::createNullSearch(std::string& part1,
                                                      std::string& part2) const {
    part1.clear();
    part2.clear();
    return true;
}

bool CTokenListReverseSearchCreator::createNoUniqueTokenSearch(int /*categoryId*/,
                                                               const std::string& /*example*/,
                                                               size_t /*maxMatchingStringLen*/,
                                                               std::string& part1,
                                                               std::string& part2) const {
    part1.clear();
    part2.clear();
    return true;
}

void CTokenListReverseSearchCreator::initStandardSearch(int /*categoryId*/,
                                                        const std::string& /*example*/,
                                                        size_t /*maxMatchingStringLen*/,
                                                        std::string& part1,
                                                        std::string& part2) const {
    part1.clear();
    part2.clear();
}

void CTokenListReverseSearchCreator::addInOrderCommonToken(const std::string& token,
                                                           bool first,
                                                           std::string& part1,
                                                           std::string& part2) const {
    if (first) {
        part2 += ".*?";
    } else {
        part1 += ' ';
        part2 += ".+?";
    }
    part1 += token;
    part2 += core::CRegex::escapeRegexSpecial(token);
}

void CTokenListReverseSearchCreator::addOutOfOrderCommonToken(const std::string& token,
                                                              std::string& part1,
                                                              std::string& /*part2*/) const {
    if (part1.empty() == false) {
        part1 += ' ';
    }
    part1 += token;
}

void CTokenListReverseSearchCreator::closeStandardSearch(std::string& /*part1*/,
                                                         std::string& part2) const {
    part2 += ".*";
}

const std::string& CTokenListReverseSearchCreator::fieldName() const {
    return m_FieldName;
}

void CTokenListReverseSearchCreator::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CTokenListReverseSearchCreator");
    core::CMemoryDebug::dynamicSize("m_FieldName", m_FieldName, mem);
}

std::size_t CTokenListReverseSearchCreator::memoryUsage() const {
    std::size_t mem = 0;
    mem += core::CMemory::dynamicSize(m_FieldName);
    return mem;
}
}
}
