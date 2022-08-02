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
#include <model/CTokenListReverseSearchCreator.h>

#include <core/CMemoryDef.h>
#include <core/CRegex.h>

namespace ml {
namespace model {

CTokenListReverseSearchCreator::CTokenListReverseSearchCreator(const std::string& fieldName)
    : m_FieldName(fieldName) {
}

std::size_t CTokenListReverseSearchCreator::availableCost() const {
    // This is pretty arbitrary, but MUST be less than the maximum length of a
    // field in ES (currently 32766 bytes), and ideally should be quite a lot
    // less as a huge reverse search is pretty unwieldy
    return 10000;
}

std::size_t CTokenListReverseSearchCreator::costOfToken(const std::string& token,
                                                        std::size_t numOccurrences) const {
    std::size_t tokenLength{token.length()};
    return (1 + tokenLength + // length of what we add to the terms
            3 + tokenLength   // length of what we add to the regex
            ) *
           numOccurrences;
}

bool CTokenListReverseSearchCreator::createNoUniqueTokenSearch(CLocalCategoryId /*categoryId*/,
                                                               const std::string& /*example*/,
                                                               std::size_t /*maxMatchingStringLen*/,
                                                               std::string& terms,
                                                               std::string& regex) const {
    terms.clear();
    regex = ".*";
    return true;
}

void CTokenListReverseSearchCreator::initStandardSearch(CLocalCategoryId /*categoryId*/,
                                                        const std::string& /*example*/,
                                                        std::size_t /*maxMatchingStringLen*/,
                                                        std::string& terms,
                                                        std::string& regex) const {
    terms.clear();
    regex.clear();
}

void CTokenListReverseSearchCreator::addInOrderCommonToken(const std::string& token,
                                                           std::string& terms,
                                                           std::string& regex) const {
    if (regex.empty()) {
        regex += ".*?";
    } else {
        regex += ".+?";
    }
    if (terms.empty() == false) {
        terms += ' ';
    }
    terms += token;
    regex += core::CRegex::escapeRegexSpecial(token);
}

void CTokenListReverseSearchCreator::addOutOfOrderCommonToken(const std::string& token,
                                                              std::string& terms,
                                                              std::string& /*regex*/) const {
    if (terms.empty() == false) {
        terms += ' ';
    }
    terms += token;
}

void CTokenListReverseSearchCreator::closeStandardSearch(std::string& /*terms*/,
                                                         std::string& regex) const {
    regex += ".*";
}

const std::string& CTokenListReverseSearchCreator::fieldName() const {
    return m_FieldName;
}

void CTokenListReverseSearchCreator::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTokenListReverseSearchCreator");
    core::memory_debug::dynamicSize("m_FieldName", m_FieldName, mem);
}

std::size_t CTokenListReverseSearchCreator::memoryUsage() const {
    return core::memory::dynamicSize(m_FieldName);
}
}
}
