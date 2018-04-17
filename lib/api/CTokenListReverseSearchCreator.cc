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
#include <api/CTokenListReverseSearchCreator.h>

#include <core/CRegex.h>

namespace ml {
namespace api {

CTokenListReverseSearchCreator::CTokenListReverseSearchCreator(const std::string& fieldName)
    : CTokenListReverseSearchCreatorIntf(fieldName) {
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

bool CTokenListReverseSearchCreator::createNoUniqueTokenSearch(int /*type*/,
                                                               const std::string& /*example*/,
                                                               size_t /*maxMatchingStringLen*/,
                                                               std::string& part1,
                                                               std::string& part2) const {
    part1.clear();
    part2.clear();
    return true;
}

void CTokenListReverseSearchCreator::initStandardSearch(int /*type*/,
                                                        const std::string& /*example*/,
                                                        size_t /*maxMatchingStringLen*/,
                                                        std::string& part1,
                                                        std::string& part2) const {
    part1.clear();
    part2.clear();
}

void CTokenListReverseSearchCreator::addCommonUniqueToken(const std::string& /*token*/,
                                                          std::string& /*part1*/,
                                                          std::string& /*part2*/) const {
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

void CTokenListReverseSearchCreator::closeStandardSearch(std::string& /*part1*/,
                                                         std::string& part2) const {
    part2 += ".*";
}
}
}
