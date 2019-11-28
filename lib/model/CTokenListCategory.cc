/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CTokenListCategory.h>

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <algorithm>
#include <functional>

namespace ml {
namespace model {

// We use short field names to reduce the state size
namespace {
const std::string BASE_STRING("a");
const std::string BASE_TOKEN_ID("b");
const std::string BASE_TOKEN_WEIGHT("c");
const std::string MAX_STRING_LEN("d");
const std::string OUT_OF_ORDER_COMMON_TOKEN_INDEX("e");
const std::string COMMON_UNIQUE_TOKEN_ID("f");
const std::string COMMON_UNIQUE_TOKEN_WEIGHT("g");
const std::string ORIG_UNIQUE_TOKEN_WEIGHT("h");
const std::string NUM_MATCHES("i");

const std::string EMPTY_STRING;

//! Functor for comparing just the first element of a pair of sizes
class CSizePairFirstElementLess {
public:
    bool operator()(CTokenListCategory::TSizeSizePr lhs, CTokenListCategory::TSizeSizePr rhs) {
        return lhs.first < rhs.first;
    }
};
}

CTokenListCategory::CTokenListCategory(bool isDryRun,
                                       const std::string& baseString,
                                       size_t rawStringLen,
                                       const TSizeSizePrVec& baseTokenIds,
                                       size_t baseWeight,
                                       const TSizeSizeMap& uniqueTokenIds)
    : m_BaseString(baseString), m_BaseTokenIds(baseTokenIds),
      m_BaseWeight(baseWeight), m_MaxStringLen(rawStringLen),
      m_OutOfOrderCommonTokenIndex(baseTokenIds.size()),
      // Note: m_CommonUniqueTokenIds is required to be in sorted order, and
      // this relies on uniqueTokenIds being in sorted order
      m_CommonUniqueTokenIds(uniqueTokenIds.begin(), uniqueTokenIds.end()),
      m_CommonUniqueTokenWeight(0), m_OrigUniqueTokenWeight(0),
      m_NumMatches(isDryRun ? 0 : 1) {
    for (TSizeSizeMapCItr iter = uniqueTokenIds.begin();
         iter != uniqueTokenIds.end(); ++iter) {
        m_CommonUniqueTokenWeight += iter->second;
    }
    m_OrigUniqueTokenWeight = m_CommonUniqueTokenWeight;
}

CTokenListCategory::CTokenListCategory(core::CStateRestoreTraverser& traverser)
    : m_BaseWeight(0), m_MaxStringLen(0), m_OutOfOrderCommonTokenIndex(0),
      m_CommonUniqueTokenWeight(0), m_OrigUniqueTokenWeight(0), m_NumMatches(0) {
    traverser.traverseSubLevel(std::bind(&CTokenListCategory::acceptRestoreTraverser,
                                         this, std::placeholders::_1));
}

bool CTokenListCategory::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    bool expectWeight(false);

    do {
        const std::string& name = traverser.name();
        if (name == BASE_STRING) {
            m_BaseString = traverser.value();
        } else if (name == BASE_TOKEN_ID) {
            TSizeSizePr tokenAndWeight(0, 0);
            if (core::CStringUtils::stringToType(traverser.value(),
                                                 tokenAndWeight.first) == false) {
                LOG_ERROR(<< "Invalid base token ID in " << traverser.value());
                return false;
            }

            m_BaseTokenIds.push_back(tokenAndWeight);
        } else if (name == BASE_TOKEN_WEIGHT) {
            if (m_BaseTokenIds.empty()) {
                LOG_ERROR(<< "Base token weight precedes base token ID in "
                          << traverser.value());
                return false;
            }

            TSizeSizePr& tokenAndWeight = m_BaseTokenIds.back();
            if (core::CStringUtils::stringToType(traverser.value(),
                                                 tokenAndWeight.second) == false) {
                LOG_ERROR(<< "Invalid base token weight in " << traverser.value());
                return false;
            }

            m_BaseWeight += tokenAndWeight.second;
        } else if (name == MAX_STRING_LEN) {
            if (core::CStringUtils::stringToType(traverser.value(), m_MaxStringLen) == false) {
                LOG_ERROR(<< "Invalid maximum string length in " << traverser.value());
                return false;
            }
        } else if (name == OUT_OF_ORDER_COMMON_TOKEN_INDEX) {
            if (core::CStringUtils::stringToType(
                    traverser.value(), m_OutOfOrderCommonTokenIndex) == false) {
                LOG_ERROR(<< "Invalid maximum string length in " << traverser.value());
                return false;
            }
        } else if (name == COMMON_UNIQUE_TOKEN_ID) {
            TSizeSizePr tokenAndWeight(0, 0);
            if (core::CStringUtils::stringToType(traverser.value(),
                                                 tokenAndWeight.first) == false) {
                LOG_ERROR(<< "Invalid common unique token ID in " << traverser.value());
                return false;
            }

            m_CommonUniqueTokenIds.push_back(tokenAndWeight);
            expectWeight = true;
        } else if (name == COMMON_UNIQUE_TOKEN_WEIGHT) {
            if (!expectWeight) {
                LOG_ERROR(<< "Common unique token weight precedes common unique token ID in "
                          << traverser.value());
                return false;
            }

            TSizeSizePr& tokenAndWeight = m_CommonUniqueTokenIds.back();
            if (core::CStringUtils::stringToType(traverser.value(),
                                                 tokenAndWeight.second) == false) {
                LOG_ERROR(<< "Invalid common unique token weight in "
                          << traverser.value());
                return false;
            }
            expectWeight = false;

            m_CommonUniqueTokenWeight += tokenAndWeight.second;
        } else if (name == ORIG_UNIQUE_TOKEN_WEIGHT) {
            if (core::CStringUtils::stringToType(traverser.value(),
                                                 m_OrigUniqueTokenWeight) == false) {
                LOG_ERROR(<< "Invalid maximum string length in " << traverser.value());
                return false;
            }
        } else if (name == NUM_MATCHES) {
            if (core::CStringUtils::stringToType(traverser.value(), m_NumMatches) == false) {
                LOG_ERROR(<< "Invalid maximum string length in " << traverser.value());
                return false;
            }
        }
    } while (traverser.next());

    return true;
}

bool CTokenListCategory::addString(bool isDryRun,
                                   const std::string& /* str */,
                                   size_t rawStringLen,
                                   const TSizeSizePrVec& tokenIds,
                                   const TSizeSizeMap& uniqueTokenIds,
                                   double /* similarity */) {
    bool changed(false);

    // Remove any token IDs from the common unique token map that aren't present
    // with the same weight in the new string, and adjust the common weight
    // accordingly
    TSizeSizePrVecItr commonIter = m_CommonUniqueTokenIds.begin();
    TSizeSizeMapCItr newIter = uniqueTokenIds.begin();
    while (commonIter != m_CommonUniqueTokenIds.end()) {
        if (newIter == uniqueTokenIds.end() || commonIter->first < newIter->first) {
            m_CommonUniqueTokenWeight -= commonIter->second;
            commonIter = m_CommonUniqueTokenIds.erase(commonIter);
            changed = true;
        } else {
            if (commonIter->first == newIter->first) {
                if (commonIter->second == newIter->second) {
                    ++commonIter;
                } else {
                    m_CommonUniqueTokenWeight -= commonIter->second;
                    commonIter = m_CommonUniqueTokenIds.erase(commonIter);
                    changed = true;
                }
            }
            ++newIter;
        }
    }

    // Reduce the out-of-order common token index if there are tokens that
    // aren't in the same order in the new string, and adjust the common weight
    // accordingly
    TSizeSizePrVecCItr testIter = tokenIds.begin();
    for (size_t index = 0; index < m_OutOfOrderCommonTokenIndex; ++index) {
        // Ignore tokens that are not in the common unique tokens
        if (std::binary_search(m_CommonUniqueTokenIds.begin(),
                               m_CommonUniqueTokenIds.end(), m_BaseTokenIds[index],
                               CSizePairFirstElementLess()) == false) {
            continue;
        }

        // Skip tokens in the test tokens until we find one that matches the
        // base token.  If we reach the end of the test tokens whilst doing
        // this, it means the test tokens don't contain the base tokens in the
        // same order, in which case the out-of-order common token index needs
        // to be reset.
        do {
            if (testIter == tokenIds.end()) {
                m_OutOfOrderCommonTokenIndex = index;
                changed = true;
                break;
            }
        } while ((testIter++)->first != m_BaseTokenIds[index].first);
    }

    if (rawStringLen > m_MaxStringLen) {
        m_MaxStringLen = rawStringLen;
        changed = true;
    }

    // Changes up to this point invalidate the cached reverse search, whereas
    // simply incrementing the number of matches doesn't
    if (changed) {
        m_ReverseSearchPart1.clear();
        m_ReverseSearchPart2.clear();
    }

    if (!isDryRun) {
        ++m_NumMatches;
        changed = true;
    }

    return changed;
}

const std::string& CTokenListCategory::baseString() const {
    return m_BaseString;
}

const CTokenListCategory::TSizeSizePrVec& CTokenListCategory::baseTokenIds() const {
    return m_BaseTokenIds;
}

size_t CTokenListCategory::baseWeight() const {
    return m_BaseWeight;
}

const CTokenListCategory::TSizeSizePrVec& CTokenListCategory::commonUniqueTokenIds() const {
    return m_CommonUniqueTokenIds;
}

size_t CTokenListCategory::commonUniqueTokenWeight() const {
    return m_CommonUniqueTokenWeight;
}

size_t CTokenListCategory::origUniqueTokenWeight() const {
    return m_OrigUniqueTokenWeight;
}

size_t CTokenListCategory::maxStringLen() const {
    return m_MaxStringLen;
}

size_t CTokenListCategory::outOfOrderCommonTokenIndex() const {
    return m_OutOfOrderCommonTokenIndex;
}

size_t CTokenListCategory::maxMatchingStringLen() const {
    // Add a 10% margin of error
    return (m_MaxStringLen * 11) / 10;
}

size_t CTokenListCategory::missingCommonTokenWeight(const TSizeSizeMap& uniqueTokenIds) const {
    size_t presentWeight(0);

    TSizeSizePrVecCItr commonIter = m_CommonUniqueTokenIds.begin();
    TSizeSizeMapCItr testIter = uniqueTokenIds.begin();
    while (commonIter != m_CommonUniqueTokenIds.end() &&
           testIter != uniqueTokenIds.end()) {
        if (commonIter->first == testIter->first) {
            // Don't increment the weight if a given token appears a different
            // number of times in the two strings
            if (commonIter->second == testIter->second) {
                presentWeight += commonIter->second;
            }
            ++commonIter;
            ++testIter;
        } else if (commonIter->first < testIter->first) {
            ++commonIter;
        } else // if (commonIter->first > testIter->first)
        {
            ++testIter;
        }
    }

    // The missing count will be the total weight less the weight of those
    // present.  Doing it this way around means we can break out of the above
    // loop earlier when there's a big mismatch in the two map sizes.
    return m_CommonUniqueTokenWeight - presentWeight;
}

bool CTokenListCategory::isMissingCommonTokenWeightZero(const TSizeSizeMap& uniqueTokenIds) const {
    // This method could be implemented as:
    // return this->missingCommonTokenWeight(uniqueTokenIds) == 0;
    //
    // However, it's much faster to return false as soon as a mismatch occurs

    TSizeSizePrVecCItr commonIter = m_CommonUniqueTokenIds.begin();
    TSizeSizeMapCItr testIter = uniqueTokenIds.begin();
    while (commonIter != m_CommonUniqueTokenIds.end() &&
           testIter != uniqueTokenIds.end()) {
        if (commonIter->first < testIter->first) {
            return false;
        }

        if (commonIter->first == testIter->first) {
            // The tokens must appear the same number of times in the two
            // strings
            if (commonIter->second != testIter->second) {
                return false;
            }
            ++commonIter;
        }

        ++testIter;
    }

    return commonIter == m_CommonUniqueTokenIds.end();
}

bool CTokenListCategory::containsCommonTokensInOrder(const TSizeSizePrVec& tokenIds) const {
    TSizeSizePrVecCItr testIter = tokenIds.begin();
    for (TSizeSizePrVecCItr baseIter = m_BaseTokenIds.begin();
         baseIter != m_BaseTokenIds.end(); ++baseIter) {
        // Ignore tokens that are not in the common unique tokens
        if (std::binary_search(m_CommonUniqueTokenIds.begin(),
                               m_CommonUniqueTokenIds.end(), *baseIter,
                               CSizePairFirstElementLess()) == false) {
            continue;
        }

        // Skip tokens in the test tokens until we find one that matches the
        // base token.  If we reach the end of the test tokens whilst doing
        // this, it means the test tokens don't contain the base tokens in the
        // correct order.
        do {
            if (testIter == tokenIds.end()) {
                return false;
            }
        } while ((testIter++)->first != baseIter->first);
    }

    return true;
}

size_t CTokenListCategory::numMatches() const {
    return m_NumMatches;
}

void CTokenListCategory::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(BASE_STRING, m_BaseString);

    for (TSizeSizePrVecCItr iter = m_BaseTokenIds.begin();
         iter != m_BaseTokenIds.end(); ++iter) {
        inserter.insertValue(BASE_TOKEN_ID, iter->first);
        inserter.insertValue(BASE_TOKEN_WEIGHT, iter->second);
    }

    inserter.insertValue(MAX_STRING_LEN, m_MaxStringLen);
    inserter.insertValue(OUT_OF_ORDER_COMMON_TOKEN_INDEX, m_OutOfOrderCommonTokenIndex);

    for (TSizeSizePrVecCItr iter = m_CommonUniqueTokenIds.begin();
         iter != m_CommonUniqueTokenIds.end(); ++iter) {
        inserter.insertValue(COMMON_UNIQUE_TOKEN_ID, iter->first);
        inserter.insertValue(COMMON_UNIQUE_TOKEN_WEIGHT, iter->second);
    }

    inserter.insertValue(ORIG_UNIQUE_TOKEN_WEIGHT, m_OrigUniqueTokenWeight);
    inserter.insertValue(NUM_MATCHES, m_NumMatches);
}

bool CTokenListCategory::cachedReverseSearch(std::string& part1, std::string& part2) const {
    part1 = m_ReverseSearchPart1;
    part2 = m_ReverseSearchPart2;

    // There's an assumption here that a valid reverse search will not have both
    // parts 1 and 2 being empty strings. If this assumption ceases to be true
    // for any type of reverse search in the future then an extra boolean member
    // should be added to indicate where the cached parts 1 and 2 represent
    // a valid reverse search.
    bool missed(part1.empty() && part2.empty());

    LOG_TRACE(<< "Reverse search cache " << (missed ? "miss" : "hit"));

    return !missed;
}

void CTokenListCategory::cacheReverseSearch(const std::string& part1,
                                            const std::string& part2) {
    m_ReverseSearchPart1 = part1;
    m_ReverseSearchPart2 = part2;
}
}
}
