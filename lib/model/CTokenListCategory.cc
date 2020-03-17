/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CTokenListCategory.h>

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <functional>

namespace ml {
namespace model {

// We use short field names to reduce the state size
namespace {
const std::string BASE_STRING{"a"};
const std::string BASE_TOKEN_ID{"b"};
const std::string BASE_TOKEN_WEIGHT{"c"};
const std::string MAX_STRING_LEN{"d"};
const std::string ORDERED_COMMON_TOKEN_END_INDEX{"e"};
const std::string COMMON_UNIQUE_TOKEN_ID{"f"};
const std::string COMMON_UNIQUE_TOKEN_WEIGHT{"g"};
const std::string ORIG_UNIQUE_TOKEN_WEIGHT{"h"};
const std::string NUM_MATCHES{"i"};
const std::string ORDERED_COMMON_TOKEN_BEGIN_INDEX{"j"};

//! Functor for comparing token IDs that works for both simple token IDs and
//! token ID/weight pairs.
class CTokenIdLess {
public:
    bool operator()(CTokenListCategory::TSizeSizePr lhs, CTokenListCategory::TSizeSizePr rhs) {
        return lhs.first < rhs.first;
    }

    bool operator()(std::size_t lhs, CTokenListCategory::TSizeSizePr rhs) {
        return lhs < rhs.first;
    }

    bool operator()(CTokenListCategory::TSizeSizePr lhs, std::size_t rhs) {
        return lhs.first < rhs;
    }

    bool operator()(std::size_t lhs, std::size_t rhs) { return lhs < rhs; }
};
}

CTokenListCategory::CTokenListCategory(bool isDryRun,
                                       const std::string& baseString,
                                       std::size_t rawStringLen,
                                       const TSizeSizePrVec& baseTokenIds,
                                       std::size_t baseWeight,
                                       const TSizeSizeMap& uniqueTokenIds)
    : m_BaseString{baseString}, m_BaseTokenIds{baseTokenIds}, m_BaseWeight{baseWeight},
      m_MaxStringLen{rawStringLen}, m_OrderedCommonTokenBeginIndex{0},
      m_OrderedCommonTokenEndIndex{baseTokenIds.size()},
      // Note: m_CommonUniqueTokenIds is required to be in sorted order, and
      // this relies on uniqueTokenIds being in sorted order
      m_CommonUniqueTokenIds{uniqueTokenIds.begin(), uniqueTokenIds.end()},
      m_CommonUniqueTokenWeight{0}, m_OrigUniqueTokenWeight{0}, m_NumMatches{isDryRun ? 0u : 1u},
      m_Changed{!isDryRun} {
    for (auto uniqueTokenId : uniqueTokenIds) {
        m_CommonUniqueTokenWeight += uniqueTokenId.second;
    }
    m_OrigUniqueTokenWeight = m_CommonUniqueTokenWeight;
}

CTokenListCategory::CTokenListCategory(core::CStateRestoreTraverser& traverser)
    : m_BaseWeight{0}, m_MaxStringLen{0}, m_OrderedCommonTokenBeginIndex{0}, m_OrderedCommonTokenEndIndex{0},
      m_CommonUniqueTokenWeight{0}, m_OrigUniqueTokenWeight{0}, m_NumMatches{0} {
    traverser.traverseSubLevel(std::bind(&CTokenListCategory::acceptRestoreTraverser,
                                         this, std::placeholders::_1));
}

bool CTokenListCategory::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    bool expectWeight{false};

    // This won't be present in pre-7.7 state,
    // and for such versions it was always 0
    m_OrderedCommonTokenBeginIndex = 0;

    do {
        const std::string& name{traverser.name()};
        if (name == BASE_STRING) {
            m_BaseString = traverser.value();
        } else if (name == BASE_TOKEN_ID) {
            TSizeSizePr tokenAndWeight{0, 0};
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
        } else if (name == ORDERED_COMMON_TOKEN_BEGIN_INDEX) {
            if (core::CStringUtils::stringToType(
                    traverser.value(), m_OrderedCommonTokenBeginIndex) == false) {
                LOG_ERROR(<< "Invalid ordered common token start index in "
                          << traverser.value());
                return false;
            }
        } else if (name == ORDERED_COMMON_TOKEN_END_INDEX) {
            if (core::CStringUtils::stringToType(
                    traverser.value(), m_OrderedCommonTokenEndIndex) == false) {
                LOG_ERROR(<< "Invalid ordered common token end index in "
                          << traverser.value());
                return false;
            }
        } else if (name == COMMON_UNIQUE_TOKEN_ID) {
            TSizeSizePr tokenAndWeight{0, 0};
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
                                   std::size_t rawStringLen,
                                   const TSizeSizePrVec& tokenIds,
                                   const TSizeSizeMap& uniqueTokenIds) {

    // Remove any token IDs from the common unique token map that aren't present
    // with the same weight in the new string, and adjust the common weight
    // accordingly.
    bool changed{this->updateCommonUniqueTokenIds(uniqueTokenIds)};

    // Adjust the common ordered token indices if there are tokens that
    // aren't in the same order in the new string, and adjust the common weight
    // accordingly.
    changed = this->updateOrderedCommonTokenIds(tokenIds) || changed;

    // Adjust the maximum observed string length for this category.
    if (rawStringLen > m_MaxStringLen) {
        m_MaxStringLen = rawStringLen;
        changed = true;
    }

    // Changes up to this point invalidate the cached reverse search, whereas
    // simply incrementing the number of matches doesn't.
    if (changed) {
        m_ReverseSearchPart1.clear();
        m_ReverseSearchPart2.clear();
    }

    if (!isDryRun) {
        ++m_NumMatches;
        changed = true;
    }
    if (changed) {
        m_Changed = true;
    }

    return changed;
}

bool CTokenListCategory::updateCommonUniqueTokenIds(const TSizeSizeMap& newUniqueTokenIds) {

    bool changed{false};

    auto commonIter = m_CommonUniqueTokenIds.begin();
    auto newIter = newUniqueTokenIds.begin();
    while (commonIter != m_CommonUniqueTokenIds.end()) {
        if (newIter == newUniqueTokenIds.end() || commonIter->first < newIter->first) {
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
    return changed;
}

// NB: This private method makes the assumption that
// updateCommonUniqueTokenIds() was called before it in the update sequence.
bool CTokenListCategory::updateOrderedCommonTokenIds(const TSizeSizePrVec& newTokenIds) {

    bool changed{false};

    // Start by adjusting the start and end of the range to exclude any tokens
    // which are no longer common.
    while (m_OrderedCommonTokenEndIndex > m_OrderedCommonTokenBeginIndex &&
           this->isTokenCommon(m_BaseTokenIds[m_OrderedCommonTokenEndIndex - 1].first) == false) {
        --m_OrderedCommonTokenEndIndex;
        changed = true;
    }
    while (m_OrderedCommonTokenBeginIndex < m_OrderedCommonTokenEndIndex &&
           this->isTokenCommon(m_BaseTokenIds[m_OrderedCommonTokenBeginIndex].first) == false) {
        ++m_OrderedCommonTokenBeginIndex;
        changed = true;
    }

    // If the common tokens between the new tokens and the base tokens are in a
    // different order then the commonly ordered subset needs to be reduced.
    // The objectives of this process are:
    // 1. In the (likely) case where no adjustment is needed, determine this
    //    as quickly as possible.
    // 2. In the case where adjustment is needed, pick the longest subset of
    //    previously ordered common tokens that have the same order in the new
    //    tokens.
    // The algorithm used here is technically O(N^3 * log(N)), but has these
    // redeeming features:
    // 1. It does not allocate any memory.
    // 2. It is order O(N * log(N)) in the (likely) case of no change required.
    // 3. In the case where the nested loops run many times it will be reducing
    //    the value of N for subsequent calls, thus making those subsequent
    //    calls much faster.  For example, suppose we start off with 100 ordered
    //    tokens, and one call to this method runs the outer loop 50 times to
    //    reduce the ordered set to 50 tokens.  Then the next call will start
    //    with only 50 ordered tokens.
    // There's a trade-off here, because reducing the complexity would mean
    // changing the data structures, which would add complexity to state
    // persistence, or allocating temporary storage, which would increase the
    // runtime a lot in the common case where the previously ordered common
    // tokens are present in the same order in the new tokens.

    std::size_t bestOrderedCommonTokenBeginIndex{m_OrderedCommonTokenEndIndex};
    std::size_t bestOrderedCommonTokenEndIndex{m_OrderedCommonTokenEndIndex};

    // Iterate over the possible starting positions within the current ordered
    // tokens.
    for (std::size_t tryOrderedCommonTokenBeginIndex = m_OrderedCommonTokenBeginIndex;
         tryOrderedCommonTokenBeginIndex < m_OrderedCommonTokenEndIndex;
         ++tryOrderedCommonTokenBeginIndex) {

        std::size_t newIndex{0};
        for (std::size_t commonIndex = tryOrderedCommonTokenBeginIndex;
             commonIndex < m_OrderedCommonTokenEndIndex; ++commonIndex) {

            // Ignore tokens that are not in the common unique tokens
            if (this->isTokenCommon(m_BaseTokenIds[commonIndex].first) == false) {
                continue;
            }

            // Skip tokens in the test tokens until we find one that matches the
            // current base token.  If we reach the end of the test tokens while
            // doing this it means the new tokens don't contain the base tokens
            // in the same order.
            while (newIndex < newTokenIds.size() &&
                   newTokenIds[newIndex].first != m_BaseTokenIds[commonIndex].first) {
                ++newIndex;
            }
            if (newIndex == newTokenIds.size()) {
                // Record the bounds of the matched subset if it's better than
                // what we've previously seen
                if (commonIndex - tryOrderedCommonTokenBeginIndex >
                    bestOrderedCommonTokenEndIndex - bestOrderedCommonTokenBeginIndex) {
                    bestOrderedCommonTokenBeginIndex = tryOrderedCommonTokenBeginIndex;
                    bestOrderedCommonTokenEndIndex = commonIndex;
                }
                break;
            }
        }
        if (newIndex < newTokenIds.size()) {
            // With this try at the begin index we got the best possible match
            // given the starting point, but that might not be best overall.
            if (m_OrderedCommonTokenEndIndex - tryOrderedCommonTokenBeginIndex >
                bestOrderedCommonTokenEndIndex - bestOrderedCommonTokenBeginIndex) {
                bestOrderedCommonTokenBeginIndex = tryOrderedCommonTokenBeginIndex;
                bestOrderedCommonTokenEndIndex = m_OrderedCommonTokenEndIndex;
            }
            // We cannot do better by incrementing the starting token, because
            // for the current starting token we got the longest possible match,
            // so stop here
            break;
        }
    }

    if (m_OrderedCommonTokenBeginIndex != bestOrderedCommonTokenBeginIndex) {
        m_OrderedCommonTokenBeginIndex = bestOrderedCommonTokenBeginIndex;
        changed = true;
    }
    if (m_OrderedCommonTokenEndIndex != bestOrderedCommonTokenEndIndex) {
        m_OrderedCommonTokenEndIndex = bestOrderedCommonTokenEndIndex;
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

std::size_t CTokenListCategory::baseWeight() const {
    return m_BaseWeight;
}

const CTokenListCategory::TSizeSizePrVec& CTokenListCategory::commonUniqueTokenIds() const {
    return m_CommonUniqueTokenIds;
}

std::size_t CTokenListCategory::commonUniqueTokenWeight() const {
    return m_CommonUniqueTokenWeight;
}

std::size_t CTokenListCategory::origUniqueTokenWeight() const {
    return m_OrigUniqueTokenWeight;
}

std::size_t CTokenListCategory::maxStringLen() const {
    return m_MaxStringLen;
}

CTokenListCategory::TSizeSizePr CTokenListCategory::orderedCommonTokenBounds() const {
    return {m_OrderedCommonTokenBeginIndex, m_OrderedCommonTokenEndIndex};
}

std::size_t CTokenListCategory::maxMatchingStringLen() const {
    // Add a 10% margin of error
    return (m_MaxStringLen * 11) / 10;
}

std::size_t CTokenListCategory::missingCommonTokenWeight(const TSizeSizeMap& uniqueTokenIds) const {
    std::size_t presentWeight{0};

    auto commonIter = m_CommonUniqueTokenIds.begin();
    auto testIter = uniqueTokenIds.begin();
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

bool CTokenListCategory::containsCommonInOrderTokensInOrder(const TSizeSizePrVec& tokenIds) const {

    auto testIter = tokenIds.begin();
    for (std::size_t index = m_OrderedCommonTokenBeginIndex;
         index < m_OrderedCommonTokenEndIndex; ++index) {
        std::size_t baseTokenId{m_BaseTokenIds[index].first};

        // Ignore tokens that are not in the common unique tokens
        if (this->isTokenCommon(baseTokenId) == false) {
            continue;
        }

        // Skip tokens in the test tokens until we find one that matches the
        // base token.  If we reach the end of the test tokens whilst doing
        // this, it means the test tokens don't contain the common ordered base
        // tokens in the correct order.
        do {
            if (testIter == tokenIds.end()) {
                return false;
            }
        } while ((testIter++)->first != baseTokenId);
    }

    return true;
}

bool CTokenListCategory::isTokenCommon(std::size_t tokenId) const {
    return std::binary_search(m_CommonUniqueTokenIds.begin(),
                              m_CommonUniqueTokenIds.end(), tokenId, CTokenIdLess());
}

std::size_t CTokenListCategory::numMatches() const {
    return m_NumMatches;
}

void CTokenListCategory::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(BASE_STRING, m_BaseString);

    for (auto baseTokenId : m_BaseTokenIds) {
        inserter.insertValue(BASE_TOKEN_ID, baseTokenId.first);
        inserter.insertValue(BASE_TOKEN_WEIGHT, baseTokenId.second);
    }

    inserter.insertValue(MAX_STRING_LEN, m_MaxStringLen);
    inserter.insertValue(ORDERED_COMMON_TOKEN_BEGIN_INDEX, m_OrderedCommonTokenBeginIndex);
    inserter.insertValue(ORDERED_COMMON_TOKEN_END_INDEX, m_OrderedCommonTokenEndIndex);

    for (auto commonUniqueTokenId : m_CommonUniqueTokenIds) {
        inserter.insertValue(COMMON_UNIQUE_TOKEN_ID, commonUniqueTokenId.first);
        inserter.insertValue(COMMON_UNIQUE_TOKEN_WEIGHT, commonUniqueTokenId.second);
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

void CTokenListCategory::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTokenListCategory");
    core::CMemoryDebug::dynamicSize("m_BaseString", m_BaseString, mem);
    core::CMemoryDebug::dynamicSize("m_BaseTokenIds", m_BaseTokenIds, mem);
    core::CMemoryDebug::dynamicSize("m_CommonUniqueTokenIds", m_CommonUniqueTokenIds, mem);
    core::CMemoryDebug::dynamicSize("m_ReverseSearchPart1", m_ReverseSearchPart1, mem);
    core::CMemoryDebug::dynamicSize("m_ReverseSearchPart2", m_ReverseSearchPart2, mem);
}

std::size_t CTokenListCategory::memoryUsage() const {
    std::size_t mem = 0;
    mem += core::CMemory::dynamicSize(m_BaseString);
    mem += core::CMemory::dynamicSize(m_BaseTokenIds);
    mem += core::CMemory::dynamicSize(m_CommonUniqueTokenIds);
    mem += core::CMemory::dynamicSize(m_ReverseSearchPart1);
    mem += core::CMemory::dynamicSize(m_ReverseSearchPart2);
    return mem;
}

bool CTokenListCategory::isChangedAndReset() {
    if (m_Changed) {
        m_Changed = false;
        return true;
    }
    return false;
}
}
}
