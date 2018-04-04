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

#include <core/CFlatPrefixTree.h>

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CStringUtils.h>

#include <core/CContainerPrinter.h>

#include <boost/algorithm/cxx11/is_sorted.hpp>

#include <algorithm>
#include <limits>

namespace ml {
namespace core {

namespace {
const uint32_t NO_CHILD = std::numeric_limits<uint32_t>::max();
const char PADDING_NODE = '$';
const char LEAF_NODE = 'l';
const char BRANCH_NODE = 'b';
const char LEAF_AND_BRANCH_NODE = '*';
const std::string EMPTY_STRING = "";

struct SCharNotEqualTo {
    SCharNotEqualTo(char c, std::size_t pos) : s_Char(c), s_Pos(pos) {}

    bool operator()(const std::string& str) { return str[s_Pos] != s_Char; }

    char s_Char;
    std::size_t s_Pos;
};
}

CFlatPrefixTree::SNode::SNode(char c, char type, uint32_t next) : s_Char(c), s_Type(type), s_Next(next) {
}

bool CFlatPrefixTree::SNode::operator<(char rhs) const {
    return s_Char < rhs;
}

CFlatPrefixTree::SDistinctChar::SDistinctChar(char c, char type, std::size_t start, std::size_t end)
    : s_Char(c), s_Type(type), s_Start(start), s_End(end) {
}

CFlatPrefixTree::CFlatPrefixTree() : m_FlatTree() {
}

bool CFlatPrefixTree::build(const TStrVec& prefixes) {
    m_FlatTree.clear();

    if (boost::algorithm::is_sorted(prefixes) == false) {
        LOG_ERROR("FlatPrefixTree cannot be build from an unsorted vector of prefixes");
        return false;
    }

    if (prefixes.size() > 1) {
        for (std::size_t i = 0; i < prefixes.size() - 1; ++i) {
            if (prefixes[i] == prefixes[i + 1]) {
                LOG_ERROR("FlatPrefixTree cannot be build from a vector containing duplicate prefixes: " << prefixes[i]);
                return false;
            }
        }
    }

    if (prefixes.empty() == false) {
        // Ignore empty string if present
        std::size_t startIndex = prefixes[0] == EMPTY_STRING ? 1 : 0;
        this->buildRecursively(prefixes, startIndex, prefixes.size(), 0);
    }

    if (m_FlatTree.size() >= NO_CHILD) {
        LOG_ERROR("Failed to build the tree: " << m_FlatTree.size() << " nodes were required; no more than " << NO_CHILD
                                               << " are supported.");
        m_FlatTree.clear();
        return false;
    }

    LOG_TRACE("Tree = " << this->print());
    return true;
}

void CFlatPrefixTree::buildRecursively(const TStrVec& prefixes, std::size_t prefixesStart, std::size_t prefixesEnd, std::size_t charPos) {
    // First, we extract the distinct characters for the current character position and we
    // record their start/end indices in the prefixes vector.
    TDistinctCharVec distinctCharsWithRange;
    distinctCharsWithRange.reserve(256);
    this->extractDistinctCharacters(prefixes, prefixesStart, prefixesEnd, charPos, distinctCharsWithRange);

    // Now, we create the nodes of the current level: the padding node, that contains
    // the number of distinct characters, and a node for each distinct character.
    m_FlatTree.push_back(SNode(PADDING_NODE, PADDING_NODE, static_cast<uint32_t>(distinctCharsWithRange.size())));
    std::size_t treeSizeBeforeNewChars = m_FlatTree.size();
    for (std::size_t i = 0; i < distinctCharsWithRange.size(); ++i) {
        SDistinctChar& distinctChar = distinctCharsWithRange[i];
        m_FlatTree.push_back(SNode(distinctChar.s_Char, distinctChar.s_Type, NO_CHILD));
    }

    // Finally, for the nodes that have children, we set their next child index to the current
    // tree size and we recurse.
    for (std::size_t i = 0; i < distinctCharsWithRange.size(); ++i) {
        SDistinctChar& distinctChar = distinctCharsWithRange[i];
        if (distinctChar.s_Type != LEAF_NODE) {
            m_FlatTree[treeSizeBeforeNewChars + i].s_Next = static_cast<uint32_t>(m_FlatTree.size());
            this->buildRecursively(prefixes, distinctChar.s_Start, distinctChar.s_End, charPos + 1);
        }
    }
}

void CFlatPrefixTree::extractDistinctCharacters(const TStrVec& prefixes,
                                                std::size_t prefixesStart,
                                                std::size_t prefixesEnd,
                                                std::size_t charPos,
                                                TDistinctCharVec& distinctChars) {
    TStrVecCItr pos = prefixes.begin() + prefixesStart;
    TStrVecCItr end = prefixes.begin() + prefixesEnd;
    while (pos != end) {
        char leadingChar = (*pos)[charPos];
        TStrVecCItr next = std::find_if(pos, end, SCharNotEqualTo(leadingChar, charPos));
        std::size_t startIndex = pos - prefixes.begin();
        std::size_t endIndex = next - prefixes.begin();
        char type = charPos + 1 == prefixes[startIndex].length() ? LEAF_NODE : BRANCH_NODE;
        if (type == LEAF_NODE && endIndex - startIndex > 1) {
            type = LEAF_AND_BRANCH_NODE;
            ++startIndex;
        }
        distinctChars.push_back(SDistinctChar(leadingChar, type, startIndex, endIndex));

        pos = next;
    }
}

bool CFlatPrefixTree::matchesStart(const std::string& key) const {
    return this->matches(key.begin(), key.end(), false);
}

bool CFlatPrefixTree::matchesFully(const std::string& key) const {
    return this->matches(key.begin(), key.end(), true);
}

bool CFlatPrefixTree::matchesStart(TStrCItr start, TStrCItr end) const {
    return this->matches(start, end, false);
}

bool CFlatPrefixTree::matchesFully(TStrCItr start, TStrCItr end) const {
    return this->matches(start, end, true);
}

bool CFlatPrefixTree::matchesStart(TStrCRItr start, TStrCRItr end) const {
    return this->matches(start, end, false);
}

bool CFlatPrefixTree::matchesFully(TStrCRItr start, TStrCRItr end) const {
    return this->matches(start, end, true);
}

template<typename ITR>
bool CFlatPrefixTree::matches(ITR start, ITR end, bool requireFullMatch) const {
    if (m_FlatTree.empty() || start == end) {
        return false;
    }

    ITR currentStringPos = start;
    std::size_t currentTreeIndex = 0;
    TNodeVecCItr levelStart;
    TNodeVecCItr levelEnd;
    char currentChar;
    char lastMatchedType = BRANCH_NODE;
    while (currentStringPos < end && currentTreeIndex != NO_CHILD) {
        levelStart = m_FlatTree.begin() + currentTreeIndex + 1;
        levelEnd = levelStart + m_FlatTree[currentTreeIndex].s_Next;
        currentChar = *currentStringPos;
        TNodeVecCItr searchResult = std::lower_bound(levelStart, levelEnd, currentChar);
        if (searchResult == levelEnd || searchResult->s_Char != currentChar) {
            break;
        }
        ++currentStringPos;
        currentTreeIndex = searchResult->s_Next;
        lastMatchedType = searchResult->s_Type;
        if (requireFullMatch == false && lastMatchedType != BRANCH_NODE) {
            break;
        }
    }
    if (lastMatchedType != BRANCH_NODE) {
        return requireFullMatch ? currentStringPos == end : true;
    }
    return false;
}

void CFlatPrefixTree::clear() {
    m_FlatTree.clear();
}

std::string CFlatPrefixTree::print() const {
    std::string result;
    result += "[";
    for (std::size_t i = 0; i < m_FlatTree.size(); ++i) {
        result += "(";
        result += m_FlatTree[i].s_Char;
        result += ", ";
        result += m_FlatTree[i].s_Type;
        result += ", ";
        result += CStringUtils::typeToString(m_FlatTree[i].s_Next);
        result += ") ";
    }
    result += "]";
    return result;
}
}
}
