/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_model_CTokenListCategory_h
#define INCLUDED_ml_model_CTokenListCategory_h

#include <core/CMemoryUsage.h>

#include <model/ImportExport.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {

//! \brief
//! The data associated with this ML category
//!
//! DESCRIPTION:\n
//! The data associated with this ML category
//!
//! IMPLEMENTATION DECISIONS:\n
//! Base the category on a string
//!
//! Token IDs are stored rather than the actual string tokens.
//! This improves performance.  The CTokenListDataCategorizer object
//! that created this object knows the mappings between the
//! token IDs and string tokens.
//!
class MODEL_EXPORT CTokenListCategory {
public:
    //! Used to associate tokens with weightings:
    //! first -> token ID
    //! second -> weighting
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;

    //! Used for storing token ID sequences
    using TSizeSizePrVec = std::vector<TSizeSizePr>;

    //! Used for storing distinct token IDs mapped to weightings
    using TSizeSizeMap = std::map<std::size_t, std::size_t>;

public:
    //! Create a new category
    CTokenListCategory(bool isDryRun,
                       std::string baseString,
                       std::size_t rawStringLen,
                       const TSizeSizePrVec& baseTokenIds,
                       std::size_t baseWeight,
                       const TSizeSizeMap& uniqueTokenIds);

    //! Constructor used when restoring from XML
    CTokenListCategory(core::CStateRestoreTraverser& traverser);

    //! Add string to this category with a double indicating
    //! how well matched the string is
    bool addString(bool isDryRun,
                   const std::string& str,
                   std::size_t rawStringLen,
                   const TSizeSizePrVec& tokenIds,
                   const TSizeSizeMap& uniqueTokenIds);

    //! Accessors
    const std::string& baseString() const;
    const TSizeSizePrVec& baseTokenIds() const;
    std::size_t baseWeight() const;
    const TSizeSizePrVec& commonUniqueTokenIds() const;
    std::size_t commonUniqueTokenWeight() const;
    std::size_t origUniqueTokenWeight() const;
    std::size_t maxStringLen() const;
    //! \return A pair of indices indicating the beginning and end of the
    //!         common ordered tokens within the base tokens.  Importantly,
    //!         the tokens within the range may not all be common across the
    //!         category.  The consumer of these bounds must check whether each
    //!         base token in the range is common before using it for any
    //!         purpose that relies on commonality (for example creating a
    //!         reverse search).
    TSizeSizePr orderedCommonTokenBounds() const;

    //! What's the longest string we'll consider a match for this category?
    //! Currently simply 10% longer than the longest string we've seen.
    std::size_t maxMatchingStringLen() const;

    //! What is the weight of tokens in a given map that are missing from
    //! this category's common unique tokens?
    std::size_t missingCommonTokenWeight(const TSizeSizeMap& uniqueTokenIds) const;

    //! Is the weight of tokens in the provided container that are missing from
    //! this category's common unique tokens equal to zero?  It is possible to
    //! test:
    //!     if (category.missingCommonTokenWeight(uniqueTokenIds) == 0)
    //! instead of calling this method.  However, this method is much faster
    //! as it can return false as soon as a mismatch occurs.
    //! \param uniqueTokenIds A container of pairs where the first element is
    //!                       a token ID and the container is sorted into
    //!                       ascending token ID order.
    template<typename PAIR_CONTAINER>
    bool isMissingCommonTokenWeightZero(const PAIR_CONTAINER& uniqueTokenIds) const {

        auto testIter = uniqueTokenIds.begin();
        for (const auto& commonItem : m_CommonUniqueTokenIds) {
            testIter = std::find_if(testIter, uniqueTokenIds.end(),
                                    [&commonItem](const auto& testItem) {
                                        return testItem.first >= commonItem.first;
                                    });
            if (testIter == uniqueTokenIds.end() ||
                testIter->first != commonItem.first ||
                testIter->second != commonItem.second) {
                return false;
            }
            ++testIter;
        }

        return true;
    }

    //! Does the reverse search of this category match the reverse search find
    //! everything that the reverse search for another category would find?
    bool matchesSearchForCategory(const CTokenListCategory& other) const;

    //! Does the reverse search of this category match the reverse search that
    //! a category created from the supplied arguments would find?
    template<typename PAIR_CONTAINER>
    bool matchesSearchForCategory(std::size_t otherBaseWeight,
                                  std::size_t otherStringLen,
                                  const PAIR_CONTAINER& otherUniqueTokenIds,
                                  const TSizeSizePrVec& otherBaseTokenIds) const {
        return (m_BaseWeight == 0) == (otherBaseWeight == 0) &&
               this->maxMatchingStringLen() >= otherStringLen &&
               this->isMissingCommonTokenWeightZero(otherUniqueTokenIds) &&
               this->containsCommonInOrderTokensInOrder(otherBaseTokenIds);
    }

    //! Does the supplied token vector contain all our common tokens in the
    //! same order as our base token vector?
    bool containsCommonInOrderTokensInOrder(const TSizeSizePrVec& tokenIds) const;

    //! \return Does the supplied token ID represent a common unique token?
    bool isTokenCommon(std::size_t tokenId) const;

    //! How many matching strings are there?
    std::size_t numMatches() const;

    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Set the cached reverse search
    void cacheReverseSearch(std::string part1, std::string part2);

    //! Have we got a cached reverse search?
    bool hasCachedReverseSearch() const;

    //! Access the cached reverse search
    const std::string& reverseSearchPart1() const;
    const std::string& reverseSearchPart2() const;

    //! Debug the memory used by this category.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this category.
    std::size_t memoryUsage() const;

    //! Returns true if the category has changed recently and resets
    //! the changed flag to false.
    bool isChangedAndReset();

private:
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Remove any token IDs from the common unique token map that aren't
    //! present with the same weight in the new string, and adjust the common
    //! weight accordingly.
    //! \return Was a change made?
    bool updateCommonUniqueTokenIds(const TSizeSizeMap& newUniqueTokenIds);

    //! Adjust the common ordered token indices if there are tokens that
    //! aren't in the same order in the new string, and adjust the common weight
    //! accordingly.
    //! \return Was a change made?
    bool updateOrderedCommonTokenIds(const TSizeSizePrVec& newTokenIds);

private:
    //! The string and tokens we base this category on
    std::string m_BaseString;
    TSizeSizePrVec m_BaseTokenIds;

    //! Cache the total weight of the base tokens
    std::size_t m_BaseWeight = 0;

    //! The original length (i.e. before filtering) of the string this category
    //! was originally based on
    std::size_t m_BaseRawStringLen = 0;

    //! The maximum original length of all the strings that have been
    //! classified as this category.  The original length may be longer than the
    //! length of the strings in passed to the addString() method, because
    //! it will include the date.
    std::size_t m_MaxStringLen = 0;

    //! The index into the base token IDs where the subsequence of tokens that
    //! are in the same order for all strings of this category begins.
    std::size_t m_OrderedCommonTokenBeginIndex = 0;

    //! One past the index into the base token IDs where the subsequence of
    //! tokens that are in the same order for all strings of this category ends.
    std::size_t m_OrderedCommonTokenEndIndex = 0;

    //! The unique token IDs that all strings classified to be this category
    //! contain.  This vector must always be sorted into ascending order.
    TSizeSizePrVec m_CommonUniqueTokenIds;

    //! Cache the weight of the common unique tokens
    std::size_t m_CommonUniqueTokenWeight = 0;

    //! What was the weight of the original unique tokens (i.e. when the category
    //! only represented one string)?  Remembering this means we can ensure
    //! that the degree of commonality doesn't fall below a certain level as
    //! the number of strings classified as this category grows.
    std::size_t m_OrigUniqueTokenWeight = 0;

    //! Number of matched strings
    std::size_t m_NumMatches = 0;

    //! Cache reverse searches to save repeated recalculations
    std::string m_ReverseSearchPart1;
    std::string m_ReverseSearchPart2;

    //! Has the category changed recently
    bool m_Changed = false;
};
}
}

#endif // INCLUDED_ml_model_CTokenListCategory_h
