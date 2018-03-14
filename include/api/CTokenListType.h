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
#ifndef INCLUDED_ml_api_CTokenListType_h
#define INCLUDED_ml_api_CTokenListType_h

#include <api/ImportExport.h>

#include <map>
#include <string>
#include <utility>
#include <vector>


namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace api {

//! \brief
//! The data associated with this Ml 'type'
//!
//! DESCRIPTION:\n
//! The data associated with this Ml 'type'
//!
//! IMPLEMENTATION DECISIONS:\n
//! Base the type on a string
//!
//! Token IDs are stored rather than the actual string tokens.
//! This improves performance.  The CTokenListDataTyper object
//! that created this object knows the mappings between the
//! token IDs and string tokens.
//!
class API_EXPORT CTokenListType {
    public:
        //! Used to associate tokens with weightings:
        //! first -> token ID
        //! second -> weighting
        typedef std::pair<size_t, size_t>      TSizeSizePr;

        //! Used for storing token ID sequences
        typedef std::vector<TSizeSizePr>       TSizeSizePrVec;
        typedef TSizeSizePrVec::iterator       TSizeSizePrVecItr;
        typedef TSizeSizePrVec::const_iterator TSizeSizePrVecCItr;

        //! Used for storing distinct token IDs mapped to weightings
        typedef std::map<size_t, size_t>       TSizeSizeMap;
        typedef TSizeSizeMap::iterator         TSizeSizeMapItr;
        typedef TSizeSizeMap::const_iterator   TSizeSizeMapCItr;

    public:
        //! Create a new type
        CTokenListType(bool isDryRun,
                       const std::string &baseString,
                       size_t rawStringLen,
                       const TSizeSizePrVec &baseTokenIds,
                       size_t baseWeight,
                       const TSizeSizeMap &uniqueTokenIds);

        //! Constructor used when restoring from XML
        CTokenListType(core::CStateRestoreTraverser &traverser);

        //! Add string to this type with a double indicating
        //! how well matched the string is
        bool addString(bool isDryRun,
                       const std::string &str,
                       size_t rawStringLen,
                       const TSizeSizePrVec &tokenIds,
                       const TSizeSizeMap &uniqueTokenIds,
                       double similarity);

        //! Accessors
        const std::string    &baseString(void) const;
        const TSizeSizePrVec &baseTokenIds(void) const;
        size_t               baseWeight(void) const;
        const TSizeSizePrVec &commonUniqueTokenIds(void) const;
        size_t               commonUniqueTokenWeight(void) const;
        size_t               origUniqueTokenWeight(void) const;
        size_t               maxStringLen(void) const;
        size_t               outOfOrderCommonTokenIndex(void) const;

        //! What's the longest string we'll consider a match for this type?
        //! Currently simply 10% longer than the longest string we've seen.
        size_t               maxMatchingStringLen(void) const;

        //! What is the weight of tokens in a given map that are missing from
        //! this type's common unique tokens?
        size_t missingCommonTokenWeight(const TSizeSizeMap &uniqueTokenIds) const;

        //! Is the weight of tokens in a given map that are missing from this
        //! type's common unique tokens equal to zero?  It is possible to test:
        //!     if (type.missingCommonTokenWeight(uniqueTokenIds) == 0)
        //! instead of calling this method.  However, this method is much faster
        //! as it can return false as soon as a mismatch occurs.
        bool isMissingCommonTokenWeightZero(const TSizeSizeMap &uniqueTokenIds) const;

        //! Does the supplied token vector contain all our common tokens in the
        //! same order as our base token vector?
        bool containsCommonTokensInOrder(const TSizeSizePrVec &tokenIds) const;

        //! How many matching strings are there?
        size_t numMatches(void) const;

        //! Persist state by passing information to the supplied inserter
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Attempt to get cached reverse search
        bool cachedReverseSearch(std::string &part1,
                                 std::string &part2) const;

        //! Set the cached reverse search
        void cacheReverseSearch(const std::string &part1,
                                const std::string &part2);

    private:
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

    private:
        //! The string and tokens we base this type on
        std::string    m_BaseString;
        TSizeSizePrVec m_BaseTokenIds;

        //! Cache the total weight of the base tokens
        size_t         m_BaseWeight;

        //! The maximum original length of all the strings that have been
        //! classified as this type.  The original length may be longer than the
        //! length of the strings in passed to the addString() method, because
        //! it will include the date.
        size_t         m_MaxStringLen;

        //! The index into the base token IDs that we should stop at when
        //! generating an ordered regex, because subsequent common token IDs are
        //! not in the same order for all strings of this type.
        size_t         m_OutOfOrderCommonTokenIndex;

        //! The unique token IDs that all strings classified to be this type
        //! contain.  This vector must always be sorted into ascending order.
        TSizeSizePrVec m_CommonUniqueTokenIds;

        //! Cache the weight of the common unique tokens
        size_t         m_CommonUniqueTokenWeight;

        //! What was the weight of the original unique tokens (i.e. when the type
        //! only represented one string)?  Remembering this means we can ensure
        //! that the degree of commonality doesn't fall below a certain level as
        //! the number of strings classified as this type grows.
        size_t      m_OrigUniqueTokenWeight;

        //! Number of matched strings
        size_t      m_NumMatches;

        //! Cache reverse searches to save repeated recalculations
        std::string m_ReverseSearchPart1;
        std::string m_ReverseSearchPart2;
};


}
}

#endif // INCLUDED_ml_api_CTokenListType_h

