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
#ifndef INCLUDED_ml_api_CBaseTokenListDataTyper_h
#define INCLUDED_ml_api_CBaseTokenListDataTyper_h

#include <core/BoostMultiIndex.h>

#include <api/CCsvInputParser.h>
#include <api/CDataTyper.h>
#include <api/CTokenListType.h>
#include <api/ImportExport.h>

#include <boost/shared_ptr.hpp>

#include <iosfwd>
#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

class CBaseTokenListDataTyperTest;

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace api {
class CTokenListReverseSearchCreatorIntf;

//! \brief
//! Abstract base class for categorising strings based on tokens.
//!
//! DESCRIPTION:\n
//! Skeleton of a way to categorise strings by breaking them into tokens
//! and looking at how similar this list of tokens is to those obtained
//! from other strings.  The exact tokenisation algorithm must be
//! provided by a derived class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Similarity must strictly exceed the given threshold to be considered
//! a match.  (This means that a threshold of 1 implies all messages are
//! different, even if they're identical!)
//!
//! This class is not thread safe.  For efficiency, each instance should
//! only be used within a single thread.  Any multi-threaded access must
//! be serialised with an external lock.
//!
//! The reverse search creator is shallow copied on copy construction and
//! assignment, hence should have no state that changes after construction.
//! (If this rule needs to be changed then some strategy for ensuring
//! correct setting of reverse search creator state needs to be added to
//! the copy constructor and assignment operator of this class.)
//!
class API_EXPORT CBaseTokenListDataTyper : public CDataTyper {
public:
    //! Name of the field that contains pre-tokenised tokens (in CSV format)
    //! if available
    static const std::string PRETOKENISED_TOKEN_FIELD;

public:
    //! Shared pointer to reverse search creator that we're will function
    //! after being shallow copied
    using TTokenListReverseSearchCreatorIntfCPtr = boost::shared_ptr<const CTokenListReverseSearchCreatorIntf>;

    //! Used to associate tokens with weightings:
    //! first -> token ID
    //! second -> weighting
    using TSizeSizePr = std::pair<size_t, size_t>;

    //! Used for storing token ID sequences
    using TSizeSizePrVec = std::vector<TSizeSizePr>;

    //! Used for storing distinct token IDs
    using TSizeSizeMap = std::map<size_t, size_t>;

    //! Used for stream output of token IDs translated back to the original
    //! tokens
    struct API_EXPORT SIdTranslater {
        SIdTranslater(const CBaseTokenListDataTyper& typer, const TSizeSizePrVec& tokenIds, char separator);

        const CBaseTokenListDataTyper& s_Typer;
        const TSizeSizePrVec& s_TokenIds;
        char s_Separator;
    };

public:
    //! Create a data typer with threshold for how comparable types are
    //! 0.0 means everything is the same type
    //! 1.0 means things have to match exactly to be the same type
    CBaseTokenListDataTyper(const TTokenListReverseSearchCreatorIntfCPtr& reverseSearchCreator,
                            double threshold,
                            const std::string& fieldName);

    //! Dump stats
    virtual void dumpStats(void) const;

    //! Compute a type from a string.  The raw string length may be longer
    //! than the length of the passed string, because the passed string may
    //! have the date stripped out of it.  Field names/values are available
    //! to the type computation.
    virtual int computeType(bool dryRun, const TStrStrUMap& fields, const std::string& str, size_t rawStringLen);

    // Bring the other overload of computeType() into scope
    using CDataTyper::computeType;

    //! Create a search that will (more or less) just select the records
    //! that are classified as the given type.  Note that the reverse search
    //! is only approximate - it may select more records than have actually
    //! been classified as the returned type.
    virtual bool createReverseSearch(int type, std::string& part1, std::string& part2, size_t& maxMatchingLength, bool& wasCached);

    //! Has the data typer's state changed?
    virtual bool hasChanged(void) const;

    //! Populate the object from part of a state document
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persist state by passing information to the supplied inserter
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Make a function that can be called later to persist state
    virtual TPersistFunc makePersistFunc(void) const;

protected:
    //! Split the string into a list of tokens.  The result of the
    //! tokenisation is returned in \p tokenIds, \p tokenUniqueIds and
    //! \p totalWeight.  Any previous content of these variables is wiped.
    virtual void tokeniseString(const TStrStrUMap& fields,
                                const std::string& str,
                                TSizeSizePrVec& tokenIds,
                                TSizeSizeMap& tokenUniqueIds,
                                size_t& totalWeight) = 0;

    //! Take a string token, convert it to a numeric ID and a weighting and
    //! add these to the provided data structures.
    virtual void
    tokenToIdAndWeight(const std::string& token, TSizeSizePrVec& tokenIds, TSizeSizeMap& tokenUniqueIds, size_t& totalWeight) = 0;

    //! Compute similarity between two vectors
    virtual double similarity(const TSizeSizePrVec& left, size_t leftWeight, const TSizeSizePrVec& right, size_t rightWeight) const = 0;

    //! Used to hold statistics about the types we compute:
    //! first -> count of matches
    //! second -> type vector index
    using TSizeSizePrList = std::list<TSizeSizePr>;
    using TSizeSizePrListItr = TSizeSizePrList::iterator;

    //! Add a match to an existing type
    void addTypeMatch(bool isDryRun,
                      const std::string& str,
                      size_t rawStringLen,
                      const TSizeSizePrVec& tokenIds,
                      const TSizeSizeMap& tokenUniqueIds,
                      double similarity,
                      TSizeSizePrListItr& iter);

    //! Given the total token weight in a vector and a threshold, what is
    //! the minimum possible token weight in a different vector that could
    //! possibly be considered to match?
    static size_t minMatchingWeight(size_t weight, double threshold);

    //! Given the total token weight in a vector and a threshold, what is
    //! maximum possible token weight in a different vector that could
    //! possibly be considered to match?
    static size_t maxMatchingWeight(size_t weight, double threshold);

    //! Get the unique token ID for a given token (assigning one if it's
    //! being seen for the first time)
    size_t idForToken(const std::string& token);

private:
    //! Value type for the TTokenMIndex below
    class CTokenInfoItem {
    public:
        CTokenInfoItem(const std::string& str, size_t index);

        //! Accessors
        const std::string& str(void) const;
        size_t index(void) const;
        size_t typeCount(void) const;
        void typeCount(size_t typeCount);

        //! Increment the type count
        void incTypeCount(void);

    private:
        //! String value of the token
        std::string m_Str;

        //! Index of the token
        size_t m_Index;

        //! How many types use this token?
        size_t m_TypeCount;
    };

    //! Compute equality based on the first element of a pair only
    class CSizePairFirstElementEquals {
    public:
        CSizePairFirstElementEquals(size_t value);

        //! PAIRTYPE can be any struct with a data member named "first"
        //! that can be checked for equality to a size_t
        template<typename PAIRTYPE>
        bool operator()(const PAIRTYPE& lhs) const {
            return lhs.first == m_Value;
        }

    private:
        size_t m_Value;
    };

    //! Used to hold the distinct types we compute (vector reallocations are
    //! not expensive because CTokenListType is movable)
    using TTokenListTypeVec = std::vector<CTokenListType>;

    //! Tag for the token index
    struct SToken {};

    using TTokenMIndex = boost::multi_index::multi_index_container<
        CTokenInfoItem,
        boost::multi_index::indexed_by<
            boost::multi_index::random_access<>,
            boost::multi_index::hashed_unique<boost::multi_index::tag<SToken>,
                                              BOOST_MULTI_INDEX_CONST_TYPE_CONST_MEM_FUN(CTokenInfoItem, std::string, str)>>>;

private:
    //! Used by deferred persistence functions
    static void
    acceptPersistInserter(const TTokenMIndex& tokenIdLookup, const TTokenListTypeVec& types, core::CStatePersistInserter& inserter);

    //! Given a string containing comma separated pre-tokenised input, add
    //! the tokens to the working data structures in the same way as if they
    //! had been determined by the tokeniseString() method.  The result of
    //! the tokenisation is returned in \p tokenIds, \p tokenUniqueIds and
    //! \p totalWeight.  Any previous content of these variables is wiped.
    bool addPretokenisedTokens(const std::string& tokensCsv, TSizeSizePrVec& tokenIds, TSizeSizeMap& tokenUniqueIds, size_t& totalWeight);

private:
    //! Reference to the object we'll use to create reverse searches
    const TTokenListReverseSearchCreatorIntfCPtr m_ReverseSearchCreator;

    //! The lower threshold for comparison.  If another type matches this
    //! closely, we'll take it providing there's no other better match.
    double m_LowerThreshold;

    //! The upper threshold for comparison.  If another type matches this
    //! closely, we accept it immediately (i.e. don't look for a better one).
    double m_UpperThreshold;

    //! Has the data typer's state changed?
    bool m_HasChanged;

    //! The types
    TTokenListTypeVec m_Types;

    //! List of match count/index into type vector in descending order of
    //! match count
    TSizeSizePrList m_TypesByCount;

    //! Used for looking up tokens to a unique ID
    TTokenMIndex m_TokenIdLookup;

    //! Vector to use to build up sequences of token IDs.  This is a member
    //! to save repeated reallocations for different strings.
    TSizeSizePrVec m_WorkTokenIds;

    //! Set to use to build up unique token IDs.  This is a member to save
    //! repeated reallocations for different strings.
    TSizeSizeMap m_WorkTokenUniqueIds;

    //! Used to parse pre-tokenised input supplied as CSV.
    CCsvInputParser::CCsvLineParser m_CsvLineParser;

    // For unit testing
    friend class ::CBaseTokenListDataTyperTest;

    // For ostream output
    friend API_EXPORT std::ostream& operator<<(std::ostream&, const SIdTranslater&);
};

API_EXPORT std::ostream& operator<<(std::ostream& strm, const CBaseTokenListDataTyper::SIdTranslater& translator);
}
}

#endif // INCLUDED_ml_api_CBaseTokenListDataTyper_h
