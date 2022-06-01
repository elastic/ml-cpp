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
#ifndef INCLUDED_ml_model_CTokenListDataCategorizerBase_h
#define INCLUDED_ml_model_CTokenListDataCategorizerBase_h

#include <core/BoostMultiIndex.h>
#include <core/CCsvLineParser.h>

#include <model/CDataCategorizer.h>
#include <model/CTokenListCategory.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace CTokenListDataCategorizerBaseTest {
struct testMaxMatchingWeights;
struct testMinMatchingWeights;
}

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CTokenListReverseSearchCreator;

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
class MODEL_EXPORT CTokenListDataCategorizerBase : public CDataCategorizer {
public:
    //! Name of the field that contains pre-tokenised tokens (in CSV format)
    //! if available
    static const std::string PRETOKENISED_TOKEN_FIELD;

public:
    //! Shared pointer to reverse search creator that we're will function
    //! after being shallow copied
    using TTokenListReverseSearchCreatorCPtr =
        std::shared_ptr<const CTokenListReverseSearchCreator>;

    //! Used to associate tokens with weightings:
    //! first -> token ID
    //! second -> weighting
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;

    //! Used for storing token ID sequences and categories with counts
    using TSizeSizePrVec = std::vector<TSizeSizePr>;
    using TSizeSizePrVecItr = TSizeSizePrVec::iterator;
    using TSizeSizePrVecCItr = TSizeSizePrVec::const_iterator;

    //! Used for storing distinct token IDs
    using TSizeSizeMap = std::map<std::size_t, std::size_t>;

    //! Used for stream output of token IDs translated back to the original
    //! tokens
    struct MODEL_EXPORT SIdTranslater {
        SIdTranslater(const CTokenListDataCategorizerBase& categorizer,
                      const TSizeSizePrVec& tokenIds,
                      char separator);

        const CTokenListDataCategorizerBase& s_Categorizer;
        const TSizeSizePrVec& s_TokenIds;
        char s_Separator;
    };

public:
    //! Create a data categorizer with threshold for how comparable categories are
    //! 0.0 means everything is the same category
    //! 1.0 means things have to match exactly to be the same category
    CTokenListDataCategorizerBase(CLimits& limits,
                                  const TTokenListReverseSearchCreatorCPtr& reverseSearchCreator,
                                  double threshold,
                                  const std::string& fieldName);

    //! No copying allowed (because it would complicate the resource monitoring).
    CTokenListDataCategorizerBase(const CTokenListDataCategorizerBase&) = delete;
    CTokenListDataCategorizerBase& operator=(const CTokenListDataCategorizerBase&) = delete;

    //! Dump stats
    void dumpStats(const TLocalCategoryIdFormatterFunc& formatterFunc) const override;

    //! Compute a category from a string.  The raw string length may be longer
    //! than the length of the passed string, because the passed string may
    //! have the date stripped out of it.  Field names/values are available
    //! to the category computation.
    CLocalCategoryId computeCategory(bool dryRun,
                                     const TStrStrUMap& fields,
                                     const std::string& str,
                                     std::size_t rawStringLen) override;

    // Bring the other overload of computeCategory() into scope
    using CDataCategorizer::computeCategory;

    //! Ensure the reverse search information is up-to-date for the specified
    //! category.  Note that the reverse search is only approximate - it may
    //! select more records than have actually been classified as the specified
    //! category.
    //! \return Was the reverse search changed as a result of the call?
    bool cacheReverseSearch(CLocalCategoryId categoryId) override;

    //! Populate the object from part of a state document
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Make a function that can be called later to persist state in the
    //! foreground, i.e. in the knowledge that no other thread will be
    //! accessing the data structures this method accesses.
    TPersistFunc makeForegroundPersistFunc() const override;

    //! Make a function that can be called later to persist state in the
    //! background, i.e. copying any required data such that other threads
    //! may modify the original data structures while persistence is taking
    //! place.
    TPersistFunc makeBackgroundPersistFunc() const override;

    //! Get the most recent categorization status.
    model_t::ECategorizationStatus categorizationStatus() const override;

    //! Debug the memory used by this categorizer.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this categorizer.
    std::size_t memoryUsage() const override;

    //! Update categorizer stats with information from this categorizer.
    void updateCategorizerStats(SCategorizerStats& categorizerStats) const;

    //! Update the model size stats with information from this categorizer.
    void updateModelSizeStats(CResourceMonitor::SModelSizeStats& modelSizeStats) const override;

    //! Categorization status is "warn" if:
    //! - At least 100 messages have been categorized
    //! and one of the following holds:
    //! - There is only 1 category
    //! - More than 90% of categories are rare
    //! - The number of categories is greater than 50% of the number of categorized messages
    //! - There are no frequent match categories
    //! - More than 50% of categories are dead
    static model_t::ECategorizationStatus
    calculateCategorizationStatus(std::size_t categorizedMessages,
                                  std::size_t totalCategories,
                                  std::size_t frequentCategories,
                                  std::size_t rareCategories,
                                  std::size_t deadCategories);

    std::size_t numMatches(CLocalCategoryId categoryId) override;

    //! Get the categories that will never be detected again because the
    //! specified category will always be returned instead.
    TLocalCategoryIdVec usurpedCategories(CLocalCategoryId categoryId) const override;

    //! Writes information about a category using the supplied output function,
    //! if the category has changed since the last time it was written.
    //! \return Was the category written?
    bool writeCategoryIfChanged(CLocalCategoryId categoryId,
                                const TCategoryOutputFunc& outputFunc) override;

    //! Writes information about all categories that have changed since the last
    //! time they were written using the supplied output function.
    //! \return Number of categories written.
    std::size_t writeChangedCategories(const TCategoryOutputFunc& outputFunc) override;

    //! Write the latest categorizer stats using the supplied output function if
    //! they have changed since the last time they were written.
    //! \return Were the stats written?
    bool writeCategorizerStatsIfChanged(const TCategorizerStatsOutputFunc& outputFunc) override;

    //! Quickly check if a stats write is important at this time.  This method
    //! is called regularly, so should not do complex processing.
    bool isStatsWriteUrgent() const override;

    //! Number of categories this categorizer has detected.
    std::size_t numCategories() const override;

protected:
    //! Split the string into a list of tokens.  The result of the
    //! tokenisation is returned in \p tokenIds, \p tokenUniqueIds and
    //! \p totalWeight.  Any previous content of these variables is wiped.
    virtual void tokeniseString(const TStrStrUMap& fields,
                                const std::string& str,
                                TSizeSizePrVec& tokenIds,
                                TSizeSizeMap& tokenUniqueIds,
                                std::size_t& totalWeight,
                                std::size_t& minReweightedTotalWeight,
                                std::size_t& maxReweightedTotalWeight) = 0;

    //! Take a string token, convert it to a numeric ID and a weighting and
    //! add these to the provided data structures.
    virtual void tokenToIdAndWeight(const std::string& token,
                                    TSizeSizePrVec& tokenIds,
                                    TSizeSizeMap& tokenUniqueIds,
                                    std::size_t& totalWeight,
                                    std::size_t& minReweightedTotalWeight,
                                    std::size_t& maxReweightedTotalWeight) = 0;

    virtual void reset() = 0;

    //! Compute similarity between two vectors
    virtual double similarity(const TSizeSizePrVec& left,
                              std::size_t leftWeight,
                              const TSizeSizePrVec& right,
                              std::size_t rightWeight) const = 0;

    //! Add a match to an existing category
    void addCategoryMatch(bool isDryRun,
                          const std::string& str,
                          std::size_t rawStringLen,
                          const TSizeSizePrVec& tokenIds,
                          const TSizeSizeMap& tokenUniqueIds,
                          TSizeSizePrVecItr iter);

    //! Given the total token weight in a vector and a threshold, what is
    //! the minimum possible token weight in a different vector that could
    //! possibly be considered to match?
    static std::size_t minMatchingWeight(std::size_t weight, double threshold);

    //! Given the total token weight in a vector and a threshold, what is
    //! maximum possible token weight in a different vector that could
    //! possibly be considered to match?
    static std::size_t maxMatchingWeight(std::size_t weight, double threshold);

    //! Get the unique token ID for a given token (assigning one if it's
    //! being seen for the first time)
    std::size_t idForToken(const std::string& token);

    //! Is the category considered rare?
    bool isCategoryCountRare(std::size_t count) const;

    //! Is the category considered frequent?
    bool isCategoryCountFrequent(std::size_t count) const;

private:
    //! Value category for the TTokenMIndex below
    class CTokenInfoItem {
    public:
        CTokenInfoItem(const std::string& str, std::size_t index);

        //! Accessors
        const std::string& str() const;
        std::size_t index() const;
        std::size_t categoryCount() const;
        void categoryCount(std::size_t categoryCount);

        //! Increment the category count
        void incCategoryCount();

        //! Debug the memory used by this item.
        void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

        //! Get the memory used by this item.
        std::size_t memoryUsage() const;

    private:
        //! String value of the token
        std::string m_Str;

        //! Index of the token
        std::size_t m_Index;

        //! How many categories use this token?
        std::size_t m_CategoryCount;
    };

    //! Compute equality based on the first element of a pair only
    class CSizePairFirstElementEquals {
    public:
        CSizePairFirstElementEquals(std::size_t value);

        //! PAIRTYPE can be any struct with a data member named "first"
        //! that can be checked for equality to a std::size_t
        template<typename PAIRTYPE>
        bool operator()(const PAIRTYPE& lhs) const {
            return lhs.first == m_Value;
        }

    private:
        std::size_t m_Value;
    };

    //! Used to hold the distinct categories we compute (vector reallocations are
    //! not expensive because CTokenListCategory is movable)
    using TTokenListCategoryVec = std::vector<CTokenListCategory>;

    //! Tag for the token index
    struct SToken {};

    using TTokenMIndex = boost::multi_index::multi_index_container<
        CTokenInfoItem,
        boost::multi_index::indexed_by<boost::multi_index::random_access<>,
                                       boost::multi_index::hashed_unique<boost::multi_index::tag<SToken>, BOOST_MULTI_INDEX_CONST_TYPE_CONST_MEM_FUN(CTokenInfoItem, std::string, str)>>>;

private:
    //! Used by deferred persistence functions
    static void acceptPersistInserter(const TTokenMIndex& tokenIdLookup,
                                      const TTokenListCategoryVec& categories,
                                      std::size_t memoryCategorizationFailures,
                                      core::CStatePersistInserter& inserter);

    //! Given a string containing comma separated pre-tokenised input, add
    //! the tokens to the working data structures in the same way as if they
    //! had been determined by the tokeniseString() method.  The result of
    //! the tokenisation is returned in \p tokenIds, \p tokenUniqueIds and
    //! \p totalWeight.  Any previous content of these variables is wiped.
    bool addPretokenisedTokens(const std::string& tokensCsv,
                               TSizeSizePrVec& tokenIds,
                               TSizeSizeMap& tokenUniqueIds,
                               std::size_t& totalWeight,
                               std::size_t& minReweightedTotalWeight,
                               std::size_t& maxReweightedTotalWeight);

    //! Get the categories that will never be detected again because the
    //! specified category will always be returned instead.  This overload
    //! is only O(N), whereas the public usurpedCategories method is O(N^2).
    //! \param iter An iterator pointing at the element of m_CategoriesByCount
    //!             for which usurped categories are to be found.
    TLocalCategoryIdVec usurpedCategories(TSizeSizePrVecCItr iter) const;

private:
    //! Reference to the object we'll use to create reverse searches
    const TTokenListReverseSearchCreatorCPtr m_ReverseSearchCreator;

    //! The lower threshold for comparison.  If another category matches this
    //! closely, we'll take it providing there's no other better match.
    double m_LowerThreshold;

    //! The upper threshold for comparison.  If another category matches this
    //! closely, we accept it immediately (i.e. don't look for a better one).
    double m_UpperThreshold;

    //! How many messages have we failed to categorize due to lack of memory?
    std::size_t m_MemoryCategorizationFailures = 0;

    //! The categories
    TTokenListCategoryVec m_Categories;

    //! List of match count/index into category vector in descending order of
    //! match count.  Note that the second element is an index into m_Categories,
    //! not a category ID.
    TSizeSizePrVec m_CategoriesByCount;

    //! Sum of all category counts.  Equal to the sum of .first for all elements
    //! of m_CategoriesByCount.
    std::size_t m_TotalCount = 0;

    //! Number of rare categories, as defined by the isCategoryCountRare()
    //! method.
    std::size_t m_NumRareCategories = 0;

    //! Used for looking up tokens to a unique ID
    TTokenMIndex m_TokenIdLookup;

    //! Vector to use to build up sequences of token IDs.  This is a member
    //! to save repeated reallocations for different strings.
    TSizeSizePrVec m_WorkTokenIds;

    //! Set to use to build up unique token IDs.  This is a member to save
    //! repeated reallocations for different strings.
    TSizeSizeMap m_WorkTokenUniqueIds;

    //! Used to parse pre-tokenised input supplied as CSV.
    core::CCsvLineParser m_CsvLineParser;

    //! Last categorizer stats that were written.
    SCategorizerStats m_LastCategorizerStats;

    // For unit testing
    friend struct CTokenListDataCategorizerBaseTest::testMaxMatchingWeights;
    friend struct CTokenListDataCategorizerBaseTest::testMinMatchingWeights;

    // For ostream output
    friend MODEL_EXPORT std::ostream& operator<<(std::ostream&, const SIdTranslater&);
};

MODEL_EXPORT std::ostream&
operator<<(std::ostream& strm, const CTokenListDataCategorizerBase::SIdTranslater& translator);
}
}

#endif // INCLUDED_ml_model_CTokenListDataCategorizerBase_h
