/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CPerPartitionCategoryIdMapper_h
#define INCLUDED_ml_api_CPerPartitionCategoryIdMapper_h

#include <core/BoostMultiIndex.h>
#include <core/CTriple.h>

#include <api/CCategoryIdMapper.h>
#include <api/ImportExport.h>

namespace ml {
namespace api {

//! \brief
//! Category ID mapper for use with per-partition categorization.
//!
//! DESCRIPTION:\n
//! Maps between global category ID and the tuple
//! (partition field value, local category ID).
//!
//! IMPLEMENTATION DECISIONS:\n
//! This mapper is designed for jobs that do per-partition
//! categorization.
//!
//! A multi-index stores the mapping so that it can work
//! both ways without data duplication.
//!
//! Although we expect global IDs to run continuously from
//! 1 to the maximum category ID, the multi-index uses an
//! ordered tree rather than random access for the global
//! ID index.  This means that if we decide to completely
//! discard partitions in the future then the data structure
//! and persistence mechanism will support this.  For the
//! same reason the highest ever global ID is stored
//! separately instead of being obtained from the end of
//! the global ID index.
//!
//! This class is not thread-safe.  Each object should only
//! be used within a single thread.
//!
class API_EXPORT CPerPartitionCategoryIdMapper : public CCategoryIdMapper {
public:
    //! Map from a categorizer key and category ID local to that categorizer to
    //! a global category ID.  This method is not const, as it will create a
    //! new global ID if one does not exist.
    int globalCategoryIdForLocalCategoryId(const std::string& categorizerKey,
                                           int localCategoryId) override;

    //! Map from a global category ID to the key of the appropriate categorizer.
    //! Returns the empty string if the global category ID is unknown.
    const std::string& categorizerKeyForGlobalCategoryId(int globalCategoryId) const override;

    //! Map from a global category ID to the local category ID of the
    //! appropriate categorizer.  Returns the hard failure code if the
    //! global category ID is unknown.
    int localCategoryIdForGlobalCategoryId(int globalCategoryId) const override;

    //! Create a clone.
    TCategoryIdMapperUPtr clone() const override;

    //! Print enough information to fully describe the mapping in log messages.
    std::string printMapping(const std::string& categorizerKey, int localCategoryId) const override;
    std::string printMapping(int globalCategoryId) const override;

    //! Persist the mapper passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Restore the mapper reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

private:
    //! Key specifiers for the multi-index
    struct SGlobalKey {};
    struct SLocalKey {};

    //! The triples stored in the index are of the form
    //! (global category ID, partition field value, local category ID).
    using TIntStrIntTriple = core::CTriple<int, std::string, int>;

    using TMIndex = boost::multi_index::multi_index_container<
        TIntStrIntTriple,
        boost::multi_index::indexed_by<
            boost::multi_index::ordered_unique<boost::multi_index::tag<SGlobalKey>, BOOST_MULTI_INDEX_MEMBER(TIntStrIntTriple, int, first)>,
            boost::multi_index::hashed_unique<boost::multi_index::tag<SLocalKey>,
                                              boost::multi_index::composite_key<TIntStrIntTriple, BOOST_MULTI_INDEX_MEMBER(TIntStrIntTriple, std::string, second), BOOST_MULTI_INDEX_MEMBER(TIntStrIntTriple, int, third)>>>>;

private:
    //! Print enough information to fully describe the mapping in debug.
    static std::string printMapping(const TIntStrIntTriple& mapping);

private:
    //! Highest previously used global ID.
    int m_HighestGlobalId = 0;

    //! Index of global to local category IDs.
    TMIndex m_Mapper;
};
}
}

#endif // INCLUDED_ml_api_CPerPartitionCategoryIdMapper_h
