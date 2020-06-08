/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CPerPartitionCategoryIdMapper_h
#define INCLUDED_ml_api_CPerPartitionCategoryIdMapper_h

#include <api/CCategoryIdMapper.h>
#include <api/ImportExport.h>

#include <map>
#include <vector>

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
//! The mappings are stored in the form of a map from
//! partition field value to a vector of global IDs that is
//! stored ordered by local ID.  Since local ID is vector
//! index plus one in vectors of all local IDs this means
//! that the ID part of the lookup just involves getting the
//! element at the desired index from the vector.
//!
//! The highest ever global ID is stored separately to the
//! main mapping data structure so that if we decide to
//! completely discard partitions in the future then the
//! data structure and persistence mechanism will support this.
//!
//! This class is not thread-safe.  Each object should only
//! be used within a single thread.
//!
class API_EXPORT CPerPartitionCategoryIdMapper : public CCategoryIdMapper {
public:
    //! Map from a categorizer key and category ID local to that categorizer to
    //! a global category ID.  This method is not const, as it will create a
    //! new global ID if one does not exist.
    CGlobalCategoryId map(const std::string& categorizerKey,
                          model::CLocalCategoryId localCategoryId) override;

    //! Create a clone.
    TCategoryIdMapperUPtr clone() const override;

    //! Persist the mapper passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Restore the mapper reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

private:
    //! Key specifiers for the multi-index
    using TGlobalCategoryIdVec = std::vector<CGlobalCategoryId>;
    using TStrGlobalCategoryIdVecMap = std::map<std::string, TGlobalCategoryIdVec>;

private:
    //! Highest previously used global ID.
    int m_HighestGlobalId = 0;

    //! Index of global to local category IDs.  The outer map is keyed on the
    //! partition field value, then the local category IDs are indices into the
    //! vector.  The string pointers inside the global category IDs point to
    //! the strings in the map keys.
    TStrGlobalCategoryIdVecMap m_Mapper;
};
}
}

#endif // INCLUDED_ml_api_CPerPartitionCategoryIdMapper_h
