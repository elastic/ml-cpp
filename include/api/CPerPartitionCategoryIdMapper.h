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
#ifndef INCLUDED_ml_api_CPerPartitionCategoryIdMapper_h
#define INCLUDED_ml_api_CPerPartitionCategoryIdMapper_h

#include <api/CCategoryIdMapper.h>
#include <api/ImportExport.h>

#include <functional>
#include <vector>

namespace ml {
namespace api {

//! \brief
//! Category ID mapper for use with per-partition categorization.
//!
//! DESCRIPTION:\n
//! Maps between local category ID and global category ID.
//!
//! Each partition will have a separate instance of this class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This mapper is designed for jobs that do per-partition
//! categorization.
//!
//! The mappings are stored in a vector of global IDs that is
//! stored ordered by local ID.  Since local ID is vector
//! index plus one in vectors of all local IDs this means that
//! the lookup just involves getting the element at the
//! desired index from the vector.
//!
//! The highest ever global ID is obtained from a supplied
//! function, as it needs to increment across all partitions,
//! not just for the partition one object of this class is
//! mapping for.
//!
//! This class is not thread-safe.  Each object should only
//! be used within a single thread.
//!
class API_EXPORT CPerPartitionCategoryIdMapper : public CCategoryIdMapper {
public:
    using TNextGlobalIdSupplier = std::function<int()>;

public:
    CPerPartitionCategoryIdMapper(std::string categorizerKey,
                                  TNextGlobalIdSupplier nextGlobalIdSupplier);

    //! Map from a local category ID local to a global category ID.  This method
    //! is not const, as it will create a new global ID if one does not exist.
    CGlobalCategoryId map(model::CLocalCategoryId localCategoryId) override;

    //! Get the categorizer key for this mapper.
    const std::string& categorizerKey() const override;

    //! Create a clone.
    TCategoryIdMapperPtr clone() const override;

    //! Persist the mapper passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Restore the mapper reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

private:
    //! Key specifiers for the multi-index
    using TGlobalCategoryIdVec = std::vector<CGlobalCategoryId>;

private:
    //! Highest previously used global ID.
    const std::string m_CategorizerKey;

    //! Supplier for next global ID.
    TNextGlobalIdSupplier m_NextGlobalIdSupplier;

    //! Index of local to global category IDs.  Each global ID is at the index
    //! of the index() method of the corresponding local ID.  The string
    //! pointers inside the global category IDs point to m_CategorizerKey.
    TGlobalCategoryIdVec m_Mappings;
};
}
}

#endif // INCLUDED_ml_api_CPerPartitionCategoryIdMapper_h
