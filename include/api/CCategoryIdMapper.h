/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CCategoryIdMapper_h
#define INCLUDED_ml_api_CCategoryIdMapper_h

#include <model/CLocalCategoryId.h>

#include <api/CGlobalCategoryId.h>
#include <api/ImportExport.h>

#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace api {

//! \brief
//! Interface for mappers between local and global category IDs.
//!
//! DESCRIPTION:\n
//! Classes that map between local and global category IDs should
//! derive from this interface.
//!
//! IMPLEMENTATION DECISIONS:\n
//! By default persist/restore is a no-op.  Derived classes must
//! override if required.
//!
class API_EXPORT CCategoryIdMapper {
public:
    using TCategoryIdMapperPtr = std::shared_ptr<CCategoryIdMapper>;

    using TLocalCategoryIdVec = std::vector<model::CLocalCategoryId>;
    using TGlobalCategoryIdVec = std::vector<CGlobalCategoryId>;

public:
    virtual ~CCategoryIdMapper() = default;

    //! Map from a local category ID local to a global category ID.  This method
    //! is not const, as it will create a new global ID if one does not exist.
    virtual CGlobalCategoryId map(model::CLocalCategoryId localCategoryId) = 0;

    //! Map from a vector of local category IDs to a vector of global category
    //! IDs.  This method is not const, as it will create new global IDs if any
    //! that are required do not already exist.
    TGlobalCategoryIdVec mapVec(const TLocalCategoryIdVec& localCategoryIds);

    //! Get the categorizer key for this mapper.
    virtual const std::string& categorizerKey() const = 0;

    //! Create a clone.
    virtual TCategoryIdMapperPtr clone() const = 0;

    //! Persist the mapper passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore the mapper reading state from \p traverser.
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
};
}
}

#endif // INCLUDED_ml_api_CCategoryIdMapper_h
