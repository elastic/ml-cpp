/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CCategoryIdMapper_h
#define INCLUDED_ml_api_CCategoryIdMapper_h

#include <api/ImportExport.h>

#include <memory>
#include <string>

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
    using TCategoryIdMapperUPtr = std::unique_ptr<CCategoryIdMapper>;

public:
    virtual ~CCategoryIdMapper() = default;

    //! Map from a categorizer key and category ID local to that categorizer to
    //! a global category ID.  This method is not const, as it will create a
    //! new global ID if one does not exist.
    virtual int globalCategoryIdForLocalCategoryId(const std::string& categorizerKey,
                                                   int localCategoryId) = 0;

    //! Map from a global category ID to the key of the appropriate categorizer.
    virtual const std::string&
    categorizerKeyForGlobalCategoryId(int globalCategoryId) const = 0;

    //! Map from a global category ID to the local category ID of the
    //! appropriate categorizer.
    virtual int localCategoryIdForGlobalCategoryId(int globalCategoryId) const = 0;

    //! Create a clone.
    virtual TCategoryIdMapperUPtr clone() const = 0;

    //! Print enough information to fully describe the mapping in log messages.
    virtual std::string printMapping(const std::string& categorizerKey,
                                     int localCategoryId) const = 0;
    virtual std::string printMapping(int globalCategoryId) const = 0;

    //! Persist the mapper passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore the mapper reading state from \p traverser.
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
};
}
}

#endif // INCLUDED_ml_api_CCategoryIdMapper_h
