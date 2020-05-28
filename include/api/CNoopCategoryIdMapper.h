/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CNoopCategoryIdMapper_h
#define INCLUDED_ml_api_CNoopCategoryIdMapper_h

#include <api/CCategoryIdMapper.h>
#include <api/ImportExport.h>

namespace ml {
namespace api {

//! \brief
//! Category ID mapper for use when local and global category IDs
//! are the same.
//!
//! DESCRIPTION:\n
//! All mappings are no-ops and the categorizer key for every global
//! category ID is the empty string.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This mapper is designed for the case where there is a single
//! categorizer, which includes all jobs created before 7.9.
//!
class API_EXPORT CNoopCategoryIdMapper : public CCategoryIdMapper {
public:
    //! Map from a categorizer key and category ID local to that categorizer to
    //! a global category ID.  This method is not const, as it will create a
    //! new global ID if one does not exist.
    int globalCategoryIdForLocalCategoryId(const std::string& categorizerKey,
                                           int localCategoryId) override;

    //! Map from a global category ID to the key of the appropriate categorizer.
    const std::string& categorizerKeyForGlobalCategoryId(int globalCategoryId) const override;

    //! Map from a global category ID to the local category ID of the
    //! appropriate categorizer.
    int localCategoryIdForGlobalCategoryId(int globalCategoryId) const override;

    //! Create a clone.
    TCategoryIdMapperUPtr clone() const override;

    //! Print enough information to fully describe the mapping in log messages.
    std::string printMapping(const std::string& categorizerKey, int localCategoryId) const override;
    std::string printMapping(int globalCategoryId) const override;
};
}
}

#endif // INCLUDED_ml_api_CNoopCategoryIdMapper_h
