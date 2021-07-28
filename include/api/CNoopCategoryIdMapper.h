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
//! All mappings are no-ops and the categorizer key is the empty
//! string.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This mapper is designed for the case where there is a single
//! categorizer, which includes all jobs created before 7.9.
//!
class API_EXPORT CNoopCategoryIdMapper : public CCategoryIdMapper {
public:
    //! Map from a local category ID local to a global category ID.  This method
    //! is not const, as it will create a new global ID if one does not exist.
    CGlobalCategoryId map(model::CLocalCategoryId localCategoryId) override;

    //! Get the categorizer key for this mapper.
    const std::string& categorizerKey() const override;

    //! Create a clone.
    TCategoryIdMapperPtr clone() const override;
};
}
}

#endif // INCLUDED_ml_api_CNoopCategoryIdMapper_h
