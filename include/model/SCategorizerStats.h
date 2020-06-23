/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_SCategorizerStats_h
#define INCLUDED_ml_model_SCategorizerStats_h

#include <model/ImportExport.h>
#include <model/ModelTypes.h>

namespace ml {
namespace model {
//! \brief
//! Stats that summarise what a categorizer has done.
//!
//! DESCIRIPTION:\n
//! Stats that summarise what a categorizer has done.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Timestamps are not stored.  When written as an output document,
//! timestamps are expected to be added somewhere along the output
//! path.
//!
struct MODEL_EXPORT SCategorizerStats {

    std::size_t s_CategorizedMessages = 0;
    std::size_t s_TotalCategories = 0;
    std::size_t s_FrequentCategories = 0;
    std::size_t s_RareCategories = 0;
    std::size_t s_DeadCategories = 0;
    std::size_t s_MemoryCategorizationFailures = 0;
    model_t::ECategorizationStatus s_CategorizationStatus = model_t::E_CategorizationStatusOk;

    //! Equality comparison
    bool operator==(const SCategorizerStats& other) const {
        return s_CategorizedMessages == other.s_CategorizedMessages &&
               s_TotalCategories == other.s_TotalCategories &&
               s_FrequentCategories == other.s_FrequentCategories &&
               s_RareCategories == other.s_RareCategories &&
               s_DeadCategories == other.s_DeadCategories &&
               s_MemoryCategorizationFailures == other.s_MemoryCategorizationFailures &&
               s_CategorizationStatus == other.s_CategorizationStatus;
    }
};
}
}

#endif // INCLUDED_ml_model_SCategorizerStats_h
