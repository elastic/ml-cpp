/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CGlobalCategoryId_h
#define INCLUDED_ml_api_CGlobalCategoryId_h

#include <model/CLocalCategoryId.h>

#include <api/ImportExport.h>

#include <iosfwd>
#include <string>

namespace ml {
namespace api {

//! \brief
//! Adds type safety to a global category ID.
//!
//! DESCRIPTION:\n
//! Global category IDs are the IDs reported externally by the
//! categorization functionality.  Global category IDs run from 1 to
//! the total number of categories across all model library
//! categorizers.
//!
//! Every global ID corresponds to a unique value of the pair
//! (categorizer key, local category ID).
//!
//! IMPLEMENTATION DECISIONS:\n
//! Global ID objects remember which categorizer key and local
//! category ID they correspond to.  This avoids the need to map
//! in both directions between local and global IDs; once a local
//! ID has been mapped to a global ID all the information is kept
//! together.
//!
//! Categorizer keys are held by pointer to avoid excessive string
//! copying.  Global ID objects are expected to be relatively
//! short-lived so that there is little risk of the string that is
//! pointed to being destroyed before this object.
//!
class API_EXPORT CGlobalCategoryId {
public:
    //! Create a category representing a soft failure.
    CGlobalCategoryId();

    //! Create a new category from its numeric ID in the case where local ID is
    //! the same and categorizer key is unimportant.
    explicit CGlobalCategoryId(int globalId);

    //! Create a new category from its numeric ID, also storing the
    //! corresponding categorizer key and local ID.
    CGlobalCategoryId(int globalId,
                      const std::string& categorizerKey,
                      model::CLocalCategoryId localCategoryId);

    //! Overload that aborts to prevent string literals being passed as the
    //! categorizer key (since we store a pointer to the string object, which
    //! would be a temporary if converted from const char*).  This is mainly an
    //! aid to unit test writers.
    [[noreturn]] CGlobalCategoryId(int globalId,
                                   const char* categorizerKey,
                                   model::CLocalCategoryId localCategoryId);

    //! Get an object representing a soft failure.
    static CGlobalCategoryId softFailure();

    //! Get an object representing a hard failure.
    static CGlobalCategoryId hardFailure();

    //! Accessor for numeric global ID.
    int globalId() const { return m_GlobalId; }

    //! Accessor for categorizer key.  Returns an empty string if unimportant.
    const std::string& categorizerKey() const;

    //! Accessor for numeric local ID.
    model::CLocalCategoryId localId() const { return m_LocalId; }

    //! Does the object represent a valid category?
    bool isValid() const { return m_GlobalId >= 1; }

    //! Does the object represent a soft failure?
    bool isSoftFailure() const {
        return m_GlobalId == model::CLocalCategoryId::SOFT_CATEGORIZATION_FAILURE_ERROR;
    }

    //! Does the object represent a hard failure?
    bool isHardFailure() const {
        return m_GlobalId == model::CLocalCategoryId::HARD_CATEGORIZATION_FAILURE_ERROR;
    }

    //! Comparison operators only consider the global ID, as that is supposed to
    //! be globally unique.
    bool operator==(const CGlobalCategoryId& other) const;
    bool operator!=(const CGlobalCategoryId& other) const;
    bool operator<(const CGlobalCategoryId& other) const;

    //! Print for debug.
    std::string print() const;

    friend API_EXPORT std::ostream& operator<<(std::ostream&, const CGlobalCategoryId&);

private:
    //! The numeric global ID.
    int m_GlobalId;

    //! The key of the categorizer that the local ID corresponds to.
    //! This may be NULL if it's unimportant (for example in failure
    //! cases or when it is known that there will only ever be one
    //! categorizer program-wide).
    const std::string* m_CategorizerKey;

    //! The corresponding local ID.
    model::CLocalCategoryId m_LocalId;
};

//! The hash only considers the global ID, as that is supposed to be globally
//! unique.
inline std::size_t hash_value(const CGlobalCategoryId& categoryId) {
    return static_cast<std::size_t>(categoryId.globalId());
}

API_EXPORT
std::ostream& operator<<(std::ostream& strm, const CGlobalCategoryId& categoryId);
}
}

#endif // INCLUDED_ml_api_CGlobalCategoryId_h
