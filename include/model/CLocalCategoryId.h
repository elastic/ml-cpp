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
#ifndef INCLUDED_ml_model_CLocalCategoryId_h
#define INCLUDED_ml_model_CLocalCategoryId_h

#include <model/ImportExport.h>

#include <iosfwd>
#include <string>

namespace ml {
namespace model {

//! \brief
//! Adds type safety to a local category ID.
//!
//! DESCRIPTION:\n
//! Local category IDs are the IDs used by the categorization classes
//! in the model library.  Each model library categorizer will have
//! local category IDs running from 1 to the number of categories
//! found.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This is basically a wrapper around an int, but by using a dedicated
//! class there is less risk of accidentally confusing other ints.
//!
class MODEL_EXPORT CLocalCategoryId {
public:
    //! A soft categorization failure means downstream components can continue,
    //! by considering the input record to be in some "uncategorizable"
    //! category.
    static const int SOFT_CATEGORIZATION_FAILURE_ERROR;

    //! A hard categorization failure means processing of the input record must
    //! cease.  (This is generally used to prevent excessive memory
    //! accumulation.)
    static const int HARD_CATEGORIZATION_FAILURE_ERROR;

public:
    //! Create a category representing a soft failure.
    CLocalCategoryId();

    //! Create a new category from its numeric ID.
    explicit CLocalCategoryId(int id);

    //! Create a new category based on a vector index.
    explicit CLocalCategoryId(std::size_t index);

    //! Get an object representing a soft failure.
    static CLocalCategoryId softFailure();

    //! Get an object representing a hard failure.
    static CLocalCategoryId hardFailure();

    //! Accessor for numeric ID.
    int id() const { return m_Id; }

    //! The appropriate vector index for the category.  The caller should
    //! ensure the object represents a valid category before calling this.
    std::size_t index() const { return static_cast<std::size_t>(m_Id - 1); }

    //! Does the object represent a valid category?
    bool isValid() const { return m_Id >= 1; }

    //! Does the object represent a soft failure?
    bool isSoftFailure() const {
        return m_Id == SOFT_CATEGORIZATION_FAILURE_ERROR;
    }

    //! Does the object represent a hard failure?
    bool isHardFailure() const {
        return m_Id == HARD_CATEGORIZATION_FAILURE_ERROR;
    }

    //! Comparison operators.
    bool operator==(const CLocalCategoryId& other) const;
    bool operator!=(const CLocalCategoryId& other) const;
    bool operator<(const CLocalCategoryId& other) const;

    //! Convert to string.
    std::string toString() const;

    //! Parse from string.  Returns false if unsuccessful.
    bool fromString(const std::string& str);

private:
    //! The numeric ID.
    int m_Id;
};

inline std::size_t hash_value(const CLocalCategoryId& categoryId) {
    return static_cast<std::size_t>(categoryId.id());
}

MODEL_EXPORT
std::ostream& operator<<(std::ostream& strm, const CLocalCategoryId& categoryId);
}
}

#endif // INCLUDED_ml_model_CLocalCategoryId_h
