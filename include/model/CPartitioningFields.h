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

#ifndef INCLUDED_ml_model_CPartitioningFields_h
#define INCLUDED_ml_model_CPartitioningFields_h

#include <model/ImportExport.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace model {

//! \brief A wrapper around the partitioning fields of a model.
//!
//! DESCTIPTION:\n
//! This wraps a list of field (name, value) pairs and encapsulates
//! constant cost access to the partition field.
class MODEL_EXPORT CPartitioningFields {
public:
    using TStrCRef = std::reference_wrapper<const std::string>;
    using TStrCRefStrCRefPr = std::pair<TStrCRef, TStrCRef>;
    using TStrCRefStrCRefPrVec = std::vector<TStrCRefStrCRefPr>;

public:
    CPartitioningFields(const std::string& partitionFieldName,
                        const std::string& partitionFieldValue);

    //! Append the field (name, value) pair (\p fieldName, \p fieldValue).
    void add(const std::string& fieldName, const std::string& fieldValue);

    //! Get the number of partitioning fields.
    std::size_t size() const;

    //! Get a read only reference to the i'th field (name, value) pair.
    const TStrCRefStrCRefPr& operator[](std::size_t i) const;
    //! Get the i'th field (name, value) pair.
    TStrCRefStrCRefPr& operator[](std::size_t i);

    //! Get a read only reference to the last field (name, value) pair.
    const TStrCRefStrCRefPr& back() const;
    //! Get the last field (name, value) pair.
    TStrCRefStrCRefPr& back();

    //! Get the partition field value.
    const std::string& partitionFieldValue() const;

private:
    //! The partitioning fields (name, value) pairs.
    TStrCRefStrCRefPrVec m_PartitioningFields;
};
}
}

#endif // INCLUDED_ml_model_CPartitioningFields_h
