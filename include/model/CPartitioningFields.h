/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifndef INCLUDED_ml_model_CPartitioningFields_h
#define INCLUDED_ml_model_CPartitioningFields_h

#include <model/ImportExport.h>

#include <boost/ref.hpp>

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
    using TStrCRef = boost::reference_wrapper<const std::string>;
    using TStrCRefStrCRefPr = std::pair<TStrCRef, TStrCRef>;
    using TStrCRefStrCRefPrVec = std::vector<TStrCRefStrCRefPr>;

public:
    CPartitioningFields(const std::string &partitionFieldName,
                        const std::string &partitionFieldValue);

    //! Append the field (name, value) pair (\p fieldName, \p fieldValue).
    void add(const std::string &fieldName, const std::string &fieldValue);

    //! Get the number of partitioning fields.
    std::size_t size(void) const;

    //! Get a read only reference to the i'th field (name, value) pair.
    const TStrCRefStrCRefPr &operator[](std::size_t i) const;
    //! Get the i'th field (name, value) pair.
    TStrCRefStrCRefPr &operator[](std::size_t i);

    //! Get a read only reference to the last field (name, value) pair.
    const TStrCRefStrCRefPr &back(void) const;
    //! Get the last field (name, value) pair.
    TStrCRefStrCRefPr &back(void);

    //! Get the partition field value.
    const std::string &partitionFieldValue(void) const;

private:
    //! The partitioning fields (name, value) pairs.
    TStrCRefStrCRefPrVec m_PartitioningFields;
};
}
}

#endif// INCLUDED_ml_model_CPartitioningFields_h
