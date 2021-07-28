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

#include <model/CPartitioningFields.h>

namespace ml {
namespace model {

CPartitioningFields::CPartitioningFields(const std::string& partitionFieldName,
                                         const std::string& partitionFieldValue) {
    m_PartitioningFields.reserve(3);
    this->add(partitionFieldName, partitionFieldValue);
}

void CPartitioningFields::add(const std::string& fieldName, const std::string& fieldValue) {
    m_PartitioningFields.emplace_back(TStrCRef(fieldName), TStrCRef(fieldValue));
}

std::size_t CPartitioningFields::size() const {
    return m_PartitioningFields.size();
}

const CPartitioningFields::TStrCRefStrCRefPr& CPartitioningFields::
operator[](std::size_t i) const {
    return m_PartitioningFields[i];
}

CPartitioningFields::TStrCRefStrCRefPr& CPartitioningFields::operator[](std::size_t i) {
    return m_PartitioningFields[i];
}

const CPartitioningFields::TStrCRefStrCRefPr& CPartitioningFields::back() const {
    return m_PartitioningFields.back();
}

CPartitioningFields::TStrCRefStrCRefPr& CPartitioningFields::back() {
    return m_PartitioningFields.back();
}

const std::string& CPartitioningFields::partitionFieldValue() const {
    return m_PartitioningFields[0].second.get();
}
}
}
