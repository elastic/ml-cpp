/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CPartitioningFields.h>

namespace ml
{
namespace model
{

CPartitioningFields::CPartitioningFields(const std::string &partitionFieldName,
                                         const std::string &partitionFieldValue)
{
    m_PartitioningFields.reserve(3);
    this->add(partitionFieldName, partitionFieldValue);
}

void CPartitioningFields::add(const std::string &fieldName, const std::string &fieldValue)
{
    m_PartitioningFields.emplace_back(TStrCRef(fieldName), TStrCRef(fieldValue));
}

std::size_t CPartitioningFields::size(void) const
{
    return m_PartitioningFields.size();
}

const CPartitioningFields::TStrCRefStrCRefPr &CPartitioningFields::operator[](std::size_t i) const
{
    return m_PartitioningFields[i];
}

CPartitioningFields::TStrCRefStrCRefPr &CPartitioningFields::operator[](std::size_t i)
{
    return m_PartitioningFields[i];
}

const CPartitioningFields::TStrCRefStrCRefPr &CPartitioningFields::back(void) const
{
    return m_PartitioningFields.back();
}

CPartitioningFields::TStrCRefStrCRefPr &CPartitioningFields::back(void)
{
    return m_PartitioningFields.back();
}

const std::string &CPartitioningFields::partitionFieldValue(void) const
{
    return m_PartitioningFields[0].second.get();
}

}
}
