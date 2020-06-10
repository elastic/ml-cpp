/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CLocalCategoryId.h>

#include <core/CStringUtils.h>

#include <ostream>

namespace ml {
namespace model {

// Initialise statics
const int CLocalCategoryId::SOFT_CATEGORIZATION_FAILURE_ERROR{-1};
const int CLocalCategoryId::HARD_CATEGORIZATION_FAILURE_ERROR{-2};

CLocalCategoryId::CLocalCategoryId() : m_Id{SOFT_CATEGORIZATION_FAILURE_ERROR} {
}

CLocalCategoryId::CLocalCategoryId(int id) : m_Id{id} {
}

CLocalCategoryId::CLocalCategoryId(std::size_t index)
    : m_Id{static_cast<int>(index + 1)} {
}

CLocalCategoryId CLocalCategoryId::softFailure() {
    return CLocalCategoryId{SOFT_CATEGORIZATION_FAILURE_ERROR};
}
CLocalCategoryId CLocalCategoryId::hardFailure() {
    return CLocalCategoryId{HARD_CATEGORIZATION_FAILURE_ERROR};
}

bool CLocalCategoryId::operator==(const CLocalCategoryId& other) const {
    return m_Id == other.m_Id;
}

bool CLocalCategoryId::operator!=(const CLocalCategoryId& other) const {
    return m_Id != other.m_Id;
}

bool CLocalCategoryId::operator<(const CLocalCategoryId& other) const {
    return m_Id < other.m_Id;
}

std::string CLocalCategoryId::toString() const {
    return std::to_string(m_Id);
}

bool CLocalCategoryId::fromString(const std::string& str) {
    return core::CStringUtils::stringToType(str, m_Id);
}

std::ostream& operator<<(std::ostream& strm, const CLocalCategoryId& categoryId) {
    return strm << categoryId.id();
}
}
}
