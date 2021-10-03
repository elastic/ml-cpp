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

#include <api/CGlobalCategoryId.h>

#include <core/CLogger.h>

#include <sstream>

namespace {
const std::string EMPTY_STRING;
}

namespace ml {
namespace api {

CGlobalCategoryId::CGlobalCategoryId()
    : m_GlobalId{model::CLocalCategoryId::SOFT_CATEGORIZATION_FAILURE_ERROR},
      m_CategorizerKey{nullptr}, m_LocalId{model::CLocalCategoryId::softFailure()} {
}

CGlobalCategoryId::CGlobalCategoryId(int globalId)
    : m_GlobalId{globalId}, m_CategorizerKey{nullptr}, m_LocalId{globalId} {
}

CGlobalCategoryId::CGlobalCategoryId(int globalId,
                                     const std::string& categorizerKey,
                                     model::CLocalCategoryId localCategoryId)
    : m_GlobalId{globalId}, m_CategorizerKey{&categorizerKey}, m_LocalId{localCategoryId} {
    // Enforce the invariant that global ID and local ID are the same for
    // failure states
    if (this->isValid() == false) {
        m_CategorizerKey = nullptr;
        m_LocalId = model::CLocalCategoryId{m_GlobalId};
    }
}

CGlobalCategoryId::CGlobalCategoryId(int globalId,
                                     const char* /*categorizerKey*/,
                                     model::CLocalCategoryId localCategoryId)
    : m_GlobalId{globalId}, m_CategorizerKey{nullptr}, m_LocalId{localCategoryId} {
    LOG_ABORT(<< "Programmatic error: CGlobalCategoryId called with const char* categorizer key");
}

CGlobalCategoryId CGlobalCategoryId::softFailure() {
    return CGlobalCategoryId{};
}

CGlobalCategoryId CGlobalCategoryId::hardFailure() {
    return CGlobalCategoryId{model::CLocalCategoryId::HARD_CATEGORIZATION_FAILURE_ERROR};
}

const std::string& CGlobalCategoryId::categorizerKey() const {
    return (m_CategorizerKey == nullptr) ? EMPTY_STRING : *m_CategorizerKey;
}

bool CGlobalCategoryId::operator==(const CGlobalCategoryId& other) const {
    return m_GlobalId == other.m_GlobalId;
}

bool CGlobalCategoryId::operator!=(const CGlobalCategoryId& other) const {
    return m_GlobalId != other.m_GlobalId;
}

bool CGlobalCategoryId::operator<(const CGlobalCategoryId& other) const {
    return m_GlobalId < other.m_GlobalId;
}

std::string CGlobalCategoryId::print() const {
    if (m_CategorizerKey == nullptr) {
        return std::to_string(m_GlobalId);
    }
    std::ostringstream strm;
    strm << *m_CategorizerKey << '/' << m_LocalId << ';' << m_GlobalId;
    return strm.str();
}

std::ostream& operator<<(std::ostream& strm, const CGlobalCategoryId& categoryId) {
    if (categoryId.m_CategorizerKey != nullptr) {
        strm << *categoryId.m_CategorizerKey << '/' << categoryId.m_LocalId << ';';
    }
    return strm << categoryId.m_GlobalId;
}
}
}
