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
#include <core/CStoredStringPtr.h>

#include <core/CMemoryDef.h>

#include <boost/functional/hash.hpp>

#include <utility>

namespace ml {
namespace core {

const CStoredStringPtr CStoredStringPtr::NULL_STRING;

CStoredStringPtr::CStoredStringPtr() noexcept : m_String{} {
}

CStoredStringPtr::CStoredStringPtr(const std::string& str)
    : m_String{str} {
}

CStoredStringPtr::CStoredStringPtr(std::string&& str)
    : m_String{std::move(str)} {
}

CStoredStringPtr::~CStoredStringPtr() = default;

void CStoredStringPtr::swap(CStoredStringPtr& other) noexcept {
    m_String.swap(other.m_String);
}

const std::string& CStoredStringPtr::operator*() const noexcept {
    const static std::string EMPTY_STRING;
    return m_String ? m_String.value() : EMPTY_STRING;
}

const std::string* CStoredStringPtr::operator->() const noexcept {
    return m_String ? &m_String.value() : nullptr;
}

const CStoredStringPtr::TOptionalStr& CStoredStringPtr::getOptional() const noexcept {
    return m_String;
}

CStoredStringPtr::operator bool() const noexcept {
    return m_String.has_value();
}

bool CStoredStringPtr::operator==(const CStoredStringPtr& rhs) const noexcept {
    return m_String == rhs.m_String;
}

bool CStoredStringPtr::operator!=(const CStoredStringPtr& rhs) const noexcept {
    return m_String != rhs.m_String;
}

bool CStoredStringPtr::operator<(const CStoredStringPtr& rhs) const noexcept {
    return m_String < rhs.m_String;
}

std::size_t hash_value(const CStoredStringPtr& ptr) {
    return boost::hash_value(ptr.m_String);
}

void swap(CStoredStringPtr& lhs, CStoredStringPtr& rhs) {
    lhs.swap(rhs);
}

} // core
} // ml
