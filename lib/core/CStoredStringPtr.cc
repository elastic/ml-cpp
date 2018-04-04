/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
#include <core/CStoredStringPtr.h>

#include <core/CMemory.h>

#include <boost/functional/hash.hpp>
#include <boost/make_shared.hpp>

#include <utility>

namespace ml {
namespace core {

CStoredStringPtr::CStoredStringPtr() : m_String{} {
}

CStoredStringPtr::CStoredStringPtr(const std::string& str) : m_String{boost::make_shared<const std::string>(str)} {
}

CStoredStringPtr::CStoredStringPtr(std::string&& str) : m_String{boost::make_shared<const std::string>(std::move(str))} {
}

void CStoredStringPtr::swap(CStoredStringPtr& other) {
    m_String.swap(other.m_String);
}

const std::string& CStoredStringPtr::operator*() const {
    return *m_String;
}

const std::string* CStoredStringPtr::operator->() const {
    return m_String.get();
}

const std::string* CStoredStringPtr::get() const {
    return m_String.get();
}

CStoredStringPtr::operator bool() const {
    return m_String.get() != nullptr;
}

bool CStoredStringPtr::isUnique() const {
    return m_String.unique();
}

bool CStoredStringPtr::operator==(std::nullptr_t rhs) const {
    return m_String == rhs;
}

bool CStoredStringPtr::operator!=(std::nullptr_t rhs) const {
    return m_String != rhs;
}

bool CStoredStringPtr::operator==(const CStoredStringPtr& rhs) const {
    return m_String == rhs.m_String;
}

bool CStoredStringPtr::operator!=(const CStoredStringPtr& rhs) const {
    return m_String != rhs.m_String;
}

bool CStoredStringPtr::operator<(const CStoredStringPtr& rhs) const {
    return m_String < rhs.m_String;
}

std::size_t CStoredStringPtr::actualMemoryUsage() const {
    // We convert to a raw pointer here to avoid the "divide by use count"
    // feature of CMemory's shared_ptr handling
    return CMemory::dynamicSize(m_String.get());
}

void CStoredStringPtr::debugActualMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const {
    // This is NOT the standard way to account for the memory of a
    // shared_ptr - do NOT copy this to other classes with shared_ptr members
    mem->addItem("m_String", this->actualMemoryUsage());
}

CStoredStringPtr CStoredStringPtr::makeStoredString(const std::string& str) {
    return CStoredStringPtr(str);
}

CStoredStringPtr CStoredStringPtr::makeStoredString(std::string&& str) {
    return CStoredStringPtr(std::move(str));
}

std::size_t hash_value(const CStoredStringPtr& ptr) {
    return boost::hash_value(ptr.m_String);
}

void swap(CStoredStringPtr& lhs, CStoredStringPtr& rhs) {
    lhs.swap(rhs);
}

} // core
} // ml
