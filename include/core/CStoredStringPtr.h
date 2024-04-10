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
#ifndef INCLUDED_ml_core_CStoredStringPtr_h
#define INCLUDED_ml_core_CStoredStringPtr_h

#include <core/CMemoryUsage.h>
#include <core/ImportExport.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace ml {
namespace core {

//! \brief
//! A pointer-like proxy for strings owned by a string store.
//!
//! DESCRIPTION:\n
//! A replacement for shared_ptr for strings stored in a string store.
//! Using shared_ptr directly causes problems for the memory usage
//! calculation.  Using this class to wrap the stored strings means
//! that the majority of memory usage calculations can ignore the
//! stored strings completely, and then the string store is solely
//! responsible for tracking their memory usage.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The private constructors make it hard to accidentally construct
//! stored string pointers that are not managed by a string store.
//!
class CORE_EXPORT CStoredStringPtr {
public:
    const static CStoredStringPtr NULL_STRING;

    //! NULL constructor.
    CStoredStringPtr() noexcept;
    ~CStoredStringPtr();

    void swap(CStoredStringPtr& other) noexcept;

    //! Get a reference to the string.
    //! Returns empty string for a null pointer.
    const std::string& operator*() const noexcept;

    //! Get a pointer to the string.
    const std::string* operator->() const noexcept;

    //! Get a pointer to the string.
    const std::string* get() const noexcept;

    //! Is the pointer non-NULL?
    explicit operator bool() const noexcept;

    //! Is there only one pointer for this stored string?
    //! This method is inefficient and should only be used in unit test
    //! code (and then only infrequently).
    bool isUnique() const noexcept;

    //! Equality operator for NULL.
    bool operator==(std::nullptr_t rhs) const noexcept;
    bool operator!=(std::nullptr_t rhs) const noexcept;

    //! Equality operator.
    bool operator==(const CStoredStringPtr& rhs) const noexcept;
    bool operator!=(const CStoredStringPtr& rhs) const noexcept;

    //! Less than operator.
    bool operator<(const CStoredStringPtr& rhs) const noexcept;

    //! Get the actual memory usage of the string.  For use by the string
    //! store.
    std::size_t actualMemoryUsage() const;
    void debugActualMemoryUsage(const CMemoryUsage::TMemoryUsagePtr& mem) const;

    explicit CStoredStringPtr(const std::string& str);
    explicit CStoredStringPtr(std::string&& str);

private:
    using TOptionalStr = std::optional<std::string>;

    TOptionalStr m_String;

    friend CORE_EXPORT std::size_t hash_value(const CStoredStringPtr&);
};

//! Hash function named such that it will work automatically with Boost
//! unordered containers.
CORE_EXPORT
std::size_t hash_value(const CStoredStringPtr& ptr);

//! Swap for use by generic code.
CORE_EXPORT
void swap(CStoredStringPtr& lhs, CStoredStringPtr& rhs);

} // core
} // ml

#endif // INCLUDED_ml_core_CStoredStringPtr_h
