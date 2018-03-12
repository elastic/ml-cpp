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
#ifndef INCLUDED_ml_core_CStoredStringPtr_h
#define INCLUDED_ml_core_CStoredStringPtr_h

#include <core/CMemoryUsage.h>
#include <core/ImportExport.h>

#include <boost/shared_ptr.hpp>

#include <cstddef>
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
        //! NULL constructor.
        CStoredStringPtr() noexcept;

        void swap(CStoredStringPtr &other) noexcept;

        //! Get a reference to the string.
        const std::string &operator*() const noexcept;

        //! Get a pointer to the string.
        const std::string *operator->() const noexcept;

        //! Get a pointer to the string.
        const std::string *get() const noexcept;

        //! Is the pointer non-NULL?
        explicit operator bool() const noexcept;

        //! Is there only one pointer for this stored string?
        bool isUnique() const noexcept;

        //! Equality operator for NULL.
        bool operator==(std::nullptr_t rhs) const noexcept;
        bool operator!=(std::nullptr_t rhs) const noexcept;

        //! Equality operator.
        bool operator==(const CStoredStringPtr &rhs) const noexcept;
        bool operator!=(const CStoredStringPtr &rhs) const noexcept;

        //! Less than operator.
        bool operator<(const CStoredStringPtr &rhs) const noexcept;

        //! Claim memory usage is 0 in the main memory usage calculation, on the
        //! assumption that the actual memory usage will be accounted for in a
        //! string store.
        static bool dynamicSizeAlwaysZero() { return true; }

        //! Get the actual memory usage of the string.  For use by the string
        //! store.
        std::size_t actualMemoryUsage() const;
        void debugActualMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const;

        //! These factory methods return a stored string pointer given a string.
        //! They must only be used within string store classes that contain code
        //! to account for memory usage outside of the main memory usage
        //! calculation.
        static CStoredStringPtr makeStoredString(const std::string &str);
        static CStoredStringPtr makeStoredString(std::string &&str);

    private:
        //! Non-NULL constructors are private to prevent accidental construction
        //! outside of a string store.
        explicit CStoredStringPtr(const std::string &str);
        explicit CStoredStringPtr(std::string &&str);

    private:
        using TStrCPtr = boost::shared_ptr<const std::string>;

        //! The wrapped shared_ptr.
        TStrCPtr m_String;

        friend CORE_EXPORT std::size_t hash_value(const CStoredStringPtr &);
};

//! Hash function named such that it will work automatically with Boost
//! unordered containers.
CORE_EXPORT
std::size_t hash_value(const CStoredStringPtr &ptr);

//! Swap for use by generic code.
CORE_EXPORT
void swap(CStoredStringPtr &lhs, CStoredStringPtr &rhs);

} // core
} // ml

#endif // INCLUDED_ml_core_CStoredStringPtr_h
