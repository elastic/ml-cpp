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
#ifndef INCLUDED_ml_core_CBoostJsonPoolAllocator_h
#define INCLUDED_ml_core_CBoostJsonPoolAllocator_h

#include <boost/json.hpp>

#include <memory>
#include <vector>

namespace json = boost::json;

namespace ml {
namespace core {
//! \brief
//! A boost::json memory allocator using a fixed size buffer
//!
//! DESCRIPTION:\n
//! Encapsulates a boost::json monotonic_resource optimized with a fixed size buffer, see https://www.boost.org/doc/libs/1_84_0/libs/json/doc/html/json/allocators/storage_ptr.html
//!
//! IMPLEMENTATION DECISIONS:\n
//! Use a fixed size buffer for the allocator for performance reasons
//!
//! Retain documents created to ensure that the associated memory allocator exists for the documents
//! lifetime
//!
//! Clear the allocator on destruction
//!
class CBoostJsonPoolAllocator {
public:
    using TDocumentWeakPtr = std::weak_ptr<json::object>;
    using TDocumentPtr = std::shared_ptr<json::object>;
    using TDocumentPtrVec = std::vector<TDocumentPtr>;

public:
    CBoostJsonPoolAllocator() {}

    ~CBoostJsonPoolAllocator() {}

    //! \return document pointer suitable for storing in a container
    //! Note: The document memory is cleaned up once all references to it are destroyed.
    TDocumentWeakPtr makeStorableDoc() {
        TDocumentPtr newDoc = std::make_shared<json::object>(m_JsonStoragePointer);
        m_JsonDocumentStore.push_back(newDoc);
        return TDocumentWeakPtr(newDoc);
    }

    //! \return const reference to the underlying storage pointer
    const json::storage_ptr& get() const { return m_JsonStoragePointer; }

    //! \return reference to the underlying storage pointer
    json::storage_ptr& get() { return m_JsonStoragePointer; }

private:
    //! Size of the fixed buffer to allocate for parsing JSON
    static const size_t FIXED_BUFFER_SIZE = 4096;

private:
    //! fixed size memory buffer used to optimize allocator performance
    unsigned char m_FixedBuffer[FIXED_BUFFER_SIZE];

    //! storage pointer to use for allocating boost::json objects
    json::storage_ptr m_JsonStoragePointer{
        json::make_shared_resource<json::monotonic_resource>(m_FixedBuffer)};

    //! Container used to persist boost::json documents
    TDocumentPtrVec m_JsonDocumentStore;
};
}
}
#endif // INCLUDED_ml_core_CBoostJsonPoolAllocator_h
