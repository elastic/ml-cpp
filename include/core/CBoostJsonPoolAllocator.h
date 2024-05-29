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

namespace {

class custom_resource : public boost::container::pmr::memory_resource {
private:
    void* do_allocate(std::size_t bytes, std::size_t /*align*/) override {
        return ::operator new(bytes);
    }

    void do_deallocate(void* ptr, std::size_t /*bytes*/, std::size_t /*align*/) override {
        return ::operator delete(ptr);
    }

    bool do_is_equal(memory_resource const& other) const noexcept override {
        // since the global allocation and de-allocation functions are used,
        // any instance of a custom_resource can deallocate memory allocated
        // by another instance of a logging_resource
        return dynamic_cast<custom_resource const*>(&other) != nullptr;
    }
};
}
//! \brief
//! A custom boost::json memory allocator
//!
//! DESCRIPTION:\n
//! Encapsulates a custom boost::json memory_resource, see https://www.boost.org/doc/libs/1_83_0/libs/json/doc/html/json/allocators/storage_ptr.html
//!
//! IMPLEMENTATION DECISIONS:\n
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
    //! storage pointer to use for allocating boost::json objects
    //! We use a custom resource allocator for more predictable
    //! and timely allocation/de-allocations, see
    //! https://www.boost.org/doc/libs/1_83_0/libs/json/doc/html/json/allocators/storage_ptr.html#json.allocators.storage_ptr.user_defined_resource
    //! for more details.
    json::storage_ptr m_JsonStoragePointer{json::make_shared_resource<custom_resource>()};

    //! Container used to persist boost::json documents
    TDocumentPtrVec m_JsonDocumentStore;
};
}
}
#endif // INCLUDED_ml_core_CBoostJsonPoolAllocator_h
