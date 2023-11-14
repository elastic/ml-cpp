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

#include "ImportExport.h"

namespace json = boost::json;

namespace ml {
namespace core {
//! \brief
//! A boost::json memory allocator using a fixed size buffer
//!
//! DESCRIPTION:\n
//! Encapsulates a boost::json static_resource optimized with a fixed size buffer
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
    using TDocumentWeakPtr = std::weak_ptr<boost::json::value>;
    using TDocumentPtr = std::shared_ptr<boost::json::value>;
    using TDocumentPtrVec = std::vector<TDocumentPtr>;

public:
    CBoostJsonPoolAllocator()
        : m_JsonPoolAllocator(m_FixedBuffer) {}

    ~CBoostJsonPoolAllocator() { this->clear(); }

    void clear() { m_JsonPoolAllocator.release(); }

    //! \return document pointer suitable for storing in a container
    //! Note: The API is designed to emphasise that the client does not own the document memory
    //! i.e. The document will be invalidated on destruction of this allocator
    TDocumentWeakPtr makeStorableDoc() {
        TDocumentPtr newDoc = std::make_shared<boost::json::value>(&m_JsonPoolAllocator);
        m_JsonDocumentStore.push_back(newDoc);
        return TDocumentWeakPtr(newDoc);
    }

    //! \return const reference to the underlying memory pool allocator
    const boost::json::memory_resource& get() const {
        return m_JsonPoolAllocator;
    }

    //! \return reference to the underlying memory pool allocator
    boost::json::memory_resource& get() { return m_JsonPoolAllocator; }

private:
    //! Size of the fixed buffer to allocate
    static const size_t FIXED_BUFFER_SIZE = 4096;

private:
    //! fixed size memory buffer used to optimize allocator performance
    unsigned char m_FixedBuffer[FIXED_BUFFER_SIZE];

    //! memory pool to use for allocating boost::json objects
    boost::json::monotonic_resource m_JsonPoolAllocator;

    //! Container used to persist boost::json documents
    TDocumentPtrVec m_JsonDocumentStore;
};
}
}
#endif // INCLUDED_ml_core_CBoostJsonPoolAllocator_h
