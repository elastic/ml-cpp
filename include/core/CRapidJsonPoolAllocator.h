/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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
#ifndef INCLUDED_ml_core_CRapidJsonPoolAllocator_h
#define INCLUDED_ml_core_CRapidJsonPoolAllocator_h

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <memory>

namespace ml {
namespace core {
//! \brief
//! A rapidjson memory allocator using a fixed size buffer
//!
//! DESCRIPTION:\n
//! Encapsulates a rapidjson MemoryAllocator optimized with a fixed size buffer
//!
//! IMPLEMENTATION DECISIONS:\n
//! Use a fixed size buffer for the allocator for performance reasons
//!
//! Retain documents created to ensure that the associated memory allocator exists for the documents
//! lifetime
//!
//! Clear the allocator on destruction
//!
class CRapidJsonPoolAllocator {
public:
    using TDocumentWeakPtr = std::weak_ptr<rapidjson::Document>;
    using TDocumentPtr = std::shared_ptr<rapidjson::Document>;
    using TDocumentPtrVec = std::vector<TDocumentPtr>;

public:
    CRapidJsonPoolAllocator()
        : m_JsonPoolAllocator(m_FixedBuffer, FIXED_BUFFER_SIZE) {}

    ~CRapidJsonPoolAllocator() { this->clear(); }

    void clear() { m_JsonPoolAllocator.Clear(); }

    //! \return document pointer suitable for storing in a container
    //! Note: The API is designed to emphasise that the client does not own the document memory
    //! i.e. The document will be invalidated on destruction of this allocator
    TDocumentWeakPtr makeStorableDoc() {
        TDocumentPtr newDoc = std::make_shared<rapidjson::Document>(&m_JsonPoolAllocator);
        newDoc->SetObject();
        m_JsonDocumentStore.push_back(newDoc);
        return TDocumentWeakPtr(newDoc);
    }

    //! \return const reference to the underlying memory pool allocator
    const rapidjson::MemoryPoolAllocator<>& get() const {
        return m_JsonPoolAllocator;
    }

    //! \return reference to the underlying memory pool allocator
    rapidjson::MemoryPoolAllocator<>& get() { return m_JsonPoolAllocator; }

private:
    //! Size of the fixed buffer to allocate
    static const size_t FIXED_BUFFER_SIZE = 4096;

private:
    //! fixed size memory buffer used to optimize allocator performance
    char m_FixedBuffer[FIXED_BUFFER_SIZE];

    //! memory pool to use for allocating rapidjson objects
    rapidjson::MemoryPoolAllocator<> m_JsonPoolAllocator;

    //! Container used to persist rapidjson documents
    TDocumentPtrVec m_JsonDocumentStore;
};
}
}
#endif // INCLUDED_ml_core_CRapidJsonPoolAllocator_h
