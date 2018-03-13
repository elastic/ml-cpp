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
#ifndef INCLUDED_ml_core_CScopedRapidJsonPoolAllocator_h
#define INCLUDED_ml_core_CScopedRapidJsonPoolAllocator_h

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ml {
namespace core {
//! \brief
//! A scoped rapidjson memory allocator
//!
//! DESCRIPTION:\n
//! Control the lifetime of the memory allocator associated with a generic json output writer
//!
//! IMPLEMENTATION DECISIONS:\n
//! Parameterized on json writer type
//! Remove the allocator from the writer and release resources on destruction.
//!
template <typename T> class CScopedRapidJsonPoolAllocator {
public:
    //! \p allocatorName Unique identifier for the allocator
    //! \p jsonOutputWriter JSON output writer that will make use of the allocator
    CScopedRapidJsonPoolAllocator(const std::string &allocatorName, T &writer) : m_Writer(writer) {
        m_Writer.pushAllocator(allocatorName);
    }

    ~CScopedRapidJsonPoolAllocator() { m_Writer.popAllocator(); }

private:
    T &m_Writer;
};
}
}
#endif// INCLUDED_ml_core_CScopedRapidJsonPoolAllocator_h
