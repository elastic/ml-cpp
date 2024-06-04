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
#ifndef INCLUDED_ml_core_CScopedBoostJsonPoolAllocator_h
#define INCLUDED_ml_core_CScopedBoostJsonPoolAllocator_h

#include <boost/json.hpp>
#include <string>

namespace ml {
namespace core {
//! \brief
//! A scoped json memory allocator
//!
//! DESCRIPTION:\n
//! Control the lifetime of the memory allocator associated with a generic json output writer
//!
//! IMPLEMENTATION DECISIONS:\n
//! Parameterized on json writer type
//! Remove the allocator from the writer and release resources on destruction.
//!
template<typename T>
class CScopedBoostJsonPoolAllocator {
public:
    //! \p allocatorName Unique identifier for the allocator
    //! \p jsonOutputWriter JSON output writer that will make use of the allocator
    CScopedBoostJsonPoolAllocator(const std::string& allocatorName, T& writer)
        : m_Writer(writer), m_AllocatorName(allocatorName) {
        m_Writer.pushAllocator(allocatorName);
    }

    ~CScopedBoostJsonPoolAllocator() {
        m_Writer.popAllocator();
        m_Writer.releaseAllocator(m_AllocatorName);
    }

private:
    T& m_Writer;
    std::string m_AllocatorName;
};
}
}
#endif // INCLUDED_ml_core_CScopedBoostJsonPoolAllocator_h
