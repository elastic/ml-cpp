/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CScopedRapidJsonPoolAllocator_h
#define INCLUDED_ml_core_CScopedRapidJsonPoolAllocator_h

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ml
{
namespace core
{
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
template<typename T>
class CScopedRapidJsonPoolAllocator
{
    public:
        //! \p allocatorName Unique identifier for the allocator
        //! \p jsonOutputWriter JSON output writer that will make use of the allocator
        CScopedRapidJsonPoolAllocator(const std::string &allocatorName, T &writer) 
            : m_Writer(writer)
        {
            m_Writer.pushAllocator(allocatorName);
        }

        ~CScopedRapidJsonPoolAllocator()
        {
            m_Writer.popAllocator();
        }

    private:
        T &m_Writer;
};

}
}
#endif // INCLUDED_ml_core_CScopedRapidJsonPoolAllocator_h
