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
#ifndef INCLUDED_ml_core_CMemoryUsage_h
#define INCLUDED_ml_core_CMemoryUsage_h

#include <core/ImportExport.h>

#include <memory>
#include <ostream>
#include <string>

namespace ml {
namespace core {

class CMemoryUsageJsonWriter;

//! \brief A memory usage tracking class
//!
//! DESCRIPTION:\n
//! This is a tree structure designed to be passed to a component
//! containing subcomponents, so that each component can fill in
//! its memory usage
class CORE_EXPORT CMemoryUsage {
public:
    //! A collection of data to record memory usage information for
    //! arbitrary components
    struct SMemoryUsage {
        SMemoryUsage(const std::string& name, std::size_t memory)
            : s_Name(name), s_Memory(memory), s_Unused(0) {}

        SMemoryUsage(const std::string& name, std::size_t memory, std::size_t unused)
            : s_Name(name), s_Memory(memory), s_Unused(unused) {}

        //! Name of the component
        std::string s_Name;

        //! Bytes used by the component
        std::size_t s_Memory;

        //! For dynamic pre-allocation containers, the capacity - actual used bytes
        //! which equates to pre-allocated space, unused
        std::size_t s_Unused;
    };

    using TMemoryUsagePtr = std::shared_ptr<CMemoryUsage>;

public:
    CMemoryUsage();
    ~CMemoryUsage();
    CMemoryUsage(const CMemoryUsage&) = delete;
    CMemoryUsage& operator=(const CMemoryUsage&) = delete;

    //! Create a child node
    TMemoryUsagePtr addChild();

    //! Create a child node with a pre-computed size offset - this is to
    //! allow sizeof(T) to be removed from items within containers
    TMemoryUsagePtr addChild(std::size_t initialAmount);

    //! Create a memory usage member item
    void addItem(const SMemoryUsage& item);

    // Create a memory usage member item
    void addItem(const std::string& name, std::size_t memory);

    //! Set the name and size of this node
    void setName(const SMemoryUsage& item);

    // Set the name and size of this node
    void setName(const std::string& name, std::size_t memory);

    // Set the name and size of this node
    void setName(const std::string& name);

    //! Get the memory used by this node and all child nodes
    std::size_t usage() const;

    //! Get the unused memory wasted by this node and all child nodes
    std::size_t unusage() const;

    //! Aggregate big collections of child items together
    void compress();

    //! Format the memory used by this node and all child nodes
    //! into a JSON stream
    void print(std::ostream& outStream) const;

private:
    class CImpl;
    using TImplPtr = std::unique_ptr<CImpl>;

private:
    TImplPtr m_Impl;
};
}
}

#endif // INCLUDED_ml_core_CMemoryUsage_h
