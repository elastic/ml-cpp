/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CMemoryUsage_h
#define INCLUDED_ml_core_CMemoryUsage_h

#include <core/ImportExport.h>

#include <boost/shared_ptr.hpp>

#include <list>
#include <string>
#include <vector>

namespace ml {
namespace core {

namespace memory_detail {
class CMemoryUsageComparison;
class CMemoryUsageComparisonTwo;
}

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
    struct CORE_EXPORT SMemoryUsage {
        SMemoryUsage(const std::string& name, std::size_t memory) : s_Name(name), s_Memory(memory), s_Unused(0) {}

        SMemoryUsage(const std::string& name, std::size_t memory, std::size_t unused) : s_Name(name), s_Memory(memory), s_Unused(unused) {}

        //! Name of the component
        std::string s_Name;

        //! Bytes used by the component
        std::size_t s_Memory;

        //! For dynamic pre-allocation containers, the capacity - actual used bytes
        //! which equates to pre-allocated space, unused
        std::size_t s_Unused;
    };

    using TMemoryUsagePtr = CMemoryUsage*;
    using TMemoryUsagePtrList = std::list<TMemoryUsagePtr>;
    using TMemoryUsagePtrListCItr = TMemoryUsagePtrList::const_iterator;
    using TMemoryUsagePtrListItr = TMemoryUsagePtrList::iterator;
    using TMemoryUsageVec = std::vector<SMemoryUsage>;
    using TMemoryUsageVecCitr = TMemoryUsageVec::const_iterator;

public:
    //! Constructor
    CMemoryUsage();

    //! Destructor
    ~CMemoryUsage();

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

    //! Format the memory used by this node and all child nodes
    //! into a JSON stream
    void print(std::ostream& outStream) const;

    //! Aggregate big collections of child items together
    void compress();

private:
    //! Give out data to the JSON writer to format, recursively
    void summary(CMemoryUsageJsonWriter& writer) const;

    //! Collection of child items
    TMemoryUsagePtrList m_Children;

    //! Collection of component items within this node
    TMemoryUsageVec m_Items;

    //! Description of this item
    SMemoryUsage m_Description;

    friend class memory_detail::CMemoryUsageComparison;
    friend class memory_detail::CMemoryUsageComparisonTwo;
};

} // core

} // ml

#endif // INCLUDED_ml_core_CMemoryUsage_h
