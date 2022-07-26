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

#ifndef INCLUDED_ml_core_CMemoryDefStd_h
#define INCLUDED_ml_core_CMemoryDefStd_h

#include <core/CMemoryDec.h>
#include <core/CMemoryDecStd.h>
#include <core/CMemoryDef.h>

namespace ml {
namespace core {
namespace memory_detail {
// Windows creates an extra map/list node per map/list
#ifdef Windows
constexpr std::size_t EXTRA_NODES{1};
#else
constexpr std::size_t EXTRA_NODES{0};
#endif

// Big variations in deque page size!
#ifdef Windows
constexpr std::size_t MIN_DEQUE_PAGE_SIZE{16};
constexpr std::size_t MIN_DEQUE_PAGE_VEC_ENTRIES{8};
#elif defined(MacOSX)
constexpr std::size_t MIN_DEQUE_PAGE_SIZE{4096};
constexpr std::size_t MIN_DEQUE_PAGE_VEC_ENTRIES{1};
#else
constexpr std::size_t MIN_DEQUE_PAGE_SIZE{512};
constexpr std::size_t MIN_DEQUE_PAGE_VEC_ENTRIES{8};
#endif
}

namespace CMemory {

template<typename T, typename A>
std::size_t dynamicSize(const std::list<T, A>& t) {
    return CMemory::elementDynamicSize(t) +
           (memory_detail::EXTRA_NODES + t.size()) *
               (sizeof(T) + CMemory::storageNodeOverhead(t));
}

template<typename T, typename A>
std::size_t dynamicSize(const std::deque<T, A>& t) {
    // std::deque is a pointer to an array of pointers to pages
    std::size_t pageSize = std::max(sizeof(T), memory_detail::MIN_DEQUE_PAGE_SIZE);
    std::size_t itemsPerPage = pageSize / sizeof(T);
    // This could be an underestimate if items have been removed
    std::size_t numPages = (t.size() + itemsPerPage - 1) / itemsPerPage;
    // This could also be an underestimate if items have been removed
    std::size_t pageVecEntries = std::max(numPages, memory_detail::MIN_DEQUE_PAGE_VEC_ENTRIES);

    return CMemory::elementDynamicSize(t) +
           pageVecEntries * sizeof(std::size_t) + numPages * pageSize;
}

template<typename K, typename V, typename C, typename A>
std::size_t dynamicSize(const std::map<K, V, C, A>& t) {
    return CMemory::elementDynamicSize(t) +
           (memory_detail::EXTRA_NODES + t.size()) *
               (sizeof(K) + sizeof(V) + CMemory::storageNodeOverhead(t));
}

template<typename K, typename V, typename C, typename A>
std::size_t dynamicSize(const std::multimap<K, V, C, A>& t) {
    // In practice, both std::multimap and std::map use the same
    // rb tree implementation.
    return CMemory::elementDynamicSize(t) +
           (memory_detail::EXTRA_NODES + t.size()) *
               (sizeof(K) + sizeof(V) + CMemory::storageNodeOverhead(t));
}

template<typename T, typename C, typename A>
std::size_t dynamicSize(const std::set<T, C, A>& t) {
    return CMemory::elementDynamicSize(t) +
           (memory_detail::EXTRA_NODES + t.size()) *
               (sizeof(T) + CMemory::storageNodeOverhead(t));
}

template<typename T, typename C, typename A>
std::size_t dynamicSize(const std::multiset<T, C, A>& t) {
    // In practice, both std::multiset and std::set use the same
    // rb tree implementation.
    return CMemory::elementDynamicSize(t) +
           (memory_detail::EXTRA_NODES + t.size()) *
               (sizeof(T) + CMemory::storageNodeOverhead(t));
}
}

namespace CMemoryDebug {

template<typename T, typename A>
void dynamicSize(const char* name,
                 const std::list<T, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    // std::list appears to use 2 pointers per list node
    // (prev and next pointers).
    std::string componentName(name);
    componentName += "_list";

    std::size_t listSize = (memory_detail::EXTRA_NODES + t.size()) *
                           (sizeof(T) + CMemory::storageNodeOverhead(t));

    CMemoryUsage::SMemoryUsage usage(componentName, listSize);
    CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
    ptr->setName(usage);

    CMemoryDebug::elementDynamicSize(std::move(componentName), t, mem);
}

template<typename T, typename C, typename A>
void dynamicSize(const char* name,
                 const std::deque<T, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    // std::deque is a pointer to an array of pointers to pages
    std::string componentName(name);
    componentName += "_deque";

    std::size_t pageSize = std::max(sizeof(T), memory_detail::MIN_DEQUE_PAGE_SIZE);
    std::size_t itemsPerPage = pageSize / sizeof(T);
    // This could be an underestimate if items have been removed
    std::size_t numPages = (t.size() + itemsPerPage - 1) / itemsPerPage;
    // This could also be an underestimate if items have been removed
    std::size_t pageVecEntries = std::max(numPages, memory_detail::MIN_DEQUE_PAGE_VEC_ENTRIES);

    std::size_t dequeTotal = pageVecEntries * sizeof(std::size_t) + numPages * pageSize;
    std::size_t dequeUsed = numPages * sizeof(std::size_t) + t.size() * sizeof(T);

    CMemoryUsage::SMemoryUsage usage(componentName, dequeTotal, dequeTotal - dequeUsed);
    CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
    ptr->setName(usage);

    CMemoryDebug::elementDynamicSize(std::move(componentName), t, mem);
}

template<typename K, typename V, typename C, typename A>
void dynamicSize(const char* name,
                 const std::map<K, V, C, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    // std::map appears to use 4 pointers/size_ts per tree node
    // (colour, parent, left and right child pointers).
    std::string componentName(name);
    componentName += "_map";

    std::size_t mapSize = (memory_detail::EXTRA_NODES + t.size()) *
                          (sizeof(K) + sizeof(V) + CMemory::storageNodeOverhead(t));

    CMemoryUsage::SMemoryUsage usage(componentName, mapSize);
    CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
    ptr->setName(usage);

    CMemoryDebug::associativeElementDynamicSize(std::move(componentName), t, mem);
}

template<typename K, typename V, typename C, typename A>
void dynamicSize(const char* name,
                 const std::multimap<K, V, C, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    // In practice, both std::multimap and std::map use the same
    // rb tree implementation.
    std::string componentName(name);
    componentName += "_map";

    std::size_t mapSize = (memory_detail::EXTRA_NODES + t.size()) *
                          (sizeof(K) + sizeof(V) + CMemory::storageNodeOverhead(t));

    CMemoryUsage::SMemoryUsage usage(componentName, mapSize);
    CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
    ptr->setName(usage);

    CMemoryDebug::associativeElementDynamicSize(std::move(componentName), t, mem);
}

template<typename T, typename C, typename A>
void dynamicSize(const char* name,
                 const std::set<T, C, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    // std::set appears to use 4 pointers/size_ts per tree node
    // (colour, parent, left and right child pointers).
    std::string componentName(name);
    componentName += "_set";

    std::size_t setSize = (memory_detail::EXTRA_NODES + t.size()) *
                          (sizeof(T) + CMemory::storageNodeOverhead(t));

    CMemoryUsage::SMemoryUsage usage(componentName, setSize);
    CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
    ptr->setName(usage);

    CMemoryDebug::elementDynamicSize(std::move(componentName), t, mem);
}

template<typename T, typename C, typename A>
void dynamicSize(const char* name,
                 const std::multiset<T, C, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    // In practice, both std::multimap and std::map use the same
    // rb tree implementation.
    std::string componentName(name);
    componentName += "_set";

    std::size_t setSize = (memory_detail::EXTRA_NODES + t.size()) *
                          (sizeof(T) + CMemory::storageNodeOverhead(t));

    CMemoryUsage::SMemoryUsage usage(componentName, setSize);
    CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
    ptr->setName(usage);

    CMemoryDebug::elementDynamicSize(std::move(componentName), t, mem);
}
}
}
}

#endif // INCLUDED_ml_core_CMemoryDefStd_h
