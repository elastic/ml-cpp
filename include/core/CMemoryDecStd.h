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

#ifndef INCLUDED_ml_core_CMemoryDecStd_h
#define INCLUDED_ml_core_CMemoryDecStd_h

#include <core/CMemoryDec.h>

#include <deque>
#include <list>
#include <map>
#include <set>

namespace ml {
namespace core {
namespace CMemory {

template<typename T, typename A>
constexpr std::size_t storageNodeOverhead(const std::list<T, A>&) {
    // std::list uses 2 pointers per list node (prev and next pointers).
    return 2 * sizeof(std::size_t);
}

template<typename K, typename V, typename C, typename A>
constexpr std::size_t storageNodeOverhead(const std::map<K, V, C, A>&) {
    // std::map appears to use 4 pointers/size_ts per tree node
    // (colour, parent, left and right child pointers).
    return 4 * sizeof(std::size_t);
}

template<typename K, typename V, typename C, typename A>
constexpr std::size_t storageNodeOverhead(const std::multimap<K, V, C, A>&) {
    // In practice, both std::multimap and std::map use the same
    // rb tree implementation.
    return 4 * sizeof(std::size_t);
}

template<typename T, typename C, typename A>
constexpr std::size_t storageNodeOverhead(const std::set<T, C, A>&) {
    // std::set appears to use 4 pointers/size_ts per tree node
    // (colour, parent, left and right child pointers).
    return 4 * sizeof(std::size_t);
}

template<typename T, typename C, typename A>
constexpr std::size_t storageNodeOverhead(const std::multiset<T, C, A>&) {
    // In practice, both std::multiset and std::set use the same
    // rb tree implementation.
    return 4 * sizeof(std::size_t);
}

template<typename T, typename A>
std::size_t dynamicSize(const std::list<T, A>& t);

template<typename T, typename A>
std::size_t dynamicSize(const std::deque<T, A>& t);

template<typename K, typename V, typename C, typename A>
std::size_t dynamicSize(const std::map<K, V, C, A>& t);

template<typename K, typename V, typename C, typename A>
std::size_t dynamicSize(const std::multimap<K, V, C, A>& t);

template<typename T, typename C, typename A>
std::size_t dynamicSize(const std::set<T, C, A>& t);

template<typename T, typename C, typename A>
std::size_t dynamicSize(const std::multiset<T, C, A>& t);
}

namespace CMemoryDebug {

template<typename T, typename A>
void dynamicSize(const char* name,
                 const std::list<T, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T, typename C, typename A>
void dynamicSize(const char* name,
                 const std::deque<T, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename K, typename V, typename C, typename A>
void dynamicSize(const char* name,
                 const std::map<K, V, C, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename K, typename V, typename C, typename A>
void dynamicSize(const char* name,
                 const std::multimap<K, V, C, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T, typename C, typename A>
void dynamicSize(const char* name,
                 const std::set<T, C, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T, typename C, typename A>
void dynamicSize(const char* name,
                 const std::multiset<T, C, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);
}
}
}

#endif // INCLUDED_ml_core_CMemoryDecStd_h
