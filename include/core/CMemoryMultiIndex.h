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

#ifndef INCLUDED_ml_core_CMemoryMultiIndex_h
#define INCLUDED_ml_core_CMemoryMultiIndex_h

#include <core/BoostMultiIndex.h>
#include <core/CMemory.h>

namespace ml {
namespace core {
template<typename T, typename I, typename A>
std::size_t
CMemory::dynamicSize(const boost::multi_index::multi_index_container<T, I, A>& t) {
    return elementDynamicSize(t) + t.size() * (sizeof(T) + storageNodeOverhead(t));
}

template<typename T, typename I, typename A>
std::size_t
CMemory::storageNodeOverhead(const boost::multi_index::multi_index_container<T, I, A>& t) {
    // It's tricky to determine the container overhead of a multi-index
    // container.  It can have an arbitrary number of indices, each of which
    // can be of a different type.  To accurately determine the overhead
    // would require some serious template metaprogramming to interpret the
    // "typename I" template argument, and it's just not worth it given the
    // infrequent and relatively simple usage (generally just two indices
    // in our current codebase).  Therefore there's an approximation here
    // that the overhead is 2 pointers per entry per index.
    using TMultiIndex = boost::multi_index::multi_index_container<T, I, A>;
    constexpr std::size_t indexCount{
        boost::mpl::size<typename TMultiIndex::index_type_list>::value};
    return 2 * indexCount * sizeof(std::size_t);
}

template<typename T, typename I, typename A>
void CMemoryDebug::dynamicSize(const char* name,
                        const boost::multi_index::multi_index_container<T, I, A>& t,
                        const CMemoryUsage::TMemoryUsagePtr& mem) {
    // It's tricky to determine the container overhead of a multi-index
    // container.  It can have an arbitrary number of indices, each of which
    // can be of a different type.  To accurately determine the overhead
    // would require some serious template metaprogramming to interpret the
    // "typename I" template argument, and it's just not worth it given the
    // infrequent and relatively simple usage (generally just two indices
    // in our current codebase).  Therefore there's an approximation here
    // that the overhead is 2 pointers per entry per index.
    using TMultiIndex = boost::multi_index::multi_index_container<T, I, A>;
    constexpr std::size_t indexCount{
        boost::mpl::size<typename TMultiIndex::index_type_list>::value};
    std::string componentName(name);

    std::size_t items = t.size();
    CMemoryUsage::SMemoryUsage usage(
        componentName + "::" + typeid(T).name(),
        items * (sizeof(T) + 2 * indexCount * sizeof(std::size_t)));
    CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
    ptr->setName(usage);

    elementDynamicSize(std::move(componentName), t, mem);
}
}
}

#endif // INCLUDED_ml_core_CMemoryMultiIndex_h
