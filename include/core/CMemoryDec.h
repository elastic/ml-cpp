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

#ifndef INCLUDED_ml_core_CMemoryDec_h
#define INCLUDED_ml_core_CMemoryDec_h

#include <core/CMemoryFwd.h>

#include <boost/circular_buffer_fwd.hpp>
#include <boost/container/container_fwd.hpp>
#include <boost/multi_index_container_fwd.hpp>
#include <boost/unordered/unordered_map_fwd.hpp>
#include <boost/unordered/unordered_set_fwd.hpp>

#include <any>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace ml {
namespace core {
template<typename T, std::size_t N>
class CSmallVector;

namespace memory_detail {
//! \brief Default template for classes that don't sport a staticSize member.
template<typename T, typename ENABLE = void>
struct SMemoryStaticSize {
    static std::size_t dispatch(const T& /*t*/) { return sizeof(T); }
};

//! \brief Template specialisation for classes having a staticSize member:
//! used when base class pointers are passed to dynamicSize().
// clang-format off
template<typename T>
struct SMemoryStaticSize<T, std::enable_if_t<
            std::is_same_v<decltype(&T::staticSize), std::size_t (T::*)() const>>> {
    static std::size_t dispatch(const T& t) { return t.staticSize(); }
};
// clang-format on
}

//! \brief Core memory usage template class.
//!
//! DESCRIPTION:\n
//! Core memory usage template class. Provides a method for determining
//! the memory used by different ml classes and standard containers.
//!
//! ML classes can declare a public member function:
//! \code{.cpp}
//!     std::size_t memoryUsage() const;
//! \endcode
//! which should call CMemory::dynamicSize(t); on all its dynamic members.
//!
//! For virtual hierarchies, the compiler can not determine the size
//! of derived classes from the base pointer, so wherever the afore-
//! mentioned memoryUsage() function is virtual, an associated function
//! \code{.cpp}
//!     std::size_t staticSize() const;
//! \endcode
//! should be declared, returning sizeof(*this).
namespace CMemory {

//! Default implementation.
template<typename T>
std::size_t staticSize(const T& t) {
    return memory_detail::SMemoryStaticSize<T>::dispatch(t);
}

//! Default implementation.
template<typename T>
constexpr std::size_t storageNodeOverhead(const T&) {
    return 0;
}

template<typename K, typename V, typename H, typename P, typename A>
constexpr std::size_t storageNodeOverhead(const boost::unordered_map<K, V, H, P, A>&) {
    return 2 * sizeof(std::size_t);
}

template<typename T, typename H, typename P, typename A>
constexpr std::size_t storageNodeOverhead(const boost::unordered_set<T, H, P, A>&) {
    return 2 * sizeof(std::size_t);
}

//! Default implementation for non-pointer types.
template<typename T>
std::size_t dynamicSize(const T& t, std::enable_if_t<!std::is_pointer_v<T>>* = nullptr);

//! Default implementation for pointer types.
template<typename T>
std::size_t dynamicSize(const T& t, std::enable_if_t<std::is_pointer_v<T>>* = nullptr);

template<typename T, typename DELETER>
std::size_t dynamicSize(const std::unique_ptr<T, DELETER>& t);

template<typename T>
std::size_t dynamicSize(const std::shared_ptr<T>& t);

template<typename T, std::size_t N>
std::size_t dynamicSize(const std::array<T, N>& t);

template<typename T, typename A>
std::size_t dynamicSize(const std::vector<T, A>& t);

template<typename T, std::size_t N>
std::size_t dynamicSize(const CSmallVector<T, N>& t);

template<typename K, typename V, typename H, typename P, typename A>
std::size_t dynamicSize(const boost::unordered_map<K, V, H, P, A>& t);

template<typename K, typename V, typename C, typename A>
std::size_t dynamicSize(const boost::container::flat_map<K, V, C, A>& t);

template<typename T, typename H, typename P, typename A>
std::size_t dynamicSize(const boost::unordered_set<T, H, P, A>& t);

template<typename T, typename C, typename A>
std::size_t dynamicSize(const boost::container::flat_set<T, C, A>& t);

template<typename T, typename A>
std::size_t dynamicSize(const boost::circular_buffer<T, A>& t);

template<typename T>
std::size_t dynamicSize(const std::optional<T>& t);

template<typename T>
std::size_t dynamicSize(const std::reference_wrapper<T>& /*t*/);

template<typename T, typename V>
std::size_t dynamicSize(const std::pair<T, V>& t);

CORE_EXPORT
std::size_t dynamicSize(const std::string& t);

CORE_EXPORT
std::size_t dynamicSize(const std::any& t);

template<typename T, typename I, typename A>
std::size_t dynamicSize(const boost::multi_index::multi_index_container<T, I, A>& t);

//! Helper to compute container element dynamic memory usage.
template<typename CONTAINER>
std::size_t elementDynamicSize(const CONTAINER& t);
}

//! \brief Core memory debug usage template class.
//!
//! DESCRIPTION:\n
//! Core memory debug usage template class. Provides an extension to the
//! CMemory class for creating a detailed breakdown of memory used by
//! classes and containers, utilising the CMemoryUsage class.
//!
//! ML classes can declare a public member function:
//! \code{.cpp}
//!     void debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr&) const;
//! \endcode
//! which should call CMemoryDebug::dynamicSize("t_name", t, memUsagePtr)
//! on all its dynamic members.
namespace CMemoryDebug {

//! Default implementation for non-pointer types.
template<typename T>
void dynamicSize(const char* name,
                 const T& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem,
                 std::enable_if_t<!std::is_pointer_v<T>>* = nullptr);

//! Default implementation for pointer types.
template<typename T>
void dynamicSize(const char* name,
                 const T& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem,
                 std::enable_if_t<std::is_pointer_v<T>>* = nullptr);

template<typename T>
void dynamicSize(const char* name,
                 const std::unique_ptr<T>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T>
void dynamicSize(const char* name,
                 const std::shared_ptr<T>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T, std::size_t N>
void dynamicSize(const char* name,
                 const std::array<T, N>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T, typename A>
void dynamicSize(const char* name,
                 const std::vector<T, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T, std::size_t N>
void dynamicSize(const char* name,
                 const CSmallVector<T, N>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename K, typename V, typename H, typename P, typename A>
void dynamicSize(const char* name,
                 const boost::unordered_map<K, V, H, P, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename K, typename V, typename C, typename A>
void dynamicSize(const char* name,
                 const boost::container::flat_map<K, V, C, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T, typename H, typename P, typename A>
void dynamicSize(const char* name,
                 const boost::unordered_set<T, H, P, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T, typename C, typename A>
void dynamicSize(const char* name,
                 const boost::container::flat_set<T, C, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T, typename A>
void dynamicSize(const char* name,
                 const boost::circular_buffer<T, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T>
void dynamicSize(const char* name,
                 const std::optional<T>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T>
void dynamicSize(const char* /*name*/,
                 const std::reference_wrapper<T>& /*t*/,
                 const CMemoryUsage::TMemoryUsagePtr& /*mem*/);

template<typename U, typename V>
void dynamicSize(const char* name,
                 const std::pair<U, V>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

CORE_EXPORT
void dynamicSize(const char* name, const std::string& t, const CMemoryUsage::TMemoryUsagePtr& mem);

CORE_EXPORT
void dynamicSize(const char* name, const std::any& t, const CMemoryUsage::TMemoryUsagePtr& mem);

template<typename T, typename I, typename A>
void dynamicSize(const char* name,
                 const boost::multi_index::multi_index_container<T, I, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem);

//! Helper to debug container element dynamic memory usage.
template<typename CONTAINER>
void elementDynamicSize(std::string name,
                        const CONTAINER& t,
                        const CMemoryUsage::TMemoryUsagePtr& mem);

//! Helper to debug associative container element dynamic memory usage.
template<typename CONTAINER>
void associativeElementDynamicSize(std::string name,
                                   const CONTAINER& t,
                                   const CMemoryUsage::TMemoryUsagePtr& mem);
}
}
}

#endif // INCLUDED_ml_core_CMemoryDec_h
