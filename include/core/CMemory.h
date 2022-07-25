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

#ifndef INCLUDED_ml_core_CMemory_h
#define INCLUDED_ml_core_CMemory_h

#include <core/CLogger.h>
#include <core/CMemoryFwd.h>
#include <core/CNonInstantiatable.h>
#include <core/UnwrapRef.h>

#include <boost/any.hpp>
#include <boost/circular_buffer_fwd.hpp>
#include <boost/container/container_fwd.hpp>
#include <boost/multi_index_container_fwd.hpp>
#include <boost/unordered/unordered_map_fwd.hpp>
#include <boost/unordered/unordered_set_fwd.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace ml {
namespace core {

template<typename T, std::size_t N>
class CSmallVector;
using TTypeInfoCRef = std::reference_wrapper<const std::type_info>;

namespace memory_detail {

// Windows creates an extra map/list node per map/list
#ifdef Windows
const std::size_t EXTRA_NODES = 1;
#else
const std::size_t EXTRA_NODES = 0;
#endif

// Big variations in deque page size!
#ifdef Windows
const std::size_t MIN_DEQUE_PAGE_SIZE = 16;
const std::size_t MIN_DEQUE_PAGE_VEC_ENTRIES = 8;
#elif defined(MacOSX)
const std::size_t MIN_DEQUE_PAGE_SIZE = 4096;
const std::size_t MIN_DEQUE_PAGE_VEC_ENTRIES = 1;
#else
const std::size_t MIN_DEQUE_PAGE_SIZE = 512;
const std::size_t MIN_DEQUE_PAGE_VEC_ENTRIES = 8;
#endif

//! \brief Default template declaration for CMemoryDynamicSize::dispatch.
template<typename T, typename = void>
struct SMemoryDynamicSize {
    static std::size_t dispatch(const T&) { return 0; }
};

//! \brief Template specialisation where T has member function "memoryUsage()".
// clang-format off
template<typename T>
struct SMemoryDynamicSize<T, std::enable_if_t<
            std::is_same_v<decltype(&T::memoryUsage), std::size_t (T::*)() const>>> {
    static std::size_t dispatch(const T& t) { return t.memoryUsage(); }
};
// clang-format on

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

//! \brief Total ordering of type_info objects.
struct STypeInfoLess {
    template<typename T>
    bool operator()(const std::pair<TTypeInfoCRef, T>& lhs,
                    const std::pair<TTypeInfoCRef, T>& rhs) const {
        return unwrap_ref(lhs.first).before(unwrap_ref(rhs.first));
    }
    template<typename T>
    bool operator()(const std::pair<TTypeInfoCRef, T>& lhs, TTypeInfoCRef rhs) const {
        return unwrap_ref(lhs.first).before(unwrap_ref(rhs));
    }
    template<typename T>
    bool operator()(TTypeInfoCRef lhs, const std::pair<TTypeInfoCRef, T>& rhs) const {
        return unwrap_ref(lhs).before(unwrap_ref(rhs.first));
    }
};

//! Check if a small vector is using in-place storage.
//!
//! A small vector is only using in-place storage if the end of its
//! current capacity is closer to its address than its size. Note
//! that we can't simply check if capacity > N because N is treated
//! as a guideline.
template<typename T, std::size_t N>
bool inplace(const CSmallVector<T, N>& t) {
    const char* address = reinterpret_cast<const char*>(&t);
    const char* storage = reinterpret_cast<const char*>(t.data());
    return storage >= address && storage < address + sizeof t;
}

} // memory_detail::

//! \brief Core memory usage template class.
//!
//! DESCRIPTION:\n
//! Core memory usage template class. Provides a method for determining
//! the memory used by different ml classes and standard containers.
//!
//! ML classes can declare a public member function:
//!     std::size_t memoryUsage() const;
//!
//! which should call CMemory::dynamicSize(t); on all its dynamic members.
//!
//! For virtual hierarchies, the compiler can not determine the size
//! of derived classes from the base pointer, so wherever the afore-
//! mentioned memoryUsage() function is virtual, an associated function
//!
//!     std::size_t staticSize() const;
//!
//! should be declared, returning sizeof(*this).
//!
//! IMPLEMENTATION DECISIONS:\n
//! Template classes and overloads allow the compiler to determine the
//! correct method for arbitrary types.
//!
//! Only contains static members, this should not be instantiated.
//!
class CORE_EXPORT CMemory : private CNonInstantiatable {
public:
    //! Implements a visitor pattern for computing the size of types
    //! stored in boost::any.
    //!
    //! DESCRIPTION:\n
    //! The idea of this class is that the user of dynamicSize should
    //! register call backs to compute the size of objects which are
    //! stored in boost::any. Provided all registered types which will
    //! be visited have been registered then this should correctly
    //! compute the dynamic size used by objects stored in boost::any.
    //! It will warn if a type is visited which is not registered.
    //! There is a singleton visitor available from CMemory. Example
    //! usage is as follows:
    //! \code{.cpp}
    //!   CMemory::anyVisitor().insertCallback<std::vector<double>>();
    //!   std::vector<boost::any> variables;
    //!   variables.push_back(TDoubleVec(10));
    //!   std::size_t size = CMemory::dynamicSize(variables, visitor);
    //! \endcode
    class CORE_EXPORT CAnyVisitor {
    public:
        using TDynamicSizeFunc = std::size_t (*)(const boost::any& any);
        using TTypeInfoDynamicSizeFuncPr = std::pair<TTypeInfoCRef, TDynamicSizeFunc>;
        using TTypeInfoDynamicSizeFuncPrVec = std::vector<TTypeInfoDynamicSizeFuncPr>;

        //! Insert a callback to compute the size of the type T
        //! if it is stored in boost::any.
        template<typename T>
        bool registerCallback() {
            auto i = std::lower_bound(m_Callbacks.begin(), m_Callbacks.end(),
                                      std::cref(typeid(T)),
                                      memory_detail::STypeInfoLess());
            if (i == m_Callbacks.end()) {
                m_Callbacks.emplace_back(std::cref(typeid(T)),
                                         &CAnyVisitor::dynamicSizeCallback<T>);
                return true;
            }
            if (i->first.get() != typeid(T)) {
                m_Callbacks.insert(i, {std::cref(typeid(T)),
                                       &CAnyVisitor::dynamicSizeCallback<T>});
                return true;
            }
            return false;
        }

        //! Calculate the dynamic size of x if a callback has been
        //! registered for its type.
        std::size_t dynamicSize(const boost::any& x) const {
            if (!x.empty()) {
                auto i = std::lower_bound(m_Callbacks.begin(),
                                          m_Callbacks.end(), std::cref(x.type()),
                                          memory_detail::STypeInfoLess());
                if (i != m_Callbacks.end() && i->first.get() == x.type()) {
                    return (*i->second)(x);
                }
                LOG_ERROR(<< "No callback registered for " << x.type().name());
            }
            return 0;
        }

    private:
        //! Wraps up call to any_cast and dynamicSize.
        template<typename T>
        static std::size_t dynamicSizeCallback(const boost::any& any) {
            try {
                return sizeof(T) + CMemory::dynamicSize(boost::any_cast<const T&>(any));
            } catch (const std::exception& e) {
                LOG_ERROR(<< "Failed to calculate size " << e.what());
            }
            return 0;
        }

        TTypeInfoDynamicSizeFuncPrVec m_Callbacks;
    };

public:
    //! Default implementation for non-pointer types.
    template<typename T>
    static std::size_t
    dynamicSize(const T& t, std::enable_if_t<!std::is_pointer_v<T>>* = nullptr) {
        if constexpr (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            return memory_detail::SMemoryDynamicSize<T>::dispatch(t);
        }
        return 0;
    }

    //! Default implementation.
    template<typename T>
    static constexpr std::size_t storageNodeSize(const T&) {
        return 0;
    }

    //! Default implementation for pointer types.
    template<typename T>
    static std::size_t
    dynamicSize(const T& t, std::enable_if_t<std::is_pointer_v<T>>* = nullptr) {
        return t == nullptr ? 0 : staticSize(*t) + dynamicSize(*t);
    }

    template<typename T, typename DELETER>
    static std::size_t dynamicSize(const std::unique_ptr<T, DELETER>& t) {
        return t == nullptr ? 0 : staticSize(*t) + dynamicSize(*t);
    }

    template<typename T>
    static std::size_t dynamicSize(const std::shared_ptr<T>& t) {
        // The check for nullptr here may seem unnecessary but there are situations
        // where an unset shared_ptr can have a use_count greater than 0, see
        // https://stackoverflow.com/questions/48885252/c-sharedptr-use-count-for-nullptr/48885643
        long uc{t == nullptr ? 0 : t.use_count()};
        if (uc == 0) {
            return 0;
        }
        // Note we add on sizeof(long) here to account for the memory
        // used by the shared reference count. Also, round up.
        return (sizeof(long) + staticSize(*t) + dynamicSize(*t) + std::size_t(uc - 1)) / uc;
    }

    template<typename T, std::size_t N>
    static std::size_t dynamicSize(const std::array<T, N>& t) {
        return elementDynamicSize(t);
    }

    template<typename T, typename A>
    static std::size_t dynamicSize(const std::vector<T, A>& t) {
        return elementDynamicSize(t) + sizeof(T) * t.capacity();
    }

    template<typename T, std::size_t N>
    static std::size_t dynamicSize(const CSmallVector<T, N>& t) {
        return elementDynamicSize(t) +
               (memory_detail::inplace(t) ? 0 : t.capacity()) * sizeof(T);
    }

    static std::size_t dynamicSize(const std::string& t) {
        std::size_t capacity = t.capacity();
        // The different STLs we use on various platforms all have different
        // allocation strategies for strings
        // These are hard-coded here, on the assumption that they will not
        // change frequently - but checked by unittests that do runtime
        // verification
#ifdef MacOSX
        if (capacity <= 22) {
            // For lengths up to 22 bytes there is no allocation
            return 0;
        }
#else // Linux (with C++11 ABI) and Windows
        if (capacity <= 15) {
            // For lengths up to 15 bytes there is no allocation
            return 0;
        }
#endif
        return capacity + 1;
    }

    template<typename K, typename V, typename H, typename P, typename A>
    static std::size_t dynamicSize(const boost::unordered_map<K, V, H, P, A>& t) {
        return elementDynamicSize(t) +
               (t.bucket_count() * CMemory::storageNodeOverhead(t)) +
               (t.size() * (sizeof(K) + sizeof(V) + storageNodeOverhead(t)));
    }

    template<typename K, typename V, typename H, typename P, typename A>
    static constexpr std::size_t
    storageNodeOverhead(const boost::unordered_map<K, V, H, P, A>&) {
        return 2 * sizeof(std::size_t);
    }

    template<typename K, typename V, typename C, typename A>
    static std::size_t dynamicSize(const std::map<K, V, C, A>& t) {
        return elementDynamicSize(t) +
               (memory_detail::EXTRA_NODES + t.size()) *
                   (sizeof(K) + sizeof(V) + storageNodeOverhead(t));
    }

    template<typename K, typename V, typename C, typename A>
    static constexpr std::size_t storageNodeOverhead(const std::map<K, V, C, A>&) {
        // std::map appears to use 4 pointers/size_ts per tree node
        // (colour, parent, left and right child pointers).
        return 4 * sizeof(std::size_t);
    }

    template<typename K, typename V, typename C, typename A>
    static std::size_t dynamicSize(const std::multimap<K, V, C, A>& t) {
        // In practice, both std::multimap and std::map use the same
        // rb tree implementation.
        return elementDynamicSize(t) +
               (memory_detail::EXTRA_NODES + t.size()) *
                   (sizeof(K) + sizeof(V) + storageNodeOverhead(t));
    }

    template<typename K, typename V, typename C, typename A>
    static constexpr std::size_t storageNodeOverhead(const std::multimap<K, V, C, A>&) {
        // In practice, both std::multimap and std::map use the same
        // rb tree implementation.
        return 4 * sizeof(std::size_t);
    }

    template<typename K, typename V, typename C, typename A>
    static std::size_t dynamicSize(const boost::container::flat_map<K, V, C, A>& t) {
        return elementDynamicSize(t) + t.capacity() * sizeof(std::pair<K, V>);
    }

    template<typename T, typename H, typename P, typename A>
    static std::size_t dynamicSize(const boost::unordered_set<T, H, P, A>& t) {
        return elementDynamicSize(t) + (t.bucket_count() * sizeof(std::size_t) * 2) +
               (t.size() * (sizeof(T) + storageNodeOverhead(t)));
    }

    template<typename T, typename H, typename P, typename A>
    static constexpr std::size_t
    storageNodeOverhead(const boost::unordered_set<T, H, P, A>&) {
        return 2 * sizeof(std::size_t);
    }

    template<typename T, typename C, typename A>
    static std::size_t dynamicSize(const std::set<T, C, A>& t) {
        return elementDynamicSize(t) + (memory_detail::EXTRA_NODES + t.size()) *
                                           (sizeof(T) + storageNodeOverhead(t));
    }

    template<typename T, typename C, typename A>
    static constexpr std::size_t storageNodeOverhead(const std::set<T, C, A>&) {
        // std::set appears to use 4 pointers/size_ts per tree node
        // (colour, parent, left and right child pointers).
        return 4 * sizeof(std::size_t);
    }

    template<typename T, typename C, typename A>
    static std::size_t dynamicSize(const std::multiset<T, C, A>& t) {
        // In practice, both std::multiset and std::set use the same
        // rb tree implementation.
        return elementDynamicSize(t) + (memory_detail::EXTRA_NODES + t.size()) *
                                           (sizeof(T) + storageNodeOverhead(t));
    }

    template<typename T, typename C, typename A>
    static constexpr std::size_t storageNodeOverhead(const std::multiset<T, C, A>&) {
        // In practice, both std::multiset and std::set use the same
        // rb tree implementation.
        return 4 * sizeof(std::size_t);
    }

    template<typename T, typename C, typename A>
    static std::size_t dynamicSize(const boost::container::flat_set<T, C, A>& t) {
        return elementDynamicSize(t) + t.capacity() * sizeof(T);
    }

    template<typename T, typename A>
    static std::size_t dynamicSize(const std::list<T, A>& t) {
        return elementDynamicSize(t) + (memory_detail::EXTRA_NODES + t.size()) *
                                           (sizeof(T) + storageNodeOverhead(t));
    }

    template<typename T, typename A>
    static constexpr std::size_t storageNodeOverhead(const std::list<T, A>&) {
        // std::list appears to use 2 pointers per list node
        // (prev and next pointers).
        return 2 * sizeof(std::size_t);
    }

    template<typename T, typename A>
    static std::size_t dynamicSize(const std::deque<T, A>& t) {
        // std::deque is a pointer to an array of pointers to pages
        std::size_t pageSize = std::max(sizeof(T), memory_detail::MIN_DEQUE_PAGE_SIZE);
        std::size_t itemsPerPage = pageSize / sizeof(T);
        // This could be an underestimate if items have been removed
        std::size_t numPages = (t.size() + itemsPerPage - 1) / itemsPerPage;
        // This could also be an underestimate if items have been removed
        std::size_t pageVecEntries = std::max(numPages, memory_detail::MIN_DEQUE_PAGE_VEC_ENTRIES);

        return elementDynamicSize(t) + pageVecEntries * sizeof(std::size_t) +
               numPages * pageSize;
    }

    template<typename T, typename I, typename A>
    static std::size_t
    dynamicSize(const boost::multi_index::multi_index_container<T, I, A>& t);

    template<typename T, typename I, typename A>
    static std::size_t
    storageNodeOverhead(const boost::multi_index::multi_index_container<T, I, A>& t);

    template<typename T, typename A>
    static std::size_t dynamicSize(const boost::circular_buffer<T, A>& t) {
        return elementDynamicSize(t) + t.capacity() * sizeof(T);
    }

    template<typename T>
    static std::size_t dynamicSize(const std::optional<T>& t) {
        if (!t) {
            return 0;
        }
        return dynamicSize(*t);
    }

    template<typename T>
    static std::size_t dynamicSize(const std::reference_wrapper<T>& /*t*/) {
        return 0;
    }

    template<typename T, typename V>
    static std::size_t dynamicSize(const std::pair<T, V>& t) {
        std::size_t mem = 0;
        if constexpr (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            mem += dynamicSize(t.first);
        }
        if constexpr (!memory_detail::SDynamicSizeAlwaysZero<V>::value()) {
            mem += dynamicSize(t.second);
        }
        return mem;
    }

    static std::size_t dynamicSize(const boost::any& t) {
        // boost::any holds a pointer to a new'd item.
        return ms_AnyVisitor.dynamicSize(t);
    }

    //! Default template.
    template<typename T>
    static std::size_t staticSize(const T& t) {
        return memory_detail::SMemoryStaticSize<T>::dispatch(t);
    }

    //! Get the any visitor singleton.
    static CAnyVisitor& anyVisitor() { return ms_AnyVisitor; }

private:
    template<typename CONTAINER>
    static std::size_t elementDynamicSize(const CONTAINER& t) {
        std::size_t mem = 0;
        if constexpr (!memory_detail::SDynamicSizeAlwaysZero<typename CONTAINER::value_type>::value()) {
            for (const auto& v : t) {
                mem += dynamicSize(v);
            }
        }
        return mem;
    }

private:
    static CAnyVisitor ms_AnyVisitor;
};

namespace memory_detail {

//! Default template declaration for SDebugMemoryDynamicSize::dispatch.
template<typename T, typename = void>
struct SDebugMemoryDynamicSize {
    static void dispatch(const char* name, const T& t, const CMemoryUsage::TMemoryUsagePtr& mem) {
        std::size_t used = CMemory::dynamicSize(t);
        if (used > 0) {
            std::string description(name);
            description += "::";
            description += typeid(T).name();
            mem->addItem(description, used);
        }
    }
};

//! Template specialisation for when T has a debugMemoryUsage member function.
// clang-format off
template<typename T>
struct SDebugMemoryDynamicSize<T, std::enable_if_t<
            std::is_same_v<decltype(&T::debugMemoryUsage), void (T::*)(const CMemoryUsage::TMemoryUsagePtr&) const>>> {
    static void dispatch(const char*, const T& t, const CMemoryUsage::TMemoryUsagePtr& mem) {
        t.debugMemoryUsage(mem->addChild());
    }
};
// clang-format on
}

//! \brief Core memory debug usage template class.
//!
//! DESCRIPTION:\n
//! Core memory debug usage template class. Provides an extension to the
//! CMemory class for creating a detailed breakdown of memory used by
//! classes and containers, utilising the CMemoryUsage class.
//!
//! ML classes can declare a public member function:
//!     void debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr&) const;
//!
//! which should call CMemoryDebug::dynamicSize("t_name", t, memUsagePtr)
//! on all its dynamic members.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Template classes and overloads to allow the compiler to determine the
//! correct method for arbitrary types.
//!
//! Only contains static members, this should not be instantiated.
//!
class CORE_EXPORT CMemoryDebug : private CNonInstantiatable {
public:
    //! Implements a visitor pattern for computing the size of types
    //! stored in boost::any.
    //!
    //! DESCRIPTION:\n
    //! See CMemory::CAnyVisitor for details.
    class CORE_EXPORT CAnyVisitor {
    public:
        using TDynamicSizeFunc = void (*)(const char*,
                                          const boost::any& any,
                                          const CMemoryUsage::TMemoryUsagePtr& mem);
        using TTypeInfoDynamicSizeFuncPr = std::pair<TTypeInfoCRef, TDynamicSizeFunc>;
        using TTypeInfoDynamicSizeFuncPrVec = std::vector<TTypeInfoDynamicSizeFuncPr>;

        //! Insert a callback to compute the size of the type T
        //! if it is stored in boost::any.
        template<typename T>
        bool registerCallback() {
            auto i = std::lower_bound(m_Callbacks.begin(), m_Callbacks.end(),
                                      std::cref(typeid(T)),
                                      memory_detail::STypeInfoLess());
            if (i == m_Callbacks.end()) {
                m_Callbacks.emplace_back(std::cref(typeid(T)),
                                         &CAnyVisitor::dynamicSizeCallback<T>);
                return true;
            }
            if (i->first.get() != typeid(T)) {
                m_Callbacks.insert(i, {std::cref(typeid(T)),
                                       &CAnyVisitor::dynamicSizeCallback<T>});
                return true;
            }
            return false;
        }

        //! Calculate the dynamic size of x if a callback has been
        //! registered for its type.
        void dynamicSize(const char* name,
                         const boost::any& x,
                         const CMemoryUsage::TMemoryUsagePtr& mem) const {
            if (!x.empty()) {
                auto i = std::lower_bound(m_Callbacks.begin(),
                                          m_Callbacks.end(), std::cref(x.type()),
                                          memory_detail::STypeInfoLess());
                if (i != m_Callbacks.end() && i->first.get() == x.type()) {
                    (*i->second)(name, x, mem);
                    return;
                }
                LOG_ERROR(<< "No callback registered for " << x.type().name());
            }
        }

    private:
        //! Wraps up call to any_cast and dynamicSize.
        template<typename T>
        static void dynamicSizeCallback(const char* name,
                                        const boost::any& any,
                                        const CMemoryUsage::TMemoryUsagePtr& mem) {
            try {
                mem->addItem(name, sizeof(T));
                CMemoryDebug::dynamicSize(name, boost::any_cast<const T&>(any), mem);
            } catch (const std::exception& e) {
                LOG_ERROR(<< "Failed to calculate size " << e.what());
            }
        }

        TTypeInfoDynamicSizeFuncPrVec m_Callbacks;
    };

public:
    //! Default implementation for non-pointer types.
    template<typename T>
    static void dynamicSize(const char* name,
                            const T& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem,
                            std::enable_if_t<!std::is_pointer_v<T>>* = nullptr) {
        memory_detail::SDebugMemoryDynamicSize<T>::dispatch(name, t, mem);
    }

    //! Default implementation for pointer types.
    template<typename T>
    static void dynamicSize(const char* name,
                            const T& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem,
                            std::enable_if_t<std::is_pointer_v<T>>* = nullptr) {
        if (t != nullptr) {
            std::string ptrName(name);
            ptrName += "_ptr";
            mem->addItem(ptrName.c_str(), CMemory::staticSize(*t));
            memory_detail::SDebugMemoryDynamicSize<T>::dispatch(name, *t, mem);
        }
    }

    template<typename T>
    static void dynamicSize(const char* name,
                            const std::unique_ptr<T>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        if (t != nullptr) {
            std::string ptrName(name);
            ptrName += "_ptr";
            mem->addItem(ptrName.c_str(), CMemory::staticSize(*t));
            memory_detail::SDebugMemoryDynamicSize<T>::dispatch(name, *t, mem);
        }
    }

    template<typename T>
    static void dynamicSize(const char* name,
                            const std::shared_ptr<T>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        // The check for nullptr here may seem unnecessary but there are situations
        // where an unset shared_ptr can have a use_count greater than 0, see
        // https://stackoverflow.com/questions/48885252/c-sharedptr-use-count-for-nullptr/48885643
        long uc{t == nullptr ? 0 : t.use_count()};
        if (uc == 0) {
            return;
        }
        // If the pointer is shared by multiple users, each one
        // might count it, so divide by the number of users.
        // However, if only 1 user has it, do a full debug.
        std::string ptrName(name);
        if (uc == 1) {
            // Note we add on sizeof(long) here to account for
            // the memory used by the shared reference count.
            ptrName += "_shared_ptr";
            // Note we add on sizeof(long) here to account for
            // the memory used by the shared reference count.
            mem->addItem(ptrName, sizeof(long) + CMemory::staticSize(*t));
            CMemoryDebug::dynamicSize(name, *t, mem);
        } else {
            ptrName += "shared_ptr (x" + std::to_string(uc) + ")";
            // Note we add on sizeof(long) here to account for
            // the memory used by the shared reference count.
            // Also, round up.
            mem->addItem(ptrName, (sizeof(long) + CMemory::staticSize(*t) +
                                   CMemory::dynamicSize(*t) + std::size_t(uc - 1)) /
                                      uc);
        }
    }

    template<typename T, std::size_t N>
    static void dynamicSize(const char* name,
                            const std::array<T, N>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        if constexpr (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            std::string elementName{name};
            CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
            elementDynamicSize(std::move(elementName), t, mem);
        }
    }

    template<typename T, typename A>
    static void dynamicSize(const char* name,
                            const std::vector<T, A>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        std::string componentName(name);

        std::size_t items = t.size();
        std::size_t capacity = t.capacity();
        CMemoryUsage::SMemoryUsage usage(componentName + "::" + typeid(T).name(),
                                         capacity * sizeof(T),
                                         (capacity - items) * sizeof(T));
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        elementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename T, std::size_t N>
    static void dynamicSize(const char* name,
                            const CSmallVector<T, N>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        std::string componentName(name);

        std::size_t items = memory_detail::inplace(t) ? 0 : t.size();
        std::size_t capacity = memory_detail::inplace(t) ? 0 : t.capacity();
        CMemoryUsage::SMemoryUsage usage(componentName + "::" + typeid(T).name(),
                                         capacity * sizeof(T),
                                         (capacity - items) * sizeof(T));
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        elementDynamicSize(std::move(componentName), t, mem);
    }

    static void dynamicSize(const char* name,
                            const std::string& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        std::string componentName(name);
        componentName += "_string";
        std::size_t length = t.size();
        std::size_t capacity = t.capacity();
        std::size_t unused = 0;
#ifdef MacOSX
        if (capacity > 22) {
            unused = capacity - length;
            ++capacity;
        } else {
            // For lengths up to 22 bytes there is no allocation
            capacity = 0;
        }
#else // Linux (with C++11 ABI) and Windows
        if (capacity > 15) {
            unused = capacity - length;
            ++capacity;
        } else {
            // For lengths up to 15 bytes there is no allocation
            capacity = 0;
        }
#endif
        CMemoryUsage::SMemoryUsage usage(componentName, capacity, unused);
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);
    }

    template<typename K, typename V, typename H, typename P, typename A>
    static void dynamicSize(const char* name,
                            const boost::unordered_map<K, V, H, P, A>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        std::string componentName(name);
        componentName += "_umap";

        std::size_t mapSize =
            (t.bucket_count() * sizeof(std::size_t) * 2) +
            (t.size() * (sizeof(K) + sizeof(V) + 2 * sizeof(std::size_t)));

        CMemoryUsage::SMemoryUsage usage(componentName, mapSize);
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        associativeElementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename K, typename V, typename C, typename A>
    static void dynamicSize(const char* name,
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

        associativeElementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename K, typename V, typename C, typename A>
    static void dynamicSize(const char* name,
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

        associativeElementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename K, typename V, typename C, typename A>
    static void dynamicSize(const char* name,
                            const boost::container::flat_map<K, V, C, A>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        std::string componentName(name);
        componentName += "_fmap";

        std::size_t items = t.size();
        std::size_t capacity = t.capacity();

        CMemoryUsage::SMemoryUsage usage(
            componentName + "::" + typeid(std::pair<K, V>).name(),
            capacity * sizeof(std::pair<K, V>),
            (capacity - items) * sizeof(std::pair<K, V>));
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        associativeElementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename T, typename H, typename P, typename A>
    static void dynamicSize(const char* name,
                            const boost::unordered_set<T, H, P, A>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        std::string componentName(name);
        componentName += "_uset";

        std::size_t setSize = (t.bucket_count() * CMemory::storageNodeOverhead(t)) +
                              (t.size() * (sizeof(T) + CMemory::storageNodeOverhead(t)));

        CMemoryUsage::SMemoryUsage usage(componentName, setSize);
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        elementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename T, typename C, typename A>
    static void dynamicSize(const char* name,
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

        elementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename T, typename C, typename A>
    static void dynamicSize(const char* name,
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

        elementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename T, typename C, typename A>
    static void dynamicSize(const char* name,
                            const boost::container::flat_set<T, C, A>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        std::string componentName(name);
        componentName += "_fset";

        std::size_t items = t.size();
        std::size_t capacity = t.capacity();

        CMemoryUsage::SMemoryUsage usage(componentName + "::" + typeid(T).name(),
                                         capacity * sizeof(T),
                                         (capacity - items) * sizeof(T));
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        elementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename T, typename A>
    static void dynamicSize(const char* name,
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

        elementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename T, typename C, typename A>
    static void dynamicSize(const char* name,
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

        elementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename T, typename I, typename A>
    static void dynamicSize(const char* name,
                            const boost::multi_index::multi_index_container<T, I, A>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem);

    template<typename T, typename A>
    static void dynamicSize(const char* name,
                            const boost::circular_buffer<T, A>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        std::string componentName(name);

        std::size_t items = t.size();
        std::size_t capacity = t.capacity();
        CMemoryUsage::SMemoryUsage usage(componentName + "::" + typeid(T).name(),
                                         capacity * sizeof(T),
                                         (capacity - items) * sizeof(T));
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        elementDynamicSize(std::move(componentName), t, mem);
    }

    template<typename T>
    static void dynamicSize(const char* name,
                            const std::optional<T>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        if (t) {
            dynamicSize(name, *t, mem);
        }
    }

    template<typename T>
    static void dynamicSize(const char* /*name*/,
                            const std::reference_wrapper<T>& /*t*/,
                            const CMemoryUsage::TMemoryUsagePtr& /*mem*/) {}

    template<typename T, typename V>
    static void dynamicSize(const char* name,
                            const std::pair<T, V>& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        std::string keyName(name);
        keyName += "_key";
        std::string valueName(name);
        valueName += "_value";
        dynamicSize(keyName.c_str(), t.first, mem);
        dynamicSize(valueName.c_str(), t.second, mem);
    }

    static void dynamicSize(const char* name,
                            const boost::any& t,
                            const CMemoryUsage::TMemoryUsagePtr& mem) {
        // boost::any holds a pointer to a new'd item.
        ms_AnyVisitor.dynamicSize(name, t, mem);
    }

    //! Get the any visitor singleton.
    static CAnyVisitor& anyVisitor() { return ms_AnyVisitor; }

private:
    template<typename CONTAINER>
    static void elementDynamicSize(std::string name,
                                   const CONTAINER& t,
                                   const CMemoryUsage::TMemoryUsagePtr& mem) {
        if constexpr (!memory_detail::SDynamicSizeAlwaysZero<typename CONTAINER::value_type>::value()) {
            std::string elementName{name};
            elementName += "_item";
            for (const auto& v : t) {
                dynamicSize(elementName.c_str(), v, mem);
            }
        }
    }

    template<typename CONTAINER>
    static void associativeElementDynamicSize(std::string name,
                                              const CONTAINER& t,
                                              const CMemoryUsage::TMemoryUsagePtr& mem) {
        if constexpr (!memory_detail::SDynamicSizeAlwaysZero<typename CONTAINER::value_type>::value()) {
            std::string keyName{name + "_key"};
            std::string valueName{name + "_value"};
            for (const auto & [ key, value ] : t) {
                dynamicSize(keyName.c_str(), key, mem);
                dynamicSize(valueName.c_str(), value, mem);
            }
        }
    }

private:
    static CAnyVisitor ms_AnyVisitor;
};
}
}

#endif // INCLUDED_ml_core_CMemory_h
