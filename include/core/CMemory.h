/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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
#ifndef INCLUDED_ml_core_CMemory_h
#define INCLUDED_ml_core_CMemory_h

#include <core/CLogger.h>
#include <core/CMemoryUsage.h>
#include <core/CNonInstantiatable.h>

#include <boost/any.hpp>
#include <boost/array.hpp>
#include <boost/circular_buffer_fwd.hpp>
#include <boost/container/container_fwd.hpp>
#include <boost/optional/optional_fwd.hpp>
#include <boost/ref.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_pod.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/unordered/unordered_map_fwd.hpp>
#include <boost/unordered/unordered_set_fwd.hpp>
#include <boost/utility/enable_if.hpp>

#include <cstddef>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>

namespace ml {
namespace core {

template<typename T, std::size_t N>
class CSmallVector;
using TTypeInfoCRef = boost::reference_wrapper<const std::type_info>;

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

template<typename T, std::size_t (T::*)() const, typename R = void>
struct enable_if_member_function {
    using type = R;
};

template<bool (*)(), typename R = void>
struct enable_if_function {
    using type = R;
};

//! \brief Default template declaration for CMemoryDynamicSize::dispatch.
template<typename T, typename ENABLE = void>
struct SMemoryDynamicSize {
    static std::size_t dispatch(const T&) { return 0; }
};

//! \brief Template specialisation where T has member function "memoryUsage()"
template<typename T>
struct SMemoryDynamicSize<T, typename enable_if_member_function<T, &T::memoryUsage>::type> {
    static std::size_t dispatch(const T& t) { return t.memoryUsage(); }
};

//! \brief Default template for classes that don't sport a staticSize member.
template<typename T, typename ENABLE = void>
struct SMemoryStaticSize {
    static std::size_t dispatch(const T& /*t*/) { return sizeof(T); }
};

//! \brief Template specialisation for classes having a staticSize member:
//! used when base class pointers are passed to dynamicSize().
template<typename T>
struct SMemoryStaticSize<T, typename enable_if_member_function<T, &T::staticSize>::type> {
    static std::size_t dispatch(const T& t) { return t.staticSize(); }
};

//! \brief Base implementation checks for POD.
template<typename T, typename ENABLE = void>
struct SDynamicSizeAlwaysZero {
    static inline bool value() { return boost::is_pod<T>::value; }
};

//! \brief Checks types in pair.
template<typename U, typename V>
struct SDynamicSizeAlwaysZero<std::pair<U, V>> {
    static inline bool value() { return SDynamicSizeAlwaysZero<U>::value() && SDynamicSizeAlwaysZero<V>::value(); }
};

//! \brief Specialisation for std::less always true.
template<typename T>
struct SDynamicSizeAlwaysZero<std::less<T>> {
    static inline bool value() { return true; }
};

//! \brief Specialisation for std::greater always true.
template<typename T>
struct SDynamicSizeAlwaysZero<std::greater<T>> {
    static inline bool value() { return true; }
};

//! \brief Checks type in optional.
template<typename T>
struct SDynamicSizeAlwaysZero<boost::optional<T>> {
    static inline bool value() { return SDynamicSizeAlwaysZero<T>::value(); }
};

//! \brief Check for member dynamicSizeAlwaysZero function.
template<typename T>
struct SDynamicSizeAlwaysZero<T, typename enable_if_function<&T::dynamicSizeAlwaysZero>::type> {
    static inline bool value() { return T::dynamicSizeAlwaysZero(); }
};

//! \brief Total ordering of type_info objects.
struct STypeInfoLess {
    template<typename T>
    bool operator()(const std::pair<TTypeInfoCRef, T>& lhs, const std::pair<TTypeInfoCRef, T>& rhs) const {
        return boost::unwrap_ref(lhs.first).before(boost::unwrap_ref(rhs.first));
    }
    template<typename T>
    bool operator()(const std::pair<TTypeInfoCRef, T>& lhs, TTypeInfoCRef rhs) const {
        return boost::unwrap_ref(lhs.first).before(boost::unwrap_ref(rhs));
    }
    template<typename T>
    bool operator()(TTypeInfoCRef lhs, const std::pair<TTypeInfoCRef, T>& rhs) const {
        return boost::unwrap_ref(lhs).before(boost::unwrap_ref(rhs.first));
    }
};

//! Check if a small vector is using in-place storage.
//!
//! A small vector is only using in-place storage if the end of its
//! current capacity is closer to its address than its size. Note
//! that we can't simply check if capacity > N because N is treated
//! as a guideline.
template<typename T, std::size_t N>
static bool inplace(const CSmallVector<T, N>& t) {
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
private:
    static const std::string EMPTY_STRING;

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
    //! \code{cpp}
    //! CMemory::anyVisitor().insertCallback<std::vector<double>>();
    //! std::vector<boost::any> variables;
    //! variables.push_back(TDoubleVec(10));
    //! std::size_t size = CMemory::dynamicSize(variables, visitor);
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
            auto i = std::lower_bound(m_Callbacks.begin(), m_Callbacks.end(), boost::cref(typeid(T)), memory_detail::STypeInfoLess());
            if (i == m_Callbacks.end()) {
                m_Callbacks.emplace_back(boost::cref(typeid(T)), &CAnyVisitor::dynamicSizeCallback<T>);
                return true;
            } else if (i->first.get() != typeid(T)) {
                m_Callbacks.insert(i, {boost::cref(typeid(T)), &CAnyVisitor::dynamicSizeCallback<T>});
                return true;
            }
            return false;
        }

        //! Calculate the dynamic size of x if a callback has been
        //! registered for its type.
        std::size_t dynamicSize(const boost::any& x) const {
            if (!x.empty()) {
                auto i = std::lower_bound(m_Callbacks.begin(), m_Callbacks.end(), boost::cref(x.type()), memory_detail::STypeInfoLess());
                if (i != m_Callbacks.end() && i->first.get() == x.type()) {
                    return (*i->second)(x);
                }
                LOG_ERROR("No callback registered for " << x.type().name());
            }
            return 0;
        }

    private:
        //! Wraps up call to any_cast and dynamicSize.
        template<typename T>
        static std::size_t dynamicSizeCallback(const boost::any& any) {
            try {
                return sizeof(T) + CMemory::dynamicSize(boost::any_cast<const T&>(any));
            } catch (const std::exception& e) { LOG_ERROR("Failed to calculate size " << e.what()); }
            return 0;
        }

        TTypeInfoDynamicSizeFuncPrVec m_Callbacks;
    };

public:
    //! Default template.
    template<typename T>
    static std::size_t dynamicSize(const T& t, typename boost::disable_if<typename boost::is_pointer<T>>::type* = nullptr) {
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            mem += memory_detail::SMemoryDynamicSize<T>::dispatch(t);
        }
        return mem;
    }

    //! Overload for pointer.
    template<typename T>
    static std::size_t dynamicSize(const T& t, typename boost::enable_if<typename boost::is_pointer<T>>::type* = nullptr) {
        if (t == nullptr) {
            return 0;
        }
        return staticSize(*t) + dynamicSize(*t);
    }

    //! Overload for boost::shared_ptr.
    template<typename T>
    static std::size_t dynamicSize(const boost::shared_ptr<T>& t) {
        if (!t) {
            return 0;
        }
        long uc = t.use_count();
        // Round up
        return (staticSize(*t) + dynamicSize(*t) + std::size_t(uc - 1)) / uc;
    }

    //! Overload for boost::array.
    template<typename T, std::size_t N>
    static std::size_t dynamicSize(const boost::array<T, N>& t) {
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        return mem;
    }

    //! Overload for std::vector.
    template<typename T>
    static std::size_t dynamicSize(const std::vector<T>& t) {
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        return mem + sizeof(T) * t.capacity();
    }

    //! Overload for small vector.
    template<typename T, std::size_t N>
    static std::size_t dynamicSize(const CSmallVector<T, N>& t) {
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        return mem + (memory_detail::inplace(t) ? 0 : t.capacity()) * sizeof(T);
    }

    //! Overload for std::string.
    static std::size_t dynamicSize(const std::string& t) {
        std::size_t capacity = t.capacity();
        // The different STLs we use on various platforms all have different
        // allocation strategies for strings
        // These are hard-coded here, on the assumption that they will not
        // change frequently - but checked by unittests that do runtime
        // verification
        // See http://linux/wiki/index.php/Technical_design_issues#std::string
#ifdef MacOSX
        if (capacity <= 22) {
            // For lengths up to 22 bytes there is no allocation
            return 0;
        }
        return capacity + 1;

#elif defined(Linux) && (!defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI == 0)
        // All sizes > 0 use the heap, and the string structure is
        // 1 pointer + 2 sizes + 1 null terminator
        // We don't handle the reference counting, so may overestimate
        // Even some 0 length strings may use the heap - see
        // http://info.prelert.com/blog/clearing-strings
        if (capacity == 0 && t.data() == EMPTY_STRING.data()) {
            return 0;
        }
        return capacity + sizeof(void*) + (2 * sizeof(std::size_t)) + 1;

#else // Linux with C++11 ABI and Windows
        if (capacity <= 15) {
            // For lengths up to 15 bytes there is no allocation
            return 0;
        }
        return capacity + 1;
#endif
    }

    //! Overload for boost::unordered_map.
    template<typename K, typename V, typename H, typename P, typename A>
    static std::size_t dynamicSize(const boost::unordered_map<K, V, H, P, A>& t) {
        std::size_t mem = 0;
        if (!(memory_detail::SDynamicSizeAlwaysZero<K>::value() && memory_detail::SDynamicSizeAlwaysZero<V>::value())) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        return mem + (t.bucket_count() * sizeof(std::size_t) * 2) + (t.size() * (sizeof(K) + sizeof(V) + 2 * sizeof(std::size_t)));
    }

    //! Overload for std::map.
    template<typename K, typename V, typename C, typename A>
    static std::size_t dynamicSize(const std::map<K, V, C, A>& t) {
        // std::map appears to use 4 pointers/size_ts per tree node
        // (colour, parent, left and right child pointers).
        std::size_t mem = 0;
        if (!(memory_detail::SDynamicSizeAlwaysZero<K>::value() && memory_detail::SDynamicSizeAlwaysZero<V>::value())) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        return mem + (memory_detail::EXTRA_NODES + t.size()) * (sizeof(K) + sizeof(V) + 4 * sizeof(std::size_t));
    }

    //! Overload for boost::container::flat_map.
    template<typename K, typename V, typename C, typename A>
    static std::size_t dynamicSize(const boost::container::flat_map<K, V, C, A>& t) {
        std::size_t mem = 0;
        if (!(memory_detail::SDynamicSizeAlwaysZero<K>::value() && memory_detail::SDynamicSizeAlwaysZero<V>::value())) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        return mem + t.capacity() * sizeof(std::pair<K, V>);
    }

    //! Overload for boost::unordered_set.
    template<typename T, typename H, typename P, typename A>
    static std::size_t dynamicSize(const boost::unordered_set<T, H, P, A>& t) {
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        return mem + (t.bucket_count() * sizeof(std::size_t) * 2) + (t.size() * (sizeof(T) + 2 * sizeof(std::size_t)));
    }

    //! Overload for std::set.
    template<typename T, typename C, typename A>
    static std::size_t dynamicSize(const std::set<T, C, A>& t) {
        // std::set appears to use 4 pointers/size_ts per tree node
        // (colour, parent, left and right child pointers).
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        return mem + (memory_detail::EXTRA_NODES + t.size()) * (sizeof(T) + 4 * sizeof(std::size_t));
    }

    //! Overload for boost::container::flat_set.
    template<typename T, typename C, typename A>
    static std::size_t dynamicSize(const boost::container::flat_set<T, C, A>& t) {
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        return mem + t.capacity() * sizeof(T);
    }

    //! Overload for std::list.
    template<typename T, typename A>
    static std::size_t dynamicSize(const std::list<T, A>& t) {
        // std::list appears to use 2 pointers per list node
        // (prev and next pointers).
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        return mem + (memory_detail::EXTRA_NODES + t.size()) * (sizeof(T) + 2 * sizeof(std::size_t));
    }

    //! Overload for std::deque.
    template<typename T, typename A>
    static std::size_t dynamicSize(const std::deque<T, A>& t) {
        // std::deque is a pointer to an array of pointers to pages
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            for (auto i = t.begin(); i != t.end(); ++i) {
                mem += dynamicSize(*i);
            }
        }
        std::size_t pageSize = std::max(sizeof(T), memory_detail::MIN_DEQUE_PAGE_SIZE);
        std::size_t itemsPerPage = pageSize / sizeof(T);
        // This could be an underestimate if items have been removed
        std::size_t numPages = (t.size() + itemsPerPage - 1) / itemsPerPage;
        // This could also be an underestimate if items have been removed
        std::size_t pageVecEntries = std::max(numPages, memory_detail::MIN_DEQUE_PAGE_VEC_ENTRIES);

        return mem + pageVecEntries * sizeof(std::size_t) + numPages * pageSize;
    }

    //! Overload for boost::circular_buffer.
    template<typename T, typename A>
    static std::size_t dynamicSize(const boost::circular_buffer<T, A>& t) {
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            for (std::size_t i = 0; i < t.size(); ++i) {
                mem += dynamicSize(t[i]);
            }
        }
        return mem + t.capacity() * sizeof(T);
    }

    //! Overload for boost::optional.
    template<typename T>
    static std::size_t dynamicSize(const boost::optional<T>& t) {
        if (!t) {
            return 0;
        }
        return dynamicSize(*t);
    }

    //! Overload for boost::reference_wrapper.
    template<typename T>
    static std::size_t dynamicSize(const boost::reference_wrapper<T>& /*t*/) {
        return 0;
    }

    //! Overload for std::pair.
    template<typename T, typename V>
    static std::size_t dynamicSize(const std::pair<T, V>& t) {
        std::size_t mem = 0;
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            mem += dynamicSize(t.first);
        }
        if (!memory_detail::SDynamicSizeAlwaysZero<V>::value()) {
            mem += dynamicSize(t.second);
        }
        return mem;
    }

    //! Overload for boost::any.
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
    static CAnyVisitor ms_AnyVisitor;
};

namespace memory_detail {

template<typename T, void (T::*)(CMemoryUsage::TMemoryUsagePtr) const, typename R = void>
struct enable_if_member_debug_function {
    using type = R;
};

//! Default template declaration for SDebugMemoryDynamicSize::dispatch.
template<typename T, typename ENABLE = void>
struct SDebugMemoryDynamicSize {
    static void dispatch(const char* name, const T& t, CMemoryUsage::TMemoryUsagePtr mem) {
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
template<typename T>
struct SDebugMemoryDynamicSize<T, typename enable_if_member_debug_function<T, &T::debugMemoryUsage>::type> {
    static void dispatch(const char*, const T& t, CMemoryUsage::TMemoryUsagePtr mem) { t.debugMemoryUsage(mem->addChild()); }
};

} // memory_detail

//! \brief Core memory debug usage template class.
//!
//! DESCRIPTION:\n
//! Core memory debug usage template class. Provides an extension to the
//! CMemory class for creating a detailed breakdown of memory used by
//! classes and containers, utilising the CMemoryUsage class.
//!
//! ML classes can declare a public member function:
//!     void debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr) const;
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
private:
    static const std::string EMPTY_STRING;

public:
    //! Implements a visitor pattern for computing the size of types
    //! stored in boost::any.
    //!
    //! DESCRIPTION:\n
    //! See CMemory::CAnyVisitor for details.
    class CORE_EXPORT CAnyVisitor {
    public:
        using TDynamicSizeFunc = void (*)(const char*, const boost::any& any, CMemoryUsage::TMemoryUsagePtr mem);
        using TTypeInfoDynamicSizeFuncPr = std::pair<TTypeInfoCRef, TDynamicSizeFunc>;
        using TTypeInfoDynamicSizeFuncPrVec = std::vector<TTypeInfoDynamicSizeFuncPr>;

        //! Insert a callback to compute the size of the type T
        //! if it is stored in boost::any.
        template<typename T>
        bool registerCallback() {
            auto i = std::lower_bound(m_Callbacks.begin(), m_Callbacks.end(), boost::cref(typeid(T)), memory_detail::STypeInfoLess());
            if (i == m_Callbacks.end()) {
                m_Callbacks.emplace_back(boost::cref(typeid(T)), &CAnyVisitor::dynamicSizeCallback<T>);
                return true;
            } else if (i->first.get() != typeid(T)) {
                m_Callbacks.insert(i, {boost::cref(typeid(T)), &CAnyVisitor::dynamicSizeCallback<T>});
                return true;
            }
            return false;
        }

        //! Calculate the dynamic size of x if a callback has been
        //! registered for its type.
        void dynamicSize(const char* name, const boost::any& x, CMemoryUsage::TMemoryUsagePtr mem) const {
            if (!x.empty()) {
                auto i = std::lower_bound(m_Callbacks.begin(), m_Callbacks.end(), boost::cref(x.type()), memory_detail::STypeInfoLess());
                if (i != m_Callbacks.end() && i->first.get() == x.type()) {
                    (*i->second)(name, x, mem);
                    return;
                }
                LOG_ERROR("No callback registered for " << x.type().name());
            }
        }

    private:
        //! Wraps up call to any_cast and dynamicSize.
        template<typename T>
        static void dynamicSizeCallback(const char* name, const boost::any& any, CMemoryUsage::TMemoryUsagePtr mem) {
            try {
                mem->addItem(name, sizeof(T));
                CMemoryDebug::dynamicSize(name, boost::any_cast<const T&>(any), mem);
            } catch (const std::exception& e) { LOG_ERROR("Failed to calculate size " << e.what()); }
        }

        TTypeInfoDynamicSizeFuncPrVec m_Callbacks;
    };

public:
    //! Default template.
    template<typename T>
    static void dynamicSize(const char* name,
                            const T& t,
                            CMemoryUsage::TMemoryUsagePtr mem,
                            typename boost::disable_if<typename boost::is_pointer<T>>::type* = nullptr) {
        memory_detail::SDebugMemoryDynamicSize<T>::dispatch(name, t, mem);
    }

    //! Overload for pointer.
    template<typename T>
    static void dynamicSize(const char* name,
                            const T& t,
                            CMemoryUsage::TMemoryUsagePtr mem,
                            typename boost::enable_if<typename boost::is_pointer<T>>::type* = nullptr) {
        if (t != nullptr) {
            mem->addItem("ptr", CMemory::staticSize(*t));
            memory_detail::SDebugMemoryDynamicSize<T>::dispatch(name, *t, mem);
        }
    }

    //! Overload for boost::shared_ptr.
    template<typename T>
    static void dynamicSize(const char* name, const boost::shared_ptr<T>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        if (t) {
            long uc = t.use_count();
            // If the pointer is shared by multiple users, each one
            // might count it, so divide by the number of users.
            // However, if only 1 user has it, do a full debug.
            if (uc == 1) {
                mem->addItem("shared_ptr", CMemory::staticSize(*t));
                dynamicSize(name, *t, mem);
            } else {
                std::ostringstream ss;
                ss << "shared_ptr (x" << uc << ')';
                // Round up
                mem->addItem(ss.str(), (CMemory::staticSize(*t) + CMemory::dynamicSize(*t) + std::size_t(uc - 1)) / uc);
            }
        }
    }

    //! Overload for boost::array.
    template<typename T, std::size_t N>
    static void dynamicSize(const char* name, const boost::array<T, N>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        if (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
            std::string componentName(name);
            componentName += "_item";

            CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
            for (auto i = t.begin(); i != t.end(); ++i) {
                dynamicSize(componentName.c_str(), *i, ptr);
            }
        }
    }

    //! Overload for std::vector.
    template<typename T>
    static void dynamicSize(const char* name, const std::vector<T>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        std::string componentName(name);

        std::size_t items = t.size();
        std::size_t capacity = t.capacity();
        CMemoryUsage::SMemoryUsage usage(componentName + "::" + typeid(T).name(), capacity * sizeof(T), (capacity - items) * sizeof(T));
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        componentName += "_item";
        for (auto i = t.begin(); i != t.end(); ++i) {
            dynamicSize(componentName.c_str(), *i, ptr);
        }
    }

    //! Overload for small vector.
    template<typename T, std::size_t N>
    static void dynamicSize(const char* name, const CSmallVector<T, N>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        std::string componentName(name);

        std::size_t items = memory_detail::inplace(t) ? 0 : t.size();
        std::size_t capacity = memory_detail::inplace(t) ? 0 : t.capacity();
        CMemoryUsage::SMemoryUsage usage(componentName + "::" + typeid(T).name(), capacity * sizeof(T), (capacity - items) * sizeof(T));
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        componentName += "_item";
        for (auto i = t.begin(); i != t.end(); ++i) {
            dynamicSize(componentName.c_str(), *i, ptr);
        }
    }

    //! Overload for std::string.
    static void dynamicSize(const char* name, const std::string& t, CMemoryUsage::TMemoryUsagePtr mem) {
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

#elif defined(Linux) && (!defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI == 0)
        // All sizes > 0 use the heap, and the string structure is
        // 1 pointer + 2 sizes + 1 null terminator
        // We don't handle the reference counting, so may overestimate
        // Even some 0 length strings may use the heap - see
        // http://info.prelert.com/blog/clearing-strings
        if (capacity > 0 || t.data() != EMPTY_STRING.data()) {
            unused = capacity - length;
            capacity += sizeof(void*) + (2 * sizeof(std::size_t)) + 1;
        }

#else // Linux with C++11 ABI and Windows
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

    //! Overload for boost::unordered_map.
    template<typename K, typename V, typename H, typename P, typename A>
    static void dynamicSize(const char* name, const boost::unordered_map<K, V, H, P, A>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        std::string componentName(name);
        componentName += "_umap";

        std::size_t mapSize = (t.bucket_count() * sizeof(std::size_t) * 2) + (t.size() * (sizeof(K) + sizeof(V) + 2 * sizeof(std::size_t)));

        CMemoryUsage::SMemoryUsage usage(componentName, mapSize);
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        for (auto i = t.begin(); i != t.end(); ++i) {
            dynamicSize("key", i->first, ptr);
            dynamicSize("value", i->second, ptr);
        }
    }

    //! Overload for std::map.
    template<typename K, typename V, typename C, typename A>
    static void dynamicSize(const char* name, const std::map<K, V, C, A>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        // std::map appears to use 4 pointers/size_ts per tree node
        // (colour, parent, left and right child pointers)
        std::string componentName(name);
        componentName += "_map";

        std::size_t mapSize = (memory_detail::EXTRA_NODES + t.size()) * (sizeof(K) + sizeof(V) + 4 * sizeof(std::size_t));

        CMemoryUsage::SMemoryUsage usage(componentName, mapSize);
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        for (auto i = t.begin(); i != t.end(); ++i) {
            dynamicSize("key", i->first, ptr);
            dynamicSize("value", i->second, ptr);
        }
    }

    //! Overload for boost::container::flat_map.
    template<typename K, typename V, typename C, typename A>
    static void dynamicSize(const char* name, const boost::container::flat_map<K, V, C, A>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        std::string componentName(name);
        componentName += "_fmap";

        std::size_t items = t.size();
        std::size_t capacity = t.capacity();

        CMemoryUsage::SMemoryUsage usage(componentName + "::" + typeid(std::pair<K, V>).name(),
                                         capacity * sizeof(std::pair<K, V>),
                                         (capacity - items) * sizeof(std::pair<K, V>));
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        for (auto i = t.begin(); i != t.end(); ++i) {
            dynamicSize("key", i->first, ptr);
            dynamicSize("value", i->second, ptr);
        }
    }

    //! Overload for boost::unordered_set.
    template<typename T, typename H, typename P, typename A>
    static void dynamicSize(const char* name, const boost::unordered_set<T, H, P, A>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        std::string componentName(name);
        componentName += "_uset";

        std::size_t setSize = (t.bucket_count() * sizeof(std::size_t) * 2) + (t.size() * (sizeof(T) + 2 * sizeof(std::size_t)));

        CMemoryUsage::SMemoryUsage usage(componentName, setSize);
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        for (auto i = t.begin(); i != t.end(); ++i) {
            dynamicSize("value", *i, ptr);
        }
    }

    //! Overload for std::set.
    template<typename T, typename C, typename A>
    static void dynamicSize(const char* name, const std::set<T, C, A>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        // std::set appears to use 4 pointers/size_ts per tree node
        // (colour, parent, left and right child pointers)
        std::string componentName(name);
        componentName += "_set";

        std::size_t setSize = (memory_detail::EXTRA_NODES + t.size()) * (sizeof(T) + 4 * sizeof(std::size_t));

        CMemoryUsage::SMemoryUsage usage(componentName, setSize);
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        for (auto i = t.begin(); i != t.end(); ++i) {
            dynamicSize("value", *i, ptr);
        }
    }

    //! Overload for boost::container::flat_set.
    template<typename T, typename C, typename A>
    static void dynamicSize(const char* name, const boost::container::flat_set<T, C, A>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        std::string componentName(name);
        componentName += "_fset";

        std::size_t items = t.size();
        std::size_t capacity = t.capacity();

        CMemoryUsage::SMemoryUsage usage(componentName + "::" + typeid(T).name(), capacity * sizeof(T), (capacity - items) * sizeof(T));
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        for (auto i = t.begin(); i != t.end(); ++i) {
            dynamicSize("value", *i, ptr);
        }
    }

    //! Overload for std::list.
    template<typename T, typename A>
    static void dynamicSize(const char* name, const std::list<T, A>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        // std::list appears to use 2 pointers per list node
        // (prev and next pointers).
        std::string componentName(name);
        componentName += "_list";

        std::size_t listSize = (memory_detail::EXTRA_NODES + t.size()) * (sizeof(T) + 4 * sizeof(std::size_t));

        CMemoryUsage::SMemoryUsage usage(componentName, listSize);
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        for (auto i = t.begin(); i != t.end(); ++i) {
            dynamicSize("value", *i, ptr);
        }
    }

    //! Overload for std::deque.
    template<typename T, typename C, typename A>
    static void dynamicSize(const char* name, const std::deque<T, A>& t, CMemoryUsage::TMemoryUsagePtr mem) {
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

        for (auto i = t.begin(); i != t.end(); ++i) {
            dynamicSize("value", *i, ptr);
        }
    }

    //! Overload for boost::circular_buffer.
    template<typename T, typename A>
    static void dynamicSize(const char* name, const boost::circular_buffer<T, A>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        std::string componentName(name);

        std::size_t items = t.size();
        std::size_t capacity = t.capacity();
        CMemoryUsage::SMemoryUsage usage(componentName + "::" + typeid(T).name(), capacity * sizeof(T), (capacity - items) * sizeof(T));
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        ptr->setName(usage);

        componentName += "_item";
        for (std::size_t i = 0; i < items; ++i) {
            dynamicSize(componentName.c_str(), t[i], ptr);
        }
    }

    //! Overload for boost::optional.
    template<typename T>
    static void dynamicSize(const char* name, const boost::optional<T>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        if (t) {
            dynamicSize(name, *t, mem);
        }
    }

    //! Overload for boost::reference_wrapper.
    template<typename T>
    static void dynamicSize(const char* /*name*/, const boost::reference_wrapper<T>& /*t*/, CMemoryUsage::TMemoryUsagePtr /*mem*/) {
        return;
    }

    //! Overload for std::pair.
    template<typename T, typename V>
    static void dynamicSize(const char* name, const std::pair<T, V>& t, CMemoryUsage::TMemoryUsagePtr mem) {
        std::string keyName(name);
        keyName += "_key";
        std::string valueName(name);
        valueName += "_value";
        dynamicSize(keyName.c_str(), t.first, mem);
        dynamicSize(valueName.c_str(), t.second, mem);
    }

    //! Overload for boost::any.
    static void dynamicSize(const char* name, const boost::any& t, CMemoryUsage::TMemoryUsagePtr mem) {
        // boost::any holds a pointer to a new'd item.
        ms_AnyVisitor.dynamicSize(name, t, mem);
    }

    //! Get the any visitor singleton.
    static CAnyVisitor& anyVisitor() { return ms_AnyVisitor; }

private:
    static CAnyVisitor ms_AnyVisitor;
};

} // core
} // ml

#endif // INCLUDED_ml_core_CMemory_h
