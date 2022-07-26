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

#ifndef INCLUDED_ml_core_CMemoryDef_h
#define INCLUDED_ml_core_CMemoryDef_h

#include <core/CMemoryDec.h>

#include <core/CLogger.h>
#include <core/UnwrapRef.h>

#include <algorithm>
#include <functional>
#include <typeinfo>

namespace ml {
namespace core {
using TTypeInfoCRef = std::reference_wrapper<const std::type_info>;

namespace memory_detail {
//! \name Helpers for expectMemoryOverload.
//!{
// clang-format off
template<typename T, typename = void>
struct SContainerLike : std::false_type {};
template<typename T>
struct SContainerLike<T, std::void_t<typename T::const_iterator>> : std::true_type {}; 
template<typename T, typename = void>
struct SUserSuppliedMemoryUsageFunction : std::false_type {};
template<typename T>
struct SUserSuppliedMemoryUsageFunction<T, std::enable_if_t<
            std::is_same_v<decltype(&T::memoryUsage), std::size_t (T::*)() const>>>
        : std::true_type {
};
// clang-format on
//@}

//! Check if we expect an overloaded dynamicSize function.
template<typename T>
constexpr bool expectMemoryOverload() {
    return (SContainerLike<T>::value || std::is_same<T, boost::any>::value) &&
           !SUserSuppliedMemoryUsageFunction<T>::value &&
           !SDynamicSizeAlwaysZero<T>::value();
}

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

//! \name Helpers for expectDebugMemoryOverload.
//@{
// clang-format off
template<typename T, typename = void>
struct SUserSuppliedDebugMemoryUsageFunction : std::false_type {};
template<typename T>
struct SUserSuppliedDebugMemoryUsageFunction<T, std::enable_if_t<
            std::is_same_v<decltype(&T::debugMemoryUsage), void (T::*)(const CMemoryUsage::TMemoryUsagePtr&) const>>>
        : std::true_type {
};
// clang-format on
//@}

//! Check if we expect an overloaded dynamicSize function.
template<typename T>
constexpr bool expectDebugMemoryOverload() {
    return (SContainerLike<T>::value || std::is_same<T, boost::any>::value) &&
           !SUserSuppliedDebugMemoryUsageFunction<T>::value &&
           !SDynamicSizeAlwaysZero<T>::value();
}

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

//! \brief Total ordering of type_info objects.
struct CORE_EXPORT STypeInfoLess {
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
}

namespace CMemory {

//! Default implementation for non-pointer types.
template<typename T>
std::size_t dynamicSize(const T& t, std::enable_if_t<!std::is_pointer_v<T>>*) {
    // If you hit this assert there are one of three reasons:
    //   1. You need to add a memoryUsage function to your type,
    //   2. You need to add a dynamicSizeAlwaysZero function to your type,
    //   3. You need to include CMemoryDefStd.h.
    static_assert(!memory_detail::expectMemoryOverload<T>(),
                  "Maybe miss accounting for memory usage");
    if constexpr (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
        return memory_detail::SMemoryDynamicSize<T>::dispatch(t);
    }
    return 0;
}

//! Default implementation for pointer types.
template<typename T>
std::size_t dynamicSize(const T& t, std::enable_if_t<std::is_pointer_v<T>>*) {
    return t == nullptr ? 0 : CMemory::staticSize(*t) + CMemory::dynamicSize(*t);
}

template<typename T, typename DELETER>
std::size_t dynamicSize(const std::unique_ptr<T, DELETER>& t) {
    return t == nullptr ? 0 : CMemory::staticSize(*t) + CMemory::dynamicSize(*t);
}

template<typename T>
std::size_t dynamicSize(const std::shared_ptr<T>& t) {
    // The check for nullptr here may seem unnecessary but there are situations
    // where an unset shared_ptr can have a use_count greater than 0, see
    // https://stackoverflow.com/questions/48885252/c-sharedptr-use-count-for-nullptr/48885643
    long uc{t == nullptr ? 0 : t.use_count()};
    if (uc == 0) {
        return 0;
    }
    // Note we add on sizeof(long) here to account for the memory
    // used by the shared reference count. Also, round up.
    return (sizeof(long) + CMemory::staticSize(*t) + CMemory::dynamicSize(*t) +
            std::size_t(uc - 1)) /
           uc;
}

template<typename T, std::size_t N>
std::size_t dynamicSize(const std::array<T, N>& t) {
    return CMemory::elementDynamicSize(t);
}

template<typename T, typename A>
std::size_t dynamicSize(const std::vector<T, A>& t) {
    return CMemory::elementDynamicSize(t) + sizeof(T) * t.capacity();
}

template<typename T, std::size_t N>
std::size_t dynamicSize(const CSmallVector<T, N>& t) {
    return CMemory::elementDynamicSize(t) +
           (memory_detail::inplace(t) ? 0 : t.capacity()) * sizeof(T);
}

template<typename K, typename V, typename H, typename P, typename A>
std::size_t dynamicSize(const boost::unordered_map<K, V, H, P, A>& t) {
    return CMemory::elementDynamicSize(t) +
           (t.bucket_count() * CMemory::storageNodeOverhead(t)) +
           (t.size() * (sizeof(K) + sizeof(V) + CMemory::storageNodeOverhead(t)));
}

template<typename K, typename V, typename C, typename A>
std::size_t dynamicSize(const boost::container::flat_map<K, V, C, A>& t) {
    return CMemory::elementDynamicSize(t) + t.capacity() * sizeof(std::pair<K, V>);
}

template<typename T, typename H, typename P, typename A>
std::size_t dynamicSize(const boost::unordered_set<T, H, P, A>& t) {
    return CMemory::elementDynamicSize(t) +
           (t.bucket_count() * sizeof(std::size_t) * 2) +
           (t.size() * (sizeof(T) + CMemory::storageNodeOverhead(t)));
}

template<typename T, typename C, typename A>
std::size_t dynamicSize(const boost::container::flat_set<T, C, A>& t) {
    return CMemory::elementDynamicSize(t) + t.capacity() * sizeof(T);
}

template<typename T, typename A>
std::size_t dynamicSize(const boost::circular_buffer<T, A>& t) {
    return CMemory::elementDynamicSize(t) + t.capacity() * sizeof(T);
}

template<typename T>
std::size_t dynamicSize(const std::optional<T>& t) {
    return !t ? 0 : CMemory::dynamicSize(*t);
}

template<typename T>
std::size_t dynamicSize(const std::reference_wrapper<T>& /*t*/) {
    return 0;
}

template<typename T, typename V>
std::size_t dynamicSize(const std::pair<T, V>& t) {
    std::size_t mem = 0;
    if constexpr (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
        mem += CMemory::dynamicSize(t.first);
    }
    if constexpr (!memory_detail::SDynamicSizeAlwaysZero<V>::value()) {
        mem += CMemory::dynamicSize(t.second);
    }
    return mem;
}

std::size_t dynamicSize(const std::string& t);

std::size_t dynamicSize(const boost::any& t);

template<typename T, typename I, typename A>
std::size_t dynamicSize(const boost::multi_index::multi_index_container<T, I, A>& t);

template<typename CONTAINER>
std::size_t elementDynamicSize(const CONTAINER& t) {
    std::size_t mem = 0;
    if constexpr (!memory_detail::SDynamicSizeAlwaysZero<typename CONTAINER::value_type>::value()) {
        for (const auto& v : t) {
            mem += CMemory::dynamicSize(v);
        }
    }
    return mem;
}

//! Implements a visitor pattern for computing the size of types stored in
//! boost::any.
//!
//! DESCRIPTION:\n
//! The idea of this class is a place for users of dynamicSize to register
//! callbacks to compute the size of objects stored in boost::any. The user
//! must ensure that all types have registered callbacks before trying to
//! compute their memory usage. It will warn if a type is visited which is
//! not registered. Example usage:
//! \code{.cpp}
//!   CMemory::anyVisitor().insertCallback<std::vector<double>>();
//!   std::vector<boost::any> variables;
//!   variables.push_back(TDoubleVec(10));
//!   std::size_t size{CMemory::dynamicSize(variables, visitor)};
//! \endcode
//!
//! IMPLEMENTATION DECISIONS:\n
//! There is no locking to modify the vistor. This is because we expect that
//! callbacks are registered once as part of static initialisation.
class CORE_EXPORT CAnyVisitor {
public:
    using TDynamicSizeFunc = std::size_t (*)(const boost::any& any);
    using TTypeInfoDynamicSizeFuncPr = std::pair<TTypeInfoCRef, TDynamicSizeFunc>;
    using TTypeInfoDynamicSizeFuncPrVec = std::vector<TTypeInfoDynamicSizeFuncPr>;

public:
    static CAnyVisitor& instance() {
        static CAnyVisitor instance;
        return instance;
    }

public:
    CAnyVisitor(const CAnyVisitor&) = delete;
    CAnyVisitor& operator=(const CAnyVisitor&) = delete;

    //! Insert a callback to compute the size of the type T
    //! if it is stored in boost::any.
    template<typename T>
    bool registerCallback() {
        auto i = std::lower_bound(m_Callbacks.begin(), m_Callbacks.end(),
                                  std::cref(typeid(T)), memory_detail::STypeInfoLess());
        if (i == m_Callbacks.end()) {
            m_Callbacks.emplace_back(std::cref(typeid(T)),
                                     &CAnyVisitor::dynamicSizeCallback<T>);
            return true;
        }
        if (i->first.get() != typeid(T)) {
            m_Callbacks.insert(
                i, {std::cref(typeid(T)), &CAnyVisitor::dynamicSizeCallback<T>});
            return true;
        }
        return false;
    }

    //! Calculate the dynamic size of x if a callback has been
    //! registered for its type.
    std::size_t dynamicSize(const boost::any& x) const {
        if (!x.empty()) {
            auto i = std::lower_bound(m_Callbacks.begin(), m_Callbacks.end(),
                                      std::cref(x.type()),
                                      memory_detail::STypeInfoLess());
            if (i != m_Callbacks.end() && i->first.get() == x.type()) {
                return (*i->second)(x);
            }
            LOG_ERROR(<< "No callback registered for " << x.type().name());
        }
        return 0;
    }

private:
    CAnyVisitor() = default;

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

private:
    TTypeInfoDynamicSizeFuncPrVec m_Callbacks;
};

//! Get the any visitor singleton.
CAnyVisitor& anyVisitor();
}

namespace CMemoryDebug {

//! Default implementation for non-pointer types.
template<typename T>
void dynamicSize(const char* name,
                 const T& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem,
                 std::enable_if_t<!std::is_pointer_v<T>>*) {
    // If you hit this assert there are one of three reasons:
    //   1. You need to add a debugMemoryUsage function to your type,
    //   2. You need to add a dynamicSizeAlwaysZero function to your type,
    //   3. You need to include CMemoryDefStd.h.
    static_assert(!memory_detail::expectDebugMemoryOverload<T>(),
                  "Maybe miss accounting for memory usage");
    memory_detail::SDebugMemoryDynamicSize<T>::dispatch(name, t, mem);
}

//! Default implementation for pointer types.
template<typename T>
void dynamicSize(const char* name,
                 const T& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem,
                 std::enable_if_t<std::is_pointer_v<T>>*) {
    if (t != nullptr) {
        std::string ptrName(name);
        ptrName += "_ptr";
        mem->addItem(ptrName.c_str(), CMemory::staticSize(*t));
        memory_detail::SDebugMemoryDynamicSize<T>::dispatch(name, *t, mem);
    }
}

template<typename T>
void dynamicSize(const char* name,
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
void dynamicSize(const char* name,
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
void dynamicSize(const char* name,
                 const std::array<T, N>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    if constexpr (!memory_detail::SDynamicSizeAlwaysZero<T>::value()) {
        std::string elementName{name};
        CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
        CMemoryDebug::elementDynamicSize(std::move(elementName), t, mem);
    }
}

template<typename T, typename A>
void dynamicSize(const char* name,
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

    CMemoryDebug::elementDynamicSize(std::move(componentName), t, mem);
}

template<typename T, std::size_t N>
void dynamicSize(const char* name,
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

    CMemoryDebug::elementDynamicSize(std::move(componentName), t, mem);
}

template<typename K, typename V, typename H, typename P, typename A>
void dynamicSize(const char* name,
                 const boost::unordered_map<K, V, H, P, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    std::string componentName(name);
    componentName += "_umap";

    std::size_t mapSize = (t.bucket_count() * sizeof(std::size_t) * 2) +
                          (t.size() * (sizeof(K) + sizeof(V) + 2 * sizeof(std::size_t)));

    CMemoryUsage::SMemoryUsage usage(componentName, mapSize);
    CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
    ptr->setName(usage);

    CMemoryDebug::associativeElementDynamicSize(std::move(componentName), t, mem);
}

template<typename K, typename V, typename C, typename A>
void dynamicSize(const char* name,
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

    CMemoryDebug::associativeElementDynamicSize(std::move(componentName), t, mem);
}

template<typename T, typename H, typename P, typename A>
void dynamicSize(const char* name,
                 const boost::unordered_set<T, H, P, A>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    std::string componentName(name);
    componentName += "_uset";

    std::size_t setSize = (t.bucket_count() * CMemory::storageNodeOverhead(t)) +
                          (t.size() * (sizeof(T) + CMemory::storageNodeOverhead(t)));

    CMemoryUsage::SMemoryUsage usage(componentName, setSize);
    CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
    ptr->setName(usage);

    CMemoryDebug::elementDynamicSize(std::move(componentName), t, mem);
}

template<typename T, typename C, typename A>
void dynamicSize(const char* name,
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

    CMemoryDebug::elementDynamicSize(std::move(componentName), t, mem);
}

template<typename T, typename A>
void dynamicSize(const char* name,
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

    CMemoryDebug::elementDynamicSize(std::move(componentName), t, mem);
}

template<typename T>
void dynamicSize(const char* name,
                 const std::optional<T>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    if (t) {
        CMemoryDebug::dynamicSize(name, *t, mem);
    }
}

template<typename T>
void dynamicSize(const char* /*name*/,
                 const std::reference_wrapper<T>& /*t*/,
                 const CMemoryUsage::TMemoryUsagePtr& /*mem*/) {
}

template<typename U, typename V>
void dynamicSize(const char* name,
                 const std::pair<U, V>& t,
                 const CMemoryUsage::TMemoryUsagePtr& mem) {
    if (!memory_detail::SDynamicSizeAlwaysZero<U>::value()) {
        std::string keyName(name);
        keyName += "_first";
        CMemoryDebug::dynamicSize(keyName.c_str(), t.first, mem);
    }
    if (!memory_detail::SDynamicSizeAlwaysZero<V>::value()) {
        std::string valueName(name);
        valueName += "_second";
        CMemoryDebug::dynamicSize(valueName.c_str(), t.second, mem);
    }
}

template<typename CONTAINER>
void associativeElementDynamicSize(std::string name,
                                   const CONTAINER& t,
                                   const CMemoryUsage::TMemoryUsagePtr& mem) {
    if constexpr (!memory_detail::SDynamicSizeAlwaysZero<typename CONTAINER::value_type>::value()) {
        std::string keyName{name + "_key"};
        std::string valueName{name + "_value"};
        for (const auto & [ key, value ] : t) {
            CMemoryDebug::dynamicSize(keyName.c_str(), key, mem);
            CMemoryDebug::dynamicSize(valueName.c_str(), value, mem);
        }
    }
}

template<typename CONTAINER>
void elementDynamicSize(std::string name,
                        const CONTAINER& t,
                        const CMemoryUsage::TMemoryUsagePtr& mem) {
    if constexpr (!memory_detail::SDynamicSizeAlwaysZero<typename CONTAINER::value_type>::value()) {
        std::string elementName{name};
        elementName += "_item";
        for (const auto& v : t) {
            CMemoryDebug::dynamicSize(elementName.c_str(), v, mem);
        }
    }
}

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

public:
    static CAnyVisitor& instance() {
        static CAnyVisitor instance;
        return instance;
    }

public:
    CAnyVisitor(const CAnyVisitor&) = delete;
    CAnyVisitor& operator=(const CAnyVisitor&) = delete;

    //! Insert a callback to compute the size of the type T
    //! if it is stored in boost::any.
    template<typename T>
    bool registerCallback() {
        auto i = std::lower_bound(m_Callbacks.begin(), m_Callbacks.end(),
                                  std::cref(typeid(T)), memory_detail::STypeInfoLess());
        if (i == m_Callbacks.end()) {
            m_Callbacks.emplace_back(std::cref(typeid(T)),
                                     &CAnyVisitor::dynamicSizeCallback<T>);
            return true;
        }
        if (i->first.get() != typeid(T)) {
            m_Callbacks.insert(
                i, {std::cref(typeid(T)), &CAnyVisitor::dynamicSizeCallback<T>});
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
            auto i = std::lower_bound(m_Callbacks.begin(), m_Callbacks.end(),
                                      std::cref(x.type()),
                                      memory_detail::STypeInfoLess());
            if (i != m_Callbacks.end() && i->first.get() == x.type()) {
                (*i->second)(name, x, mem);
                return;
            }
            LOG_ERROR(<< "No callback registered for " << x.type().name());
        }
    }

private:
    CAnyVisitor() = default;

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

private:
    TTypeInfoDynamicSizeFuncPrVec m_Callbacks;
};

//! Get the any visitor singleton.
CAnyVisitor& anyVisitor();
}
}
}

#endif // INCLUDED_ml_core_CMemoryDef_h
