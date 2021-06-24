/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CAlignment.h>
#include <core/CContainerPrinter.h>
#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CSmallVector.h>

#include <test/CRandomNumbers.h>

#include <boost/circular_buffer.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>

#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

BOOST_AUTO_TEST_SUITE(CMemoryUsageTest)

using namespace ml;

namespace {

// Subset of model_t equivalent duplicated here to avoid a dependency
// with the model library
enum EFeature {
    E_IndividualHighMeanByPerson,
    E_IndividualCountByBucketAndPerson,
    E_IndividualHighCountsByBucketAndPerson
};

using TIntVec = std::vector<int>;
using TStrVec = std::vector<std::string>;

struct SPod {
    double s_V1;
    double s_V2;
    int s_V3;
};

struct SFoo {
    static bool dynamicSizeAlwaysZero() { return true; }

    explicit SFoo(std::size_t key = 0) : s_Key(key) {}
    bool operator<(const SFoo& rhs) const { return s_Key < rhs.s_Key; }
    bool operator==(const SFoo& rhs) const { return s_Key == rhs.s_Key; }

    std::size_t s_Key;
    double s_State[100];
};

struct SFooWithMemoryUsage {
    explicit SFooWithMemoryUsage(std::size_t key = 0) : s_Key(key) {}
    bool operator<(const SFooWithMemoryUsage& rhs) const {
        return s_Key < rhs.s_Key;
    }
    bool operator==(const SFooWithMemoryUsage& rhs) const {
        return s_Key == rhs.s_Key;
    }
    std::size_t memoryUsage() const { return 0; }

    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("SFooWithMemoryUsage", 0);
    }

    std::size_t s_Key;
    double s_State[100];
};

struct SFooWrapper {
    std::size_t memoryUsage() const {
        std::size_t mem = core::CMemory::dynamicSize(s_Foo);
        return mem;
    }
    SFooWithMemoryUsage s_Foo;
};

struct SBar {
    using TFooVec = std::vector<SFoo>;

    explicit SBar(std::size_t key = 0) : s_Key(key), s_State() {}
    bool operator<(const SBar& rhs) const { return s_Key < rhs.s_Key; }
    bool operator==(const SBar& rhs) const { return s_Key == rhs.s_Key; }
    std::size_t memoryUsage() const {
        return sizeof(SFoo) * s_State.capacity();
    }

    std::size_t s_Key;
    TFooVec s_State;
};

struct SBarDebug {
    using TFooVec = std::vector<SFoo>;

    explicit SBarDebug(std::size_t key = 0) : s_Key(key), s_State() {}
    bool operator<(const SBarDebug& rhs) const { return s_Key < rhs.s_Key; }
    bool operator==(const SBarDebug& rhs) const { return s_Key == rhs.s_Key; }
    std::size_t memoryUsage() const {
        return sizeof(SFoo) * s_State.capacity();
    }

    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("SBarDebug", 0);
        core::CMemoryDebug::dynamicSize("s_State", s_State, mem);
    }

    std::size_t s_Key;
    TFooVec s_State;
};

struct SBarVectorDebug {
    using TFooVec = std::vector<SFooWithMemoryUsage>;

    explicit SBarVectorDebug(std::size_t key = 0) : s_Key(key), s_State() {}
    bool operator<(const SBarVectorDebug& rhs) const {
        return s_Key < rhs.s_Key;
    }
    bool operator==(const SBarVectorDebug& rhs) const {
        return s_Key == rhs.s_Key;
    }
    std::size_t memoryUsage() const {
        return core::CMemory::dynamicSize(s_State);
    }

    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("SBarVectorDebug", 0);
        core::CMemoryDebug::dynamicSize("s_State", s_State, mem);
    }

    std::size_t s_Key;
    TFooVec s_State;
};

struct SHash {
    std::size_t operator()(const SFoo& foo) const { return foo.s_Key; }
    std::size_t operator()(const SFooWithMemoryUsage& foo) const {
        return foo.s_Key;
    }
    std::size_t operator()(const SBar& bar) const { return bar.s_Key; }
};

class CBase {
public:
    CBase(std::size_t i) : m_Vec(i, 0) {}

    virtual ~CBase() = default;

    virtual std::size_t memoryUsage() const {
        return core::CMemory::dynamicSize(m_Vec);
    }

    virtual void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CBase", 0);
        core::CMemoryDebug::dynamicSize("m_Vec", m_Vec, mem);
    }

    virtual std::size_t staticSize() const { return sizeof(*this); }

    const std::uint64_t* fixed() const { return m_Fixed; } // suppress warning

private:
    std::uint64_t m_Fixed[5];
    TIntVec m_Vec;
};

class CDerived : public CBase {
public:
    CDerived(std::size_t i)
        : CBase(i), m_Strings(i, "This is a secret string") {}

    virtual ~CDerived() = default;

    virtual std::size_t memoryUsage() const {
        std::size_t mem = core::CMemory::dynamicSize(m_Strings);
        mem += this->CBase::memoryUsage();
        return mem;
    }

    virtual void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CDerived", 0);
        core::CMemoryDebug::dynamicSize("m_Strings", m_Strings, mem);
        this->CBase::debugMemoryUsage(mem->addChild());
    }

    virtual std::size_t staticSize() const { return sizeof(*this); }

    const std::uint64_t* fixed() const { return m_Fixed; } // suppress warning

private:
    std::uint64_t m_Fixed[50];
    TStrVec m_Strings;
};

//! A basic allocator that tracks memory usage
template<typename T>
class CTrackingAllocator {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

public:
    // convert an allocator<T> to allocator<U>
    template<typename U>
    struct rebind {
        using other = CTrackingAllocator<U>;
    };

public:
    CTrackingAllocator() = default;
    CTrackingAllocator(const CTrackingAllocator&) = default;

    template<typename U>
    inline CTrackingAllocator(const CTrackingAllocator<U>&) {}

    // address
    inline pointer address(reference r) { return &r; }

    inline const_pointer address(const_reference r) { return &r; }

    // memory allocation
    inline pointer allocate(size_type cnt,
                            typename std::allocator<void>::const_pointer = nullptr) {
        ms_Allocated += cnt;
        return reinterpret_cast<pointer>(::operator new(cnt * sizeof(T)));
    }

    inline void deallocate(pointer p, size_type cnt) {
        ms_Allocated -= cnt;
        ::operator delete(p);
    }

    // size
    inline size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    static std::size_t usage() { return ms_Allocated; }

    // construction/destruction
    inline void construct(pointer p, const T& t) { new (p) T(t); }

    inline void destroy(pointer p) { p->~T(); }

    inline bool operator==(const CTrackingAllocator&) const { return true; }

    inline bool operator!=(const CTrackingAllocator& a) const {
        return !operator==(a);
    }

private:
    static std::size_t ms_Allocated;
};

template<typename T>
std::size_t CTrackingAllocator<T>::ms_Allocated = 0;
}

BOOST_AUTO_TEST_CASE(testUsage) {
    using TFooVec = std::vector<SFoo>;
    using TFooWithMemoryVec = std::vector<SFooWithMemoryUsage>;
    using TFooList = std::list<SFoo>;
    using TFooWithMemoryList = std::list<SFooWithMemoryUsage>;
    using TFooDeque = std::deque<SFoo>;
    using TFooWithMemoryDeque = std::deque<SFooWithMemoryUsage>;
    using TFooCircBuf = boost::circular_buffer<SFoo>;
    using TFooWithMemoryCircBuf = boost::circular_buffer<SFooWithMemoryUsage>;
    using TFooFooMap = std::map<SFoo, SFoo>;
    using TFooWithMemoryFooWithMemoryMap = std::map<SFooWithMemoryUsage, SFooWithMemoryUsage>;
    using TFooFooUMap = boost::unordered_map<SFoo, SFoo, SHash>;
    using TFooFSet = boost::container::flat_set<SFoo>;
    using TFooWithMemoryFooWithMemoryUMap =
        boost::unordered_map<SFooWithMemoryUsage, SFooWithMemoryUsage, SHash>;
    using TBarVec = std::vector<SBar>;
    using TBarBarMap = std::map<SBar, SBar>;
    using TBarBarUMap = boost::unordered_map<SBar, SBar, SHash>;
    using TBarBarFMap = boost::container::flat_map<SBar, SBar>;
    using TBarPtr = std::shared_ptr<SBar>;
    using TBasePtr = std::shared_ptr<CBase>;
    using TDerivedVec = std::vector<CDerived>;
    using TBasePtrVec = std::vector<TBasePtr>;

    // We want various invariants to hold for dynamic size:
    //   1) The dynamic size is not affected by adding a memoryUsage
    //      to a class definition.
    //   2) If a member is stored by value don't double count its
    //      memory.
    //   3) The dynamic size of an object is not affected by whether
    //      it is stored in a container or not.

    {
        TFooVec foos(10);
        TFooWithMemoryVec foosWithMemory(10);

        LOG_DEBUG(<< "*** TFooVec ***");
        LOG_DEBUG(<< "dynamicSize(foos)           = " << core::CMemory::dynamicSize(foos));
        LOG_DEBUG(<< "dynamicSize(foosWithMemory) = "
                  << core::CMemory::dynamicSize(foosWithMemory));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(foos),
                            core::CMemory::dynamicSize(foosWithMemory));
    }
    {
        TFooList foos(10);
        TFooWithMemoryList foosWithMemory(10);

        LOG_DEBUG(<< "*** TFooList ***");
        LOG_DEBUG(<< "dynamicSize(foos)           = " << core::CMemory::dynamicSize(foos));
        LOG_DEBUG(<< "dynamicSize(foosWithMemory) = "
                  << core::CMemory::dynamicSize(foosWithMemory));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(foos),
                            core::CMemory::dynamicSize(foosWithMemory));
    }
    {
        TFooDeque foos(10);
        TFooWithMemoryDeque foosWithMemory(10);

        LOG_DEBUG(<< "*** TFooDeque ***");
        LOG_DEBUG(<< "dynamicSize(foos)           = " << core::CMemory::dynamicSize(foos));
        LOG_DEBUG(<< "dynamicSize(foosWithMemory) = "
                  << core::CMemory::dynamicSize(foosWithMemory));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(foos),
                            core::CMemory::dynamicSize(foosWithMemory));
    }
    {
        TFooCircBuf foos(10);
        foos.resize(5);
        TFooWithMemoryCircBuf foosWithMemory(10);
        foosWithMemory.resize(5);

        LOG_DEBUG(<< "*** TFooCircBuf ***");
        LOG_DEBUG(<< "dynamicSize(foos)           = " << core::CMemory::dynamicSize(foos));
        LOG_DEBUG(<< "dynamicSize(foosWithMemory) = "
                  << core::CMemory::dynamicSize(foosWithMemory));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(foos),
                            core::CMemory::dynamicSize(foosWithMemory));
    }
    {
        TFooFooMap foos;
        TFooWithMemoryFooWithMemoryMap foosWithMemory;

        std::size_t keys[] = {0, 1, 2, 3, 4, 5};
        for (std::size_t i = 0; i < boost::size(keys); ++i) {
            foos[SFoo(keys[i])] = SFoo(keys[i]);
            foosWithMemory[SFooWithMemoryUsage(keys[i])] = SFooWithMemoryUsage(keys[i]);
        }

        LOG_DEBUG(<< "*** TFooFooMap ***");
        LOG_DEBUG(<< "dynamicSize(foos)           = " << core::CMemory::dynamicSize(foos));
        LOG_DEBUG(<< "dynamicSize(foosWithMemory) = "
                  << core::CMemory::dynamicSize(foosWithMemory));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(foos),
                            core::CMemory::dynamicSize(foosWithMemory));
    }
    {
        TFooFooUMap foos;
        TFooWithMemoryFooWithMemoryUMap foosWithMemory;

        std::size_t keys[] = {0, 1, 2, 3, 4, 5};
        for (std::size_t i = 0; i < boost::size(keys); ++i) {
            foos[SFoo(keys[i])] = SFoo(keys[i]);
            foosWithMemory[SFooWithMemoryUsage(keys[i])] = SFooWithMemoryUsage(keys[i]);
        }

        LOG_DEBUG(<< "*** TFooFooUMap ***");
        LOG_DEBUG(<< "dynamicSize(foos)           = " << core::CMemory::dynamicSize(foos));
        LOG_DEBUG(<< "dynamicSize(foosWithMemory) = "
                  << core::CMemory::dynamicSize(foosWithMemory));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(foos),
                            core::CMemory::dynamicSize(foosWithMemory));
    }
    {
        TFooFSet foos;

        std::size_t keys[] = {0, 1, 2, 3, 4, 5};
        for (std::size_t i = 0; i < boost::size(keys); ++i) {
            foos.insert(SFoo(keys[i]));
        }

        LOG_DEBUG(<< "*** TFooFSet ***");
        LOG_DEBUG(<< "dynamicSize(foos)           = " << core::CMemory::dynamicSize(foos));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(foos),
                            foos.capacity() * sizeof(SFoo));
    }

    {
        LOG_DEBUG(<< "*** SFooWrapper ***");
        SFooWithMemoryUsage foo;
        SFooWrapper wrapper;
        LOG_DEBUG(<< "memoryUsage foo     = " << foo.memoryUsage());
        LOG_DEBUG(<< "memoryUsage wrapper = " << wrapper.memoryUsage());
        BOOST_REQUIRE_EQUAL(foo.memoryUsage(), wrapper.memoryUsage());
    }

    {
        TBarVec bars1;
        bars1.reserve(10);
        bars1.push_back(SBar());
        bars1.push_back(SBar());
        bars1[0].s_State.resize(1);
        bars1[1].s_State.resize(2);

        TBarVec bars2;
        bars2.reserve(10);
        bars2.push_back(SBar());
        bars2.push_back(SBar());
        TFooVec state21;
        state21.resize(1);
        TFooVec state22;
        state22.resize(2);

        LOG_DEBUG(<< "*** TBarVec ***");
        LOG_DEBUG(<< "dynamic size = " << core::CMemory::dynamicSize(bars1));
        LOG_DEBUG(<< "expected dynamic size = "
                  << core::CMemory::dynamicSize(bars2) + core::CMemory::dynamicSize(state21) +
                         core::CMemory::dynamicSize(state22));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(bars1),
                            core::CMemory::dynamicSize(bars2) +
                                core::CMemory::dynamicSize(state21) +
                                core::CMemory::dynamicSize(state22));
    }
    {
        SBar key;
        key.s_State.resize(3);
        SBar value;
        value.s_State.resize(2);

        TBarBarMap bars1;
        bars1[key] = value;

        TBarBarMap bars2;
        bars2[SBar()] = SBar();

        LOG_DEBUG(<< "*** TBarBarMap ***");
        LOG_DEBUG(<< "dynamic size = " << core::CMemory::dynamicSize(bars1));
        LOG_DEBUG(<< "expected dynamic size = "
                  << core::CMemory::dynamicSize(bars2) + core::CMemory::dynamicSize(key) +
                         core::CMemory::dynamicSize(value));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(bars1),
                            core::CMemory::dynamicSize(bars2) +
                                core::CMemory::dynamicSize(key) +
                                core::CMemory::dynamicSize(value));
    }
    {
        SBar key;
        key.s_State.resize(3);
        SBar value;
        value.s_State.resize(2);

        TBarBarUMap bars1;
        bars1[key] = value;

        TBarBarUMap bars2;
        bars2[SBar()] = SBar();

        LOG_DEBUG(<< "*** TBarBarUMap ***");
        LOG_DEBUG(<< "dynamic size = " << core::CMemory::dynamicSize(bars1));
        LOG_DEBUG(<< "expected dynamic size = "
                  << core::CMemory::dynamicSize(bars2) + core::CMemory::dynamicSize(key) +
                         core::CMemory::dynamicSize(value));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(bars1),
                            core::CMemory::dynamicSize(bars2) +
                                core::CMemory::dynamicSize(key) +
                                core::CMemory::dynamicSize(value));
    }
    {
        SBar key;
        key.s_State.resize(3);
        SBar value;
        value.s_State.resize(2);

        TBarBarFMap bars1;
        bars1.reserve(4);
        BOOST_TEST_REQUIRE(core::CMemory::dynamicSize(bars1) > 4 * sizeof(SBar));

        bars1[key] = value;

        TBarBarFMap bars2;
        bars2.reserve(4);
        bars2[SBar()] = SBar();

        LOG_DEBUG(<< "*** TBarBarFMap ***");
        LOG_DEBUG(<< "dynamic size = " << core::CMemory::dynamicSize(bars1));
        LOG_DEBUG(<< "expected dynamic size = "
                  << core::CMemory::dynamicSize(bars2) + core::CMemory::dynamicSize(key) +
                         core::CMemory::dynamicSize(value));
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(bars1),
                            core::CMemory::dynamicSize(bars2) +
                                core::CMemory::dynamicSize(key) +
                                core::CMemory::dynamicSize(value));
    }
    {
        SBar value;
        value.s_State.resize(3);

        TBarPtr pointer(new SBar(value));

        LOG_DEBUG(<< "*** TBarPtr ***");
        LOG_DEBUG(<< "dynamic size = " << core::CMemory::dynamicSize(pointer));
        LOG_DEBUG(<< "expected dynamic size = "
                  << sizeof(SBar) + sizeof(SFoo) * value.s_State.capacity());
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(pointer),
                            sizeof(long) + sizeof(SBar) +
                                sizeof(SFoo) * value.s_State.capacity());
    }

    {
        LOG_DEBUG(<< "*** boost::any ***");

        using TDoubleVec = std::vector<double>;
        using TAnyVec = std::vector<boost::any>;

        TDoubleVec a(10);
        TFooVec b(20);

        TAnyVec variables(1); // Empty any at index 0
        variables.push_back(a);
        variables.push_back(b);

        LOG_DEBUG(<< "wrong dynamic size = " << core::CMemory::dynamicSize(variables));
        BOOST_REQUIRE_EQUAL(variables.capacity() * sizeof(std::size_t),
                            core::CMemory::dynamicSize(variables));

        core::CMemory::CAnyVisitor& visitor = core::CMemory::anyVisitor();
        visitor.registerCallback<TDoubleVec>();
        visitor.registerCallback<TFooVec>();

        LOG_DEBUG(<< "dynamic size = " << core::CMemory::dynamicSize(variables));
        LOG_DEBUG(<< "expected dynamic size = "
                  << variables.capacity() * sizeof(std::size_t) + sizeof(a) +
                         core::CMemory::dynamicSize(a) + sizeof(b) +
                         core::CMemory::dynamicSize(b));
        BOOST_REQUIRE_EQUAL(variables.capacity() * sizeof(std::size_t) +
                                sizeof(a) + core::CMemory::dynamicSize(a) +
                                sizeof(b) + core::CMemory::dynamicSize(b),
                            core::CMemory::dynamicSize(variables));

        core::CMemoryDebug::CAnyVisitor& debugVisitor = core::CMemoryDebug::anyVisitor();
        debugVisitor.registerCallback<TDoubleVec>();
        debugVisitor.registerCallback<TFooVec>();

        auto mem = std::make_shared<core::CMemoryUsage>();
        core::CMemoryDebug::dynamicSize("", variables, mem);
        BOOST_REQUIRE_EQUAL(mem->usage(), core::CMemory::dynamicSize(variables));
        std::ostringstream ss;
        mem->print(ss);
        LOG_DEBUG(<< ss.str());
    }
    {
        CBase* base = new CBase(10);
        CBase* derived = new CDerived(10);
        {
            auto mem = std::make_shared<core::CMemoryUsage>();
            core::CMemoryDebug::dynamicSize("", *base, mem);
            BOOST_REQUIRE_EQUAL(mem->usage(), core::CMemory::dynamicSize(*base));
            std::ostringstream ss;
            mem->print(ss);
            LOG_TRACE(<< ss.str());
        }
        {
            auto mem = std::make_shared<core::CMemoryUsage>();
            core::CMemoryDebug::dynamicSize("", *derived, mem);
            BOOST_REQUIRE_EQUAL(mem->usage(), core::CMemory::dynamicSize(*derived));
            std::ostringstream ss;
            mem->print(ss);
            LOG_TRACE(<< ss.str());
        }
        BOOST_TEST_REQUIRE(core::CMemory::dynamicSize(*base) <
                           core::CMemory::dynamicSize(*derived));

        TBasePtr sharedBase(new CBase(10));
        TBasePtr sharedDerived(new CDerived(10));
        {
            auto mem = std::make_shared<core::CMemoryUsage>();
            core::CMemoryDebug::dynamicSize("", sharedBase, mem);
            BOOST_REQUIRE_EQUAL(mem->usage(), core::CMemory::dynamicSize(sharedBase));
            std::ostringstream ss;
            mem->print(ss);
            LOG_TRACE(<< ss.str());
        }
        {
            auto mem = std::make_shared<core::CMemoryUsage>();
            core::CMemoryDebug::dynamicSize("", sharedDerived, mem);
            BOOST_REQUIRE_EQUAL(mem->usage(), core::CMemory::dynamicSize(sharedDerived));
            std::ostringstream ss;
            mem->print(ss);
            LOG_TRACE(<< ss.str());
        }
        // boost:reference_wrapper should give zero
        std::reference_wrapper<CBase> baseRef(std::ref(*base));
        BOOST_REQUIRE_EQUAL(std::size_t(0), core::CMemory::dynamicSize(baseRef));
        {
            auto mem = std::make_shared<core::CMemoryUsage>();
            core::CMemoryDebug::dynamicSize("", baseRef, mem);
            BOOST_REQUIRE_EQUAL(mem->usage(), core::CMemory::dynamicSize(baseRef));
            std::ostringstream ss;
            mem->print(ss);
            LOG_TRACE(<< ss.str());
        }
    }
    {
        CBase base(5);
        BOOST_REQUIRE_EQUAL(base.memoryUsage(), core::CMemory::dynamicSize(base));

        CBase* basePtr = new CBase(5);
        BOOST_REQUIRE_EQUAL(basePtr->memoryUsage() + sizeof(*basePtr),
                            core::CMemory::dynamicSize(basePtr));

        CDerived derived(6);
        BOOST_REQUIRE_EQUAL(derived.memoryUsage(), core::CMemory::dynamicSize(derived));

        CDerived* derivedPtr = new CDerived(5);
        BOOST_REQUIRE_EQUAL(derivedPtr->memoryUsage() + sizeof(*derivedPtr),
                            core::CMemory::dynamicSize(derivedPtr));

        CBase* basederivedPtr = new CDerived(5);
        BOOST_REQUIRE_EQUAL(basederivedPtr->memoryUsage() + sizeof(CDerived),
                            core::CMemory::dynamicSize(basederivedPtr));

        TBasePtr sPtr(new CDerived(6));
        BOOST_REQUIRE_EQUAL(sPtr->memoryUsage() + sizeof(long) + sizeof(CDerived),
                            core::CMemory::dynamicSize(sPtr));
    }
    {
        TDerivedVec vec;
        vec.reserve(6);
        vec.push_back(CDerived(1));
        vec.push_back(CDerived(3));
        vec.push_back(CDerived(5));
        vec.push_back(CDerived(7));
        vec.push_back(CDerived(9));
        vec.push_back(CDerived(12));
        std::size_t total = core::CMemory::dynamicSize(vec);
        std::size_t calc = vec.capacity() * sizeof(CDerived);
        for (std::size_t i = 0; i < vec.size(); ++i) {
            calc += vec[i].memoryUsage();
        }
        BOOST_REQUIRE_EQUAL(calc, total);
    }
    {
        TBasePtrVec vec;
        vec.push_back(TBasePtr(new CBase(2)));
        vec.push_back(TBasePtr(new CBase(2)));
        vec.push_back(TBasePtr(new CBase(2)));
        vec.push_back(TBasePtr(new CBase(2)));
        vec.push_back(TBasePtr(new CBase(2)));
        vec.push_back(TBasePtr(new CBase(2)));
        vec.push_back(TBasePtr(new CDerived(44)));

        std::size_t total = core::CMemory::dynamicSize(vec);
        std::size_t calc = vec.capacity() * sizeof(TBasePtr);
        for (std::size_t i = 0; i < 6; ++i) {
            calc += sizeof(long);
            calc += static_cast<CBase*>(vec[i].get())->memoryUsage();
            calc += sizeof(CBase);
        }
        calc += sizeof(long);
        calc += static_cast<CDerived*>(vec[6].get())->memoryUsage();
        calc += sizeof(CDerived);
        BOOST_REQUIRE_EQUAL(calc, total);
    }
}

BOOST_AUTO_TEST_CASE(testDebug) {
    using TBarVec = std::vector<SBar>;
    using TBarVecPtr = std::shared_ptr<TBarVec>;

    // Check that we can get debug info out of classes with vectors of varying size
    {
        SBar sbar;
        SBarDebug sbarDebug;
        SBarVectorDebug sbarVectorDebug;
        for (unsigned i = 0; i < 9; ++i) {
            sbar.s_State.push_back(SFoo(i));
            sbarDebug.s_State.push_back(SFoo(i));
            sbarVectorDebug.s_State.push_back(SFooWithMemoryUsage(i));
            LOG_TRACE(<< "SFooWithMemoryUsage usage: "
                      << sbarVectorDebug.s_State.back().memoryUsage());
        }
        BOOST_REQUIRE_EQUAL(sbar.memoryUsage(), sbarDebug.memoryUsage());
        BOOST_REQUIRE_EQUAL(sbar.memoryUsage(), sbarVectorDebug.memoryUsage());

        {
            auto mem = std::make_shared<core::CMemoryUsage>();
            sbarDebug.debugMemoryUsage(mem);
            BOOST_REQUIRE_EQUAL(sbarDebug.memoryUsage(), mem->usage());
            std::ostringstream ss;
            mem->print(ss);
            LOG_TRACE(<< "SBarDebug: " + ss.str());
        }
        {
            auto mem = std::make_shared<core::CMemoryUsage>();
            sbarVectorDebug.debugMemoryUsage(mem);
            std::ostringstream ss;
            mem->print(ss);
            LOG_TRACE(<< "SBarVectorDebug: " + ss.str());
            LOG_TRACE(<< "memoryUsage: " << sbarVectorDebug.memoryUsage()
                      << ", debugUsage: " << mem->usage());
            BOOST_REQUIRE_EQUAL(sbarVectorDebug.memoryUsage(), mem->usage());
        }
    }
    {
        TBarVecPtr t(new TBarVec());
        t->push_back(SBar(0));
        t->push_back(SBar(1));
        t->push_back(SBar(2));
        t->push_back(SBar(3));
        t->push_back(SBar(4));

        core::CMemoryUsage memoryUsage;
        memoryUsage.setName("test", 0);
        core::CMemoryDebug::dynamicSize("TBarVecPtr", t, memoryUsage.addChild());
        std::ostringstream ss;
        memoryUsage.print(ss);
        LOG_TRACE(<< "TBarVecPtr usage: " << core::CMemory::dynamicSize(t)
                  << ", debug: " << memoryUsage.usage());
        LOG_TRACE(<< ss.str());
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(t), memoryUsage.usage());
    }
    {
        using TFeatureBarVecPtrPr = std::pair<EFeature, TBarVecPtr>;
        using TFeatureBarVecPtrPrVec = std::vector<TFeatureBarVecPtrPr>;
        TFeatureBarVecPtrPrVec t;

        TBarVecPtr vec(new TBarVec());
        vec->push_back(SBar(0));
        vec->push_back(SBar(1));
        vec->push_back(SBar(2));
        vec->push_back(SBar(3));
        vec->push_back(SBar(4));

        t.push_back(TFeatureBarVecPtrPr(E_IndividualHighMeanByPerson, vec));
        TBarVecPtr vec2(new TBarVec());
        vec2->push_back(SBar(22));
        vec2->push_back(SBar(33));
        t.push_back(TFeatureBarVecPtrPr(E_IndividualCountByBucketAndPerson, vec));

        t.push_back(TFeatureBarVecPtrPr(E_IndividualHighCountsByBucketAndPerson,
                                        TBarVecPtr()));
        core::CMemoryUsage memoryUsage;
        memoryUsage.setName("test", 0);
        core::CMemoryDebug::dynamicSize("TFeatureBarVecPtrPrVec", t, memoryUsage.addChild());
        std::ostringstream ss;
        memoryUsage.print(ss);
        LOG_TRACE(<< "TFeatureBarVecPtrPrVec usage: " << core::CMemory::dynamicSize(t)
                  << ", debug: " << memoryUsage.usage());
        LOG_TRACE(<< ss.str());
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(t), memoryUsage.usage());
    }
}

BOOST_AUTO_TEST_CASE(testDynamicSizeAlwaysZero) {
    bool test = core::memory_detail::SDynamicSizeAlwaysZero<int>::value();
    BOOST_REQUIRE_EQUAL(true, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<double>::value();
    BOOST_REQUIRE_EQUAL(true, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<SPod>::value();
    BOOST_REQUIRE_EQUAL(true, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<boost::optional<double>>::value();
    BOOST_REQUIRE_EQUAL(true, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<boost::optional<SPod>>::value();
    BOOST_REQUIRE_EQUAL(true, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<std::pair<int, int>>::value();
    BOOST_REQUIRE_EQUAL(true, test);
    test = std::is_pod<SFoo>::value;
    BOOST_REQUIRE_EQUAL(false, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<SFoo>::value();
    BOOST_REQUIRE_EQUAL(true, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<std::pair<boost::optional<double>, SFoo>>::value();
    BOOST_REQUIRE_EQUAL(true, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<core::CHashing::CUniversalHash::CUInt32Hash>::value();
    BOOST_REQUIRE_EQUAL(true, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<core::CHashing::CUniversalHash::CUInt32UnrestrictedHash>::value();
    BOOST_REQUIRE_EQUAL(true, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<std::pair<double, SFooWithMemoryUsage>>::value();
    BOOST_REQUIRE_EQUAL(false, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<SFooWithMemoryUsage>::value();
    BOOST_REQUIRE_EQUAL(false, test);
    test = core::memory_detail::SDynamicSizeAlwaysZero<SFooWrapper>::value();
    BOOST_REQUIRE_EQUAL(false, test);
}

BOOST_AUTO_TEST_CASE(testCompress) {
    {
        // Check that non-repeated entries are not removed
        core::CMemoryUsage mem;
        mem.setName("root", 1);
        mem.addChild()->setName("child1", 22);
        mem.addChild()->setName("child2", 23);
        mem.addChild()->setName("child3", 24);
        mem.addChild()->setName("child4", 25);
        mem.addItem("item1", 91);
        mem.addItem("item2", 92);
        mem.addItem("item3", 93);
        mem.addItem("item4", 94);
        mem.addItem("item5", 95);
        mem.addItem("item6", 96);
        BOOST_REQUIRE_EQUAL(std::size_t(656), mem.usage());
        std::string before;
        {
            std::ostringstream ss;
            mem.print(ss);
            before = ss.str();
        }
        mem.compress();
        BOOST_REQUIRE_EQUAL(std::size_t(656), mem.usage());
        std::string after;
        {
            std::ostringstream ss;
            mem.print(ss);
            after = ss.str();
        }
        BOOST_REQUIRE_EQUAL(before, after);
    }
    {
        // Check that repeated entries are removed
        core::CMemoryUsage mem;
        mem.setName("root", 1);
        mem.addChild()->setName("muffin", 4);
        mem.addChild()->setName("child", 3);
        auto child = mem.addChild();
        child->setName("child", 5);
        child->addChild()->setName("grandchild", 100);
        mem.addChild()->setName("child", 7);
        mem.addChild()->setName("child", 9);
        mem.addChild()->setName("child", 11);
        mem.addChild()->setName("puffin", 2);
        mem.addChild()->setName("child", 13);
        mem.addChild()->setName("child", 15);
        mem.addChild()->setName("child", 17);
        mem.addChild()->setName("child", 19);
        mem.addChild()->setName("child", 21);
        BOOST_REQUIRE_EQUAL(std::size_t(227), mem.usage());

        mem.compress();
        BOOST_REQUIRE_EQUAL(std::size_t(227), mem.usage());
        std::string after;
        {
            std::ostringstream ss;
            mem.print(ss);
            after = ss.str();
        }
        std::string expected("{\"root\":{\"memory\":1},\"subItems\":[{\"muffin\":"
                             "{\"memory\":4}},{\"child [*10]\":{\"memory\":220}},{\"puffin\":"
                             "{\"memory\":2}}]}\n");
        LOG_DEBUG(<< after);
        BOOST_REQUIRE_EQUAL(expected, after);
    }
}

// This "test" highlights the way the std::string class behaves on each
// platform we support.  Experience shows that methods like reserve(),
// clear() and operator=() don't always work the way the books suggest...
//
// There are no assertions, but the idea is that a developer should go
// through the output after switching to a new standard library
// implementation to ensure that the quirks of std::string are in that
// implementation are understood.
BOOST_AUTO_TEST_CASE(testStringBehaviour, *boost::unit_test::disabled()) {

    LOG_INFO(<< "Size of std::string is " << sizeof(std::string));

    std::string empty1;
    std::string empty2;

    LOG_INFO(<< "Two independently constructed empty strings have data at "
             << static_cast<const void*>(empty1.data()) << " and "
             << static_cast<const void*>(empty2.data()) << " and capacity "
             << empty1.capacity());
    if (empty1.data() == empty2.data()) {
        LOG_INFO(<< "All strings constructed empty probably share the same "
                    "representation on this platform");
    }

    std::string something1("something");
    std::string something2(something1);
    std::string something3;
    something3 = something2;

    LOG_INFO(<< "Non-empty string has data at "
             << static_cast<const void*>(something1.data()) << " length "
             << something1.length() << " and capacity " << something1.capacity());

    LOG_INFO(<< "Copy constructed string has data at "
             << static_cast<const void*>(something2.data()) << " length "
             << something2.length() << " and capacity " << something2.capacity());
    if (something2.data() == something1.data()) {
        LOG_INFO(<< "Copy constructor probably has a copy-on-write "
                    "implementation on this platform");
    }

    LOG_INFO(<< "Assigned string has data at "
             << static_cast<const void*>(something3.data()) << " length "
             << something3.length() << " and capacity " << something3.capacity());
    if (something3.data() == something2.data()) {
        LOG_INFO(<< "Assignment operator probably has a copy-on-write "
                    "implementation on this platform");
    }

    something1.clear();

    LOG_INFO(<< "Cleared string that was copied to two others has data at "
             << static_cast<const void*>(something1.data()) << " length "
             << something1.length() << " and capacity " << something1.capacity());
    if (something1.data() == empty1.data()) {
        LOG_INFO(<< "Cleared strings revert to shared empty representation on "
                    "this platform");
    }

    something2 = empty2;

    LOG_INFO(<< "String that was copied to another then assigned an empty string "
                "has data at "
             << static_cast<const void*>(something2.data()) << " length "
             << something2.length() << " and capacity " << something2.capacity());
    if (something2.data() == empty1.data()) {
        LOG_INFO(<< "Strings that have an empty constructed string assigned to "
                    "them share the same representation as other empty "
                    "constructed strings on this platform");
    }

    std::string uncopied("uncopied");

    LOG_INFO(<< "Non-empty uncopied string has data at "
             << static_cast<const void*>(uncopied.data()) << " length "
             << uncopied.length() << " and capacity " << uncopied.capacity());

    uncopied.clear();

    LOG_INFO(<< "Cleared uncopied string has data at "
             << static_cast<const void*>(uncopied.data()) << " length "
             << uncopied.length() << " and capacity " << uncopied.capacity());

    std::string startSmall("small");

    LOG_INFO(<< "Non-empty small string unchanged since construction has data at "
             << static_cast<const void*>(startSmall.data()) << " length "
             << startSmall.length() << " and capacity " << startSmall.capacity());

    startSmall.reserve(100);
    size_t capacity100(startSmall.capacity());

    LOG_INFO(<< "Small string after reserving 100 bytes has data at "
             << static_cast<const void*>(startSmall.data()) << " length "
             << startSmall.length() << " and capacity " << startSmall.capacity());

    startSmall.reserve(10);

    LOG_INFO(<< "Small string after reserving 10 bytes has data at "
             << static_cast<const void*>(startSmall.data()) << " length "
             << startSmall.length() << " and capacity " << startSmall.capacity());
    if (startSmall.capacity() < capacity100) {
        LOG_INFO(<< "On this platform reservations can reduce string capacity");
    }

    // We have to test clearing with a size/capacity that won't get confused by
    // the short string optimisation (if it's being used)
    std::string startLong("this_string_is_longer_than_one_that_will_take_advantage_of_the_small_string_optimisation");

    LOG_INFO(<< "Long string after initial construction has data at "
             << static_cast<const void*>(startLong.data()) << " length "
             << startLong.length() << " and capacity " << startLong.capacity());

    startLong.reserve(10000);
    size_t capacity10000(startLong.capacity());

    LOG_INFO(<< "Long string after reserving 10000 bytes has data at "
             << static_cast<const void*>(startLong.data()) << " length "
             << startLong.length() << " and capacity " << startLong.capacity());

    startLong.clear();

    LOG_INFO(<< "Long string after clearing has data at "
             << static_cast<const void*>(startLong.data()) << " length "
             << startLong.length() << " and capacity " << startLong.capacity());
    if (startLong.capacity() < capacity10000) {
        LOG_INFO(<< "On this platform clearing can reduce string capacity");
    }

    using TSizeVec = std::vector<size_t>;
    std::string grower;
    TSizeVec capacities(1, grower.capacity());
    for (size_t count = 0; count < 50000; ++count) {
        grower += 'x';
        if (grower.capacity() != capacities.back()) {
            capacities.push_back(grower.capacity());
        }
    }

    LOG_INFO(<< "Capacities during growth from 0 to 50000 characters are: "
             << core::CContainerPrinter::print(capacities));

    std::string toBeShrunk(100, 'a');
    toBeShrunk = "a lot smaller than it was";

    size_t preShrinkCapacity(toBeShrunk.capacity());
    LOG_INFO(<< "String to be shrunk has starting size " << toBeShrunk.size()
             << " and capacity " << preShrinkCapacity);

    std::string(toBeShrunk).swap(toBeShrunk);

    size_t postShrinkCapacity(toBeShrunk.capacity());
    LOG_INFO(<< "String to be shrunk has post-shrink size " << toBeShrunk.size()
             << " and capacity " << postShrinkCapacity);

    LOG_INFO(<< "The swap() trick to reduce capacity "
             << ((postShrinkCapacity < preShrinkCapacity) ? "works" : "DOESN'T WORK!"));
}

BOOST_AUTO_TEST_CASE(testStringMemory) {
    using TAllocator = CTrackingAllocator<char>;
    using TString = std::basic_string<char, std::char_traits<char>, TAllocator>;

    for (std::size_t i = 0; i < 1500; ++i) {
        BOOST_REQUIRE_EQUAL(std::size_t(0), TAllocator::usage());
        TString trackingString;
        std::string normalString;
        for (std::size_t j = 0; j < i; ++j) {
            trackingString.push_back(static_cast<char>('a' + j));
            normalString.push_back(static_cast<char>('a' + j));
        }
        LOG_TRACE(<< "String size " << core::CMemory::dynamicSize(normalString)
                  << ", allocated " << TAllocator::usage());
        BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(normalString), TAllocator::usage());
    }
}

BOOST_AUTO_TEST_CASE(testStringClear) {
    using TAllocator = CTrackingAllocator<char>;
    using TString = std::basic_string<char, std::char_traits<char>, TAllocator>;

    TString empty;
    TString something1("something");
    TString something2(something1);
    TString something3;
    something3 = something1;

    std::size_t usage3Copies = TAllocator::usage();

    something1.clear();

    // If the following assertion fails after a standard library upgrade then
    // the logic in include/core/CMemory.h needs changing as well as this test
    BOOST_REQUIRE_EQUAL(usage3Copies, TAllocator::usage());
}

BOOST_AUTO_TEST_CASE(testSharedPointer) {
    using TIntVecPtr = std::shared_ptr<TIntVec>;
    using TIntVecPtrVec = std::vector<TIntVecPtr>;
    using TStrPtr = std::shared_ptr<std::string>;
    using TStrPtrVec = std::vector<TStrPtr>;
    TStrPtrVec strings;

    TIntVecPtrVec vec1;
    TIntVecPtrVec vec2;

    vec1.push_back(TIntVecPtr(new TIntVec(20, 555)));
    vec1.push_back(TIntVecPtr(new TIntVec(30, 44)));
    vec1.push_back(vec1[0]);
    vec1.push_back(TIntVecPtr(new TIntVec(40, 22)));
    vec1.push_back(vec1[1]);

    vec2.push_back(vec1[3]);
    vec2.push_back(vec1[1]);
    vec2.push_back(vec1[0]);
    vec2.push_back(vec2[1]);
    vec2.push_back(vec2[2]);

    LOG_DEBUG(<< "shared_ptr size: " << sizeof(TIntVecPtr));
    LOG_DEBUG(<< "IntVec size: " << sizeof(TIntVec));
    LOG_DEBUG(<< "int size: " << sizeof(int));

    LOG_DEBUG(<< "vec1 size: " << core::CMemory::dynamicSize(vec1));
    LOG_DEBUG(<< "vec2 size: " << core::CMemory::dynamicSize(vec2));

    // shared_ptr size is 16
    // intvec size is 24
    // int size is 4

    // x1 vector size: 24 + 20 * 4 = 104
    // x2 vector size: 24 + 30 * 4 = 144
    // x3 vector size: 24 + 40 * 4 = 184

    // What we should have on 64-bit OS X:
    // vec1: 8 (capacity) * 16 (shared_ptr element size) + 104 + 144 + 184
    // vec2: 8 (capacity) * 16 (shared_ptr element size)
    // = 688

    std::size_t expectedSize =
        vec1.capacity() * sizeof(TIntVecPtr) + vec2.capacity() * sizeof(TIntVecPtr) +
        3 * (sizeof(long) + sizeof(TIntVec)) +
        (vec1[0]->capacity() + vec1[1]->capacity() + vec1[3]->capacity()) * sizeof(int);

    LOG_DEBUG(<< "Expected: " << expectedSize << ", actual: "
              << (core::CMemory::dynamicSize(vec1) + core::CMemory::dynamicSize(vec2)));

    BOOST_REQUIRE_EQUAL(expectedSize, core::CMemory::dynamicSize(vec1) +
                                          core::CMemory::dynamicSize(vec2));

    TStrPtrVec svec1;
    svec1.push_back(TStrPtr(new std::string("This is a string")));
    svec1.push_back(TStrPtr(new std::string(
        "Here is some more string data, a little longer than the previous one")));
    svec1.push_back(TStrPtr(new std::string("An uninteresting string, this one!")));

    TStrPtrVec svec2;
    svec2.push_back(TStrPtr());
    svec2.push_back(TStrPtr());
    svec2.push_back(TStrPtr());

    long stringSizeBefore = core::CMemory::dynamicSize(svec1) +
                            core::CMemory::dynamicSize(svec2);

    svec2[0] = svec1[2];
    svec2[1] = svec1[0];
    svec2[2] = svec1[1];

    long stringSizeAfter = core::CMemory::dynamicSize(svec1) +
                           core::CMemory::dynamicSize(svec2);

    BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(svec1),
                        core::CMemory::dynamicSize(svec2));
    // Allow for integer rounding off by 1 for each string
    BOOST_TEST_REQUIRE(std::abs(stringSizeBefore - stringSizeAfter) < 4);
}

BOOST_AUTO_TEST_CASE(testRawPointer) {
    std::string* strPtr = nullptr;
    BOOST_REQUIRE_EQUAL(std::size_t(0), core::CMemory::dynamicSize(strPtr));

    std::string foo = "abcdefghijklmnopqrstuvwxyz";
    std::size_t fooMem = core::CMemory::dynamicSize(foo);
    // We will not normally have a raw pointer on stack memory,
    // but we do so here for testing purposes.
    strPtr = &foo;

    BOOST_REQUIRE_EQUAL(fooMem + sizeof(std::string), core::CMemory::dynamicSize(strPtr));
}

BOOST_AUTO_TEST_CASE(testSmallVector) {
    using TSizeVec = std::vector<std::size_t>;
    using TDouble1Vec = core::CSmallVector<double, 2>;
    using TDouble6Vec = core::CSmallVector<double, 6>;
    using TDouble9Vec = core::CSmallVector<double, 8>;

    test::CRandomNumbers test;
    TSizeVec sizes;
    test.generateUniformSamples(0, 12, 100, sizes);

    for (auto size : sizes) {
        TDouble1Vec vec1(size);
        TDouble6Vec vec2(size);
        TDouble9Vec vec3(size);
        TSizeVec memory{core::CMemory::dynamicSize(vec1), core::CMemory::dynamicSize(vec2),
                        core::CMemory::dynamicSize(vec3)};
        // These assertions hold because the vectors never shrink
        if (size <= 2) {
            BOOST_TEST_REQUIRE(memory[0] == 0);
        }
        BOOST_REQUIRE(memory[0] == 0 || memory[0] == vec1.capacity() * sizeof(double));
        if (size <= 6) {
            BOOST_TEST_REQUIRE(memory[1] == 0);
        }
        BOOST_REQUIRE(memory[1] == 0 || memory[1] == vec2.capacity() * sizeof(double));
        if (size <= 8) {
            BOOST_TEST_REQUIRE(memory[2] == 0);
        }
        BOOST_REQUIRE(memory[2] == 0 || memory[2] == vec3.capacity() * sizeof(double));
    }

    // Test growing and shrinking
    TDouble6Vec growShrink;
    std::size_t extraMem{core::CMemory::dynamicSize(growShrink)};
    BOOST_REQUIRE_EQUAL(std::size_t(0), extraMem);
    growShrink.resize(6);
    extraMem = core::CMemory::dynamicSize(growShrink);
    BOOST_REQUIRE_EQUAL(std::size_t(0), extraMem);
    growShrink.resize(10);
    extraMem = core::CMemory::dynamicSize(growShrink);
    BOOST_TEST_REQUIRE(extraMem > 0);
    growShrink.clear();
    extraMem = core::CMemory::dynamicSize(growShrink);
    BOOST_TEST_REQUIRE(extraMem > 0);
    growShrink.shrink_to_fit();
    extraMem = core::CMemory::dynamicSize(growShrink);
    BOOST_REQUIRE_EQUAL(std::size_t(0), extraMem);
    growShrink.push_back(1.7);
    extraMem = core::CMemory::dynamicSize(growShrink);
    // Interesting (shocking?) result: once a boost::small_vector has switched
    // off of internal storage it will NEVER go back to internal storage.
    // Arguably this is a bug, and this assertion might start failing after a
    // Boost upgrade.  If that happens and changing it to assert extraMem is 0
    // fixes it then this means boost::small_vector has been improved.
    BOOST_TEST_REQUIRE(extraMem > 0);
}

BOOST_AUTO_TEST_CASE(testAlignedVector) {
    using TDoubleVec = std::vector<double>;
    using TAlignedDoubleVec = std::vector<double, core::CAlignedAllocator<double>>;

    TDoubleVec vector{10.0, 11.0, 12.0, 13.0, 14.0,
                      15.0, 16.0, 17.0, 18.0, 19.0};
    TAlignedDoubleVec alignedVector{10.0, 11.0, 12.0, 13.0, 14.0,
                                    15.0, 16.0, 17.0, 18.0, 19.0};

    LOG_DEBUG(<< "TDoubleVec usage = " << core::CMemory::dynamicSize(vector));
    LOG_DEBUG(<< "TAlignedDoubleVec usage = " << core::CMemory::dynamicSize(alignedVector));
    BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(vector),
                        core::CMemory::dynamicSize(alignedVector));

    core::CMemoryUsage memoryUsage;
    memoryUsage.setName("test", 0);
    core::CMemoryDebug::dynamicSize("TAlignedDoubleVec", vector, memoryUsage.addChild());
    std::ostringstream ss;
    memoryUsage.print(ss);
    LOG_DEBUG(<< "TAlignedDoubleVec usage debug = " << ss.str());
    BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(vector), memoryUsage.usage());
}

BOOST_AUTO_TEST_SUITE_END()
