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

#include <model/CStringStore.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CScopedFastLock.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <boost/bind.hpp>

namespace ml {
namespace model {

namespace {

//! \brief Helper class to hash a std::string.
struct SStrHash {
    std::size_t operator()(const std::string& key) const {
        boost::hash<std::string> hasher;
        return hasher(key);
    }
} STR_HASH;

//! \brief Helper class to compare a std::string and a CStoredStringPtr.
struct SStrStoredStringPtrEqual {
    bool operator()(const std::string& lhs, const core::CStoredStringPtr& rhs) const { return lhs == *rhs; }
} STR_EQUAL;

// To ensure the singletons are constructed before multiple threads may
// require them call instance() during the static initialisation phase
// of the program.  Of course, the instance may already be constructed
// before this if another static object has used it.
const CStringStore& DO_NOT_USE_THIS_VARIABLE = CStringStore::names();
const CStringStore& DO_NOT_USE_THIS_VARIABLE_EITHER = CStringStore::influencers();
}

void CStringStore::tidyUpNotThreadSafe() {
    names().pruneRemovedNotThreadSafe();
    influencers().pruneNotThreadSafe();
}

CStringStore& CStringStore::names() {
    static CStringStore namesInstance;
    return namesInstance;
}

CStringStore& CStringStore::influencers() {
    static CStringStore influencersInstance;
    return influencersInstance;
}

const core::CStoredStringPtr& CStringStore::getEmpty() const {
    return m_EmptyString;
}

core::CStoredStringPtr CStringStore::get(const std::string& value) {
    // This section is expected to be performed frequently.
    //
    // We ensure either:
    //   1) Some threads may perform an insert and no thread will perform
    //      a find until no threads can still perform an insert.
    //   2) Some threads may perform a find and no thread will perform
    //      an insert until no thread can still perform a find.
    //
    // We "leak" strings if there is contention between reading and writing,
    // which is expected to be rare because inserts are expected to be rare.

    if (value.empty()) {
        return m_EmptyString;
    }

    core::CStoredStringPtr result;

    m_Reading.fetch_add(1, atomic_t::memory_order_release);
    if (m_Writing.load(atomic_t::memory_order_consume) == 0) {
        auto i = m_Strings.find(value, STR_HASH, STR_EQUAL);
        if (i != m_Strings.end()) {
            result = *i;
            m_Reading.fetch_sub(1, atomic_t::memory_order_release);
        } else {
            m_Writing.fetch_add(1, atomic_t::memory_order_acq_rel);
            // NB: fetch_sub() returns the OLD value, and we know we added 1 in
            // this thread, hence the test for 1 rather than 0
            if (m_Reading.fetch_sub(1, atomic_t::memory_order_release) == 1) {
                // This section is expected to occur infrequently so inserts
                // are synchronized with a mutex.
                core::CScopedFastLock lock(m_Mutex);
                auto ret = m_Strings.insert(core::CStoredStringPtr::makeStoredString(value));
                result = *ret.first;
                if (ret.second) {
                    m_StoredStringsMemUse += result.actualMemoryUsage();
                }
                m_Writing.fetch_sub(1, atomic_t::memory_order_release);
            } else {
                m_Writing.fetch_sub(1, atomic_t::memory_order_relaxed);
                // This is leaked in the sense that it will never be shared and
                // won't count towards our reported memory usage.  But it is not
                // leaked in the traditional sense of the word, as its memory
                // will be freed when the single user is finished with it.
                result = core::CStoredStringPtr::makeStoredString(value);
            }
        }
    } else {
        m_Reading.fetch_sub(1, atomic_t::memory_order_relaxed);
        // This is leaked in the sense that it will never be shared and won't
        // count towards our reported memory usage.  But it is not leaked
        // in the traditional sense of the word, as its memory will be freed
        // when the single user is finished with it.
        result = core::CStoredStringPtr::makeStoredString(value);
    }

    return result;
}

void CStringStore::remove(const std::string& value) {
    core::CScopedFastLock lock(m_Mutex);
    m_Removed.push_back(value);
}

void CStringStore::pruneRemovedNotThreadSafe() {
    core::CScopedFastLock lock(m_Mutex);
    for (const auto& removed : m_Removed) {
        auto i = m_Strings.find(removed, STR_HASH, STR_EQUAL);
        if (i != m_Strings.end() && i->isUnique()) {
            m_StoredStringsMemUse -= i->actualMemoryUsage();
            m_Strings.erase(i);
        }
    }
    m_Removed.clear();
}

void CStringStore::pruneNotThreadSafe() {
    core::CScopedFastLock lock(m_Mutex);
    for (auto i = m_Strings.begin(); i != m_Strings.end(); /**/) {
        if (i->isUnique()) {
            m_StoredStringsMemUse -= i->actualMemoryUsage();
            i = m_Strings.erase(i);
        } else {
            ++i;
        }
    }
}

void CStringStore::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName(this == &CStringStore::names()
                     ? "names StringStore"
                     : (this == &CStringStore::influencers() ? "influencers StringStore" : "unknown StringStore"));
    mem->addItem("empty string ptr", m_EmptyString.actualMemoryUsage());
    core::CScopedFastLock lock(m_Mutex);
    core::CMemoryDebug::dynamicSize("stored strings", m_Strings, mem);
    core::CMemoryDebug::dynamicSize("removed strings", m_Removed, mem);
    mem->addItem("stored string ptr memory", m_StoredStringsMemUse);
}

std::size_t CStringStore::memoryUsage() const {
    std::size_t mem = m_EmptyString.actualMemoryUsage();
    core::CScopedFastLock lock(m_Mutex);
    // The assumption here is that the existence of
    // core::CStoredStringPtr::dynamicSizeAlwaysZero() combined with dead code
    // elimination will make calculating the size of m_Strings boil down to a
    // couple of simple multiplications and additions
    mem += core::CMemory::dynamicSize(m_Strings);
    // This one could be more expensive, but the assumption is that there won't
    // be many memory usage calculations while m_Removed is populated
    mem += core::CMemory::dynamicSize(m_Removed);
    // This adds back the size that was excluded from
    // core::CMemory::dynamicSize(m_Strings)
    mem += m_StoredStringsMemUse;
    return mem;
}

CStringStore::CStringStore()
    : m_Reading(0), m_Writing(0), m_EmptyString(core::CStoredStringPtr::makeStoredString(std::string())), m_StoredStringsMemUse(0) {
}

void CStringStore::clearEverythingTestOnly() {
    // For tests that assert on memory usage it's important that these
    // containers get returned to the state of a default constructed container
    TStoredStringPtrUSet emptySet;
    emptySet.swap(m_Strings);
    TStrVec emptyVec;
    emptyVec.swap(m_Removed);
    m_StoredStringsMemUse = 0;
}

} // model
} // ml
