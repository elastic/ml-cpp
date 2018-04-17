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
#ifndef INCLUDED_ml_model_CStringStore_h
#define INCLUDED_ml_model_CStringStore_h

#include <core/CFastMutex.h>
#include <core/CMemory.h>
#include <core/CNonCopyable.h>
#include <core/CStoredStringPtr.h>

#include <model/ImportExport.h>

#include <boost/unordered_set.hpp>

#include <atomic>
#include <functional>
#include <string>

class CResourceMonitorTest;
class CStringStoreTest;

namespace ml {

namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}

namespace model {

//! \brief
//! DESCRIPTION:\n
//!
//! A collection of person/attribute strings, held using CStoredStringPtr, and
//! an independent collection of influencer strings (which have a different
//! life-cycle).
//! Users of the class look up a string by "const std::string &", and
//! receive a CStoredStringPtr in its stead.
//!
//! Specific person/attribute strings will be able to be pruned, if their
//! reference count is only 1 (i.e. we hold the only reference).
//!
//! IMPLEMENTATION DECISIONS:\n
//! A singleton class: there should only be one collection strings for
//! person names/attributes, and a separate collection for influencer
//! strings.
//! Write access is locked for the benefit of future threading.
//!
class MODEL_EXPORT CStringStore : private core::CNonCopyable {
public:
    struct MODEL_EXPORT SHashStoredStringPtr {
        std::size_t operator()(const core::CStoredStringPtr& key) const {
            boost::hash<std::string> hasher;
            return hasher(*key);
        }
    };
    struct MODEL_EXPORT SStoredStringPtrEqual {
        bool operator()(const core::CStoredStringPtr& lhs,
                        const core::CStoredStringPtr& rhs) const {
            return *lhs == *rhs;
        }
    };

public:
    //! Call this to tidy up any strings no longer needed.
    static void tidyUpNotThreadSafe();

    //! Singleton pattern for person/attribute names.
    static CStringStore& names();

    //! Singleton pattern for influencer names.
    static CStringStore& influencers();

    //! Fast method to get the pointer for an empty string.
    const core::CStoredStringPtr& getEmpty() const;

    //! (Possibly) add \p value to the store and get back a pointer to it.
    core::CStoredStringPtr get(const std::string& value);

    //! (Possibly) remove \p value from the store.
    void remove(const std::string& value);

    //! Prune strings which have been removed.
    void pruneRemovedNotThreadSafe();

    //! Iterate over the string store and remove unused entries.
    void pruneNotThreadSafe();

    //! Get the memory used by this string store
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this string store
    std::size_t memoryUsage() const;

private:
    using TStoredStringPtrUSet =
        boost::unordered_set<core::CStoredStringPtr, SHashStoredStringPtr, SStoredStringPtrEqual>;
    using TStrVec = std::vector<std::string>;

private:
    //! Constructor of a Singleton is private.
    CStringStore();

    //! Bludgeoning device to delete all objects in store.
    void clearEverythingTestOnly();

private:
    //! Fence for reading operations (in which case we "leak" a string
    //! if we try to write at the same time). See get for details.
    std::atomic_int m_Reading;

    //! Fence for writing operations (in which case we "leak" a string
    //! if we try to read at the same time). See get for details.
    std::atomic_int m_Writing;

    //! The empty string is often used so we store it outside the set.
    core::CStoredStringPtr m_EmptyString;

    //! Set to keep the person/attribute string pointers
    TStoredStringPtrUSet m_Strings;

    //! A list of the strings to remove.
    TStrVec m_Removed;

    //! Running count of memory usage by stored strings.  Avoids the need to
    //! recalculate repeatedly.
    std::size_t m_StoredStringsMemUse;

    //! Locking primitive
    mutable core::CFastMutex m_Mutex;

    friend class ::CResourceMonitorTest;
    friend class ::CStringStoreTest;
};

} // model
} // ml

#endif // INCLUDED_ml_model_CStringStore_h
