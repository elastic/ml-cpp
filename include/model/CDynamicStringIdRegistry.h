/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CDynamicStringIdRegistry_h
#define INCLUDED_ml_model_CDynamicStringIdRegistry_h

#include <core/CCompressedDictionary.h>
#include <core/CMemory.h>
#include <core/CProgramCounters.h>
#include <core/CStoredStringPtr.h>
#include <core/CoreTypes.h>

#include <model/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CResourceMonitor;

//! \brief Register string names and map them to unique identifiers.
//!
//! DESCRIPTION:\n
//! This is a registry where where a string is mapped to a unique identifier.
//! The registry provides mapping from a registered string to its id and
//! vice versa. In addition, the registry provides a recycling mechanism
//! in order to reuse IDs whose mapped string is no longer relevant.
class MODEL_EXPORT CDynamicStringIdRegistry {
public:
    using TDictionary = core::CCompressedDictionary<2>;
    using TWordSizeUMap = TDictionary::TWordTUMap<std::size_t>;
    using TWordSizeUMapItr = TWordSizeUMap::iterator;
    using TWordSizeUMapCItr = TWordSizeUMap::const_iterator;
    using TSizeVec = std::vector<std::size_t>;
    using TStrVec = std::vector<std::string>;
    using TStoredStringPtrVec = std::vector<core::CStoredStringPtr>;

public:
    //! An identifier which will never be used for a real string.
    static const std::size_t INVALID_ID;

public:
    //! Constructs a registry that expects names of type \p nameType.
    //!
    //! \param[in] nameType The type of the names expected to be registered.
    //! Mainly used to disambiguate log messages.
    //! \param[in] addedCounter The counter to be increased when a new name is registered.
    //! \param[in] addNotAllowedCounter The counter to be increased when a new name failed
    //! to register because no more additions are allowed.
    //! \param[in] recycledCounter The counter to be increased when an ID is recycled.
    CDynamicStringIdRegistry(const std::string& nameType,
                             counter_t::ECounterTypes addedCounter,
                             counter_t::ECounterTypes addNotAllowedCounter,
                             counter_t::ECounterTypes recycledCounter);

    //! Create a copy that will result in the same persisted state as the
    //! original. This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose. The boolean flag is
    //! redundant except to create a signature that will not be mistaken for
    //! a general purpose copy constructor.
    CDynamicStringIdRegistry(bool isForPersistence, const CDynamicStringIdRegistry& other);

    //! Get the name identified by \p id if it exists.
    //!
    //! \param[in] id The unique identifier of the name of interest.
    //! \return The name if \p exists or \p fallback otherwise.
    const std::string& name(std::size_t id, const std::string& fallback) const;

    //! Get the name identified by \p id if it exists, as a shared pointer
    //!
    //! \param[in] id The unique identifier of the name of interest.
    //! \return The name as a string pointer
    const core::CStoredStringPtr& namePtr(size_t id) const;

    //! Get the unique identifier of a name if it exists.
    //!
    //! \param[in] name The name of interest.
    //! \param[out] result Filled in with the identifier of \p name
    //! if it exists otherwise max std::size_t.
    //! \return True if the name exists and false otherwise.
    bool id(const std::string& name, std::size_t& result) const;

    //! Get the unique identifier of an arbitrary known name.
    //! \param[out] result Filled in with the identifier of a name
    //! \return True if a name exists and false otherwise.
    bool anyId(std::size_t& result) const;

    //! Get the number of active names (not pruned).
    std::size_t numberActiveNames() const;

    //! Get the maximum identifier seen so far
    //! (some of which might have been pruned).
    std::size_t numberNames() const;

    //! Check whether an identifier is active.
    bool isIdActive(std::size_t id) const;

    //! Register a \p name and return its unique identifier.
    std::size_t addName(const std::string& name,
                        core_t::TTime time,
                        CResourceMonitor& resourceMonitor,
                        bool& addedPerson);

    //! Remove all traces of names whose identifiers are greater than
    //! or equal to \p lowestNameToRemove.
    void removeNames(std::size_t lowestNameToRemove);

    //! Recycle the unique identifiers used by the names
    //! identified by \p namesToRemove.
    void recycleNames(const TSizeVec& namesToRemove, const std::string& defaultName);

    //! Get unique identifiers of any names that have been recycled.
    TSizeVec& recycledIds();

    //! Check the class invariants.
    bool checkInvariants() const;

    //! Clear this registry.
    void clear();

    //! Get the checksum of this registry.
    uint64_t checksum() const;

    //! Debug the memory used by this registry.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this registry.
    std::size_t memoryUsage() const;

    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

private:
    //! The type of the names expected to be registered.
    std::string m_NameType;

    //! The statistic to be increased when a new name is registered.
    counter_t::ECounterTypes m_AddedCounter;

    //! The statistic to be increased when a new name failed
    //! to register because no more additions are allowed.
    counter_t::ECounterTypes m_AddNotAllowedCounter;

    //! The statistic to be increased when an ID is recycled.
    counter_t::ECounterTypes m_RecycledCounter;

    //! A compressed dictionary.
    TDictionary m_Dictionary;

    //! Holds a unique identifier for each registered name which means
    //! we can use direct address tables and fast hash maps and
    //! sets keyed by names.
    TWordSizeUMap m_Uids;

    //! Holds the name of each unique identifier.
    TStoredStringPtrVec m_Names;

    //! A list of unique identifiers which are free to reuse.
    TSizeVec m_FreeUids;

    //! A list of recycled unique identifiers.
    TSizeVec m_RecycledUids;
};
}
}

#endif // INCLUDED_ml_model_CDynamicStringIdRegistry_h
