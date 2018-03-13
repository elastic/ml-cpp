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

#include <model/CDynamicStringIdRegistry.h>

#include <core/CLogger.h>
#include <core/CPersistUtils.h>

#include <maths/CChecksum.h>
#include <maths/COrderings.h>

#include <model/CResourceMonitor.h>
#include <model/CStringStore.h>

#include <algorithm>

namespace ml {
namespace model {
namespace {
const std::string NAMES_TAG("a");
const std::string FREE_NAMES_TAG("b");
const std::string RECYCLED_NAMES_TAG("c");
}

CDynamicStringIdRegistry::CDynamicStringIdRegistry(const std::string &nameType,
                                                   stat_t::EStatTypes addedStat,
                                                   stat_t::EStatTypes addNotAllowedStat,
                                                   stat_t::EStatTypes recycledStat)
    : m_NameType(nameType),
      m_AddedStat(addedStat),
      m_AddNotAllowedStat(addNotAllowedStat),
      m_RecycledStat(recycledStat),
      m_Uids(1) {}

CDynamicStringIdRegistry::CDynamicStringIdRegistry(bool isForPersistence,
                                                   const CDynamicStringIdRegistry &other)
    : m_NameType(other.m_NameType),
      m_AddedStat(other.m_AddedStat),
      m_AddNotAllowedStat(other.m_AddNotAllowedStat),
      m_RecycledStat(other.m_RecycledStat),
      m_Dictionary(other.m_Dictionary),
      m_Uids(other.m_Uids),
      m_Names(other.m_Names),
      m_FreeUids(other.m_FreeUids),
      m_RecycledUids(other.m_RecycledUids) {
    if (!isForPersistence) {
        LOG_ABORT("This constructor only creates clones for persistence");
    }
}

const std::string &CDynamicStringIdRegistry::name(std::size_t id,
                                                  const std::string &fallback) const {
    return id >= m_Names.size() ? fallback : *m_Names[id];
}

const core::CStoredStringPtr &CDynamicStringIdRegistry::namePtr(std::size_t id) const {
    return m_Names[id];
}

bool CDynamicStringIdRegistry::id(const std::string &name, std::size_t &result) const {
    TWordSizeUMapCItr itr = m_Uids.find(m_Dictionary.word(name));
    if (itr == m_Uids.end()) {
        result = INVALID_ID;
        return false;
    }
    result = itr->second;
    return true;
}

bool CDynamicStringIdRegistry::anyId(std::size_t &result) const {
    TWordSizeUMapCItr itr = m_Uids.begin();
    if (itr == m_Uids.end()) {
        result = INVALID_ID;
        return false;
    }
    result = itr->second;
    return true;
}

std::size_t CDynamicStringIdRegistry::numberActiveNames(void) const { return m_Uids.size(); }

std::size_t CDynamicStringIdRegistry::numberNames(void) const { return m_Names.size(); }

bool CDynamicStringIdRegistry::isIdActive(std::size_t id) const {
    return id < m_Names.size() &&
           !std::binary_search(
               m_FreeUids.begin(), m_FreeUids.end(), id, std::greater<std::size_t>());
}

std::size_t CDynamicStringIdRegistry::addName(const std::string &name,
                                              core_t::TTime time,
                                              CResourceMonitor &resourceMonitor,
                                              bool &addedPerson) {
    // Get the identifier or create one if this is the
    // first time we've seen them. (Use emplace to avoid copying
    // the string if it is already in the collection.)

    std::size_t newId = m_FreeUids.empty() ? m_Names.size() : m_FreeUids.back();

    std::size_t id = 0;

    // Is there any space in the system for us to expand the models?
    if (resourceMonitor.areAllocationsAllowed()) {
        id = m_Uids.emplace(m_Dictionary.word(name), newId).first->second;
    } else {
        // In this case we can only deal with existing people
        TWordSizeUMapCItr itr = m_Uids.find(m_Dictionary.word(name));
        if (itr == m_Uids.end()) {
            LOG_TRACE("Can't add new " << m_NameType << " - allocations not allowed");
            resourceMonitor.acceptAllocationFailureResult(time);
            core::CStatistics::stat(m_AddNotAllowedStat).increment();
            return INVALID_ID;
        }
        id = itr->second;
    }

    if (id >= m_Names.size()) {
        m_Names.push_back(CStringStore::names().get(name));
        addedPerson = true;
        core::CStatistics::stat(m_AddedStat).increment();
    } else if (id == newId) {
        LOG_TRACE("Recycling " << id << " for " << m_NameType << " " << name);
        m_Names[id] = CStringStore::names().get(name);
        if (m_FreeUids.empty()) {
            LOG_ERROR("Unexpectedly missing free " << m_NameType << " entry for " << id);
        } else {
            m_FreeUids.pop_back();
        }
        m_RecycledUids.push_back(id);
        core::CStatistics::stat(m_RecycledStat).increment();
    }

    return id;
}

void CDynamicStringIdRegistry::removeNames(std::size_t lowestNameToRemove) {
    std::size_t numberNames = this->numberNames();
    if (lowestNameToRemove >= numberNames) {
        return;
    }

    for (std::size_t id = lowestNameToRemove; id < numberNames; ++id) {
        m_Uids.erase(m_Dictionary.word(*m_Names[id]));
    }
    m_Names.erase(m_Names.begin() + lowestNameToRemove, m_Names.end());
}

void CDynamicStringIdRegistry::recycleNames(const TSizeVec &namesToRemove,
                                            const std::string &defaultName) {
    for (std::size_t i = 0u; i < namesToRemove.size(); ++i) {
        std::size_t id = namesToRemove[i];
        if (id >= m_Names.size()) {
            LOG_ERROR("Unexpected " << m_NameType << " identifier " << id);
            continue;
        }
        m_FreeUids.push_back(id);
        m_Uids.erase(m_Dictionary.word(*m_Names[id]));
        CStringStore::names().remove(*m_Names[id]);
        m_Names[id] = CStringStore::names().get(defaultName);
    }
    std::sort(m_FreeUids.begin(), m_FreeUids.end(), std::greater<std::size_t>());
    m_FreeUids.erase(std::unique(m_FreeUids.begin(), m_FreeUids.end()), m_FreeUids.end());
}

CDynamicStringIdRegistry::TSizeVec &CDynamicStringIdRegistry::recycledIds(void) {
    return m_RecycledUids;
}

bool CDynamicStringIdRegistry::checkInvariants(void) const {
    typedef boost::unordered_set<std::size_t> TSizeUSet;

    bool result = true;
    if (m_Uids.size() > m_Names.size()) {
        LOG_ERROR("Unexpected extra " << (m_Uids.size() - m_Names.size()) << " " << m_NameType
                                      << " uids");
        result = false;
    }

    TSizeUSet uniqueIds;
    for (TWordSizeUMapCItr i = m_Uids.begin(); i != m_Uids.end(); ++i) {
        if (!uniqueIds.insert(i->second).second) {
            LOG_ERROR("Duplicate id " << i->second);
            result = false;
        }
        if (i->second > m_Names.size()) {
            LOG_ERROR(m_NameType << " id " << i->second << " out of range [0, " << m_Names.size()
                                 << ")");
            result = false;
        }
    }

    return result;
}

void CDynamicStringIdRegistry::clear(void) {
    m_Uids.clear();
    m_Names.clear();
    m_FreeUids.clear();
    m_RecycledUids.clear();
}

uint64_t CDynamicStringIdRegistry::checksum(void) const {
    typedef boost::reference_wrapper<const std::string> TStrCRef;
    typedef std::vector<TStrCRef> TStrCRefVec;

    TStrCRefVec people;
    people.reserve(m_Names.size());
    for (std::size_t pid = 0u; pid < m_Names.size(); ++pid) {
        if (this->isIdActive(pid)) {
            people.emplace_back(*m_Names[pid]);
        }
    }
    std::sort(people.begin(), people.end(), maths::COrderings::SReferenceLess());
    return maths::CChecksum::calculate(0, people);
}

void CDynamicStringIdRegistry::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CDynamicStringIdRegistry");
    core::CMemoryDebug::dynamicSize("m_NameType", m_NameType, mem);
    core::CMemoryDebug::dynamicSize("m_PersonUids", m_Uids, mem);
    core::CMemoryDebug::dynamicSize("m_PersonNames", m_Names, mem);
    core::CMemoryDebug::dynamicSize("m_FreePersonUids", m_FreeUids, mem);
    core::CMemoryDebug::dynamicSize("m_RecycledPersonUids", m_RecycledUids, mem);
}

std::size_t CDynamicStringIdRegistry::memoryUsage(void) const {
    std::size_t mem = core::CMemory::dynamicSize(m_Uids);
    mem += core::CMemory::dynamicSize(m_NameType);
    mem += core::CMemory::dynamicSize(m_Uids);
    mem += core::CMemory::dynamicSize(m_Names);
    mem += core::CMemory::dynamicSize(m_FreeUids);
    mem += core::CMemory::dynamicSize(m_RecycledUids);
    return mem;
}

void CDynamicStringIdRegistry::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    // Explicity save all shared strings, on the understanding that any other
    // owners will also save their copies
    for (std::size_t i = 0; i < m_Names.size(); i++) {
        inserter.insertValue(NAMES_TAG, *m_Names[i]);
    }
    core::CPersistUtils::persist(FREE_NAMES_TAG, m_FreeUids, inserter);
    core::CPersistUtils::persist(RECYCLED_NAMES_TAG, m_RecycledUids, inserter);
}

bool CDynamicStringIdRegistry::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name = traverser.name();
        if (name == NAMES_TAG) {
            m_Names.push_back(CStringStore::names().get(traverser.value()));
        } else if (name == FREE_NAMES_TAG) {
            if (!core::CPersistUtils::restore(FREE_NAMES_TAG, m_FreeUids, traverser)) {
                return false;
            }
        } else if (name == RECYCLED_NAMES_TAG) {
            if (!core::CPersistUtils::restore(RECYCLED_NAMES_TAG, m_RecycledUids, traverser)) {
                return false;
            }
        }
    } while (traverser.next());

    // Some of the entries in m_Names may relate to IDs that are free to
    // reuse. We mustn't add these to the ID maps.

    for (std::size_t id = 0; id < m_Names.size(); ++id) {
        if (std::binary_search(
                m_FreeUids.begin(), m_FreeUids.end(), id, std::greater<std::size_t>())) {
            LOG_TRACE("Restore ignoring free " << m_NameType << " name " << *m_Names[id] << " = id "
                                               << id);
        } else {
            m_Uids[m_Dictionary.word(*m_Names[id])] = id;
        }
    }

    return true;
}

const std::size_t CDynamicStringIdRegistry::INVALID_ID(std::numeric_limits<std::size_t>::max());
}
}
