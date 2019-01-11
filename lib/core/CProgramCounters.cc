/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CProgramCounters.h>

#include <core/CLogger.h>
#include <core/CRapidJsonLineWriter.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>

#include <ostream>
#include <string>

namespace ml {
namespace core {

namespace {

using TGenericLineWriter = core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>;

const std::string NAME_TYPE("name");
const std::string DESCRIPTION_TYPE("description");
const std::string COUNTER_TYPE("value");

//! Persistence tags
const std::string KEY_TAG("a");
const std::string VALUE_TAG("b");

//! Helper function to add a string/int pair to JSON writer
void addStringInt(TGenericLineWriter& writer,
                  const std::string& name,
                  const std::string& description,
                  uint64_t counter) {
    writer.StartObject();

    writer.String(NAME_TYPE);
    writer.String(name);

    writer.String(DESCRIPTION_TYPE);
    writer.String(description);

    writer.String(COUNTER_TYPE);
    writer.Uint64(counter);

    writer.EndObject();
}
}

CProgramCounters::CProgramCounters()
    : m_Counters(), m_Cache(), m_ProgramCounterTypes() {
}

CProgramCounters& CProgramCounters::instance() {
    return ms_Instance;
}

CProgramCounters::TCounter& CProgramCounters::counter(counter_t::ECounterTypes counterType) {
    if (counterType >= counter_t::E_LastEnumCounter) {
        LOG_ABORT(<< "Bad index " << counterType);
    }
    return ms_Instance.m_Counters[counterType];
}

CProgramCounters::TCounter& CProgramCounters::counter(size_t index) {
    if (index >= ms_Instance.m_Counters.size()) {
        LOG_ABORT(<< "Bad index " << index);
    }
    return ms_Instance.m_Counters[index];
}

void CProgramCounters::cacheCounters() {
    if (ms_Instance.m_Cache.size() != 0) {
        // The cache should only exist for a very brief period of time,
        // while persistence is in operation.
        // Elsewhere there are checks to ensure that only one persistence
        // operation occurs at any point in time,so the cache _should_ be
        // empty at this point.  Warn if that isn't the case.
        LOG_WARN(<< "Overwriting existing counters cache.");

        // clear the cache
        TUInt64Vec().swap(ms_Instance.m_Cache);
    }
    ms_Instance.m_Cache.assign(ms_Instance.m_Counters.begin(),
                               ms_Instance.m_Counters.end());
}

void CProgramCounters::staticsAcceptPersistInserter(CStatePersistInserter& inserter) {
    // As the analytics thread could be updating counters while this method is running, in order
    // to guarantee that consistent counters get persisted for a background persistence task we must not persist
    // live values. Instead, we access a cached set of self consistent counters for persistence.

    auto staticsAcceptPersistInserter = [&](const auto& container) {
        int i = 0;
        for (const auto& counter : container) {
            inserter.insertValue(KEY_TAG, i);
            inserter.insertValue(VALUE_TAG, counter);
            ++i;
        }
    };

    if (ms_Instance.m_Cache.size() == 0) {
        // Programmatic error, cacheCounters has not been called immediately prior to this call.
        // Choose to populate the persisted counters with the live values.
        LOG_ERROR(<< "Unexpectedly empty counters cache.");

        staticsAcceptPersistInserter(ms_Instance.m_Counters);
    } else {
        staticsAcceptPersistInserter(ms_Instance.m_Cache);

        // clear the cache
        TUInt64Vec().swap(ms_Instance.m_Cache);
    }
}

bool CProgramCounters::staticsAcceptRestoreTraverser(CStateRestoreTraverser& traverser) {
    uint64_t value = 0;
    int key = 0;
    do {
        const std::string& name = traverser.name();
        if (name == KEY_TAG) {
            value = 0;
            if (CStringUtils::stringToType(traverser.value(), key) == false) {
                LOG_ERROR(<< "Invalid key value in " << traverser.value());
                return false;
            }
        } else if (name == VALUE_TAG) {
            if (CStringUtils::stringToType(traverser.value(), value) == false) {
                LOG_ERROR(<< "Invalid counter value in " << traverser.value());
                return false;
            }
            // Only restore counter values the current program has registered interest in. Set all other values to zero.
            // If no such limited set has been registered the all values are restored.
            if (ms_Instance.m_ProgramCounterTypes.size() != 0) {
                const auto& itr = ms_Instance.m_ProgramCounterTypes.find(
                    static_cast<counter_t::ECounterTypes>(key));
                if (itr == ms_Instance.m_ProgramCounterTypes.end()) {
                    value = 0;
                }
            }
            counter(key) = value;
            key = 0;
            value = 0;
        }
    } while (traverser.next());

    return true;
}

CProgramCounters CProgramCounters::ms_Instance;

void CProgramCounters::registerProgramCounterTypes(const counter_t::TCounterTypeSet& programCounterTypes) {
    ms_Instance.m_ProgramCounterTypes = programCounterTypes;
}

std::ostream& operator<<(std::ostream& o, const CProgramCounters& counters) {
    rapidjson::OStreamWrapper writeStream(o);
    TGenericLineWriter writer(writeStream);

    writer.StartArray();

    // If the application has not specified a limited set of (counters using registerProgramCounterTypes) then print the entire set
    // Take care to print in definition order
    if (counters.m_ProgramCounterTypes.size() == 0) {
        for (const auto& ctr : counters.m_CounterDefinitions) {
            addStringInt(writer, ctr.s_Name, ctr.s_Description,
                         counters.counter(ctr.s_Type));
        }
    } else {
        for (const auto& ctr : counters.m_CounterDefinitions) {
            if (counters.m_ProgramCounterTypes.find(ctr.s_Type) !=
                counters.m_ProgramCounterTypes.end()) {
                addStringInt(writer, ctr.s_Name, ctr.s_Description,
                             counters.counter(ctr.s_Type));
            }
        }
    }

    writer.EndArray();
    writeStream.Flush();

    return o;
}

} // core
} // ml
