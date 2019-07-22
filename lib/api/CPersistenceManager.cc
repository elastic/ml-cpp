/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CPersistenceManager.h>
#include <core/CLogger.h>
#include <core/CProgramCounters.h>
#include <core/CScopedFastLock.h>
#include <core/CTimeUtils.h>

#include <string>
#include <utility>

namespace ml {
namespace api {

namespace {
const core_t::TTime PERSIST_INTERVAL_INCREMENT(300); // 5 minutes
}

CPersistenceManager::CPersistenceManager(core_t::TTime periodicPersistInterval,
                                         bool persistInForeground,
                                         core::CDataAdder& dataAdder)
    : CPersistenceManager(periodicPersistInterval, persistInForeground, dataAdder, dataAdder) {
}

CPersistenceManager::CPersistenceManager(core_t::TTime periodicPersistInterval,
                                         bool persistInForeground,
                                         core::CDataAdder& bgDataAdder,
                                         core::CDataAdder& fgDataAdder)
    : m_PeriodicPersistInterval(periodicPersistInterval),
      m_PersistInForeground(persistInForeground),
      m_LastPeriodicPersistTime(core::CTimeUtils::now()),
      m_BgDataAdder(bgDataAdder), m_FgDataAdder(fgDataAdder), m_IsBusy(false),
      m_IsShutdown(false), m_BackgroundThread(*this) {
    if (m_PeriodicPersistInterval < PERSIST_INTERVAL_INCREMENT) {
        // This may be dynamically increased further depending on how long
        // persistence takes
        m_PeriodicPersistInterval = PERSIST_INTERVAL_INCREMENT;
    }
}

CPersistenceManager::~CPersistenceManager() {
    this->waitForIdle();
}

bool CPersistenceManager::isBusy() const {
    return m_IsBusy;
}

bool CPersistenceManager::waitForIdle() {
    {
        core::CScopedFastLock lock(m_Mutex);

        if (!m_BackgroundThread.isStarted()) {
            return true;
        }
    }

    return m_BackgroundThread.waitForFinish();
}

void CPersistenceManager::useBackgroundPersistence() {
    m_PersistInForeground = false;
}

void CPersistenceManager::useForegroundPersistence() {
    m_PersistInForeground = true;
}

bool CPersistenceManager::addPersistFunc(core::CDataAdder::TPersistFunc persistFunc) {
    if (!persistFunc) {
        return false;
    }

    core::CScopedFastLock lock(m_Mutex);

    if (this->isBusy()) {
        return false;
    }

    if (m_BackgroundThread.isStarted()) {
        // This join should be fast as the busy flag is false so the thread
        // should either have already exited or be on the verge of exiting
        if (m_BackgroundThread.waitForFinish() == false) {
            return false;
        }
    }

    m_PersistFuncs.push_back(std::move(persistFunc));

    return true;
}

bool CPersistenceManager::startPersistInBackground() {
    core::CScopedFastLock lock(m_Mutex);

    if (this->isBusy()) {
        return false;
    }

    if (m_PersistFuncs.empty()) {
        return false;
    }

    if (m_BackgroundThread.isStarted()) {
        // This join should be fast as the busy flag is false so the thread
        // should either have already exited or be on the verge of exiting
        if (m_BackgroundThread.waitForFinish() == false) {
            return false;
        }
    }

    // create a cache of the counters, to ensure persistence operates
    // on a consistent collection of counters
    core::CProgramCounters::cacheCounters();
    m_IsShutdown = false;
    m_IsBusy = m_BackgroundThread.start();

    return m_IsBusy;
}

bool CPersistenceManager::clear() {
    core::CScopedFastLock lock(m_Mutex);

    if (this->isBusy()) {
        return false;
    }

    m_PersistFuncs.clear();

    return true;
}

bool CPersistenceManager::firstProcessorBackgroundPeriodicPersistFunc(
    const TFirstProcessorPeriodicPersistFunc& firstProcessorPeriodicPersistFunc) {
    core::CScopedFastLock lock(m_Mutex);

    if (this->isBusy()) {
        return false;
    }

    m_FirstProcessorBackgroundPeriodicPersistFunc = firstProcessorPeriodicPersistFunc;

    return true;
}

bool CPersistenceManager::firstProcessorForegroundPeriodicPersistFunc(
    const TFirstProcessorPeriodicPersistFunc& firstProcessorPeriodicPersistFunc) {
    core::CScopedFastLock lock(m_Mutex);

    if (this->isBusy()) {
        return false;
    }

    m_FirstProcessorForegroundPeriodicPersistFunc = firstProcessorPeriodicPersistFunc;

    return true;
}

bool CPersistenceManager::startPersistIfAppropriate() {
    core_t::TTime due(m_LastPeriodicPersistTime + m_PeriodicPersistInterval);
    core_t::TTime now(core::CTimeUtils::now());
    if (now < due) {
        // Persist is not due
        return false;
    }

    if (this->isBusy()) {
        m_PeriodicPersistInterval += PERSIST_INTERVAL_INCREMENT;

        LOG_WARN(<< "Periodic persist is due at " << due << " but previous persist started at "
                 << core::CTimeUtils::toIso8601(m_LastPeriodicPersistTime)
                 << " is still in progress - increased persistence interval to "
                 << m_PeriodicPersistInterval << " seconds");

        return false;
    }

    return this->startPersist(now);
}

bool CPersistenceManager::startPersist(core_t::TTime timeOfPersistence) {
    if (this->isBusy()) {
        LOG_WARN(<< "Cannot start persist as a previous "
                    "background persist is still in progress");
        return false;
    }

    auto firstProcessorPeriodicPersistFunc = [&]() {
        return m_PersistInForeground ? m_FirstProcessorForegroundPeriodicPersistFunc
                                     : m_FirstProcessorBackgroundPeriodicPersistFunc;
    };

    bool persistSetupOk = firstProcessorPeriodicPersistFunc()(*this);
    if (!persistSetupOk) {
        LOG_ERROR(<< "Failed to create persistence functions");
        // It's possible that some functions were added before the failure, so
        // remove these
        this->clear();
        return false;
    }

    m_LastPeriodicPersistTime = timeOfPersistence;

    if (m_PersistInForeground) {
        this->startPersist();
    } else {
        LOG_INFO(<< "Background persist starting background thread");

        if (this->startPersistInBackground() == false) {
            LOG_ERROR(<< "Failed to start background persistence");
            this->clear();
            return false;
        }
    }

    return true;
}

void CPersistenceManager::startPersist() {
    while (m_PersistFuncs.empty() == false) {
        m_PersistFuncs.front()(m_PersistInForeground ? m_FgDataAdder : m_BgDataAdder);
        m_PersistFuncs.pop_front();
    }
}

CPersistenceManager::CBackgroundThread::CBackgroundThread(CPersistenceManager& owner)
    : m_Owner(owner) {
}

void CPersistenceManager::CBackgroundThread::run() {
    // The isBusy check will prevent concurrent access to
    // m_Owner.m_PersistFuncs here
    m_Owner.startPersist();

    core::CScopedFastLock lock(m_Owner.m_Mutex);
    m_Owner.m_IsBusy = false;
}

void CPersistenceManager::CBackgroundThread::shutdown() {
    m_Owner.m_IsShutdown = true;
}
}
}
