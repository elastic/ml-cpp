/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CBackgroundPersister.h>

#include <core/CLogger.h>
#include <core/CScopedFastLock.h>
#include <core/CTimeUtils.h>

#include <string>
#include <utility>

namespace ml {
namespace api {

namespace {
const core_t::TTime PERSIST_INTERVAL_INCREMENT(300); // 5 minutes
}

CBackgroundPersister::CBackgroundPersister(core_t::TTime periodicPersistInterval, core::CDataAdder& dataAdder)
    : m_PeriodicPersistInterval(periodicPersistInterval),
      m_LastPeriodicPersistTime(core::CTimeUtils::now()),
      m_DataAdder(dataAdder),
      m_IsBusy(false),
      m_IsShutdown(false),
      m_BackgroundThread(*this) {
    if (m_PeriodicPersistInterval < PERSIST_INTERVAL_INCREMENT) {
        // This may be dynamically increased further depending on how long
        // persistence takes
        m_PeriodicPersistInterval = PERSIST_INTERVAL_INCREMENT;
    }
}

CBackgroundPersister::CBackgroundPersister(core_t::TTime periodicPersistInterval,
                                           const TFirstProcessorPeriodicPersistFunc& firstProcessorPeriodicPersistFunc,
                                           core::CDataAdder& dataAdder)
    : m_PeriodicPersistInterval(periodicPersistInterval),
      m_LastPeriodicPersistTime(core::CTimeUtils::now()),
      m_FirstProcessorPeriodicPersistFunc(firstProcessorPeriodicPersistFunc),
      m_DataAdder(dataAdder),
      m_IsBusy(false),
      m_IsShutdown(false),
      m_BackgroundThread(*this) {
    if (m_PeriodicPersistInterval < PERSIST_INTERVAL_INCREMENT) {
        // This may be dynamically increased further depending on how long
        // persistence takes
        m_PeriodicPersistInterval = PERSIST_INTERVAL_INCREMENT;
    }
}

CBackgroundPersister::~CBackgroundPersister() {
    this->waitForIdle();
}

bool CBackgroundPersister::isBusy() const {
    return m_IsBusy;
}

bool CBackgroundPersister::waitForIdle() {
    {
        core::CScopedFastLock lock(m_Mutex);

        if (!m_BackgroundThread.isStarted()) {
            return true;
        }
    }

    return m_BackgroundThread.waitForFinish();
}

bool CBackgroundPersister::addPersistFunc(core::CDataAdder::TPersistFunc persistFunc) {
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

bool CBackgroundPersister::startPersist() {
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

    m_IsShutdown = false;
    m_IsBusy = m_BackgroundThread.start();

    return m_IsBusy;
}

bool CBackgroundPersister::clear() {
    core::CScopedFastLock lock(m_Mutex);

    if (this->isBusy()) {
        return false;
    }

    m_PersistFuncs.clear();

    return true;
}

bool CBackgroundPersister::firstProcessorPeriodicPersistFunc(const TFirstProcessorPeriodicPersistFunc& firstProcessorPeriodicPersistFunc) {
    core::CScopedFastLock lock(m_Mutex);

    if (this->isBusy()) {
        return false;
    }

    m_FirstProcessorPeriodicPersistFunc = firstProcessorPeriodicPersistFunc;

    return true;
}

bool CBackgroundPersister::startBackgroundPersist() {
    if (this->isBusy()) {
        LOG_WARN("Cannot start background persist as a previous "
                 "persist is still in progress");
        return false;
    }
    return this->startBackgroundPersist(core::CTimeUtils::now());
}

bool CBackgroundPersister::startBackgroundPersistIfAppropriate() {
    core_t::TTime due(m_LastPeriodicPersistTime + m_PeriodicPersistInterval);
    core_t::TTime now(core::CTimeUtils::now());
    if (now < due) {
        // Persist is not due
        return false;
    }

    if (this->isBusy()) {
        m_PeriodicPersistInterval += PERSIST_INTERVAL_INCREMENT;

        LOG_WARN("Periodic persist is due at "
                 << due << " but previous persist started at " << core::CTimeUtils::toIso8601(m_LastPeriodicPersistTime)
                 << " is still in progress - increased persistence interval to " << m_PeriodicPersistInterval << " seconds");

        return false;
    }

    return this->startBackgroundPersist(now);
}

bool CBackgroundPersister::startBackgroundPersist(core_t::TTime timeOfPersistence) {
    bool backgroundPersistSetupOk = m_FirstProcessorPeriodicPersistFunc(*this);
    if (!backgroundPersistSetupOk) {
        LOG_ERROR("Failed to create background persistence functions");
        // It's possible that some functions were added before the failure, so
        // remove these
        this->clear();
        return false;
    }

    m_LastPeriodicPersistTime = timeOfPersistence;

    LOG_INFO("Background persist starting background thread");

    if (this->startPersist() == false) {
        LOG_ERROR("Failed to start background persistence");
        this->clear();
        return false;
    }

    return true;
}

CBackgroundPersister::CBackgroundThread::CBackgroundThread(CBackgroundPersister& owner) : m_Owner(owner) {
}

void CBackgroundPersister::CBackgroundThread::run() {
    // The isBusy check will prevent concurrent access to
    // m_Owner.m_PersistFuncs here
    while (!m_Owner.m_PersistFuncs.empty()) {
        if (!m_Owner.m_IsShutdown) {
            m_Owner.m_PersistFuncs.front()(m_Owner.m_DataAdder);
        }
        m_Owner.m_PersistFuncs.pop_front();
    }

    core::CScopedFastLock lock(m_Owner.m_Mutex);
    m_Owner.m_IsBusy = false;
}

void CBackgroundPersister::CBackgroundThread::shutdown() {
    m_Owner.m_IsShutdown = true;
}
}
}
