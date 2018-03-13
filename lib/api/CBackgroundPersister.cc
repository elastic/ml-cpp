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
#include <api/CBackgroundPersister.h>

#include <core/CLogger.h>
#include <core/CScopedFastLock.h>
#include <core/CTimeUtils.h>

#include <string>
#include <utility>

namespace ml {
namespace api {

namespace {
const core_t::TTime PERSIST_INTERVAL_INCREMENT(300);// 5 minutes
}

CBackgroundPersister::CBackgroundPersister(core_t::TTime periodicPersistInterval,
                                           core::CDataAdder &dataAdder)
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

CBackgroundPersister::CBackgroundPersister(
    core_t::TTime periodicPersistInterval,
    const TFirstProcessorPeriodicPersistFunc &firstProcessorPeriodicPersistFunc,
    core::CDataAdder &dataAdder)
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

CBackgroundPersister::~CBackgroundPersister(void) { this->waitForIdle(); }

bool CBackgroundPersister::isBusy(void) const { return m_IsBusy; }

bool CBackgroundPersister::waitForIdle(void) {
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

    if (m_IsBusy) {
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

bool CBackgroundPersister::startPersist(void) {
    if (m_PersistFuncs.empty()) {
        return false;
    }

    core::CScopedFastLock lock(m_Mutex);

    if (m_IsBusy) {
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

bool CBackgroundPersister::clear(void) {
    core::CScopedFastLock lock(m_Mutex);

    if (m_IsBusy) {
        return false;
    }

    m_PersistFuncs.clear();

    return true;
}

bool CBackgroundPersister::firstProcessorPeriodicPersistFunc(
    const TFirstProcessorPeriodicPersistFunc &firstProcessorPeriodicPersistFunc) {
    core::CScopedFastLock lock(m_Mutex);

    if (m_IsBusy) {
        return false;
    }

    m_FirstProcessorPeriodicPersistFunc = firstProcessorPeriodicPersistFunc;

    return true;
}

bool CBackgroundPersister::startBackgroundPersistIfAppropriate(void) {
    core_t::TTime due(m_LastPeriodicPersistTime + m_PeriodicPersistInterval);
    core_t::TTime now(core::CTimeUtils::now());
    if (now < due) {
        // Persist is not due
        return false;
    }

    if (this->isBusy()) {
        m_PeriodicPersistInterval += PERSIST_INTERVAL_INCREMENT;

        LOG_WARN("Periodic persist is due at "
                 << due << " but previous persist started at "
                 << core::CTimeUtils::toIso8601(m_LastPeriodicPersistTime)
                 << " is still in progress - increased persistence interval to "
                 << m_PeriodicPersistInterval << " seconds");

        return false;
    }

    bool backgroundPersistSetupOk = m_FirstProcessorPeriodicPersistFunc(*this);
    if (!backgroundPersistSetupOk) {
        LOG_ERROR("Failed to create background persistence functions");
        // It's possible that some functions were added before the failure, so
        // remove these
        this->clear();
        return false;
    }

    m_LastPeriodicPersistTime = now;

    LOG_INFO("Background persist starting background thread");

    if (this->startPersist() == false) {
        LOG_ERROR("Failed to start background persistence");
        this->clear();
        return false;
    }

    return true;
}

CBackgroundPersister::CBackgroundThread::CBackgroundThread(CBackgroundPersister &owner)
    : m_Owner(owner) {}

void CBackgroundPersister::CBackgroundThread::run(void) {

    while (!m_Owner.m_PersistFuncs.empty()) {
        if (!m_Owner.m_IsShutdown) {
            m_Owner.m_PersistFuncs.front()(m_Owner.m_DataAdder);
        }
        m_Owner.m_PersistFuncs.pop_front();
    }

    core::CScopedFastLock lock(m_Owner.m_Mutex);

    m_Owner.m_IsBusy = false;
}

void CBackgroundPersister::CBackgroundThread::shutdown(void) { m_Owner.m_IsShutdown = true; }
}
}
