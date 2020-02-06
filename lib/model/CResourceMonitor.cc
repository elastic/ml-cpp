/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CResourceMonitor.h>

#include <core/CProgramCounters.h>
#include <core/Constants.h>

#include <model/CMonitoredResource.h>
#include <model/CStringStore.h>

#include <algorithm>
#include <limits>

namespace ml {

namespace model {

// Only prune once per hour
const core_t::TTime CResourceMonitor::MINIMUM_PRUNE_FREQUENCY(60 * 60);
const std::size_t CResourceMonitor::DEFAULT_MEMORY_LIMIT_MB(4096);
const double CResourceMonitor::DEFAULT_BYTE_LIMIT_MARGIN(0.7);
const core_t::TTime
    CResourceMonitor::MAXIMUM_BYTE_LIMIT_MARGIN_PERIOD(2 * core::constants::HOUR);

CResourceMonitor::CResourceMonitor(bool persistenceInForeground, double byteLimitMargin)
    : m_AllowAllocations(true), m_ByteLimitMargin{byteLimitMargin},
      m_ByteLimitHigh(0), m_ByteLimitLow(0), m_MonitoredResourceCurrentMemory(0),
      m_ExtraMemory(0), m_PreviousTotal(this->totalMemory()), m_Peak(m_PreviousTotal),
      m_LastAllocationFailureReport(0), m_MemoryStatus(model_t::E_MemoryStatusOk),
      m_HasPruningStarted(false), m_PruneThreshold(0), m_LastPruneTime(0),
      m_PruneWindow(std::numeric_limits<std::size_t>::max()),
      m_PruneWindowMaximum(std::numeric_limits<std::size_t>::max()),
      m_PruneWindowMinimum(std::numeric_limits<std::size_t>::max()),
      m_NoLimit(false), m_CurrentBytesExceeded(0),
      m_PersistenceInForeground(persistenceInForeground) {
    this->updateMemoryLimitsAndPruneThreshold(DEFAULT_MEMORY_LIMIT_MB);
}

void CResourceMonitor::memoryUsageReporter(const TMemoryUsageReporterFunc& reporter) {
    m_MemoryUsageReporter = reporter;
}

void CResourceMonitor::registerComponent(CMonitoredResource& resource) {
    LOG_TRACE(<< "Registering component: " << &resource);
    m_Resources.emplace(&resource, std::size_t(0));
}

void CResourceMonitor::unRegisterComponent(CMonitoredResource& resource) {
    LOG_TRACE(<< "Unregistering component: " << &resource);
    auto itr = m_Resources.find(&resource);
    if (itr == m_Resources.end()) {
        LOG_ERROR(<< "Inconsistency - component has not been registered: " << &resource);
        return;
    }

    m_MonitoredResourceCurrentMemory -= itr->second;
    m_Resources.erase(itr);
    core::CProgramCounters::counter(counter_t::E_TSADMemoryUsage) = this->totalMemory();
}

void CResourceMonitor::memoryLimit(std::size_t limitMBs) {
    this->updateMemoryLimitsAndPruneThreshold(limitMBs);

    if (m_NoLimit) {
        LOG_INFO(<< "Setting no model memory limit");
    } else {
        LOG_INFO(<< "Setting model memory limit to " << limitMBs << " MB");
    }
}

void CResourceMonitor::updateMemoryLimitsAndPruneThreshold(std::size_t limitMBs) {
    // The threshold for no limit is set such that any negative limit cast to
    // a size_t (which is unsigned) will be taken to mean no limit
    if (limitMBs > std::numeric_limits<std::size_t>::max() / 2) {
        m_NoLimit = true;
        // The high limit is set to around half what it could potentially be.
        // The reason is that other code will do "what if" calculations on this
        // number, such as "what would total memory usage be if we allocated 10
        // more models?", and it causes problems if these calculations overflow.
        m_ByteLimitHigh = std::numeric_limits<std::size_t>::max() / 2 + 1;
    } else {
        m_ByteLimitHigh = static_cast<std::size_t>(
            (limitMBs * 1024 * 1024) / this->persistenceMemoryIncreaseFactor());
    }
    m_ByteLimitLow = (m_ByteLimitHigh * 49) / 50;
    m_PruneThreshold = (m_ByteLimitHigh * 3) / 5;
}

model_t::EMemoryStatus CResourceMonitor::getMemoryStatus() {
    return m_MemoryStatus;
}

void CResourceMonitor::refresh(CMonitoredResource& resource) {
    if (m_NoLimit) {
        return;
    }
    this->forceRefresh(resource);
}

void CResourceMonitor::forceRefresh(CMonitoredResource& resource) {
    this->memUsage(&resource);

    this->updateAllowAllocations();
}

void CResourceMonitor::updateAllowAllocations() {
    std::size_t total{this->totalMemory()};
    core::CProgramCounters::counter(counter_t::E_TSADMemoryUsage) = total;
    LOG_TRACE(<< "Checking allocations: currently at " << total);
    if (m_AllowAllocations) {
        if (total > this->highLimit()) {
            LOG_INFO(<< "Over current allocation high limit. " << total
                     << " bytes used, the limit is " << this->highLimit());
            m_AllowAllocations = false;
            std::size_t bytesExceeded{total - this->highLimit()};
            m_CurrentBytesExceeded = this->adjustedUsage(bytesExceeded);
        }
    } else if (total < this->lowLimit()) {
        LOG_INFO(<< "Below current allocation low limit. " << total
                 << " bytes used, the limit is " << this->lowLimit());
        m_AllowAllocations = true;
    }
}

bool CResourceMonitor::pruneIfRequired(core_t::TTime endTime) {
    // The basic idea here is that as the memory usage goes up, we
    // prune models to bring it down again. If usage declines, we
    // relax the pruning window to let it go back up again.

    std::size_t total{this->totalMemory()};
    bool aboveThreshold = total > m_PruneThreshold;

    if (m_HasPruningStarted == false && !aboveThreshold) {
        LOG_TRACE(<< "No pruning required. " << total << " / " << m_PruneThreshold);
        return false;
    }

    if (endTime < m_LastPruneTime + MINIMUM_PRUNE_FREQUENCY) {
        LOG_TRACE(<< "Too soon since last prune to prune again");
        return false;
    }

    if (m_Resources.empty()) {
        return false;
    }

    if (m_HasPruningStarted == false) {
        for (const auto& resource : m_Resources) {
            if (resource.first->supportsPruning() &&
                resource.first->initPruneWindow(m_PruneWindowMaximum, m_PruneWindowMinimum)) {
                m_PruneWindow = m_PruneWindowMaximum;
                m_HasPruningStarted = true;
                break;
            }
        }
        if (m_HasPruningStarted == false) {
            return false;
        }
        this->acceptPruningResult();
        LOG_DEBUG(<< "Pruning started. Window (buckets): " << m_PruneWindow);
    }

    if (aboveThreshold) {
        // Do a prune and see how much we got back
        // These are the expensive operations
        std::size_t usageAfter = 0;
        for (auto& resource : m_Resources) {
            if (resource.first->supportsPruning()) {
                resource.first->prune(m_PruneWindow);
                resource.second = core::CMemory::dynamicSize(resource.first);
            }
            usageAfter += resource.second;
        }
        m_MonitoredResourceCurrentMemory = usageAfter;
        total = this->totalMemory();
        this->updateAllowAllocations();
    }

    LOG_TRACE(<< "Pruning models. Usage: " << total
              << ". Current window: " << m_PruneWindow << " buckets");

    if (total < m_PruneThreshold) {
        // Expand the window
        for (const auto& resource : m_Resources) {
            if (resource.first->supportsPruning()) {
                m_PruneWindow = std::min(
                    m_PruneWindow + std::size_t((endTime - m_LastPruneTime) /
                                                resource.first->bucketLength()),
                    m_PruneWindowMaximum);
                LOG_TRACE(<< "Expanding window, to " << m_PruneWindow);
                break;
            }
        }
    } else {
        // Shrink the window
        m_PruneWindow = std::max(m_PruneWindow * 99 / 100, m_PruneWindowMinimum);
        LOG_TRACE(<< "Shrinking window, to " << m_PruneWindow);
    }

    m_LastPruneTime = endTime;
    return aboveThreshold;
}

bool CResourceMonitor::areAllocationsAllowed() const {
    return m_AllowAllocations;
}

std::size_t CResourceMonitor::allocationLimit() const {
    return this->highLimit() - std::min(this->highLimit(), this->totalMemory());
}

void CResourceMonitor::memUsage(CMonitoredResource* resource) {
    auto itr = m_Resources.find(resource);
    if (itr == m_Resources.end()) {
        LOG_ERROR(<< "Inconsistency - component has not been registered: " << resource);
        return;
    }
    std::size_t modelPreviousUsage = itr->second;
    std::size_t modelCurrentUsage = core::CMemory::dynamicSize(itr->first);
    itr->second = modelCurrentUsage;
    m_MonitoredResourceCurrentMemory += (modelCurrentUsage - modelPreviousUsage);
}

void CResourceMonitor::sendMemoryUsageReportIfSignificantlyChanged(core_t::TTime bucketStartTime) {
    if (this->needToSendReport()) {
        this->sendMemoryUsageReport(bucketStartTime);
    }
}

bool CResourceMonitor::needToSendReport() {
    // Has the usage changed by more than 1% ?
    std::size_t total{this->totalMemory()};
    if ((std::max(total, m_PreviousTotal) - std::min(total, m_PreviousTotal)) >
        m_PreviousTotal / 100) {
        return true;
    }

    if (!m_AllocationFailures.empty()) {
        core_t::TTime latestAllocationError{(--m_AllocationFailures.end())->first};
        if (latestAllocationError > m_LastAllocationFailureReport) {
            return true;
        }
    }
    return false;
}

void CResourceMonitor::sendMemoryUsageReport(core_t::TTime bucketStartTime) {
    std::size_t total{this->totalMemory()};
    m_Peak = std::max(m_Peak, total);
    if (m_MemoryUsageReporter) {
        m_MemoryUsageReporter(this->createMemoryUsageReport(bucketStartTime));
        if (!m_AllocationFailures.empty()) {
            m_LastAllocationFailureReport = m_AllocationFailures.rbegin()->first;
        }
    }
    m_PreviousTotal = total;
}

CResourceMonitor::SModelSizeStats
CResourceMonitor::createMemoryUsageReport(core_t::TTime bucketStartTime) {
    SModelSizeStats res;
    res.s_Usage = this->totalMemory();
    res.s_AdjustedUsage = this->adjustedUsage(res.s_Usage);
    res.s_BytesMemoryLimit = 2 * m_ByteLimitHigh;
    res.s_BytesExceeded = m_CurrentBytesExceeded;
    res.s_MemoryStatus = m_MemoryStatus;
    res.s_BucketStartTime = bucketStartTime;
    for (const auto& resource : m_Resources) {
        resource.first->updateModelSizeStats(res);
    }
    res.s_AllocationFailures += m_AllocationFailures.size();
    return res;
}

std::size_t CResourceMonitor::adjustedUsage(std::size_t usage) const {
    // We scale the reported memory usage by the inverse of the byte limit margin.
    // This gives the user a fairer indication of how close the job is to hitting
    // the model memory limit in a concise manner (as the limit is scaled down by
    // the margin during the beginning period of the job's existence).
    size_t adjustedUsage{static_cast<std::size_t>(usage / m_ByteLimitMargin)};

    adjustedUsage *= this->persistenceMemoryIncreaseFactor();

    return adjustedUsage;
}

std::size_t CResourceMonitor::persistenceMemoryIncreaseFactor() const {
    // Background persist causes the memory size to double due to copying
    // the models. On top of that, after the persist is done we may not
    // be able to retrieve that memory back. Thus, we report twice the
    // memory usage in order to allow for that.
    // See https://github.com/elastic/x-pack-elasticsearch/issues/1020.
    // Issue https://github.com/elastic/x-pack-elasticsearch/issues/857
    // discusses adding an option to perform only foreground persist.
    // If that gets implemented, we should only double when background
    // persist is configured.
    return m_PersistenceInForeground ? 1 : 2;
}

void CResourceMonitor::acceptAllocationFailureResult(core_t::TTime time) {
    m_MemoryStatus = model_t::E_MemoryStatusHardLimit;
    ++m_AllocationFailures[time];
}

void CResourceMonitor::acceptPruningResult() {
    if (m_MemoryStatus == model_t::E_MemoryStatusOk) {
        m_MemoryStatus = model_t::E_MemoryStatusSoftLimit;
    }
}

bool CResourceMonitor::haveNoLimit() const {
    return m_NoLimit;
}

void CResourceMonitor::addExtraMemory(std::size_t mem) {
    m_ExtraMemory += mem;
    this->updateAllowAllocations();
}

void CResourceMonitor::clearExtraMemory() {
    if (m_ExtraMemory != 0) {
        m_ExtraMemory = 0;
        this->updateAllowAllocations();
    }
}

void CResourceMonitor::decreaseMargin(core_t::TTime elapsedTime) {
    // We choose to increase the margin to close to 1 on the order
    // time it takes to detect diurnal periodic components. These
    // will be the overwhelmingly common source of additional memory
    // so the model memory should be accurate (on average) in this
    // time frame.
    double scale{1.0 - static_cast<double>(std::min(elapsedTime, MAXIMUM_BYTE_LIMIT_MARGIN_PERIOD)) /
                           static_cast<double>(core::constants::DAY)};
    m_ByteLimitMargin = 1.0 - scale * (1.0 - m_ByteLimitMargin);
}

std::size_t CResourceMonitor::highLimit() const {
    return static_cast<std::size_t>(m_ByteLimitMargin * static_cast<double>(m_ByteLimitHigh));
}

std::size_t CResourceMonitor::lowLimit() const {
    return static_cast<std::size_t>(m_ByteLimitMargin * static_cast<double>(m_ByteLimitLow));
}

std::size_t CResourceMonitor::totalMemory() const {
    return m_MonitoredResourceCurrentMemory + m_ExtraMemory +
           CStringStore::names().memoryUsage() +
           CStringStore::influencers().memoryUsage();
}

} // model
} // ml
