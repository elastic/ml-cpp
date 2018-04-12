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

#include <model/CResourceMonitor.h>

#include <core/CStatistics.h>
#include <core/Constants.h>

#include <model/CAnomalyDetector.h>
#include <model/CDataGatherer.h>
#include <model/CStringStore.h>

#include <algorithm>
#include <limits>

namespace ml {

namespace model {

// Only prune once per hour
const core_t::TTime CResourceMonitor::MINIMUM_PRUNE_FREQUENCY(60 * 60);

const std::size_t CResourceMonitor::DEFAULT_MEMORY_LIMIT_MB(4096);

CResourceMonitor::CResourceMonitor()
    : m_AllowAllocations(true), m_ByteLimitHigh(0), m_ByteLimitLow(0),
      m_CurrentAnomalyDetectorMemory(0), m_ExtraMemory(0),
      m_PreviousTotal(this->totalMemory()), m_Peak(m_PreviousTotal),
      m_LastAllocationFailureReport(0), m_MemoryStatus(model_t::E_MemoryStatusOk),
      m_HasPruningStarted(false), m_PruneThreshold(0), m_LastPruneTime(0),
      m_PruneWindow(std::numeric_limits<std::size_t>::max()),
      m_PruneWindowMaximum(std::numeric_limits<std::size_t>::max()),
      m_PruneWindowMinimum(std::numeric_limits<std::size_t>::max()), m_NoLimit(false) {
    this->updateMemoryLimitsAndPruneThreshold(DEFAULT_MEMORY_LIMIT_MB);
}

void CResourceMonitor::memoryUsageReporter(const TMemoryUsageReporterFunc& reporter) {
    m_MemoryUsageReporter = reporter;
}

void CResourceMonitor::registerComponent(CAnomalyDetector& detector) {
    LOG_TRACE(<< "Registering component: " << detector.model());
    m_Models.insert({detector.model().get(), std::size_t(0)});
}

void CResourceMonitor::unRegisterComponent(CAnomalyDetector& detector) {
    auto iter = m_Models.find(detector.model().get());
    if (iter == m_Models.end()) {
        LOG_ERROR(<< "Inconsistency - component has not been registered: "
                  << detector.model());
        return;
    }

    LOG_TRACE(<< "Unregistering component: " << detector.model());
    m_Models.erase(iter);
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
        // Background persist causes the memory size to double due to copying
        // the models. On top of that, after the persist is done we may not
        // be able to retrieve that memory back. Thus, we halve the requested
        // memory limit in order to allow for that.
        // See https://github.com/elastic/x-pack-elasticsearch/issues/1020.
        // Issue https://github.com/elastic/x-pack-elasticsearch/issues/857
        // discusses adding an option to perform only foreground persist.
        // If that gets implemented, we should only halve when background
        // persist is configured.
        m_ByteLimitHigh = static_cast<std::size_t>((limitMBs * 1024 * 1024) / 2);
    }
    m_ByteLimitLow = m_ByteLimitHigh - 1024;
    m_PruneThreshold = static_cast<std::size_t>(m_ByteLimitHigh / 5 * 3);
}

model_t::EMemoryStatus CResourceMonitor::getMemoryStatus() {
    return m_MemoryStatus;
}

void CResourceMonitor::refresh(CAnomalyDetector& detector) {
    if (m_NoLimit) {
        return;
    }
    this->forceRefresh(detector);
}

void CResourceMonitor::forceRefresh(CAnomalyDetector& detector) {
    this->memUsage(detector.model().get());
    core::CStatistics::stat(stat_t::E_MemoryUsage).set(this->totalMemory());
    LOG_TRACE(<< "Checking allocations: currently at " << this->totalMemory());
    this->updateAllowAllocations();
}

void CResourceMonitor::updateAllowAllocations() {
    std::size_t total{this->totalMemory()};
    if (m_AllowAllocations) {
        if (total > m_ByteLimitHigh) {
            LOG_INFO(<< "Over allocation limit. " << total
                     << " bytes used, the limit is " << m_ByteLimitHigh);
            m_AllowAllocations = false;
        }
    } else {
        if (total < m_ByteLimitLow) {
            LOG_INFO(<< "Below allocation limit, used " << total);
            m_AllowAllocations = true;
        }
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

    if (m_Models.empty()) {
        return false;
    }

    if (m_HasPruningStarted == false) {
        // The longest we'll consider keeping priors for is 1M buckets.
        CAnomalyDetectorModel* model = m_Models.begin()->first;
        if (model == nullptr) {
            return false;
        }
        m_PruneWindowMaximum = model->defaultPruneWindow();
        m_PruneWindow = m_PruneWindowMaximum;
        m_PruneWindowMinimum = model->minimumPruneWindow();
        m_HasPruningStarted = true;
        this->acceptPruningResult();
        LOG_DEBUG(<< "Pruning started. Window (buckets): " << m_PruneWindow);
    }

    if (aboveThreshold) {
        // Do a prune and see how much we got back
        // These are the expensive operations
        std::size_t usageAfter = 0;
        for (auto& model : m_Models) {
            model.first->prune(m_PruneWindow);
            model.second = model.first->memoryUsage();
            usageAfter += model.second;
        }
        m_CurrentAnomalyDetectorMemory = usageAfter;
        total = this->totalMemory();
        this->updateAllowAllocations();
    }

    LOG_TRACE(<< "Pruning models. Usage: " << total
              << ". Current window: " << m_PruneWindow << " buckets");

    if (total < m_PruneThreshold) {
        // Expand the window
        m_PruneWindow = std::min(
            m_PruneWindow + std::size_t((endTime - m_LastPruneTime) /
                                        m_Models.begin()->first->bucketLength()),
            m_PruneWindowMaximum);
        LOG_TRACE(<< "Expanding window, to " << m_PruneWindow);
    } else {
        // Shrink the window
        m_PruneWindow = std::max(static_cast<std::size_t>(m_PruneWindow * 99 / 100),
                                 m_PruneWindowMinimum);
        LOG_TRACE(<< "Shrinking window, to " << m_PruneWindow);
    }

    m_LastPruneTime = endTime;
    return aboveThreshold;
}

bool CResourceMonitor::areAllocationsAllowed() const {
    return m_AllowAllocations;
}

bool CResourceMonitor::areAllocationsAllowed(std::size_t size) const {
    if (m_AllowAllocations) {
        return this->totalMemory() + size < m_ByteLimitHigh;
    }
    return false;
}

std::size_t CResourceMonitor::allocationLimit() const {
    return m_ByteLimitHigh - std::min(m_ByteLimitHigh, this->totalMemory());
}

void CResourceMonitor::memUsage(CAnomalyDetectorModel* model) {
    auto iter = m_Models.find(model);
    if (iter == m_Models.end()) {
        LOG_ERROR(<< "Inconsistency - component has not been registered: " << model);
        return;
    }
    std::size_t modelPreviousUsage = iter->second;
    std::size_t modelCurrentUsage = core::CMemory::dynamicSize(iter->first);
    iter->second = modelCurrentUsage;
    m_CurrentAnomalyDetectorMemory += (modelCurrentUsage - modelPreviousUsage);
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
        core_t::TTime lastestAllocationError = (--m_AllocationFailures.end())->first;
        if (lastestAllocationError > m_LastAllocationFailureReport) {
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

CResourceMonitor::SResults CResourceMonitor::createMemoryUsageReport(core_t::TTime bucketStartTime) {
    SResults res;
    res.s_ByFields = 0;
    res.s_OverFields = 0;
    res.s_PartitionFields = 0;
    res.s_Usage = this->totalMemory();
    res.s_AllocationFailures = 0;
    res.s_MemoryStatus = m_MemoryStatus;
    res.s_BucketStartTime = bucketStartTime;
    for (const auto& model : m_Models) {
        ++res.s_PartitionFields;
        res.s_OverFields += model.first->dataGatherer().numberOverFieldValues();
        res.s_ByFields += model.first->dataGatherer().numberByFieldValues();
    }
    res.s_AllocationFailures += m_AllocationFailures.size();
    return res;
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

std::size_t CResourceMonitor::totalMemory() const {
    return m_CurrentAnomalyDetectorMemory + m_ExtraMemory +
           CStringStore::names().memoryUsage() +
           CStringStore::influencers().memoryUsage();
}

} // model
} // ml
