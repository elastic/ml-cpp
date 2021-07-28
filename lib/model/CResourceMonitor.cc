/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <model/CResourceMonitor.h>

#include <core/CProgramCounters.h>
#include <core/Constants.h>

#include <maths/CMathsFuncs.h>
#include <maths/CTools.h>

#include <model/CMonitoredResource.h>
#include <model/CStringStore.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
const std::size_t BUCKETS_FOR_ESTABLISHED_MEMORY_SIZE{20};
const double ESTABLISHED_MEMORY_CV_THRESHOLD{0.1};
}

namespace ml {
namespace model {

// Only prune once per hour
const core_t::TTime CResourceMonitor::MINIMUM_PRUNE_FREQUENCY{60 * 60};
const std::size_t CResourceMonitor::DEFAULT_MEMORY_LIMIT_MB{4096};
const double CResourceMonitor::DEFAULT_BYTE_LIMIT_MARGIN{0.7};
const core_t::TTime CResourceMonitor::MAXIMUM_BYTE_LIMIT_MARGIN_PERIOD{2 * core::constants::HOUR};

CResourceMonitor::CResourceMonitor(bool persistenceInForeground, double byteLimitMargin)
    : m_ByteLimitMargin{byteLimitMargin}, m_PreviousTotal{this->totalMemory()},
      m_PruneWindow{std::numeric_limits<std::size_t>::max()},
      m_PruneWindowMaximum{std::numeric_limits<std::size_t>::max()},
      m_PruneWindowMinimum{std::numeric_limits<std::size_t>::max()},
      m_PersistenceInForeground{persistenceInForeground} {
    this->updateMemoryLimitsAndPruneThreshold(DEFAULT_MEMORY_LIMIT_MB);
}

void CResourceMonitor::memoryUsageReporter(const TMemoryUsageReporterFunc& reporter) {
    m_MemoryUsageReporter = reporter;
}

void CResourceMonitor::registerComponent(CMonitoredResource& resource) {
    LOG_TRACE(<< "Registering component: " << &resource);
    m_Resources.emplace(&resource, 0);
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
    std::size_t total{this->totalMemory()};
    core::CProgramCounters::counter(counter_t::E_TSADMemoryUsage) = total;
    core::CProgramCounters::counter(counter_t::E_TSADPeakMemoryUsage) = std::max(
        static_cast<std::size_t>(core::CProgramCounters::counter(counter_t::E_TSADPeakMemoryUsage)),
        total);
}

void CResourceMonitor::memoryLimit(std::size_t limitMBs) {
    this->updateMemoryLimitsAndPruneThreshold(limitMBs);

    if (m_NoLimit) {
        LOG_INFO(<< "Setting no model memory limit");
    } else {
        LOG_INFO(<< "Setting model memory limit to " << limitMBs << " MB");
    }
}

std::size_t CResourceMonitor::getBytesMemoryLimit() const {
    return m_ByteLimitHigh * this->persistenceMemoryIncreaseFactor();
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
        m_ByteLimitHigh = (limitMBs * core::constants::BYTES_IN_MEGABYTES) /
                          this->persistenceMemoryIncreaseFactor();
    }
    m_ByteLimitLow = (m_ByteLimitHigh * 49) / 50;
    m_PruneThreshold = (m_ByteLimitHigh * 3) / 5;
}

model_t::EMemoryStatus CResourceMonitor::memoryStatus() const {
    return m_MemoryStatus;
}

std::size_t CResourceMonitor::categorizerAllocationFailures() const {
    return m_CategorizerAllocationFailures;
}

void CResourceMonitor::categorizerAllocationFailures(std::size_t categorizerAllocationFailures) {
    m_CategorizerAllocationFailures = categorizerAllocationFailures;
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

void CResourceMonitor::forceRefreshAll() {
    for (auto& resource : m_Resources) {
        this->memUsage(resource.first);
    }

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
                this->startPruning();
                break;
            }
        }
        if (m_HasPruningStarted == false) {
            return false;
        }
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
                m_PruneWindow = m_PruneWindow +
                                std::size_t((endTime - m_LastPruneTime) /
                                            resource.first->bucketLength());
                if (m_PruneWindow > m_PruneWindowMaximum) {
                    // If we increase the prune window to a size that's bigger
                    // than what we started with then we should stop pruning to
                    // be consistent with what would happen if the job was
                    // closed and reopened
                    m_PruneWindow = m_PruneWindowMaximum;
                    this->endPruning();
                } else {
                    LOG_TRACE(<< "Expanding window to " << m_PruneWindow);
                }
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

void CResourceMonitor::sendMemoryUsageReportIfSignificantlyChanged(core_t::TTime bucketStartTime,
                                                                   core_t::TTime bucketLength) {
    std::uint64_t assignmentMemoryBasis{
        core::CProgramCounters::counter(counter_t::E_TSADAssignmentMemoryBasis)};
    if (this->needToSendReport(static_cast<model_t::EAssignmentMemoryBasis>(assignmentMemoryBasis),
                               bucketStartTime, bucketLength)) {
        this->sendMemoryUsageReport(bucketStartTime, bucketLength);
    }
}

void CResourceMonitor::updateMoments(std::size_t totalMemory,
                                     core_t::TTime bucketStartTime,
                                     core_t::TTime bucketLength) {
    if (m_FirstMomentsUpdateTime <= 0) {
        m_FirstMomentsUpdateTime = bucketStartTime;
    } else {
        if (bucketLength > 0 && bucketStartTime >= m_LastMomentsUpdateTime + bucketLength) {
            // The idea is to age this so that observations from more than 20
            // buckets ago have little effect.  This means the end results will
            // be close to what the old Java calculation did - it literally
            // searched the last 20 buckets.  Aging at e^-0.1 seems reasonable
            // to reduce the variance at the required rate.
            double factor{std::exp(
                -static_cast<double>((bucketStartTime - m_LastMomentsUpdateTime) / bucketLength) /
                static_cast<double>(BUCKETS_FOR_ESTABLISHED_MEMORY_SIZE / 2))};
            m_ModelBytesMoments.age(factor);
        }
    }
    m_ModelBytesMoments.add(static_cast<double>(totalMemory));
    m_LastMomentsUpdateTime = bucketStartTime;
}

bool CResourceMonitor::needToSendReport(model_t::EAssignmentMemoryBasis currentAssignmentMemoryBasis,
                                        core_t::TTime bucketStartTime,
                                        core_t::TTime bucketLength) {
    std::size_t total{this->totalMemory()};

    // Update the moments that are used to determine whether memory is stable
    this->updateMoments(total, bucketStartTime, bucketLength);

    // Has the usage changed by more than 1% ?
    if ((std::max(total, m_PreviousTotal) - std::min(total, m_PreviousTotal)) >
        m_PreviousTotal / 100) {
        return true;
    }

    // Is the assignment memory basis changing?
    if ((currentAssignmentMemoryBasis == model_t::E_AssignmentBasisUnknown ||
         currentAssignmentMemoryBasis == model_t::E_AssignmentBasisModelMemoryLimit) &&
        this->isMemoryStable(bucketLength)) {
        return true;
    }

    // Have we had new allocation failures
    if (!m_AllocationFailures.empty()) {
        core_t::TTime latestAllocationError{(--m_AllocationFailures.end())->first};
        if (latestAllocationError > m_LastAllocationFailureReport) {
            return true;
        }
    }
    return false;
}

bool CResourceMonitor::isMemoryStable(core_t::TTime bucketLength) const {

    // Sanity check
    if (maths::CBasicStatistics::count(m_ModelBytesMoments) == 0.0) {
        LOG_ERROR(<< "Programmatic error: checking memory stability before adding any measurements");
        return false;
    }

    // Must have been monitoring for 20 buckets
    std::size_t bucketCount{
        static_cast<std::size_t>((m_LastMomentsUpdateTime - m_FirstMomentsUpdateTime) / bucketLength) +
        1};
    if (bucketCount < BUCKETS_FOR_ESTABLISHED_MEMORY_SIZE) {
        return false;
    }

    // Coefficient of variation must be less than 0.1
    double mean{maths::CBasicStatistics::mean(m_ModelBytesMoments)};
    double variance{maths::CBasicStatistics::variance(m_ModelBytesMoments)};
    LOG_TRACE(<< "Model memory stability at " << m_LastMomentsUpdateTime
              << ": bucket count = " << bucketCount
              << ", sample count = " << maths::CBasicStatistics::count(m_ModelBytesMoments)
              << ", mean = " << mean << ", variance = " << variance
              << ", coefficient of variation = " << (std::sqrt(variance) / mean));
    // Instead of literally testing the coefficient of variation it's more
    // robust against zeroes and NaNs to rearrange it as follows
    return maths::CMathsFuncs::isNan(variance) == false &&
           variance <= maths::CTools::pow2(ESTABLISHED_MEMORY_CV_THRESHOLD * mean);
}

void CResourceMonitor::sendMemoryUsageReport(core_t::TTime bucketStartTime,
                                             core_t::TTime bucketLength) {
    std::size_t total{this->totalMemory()};
    if (this->isMemoryStable(bucketLength)) {
        core::CProgramCounters::counter(counter_t::E_TSADAssignmentMemoryBasis) =
            static_cast<std::uint64_t>(model_t::E_AssignmentBasisCurrentModelBytes);
    }
    core::CProgramCounters::counter(counter_t::E_TSADPeakMemoryUsage) = std::max(
        static_cast<std::size_t>(core::CProgramCounters::counter(counter_t::E_TSADPeakMemoryUsage)),
        total);
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
    res.s_PeakUsage = static_cast<std::size_t>(
        core::CProgramCounters::counter(counter_t::E_TSADPeakMemoryUsage));
    res.s_AdjustedPeakUsage = this->adjustedUsage(res.s_PeakUsage);
    res.s_BytesMemoryLimit = this->persistenceMemoryIncreaseFactor() * m_ByteLimitHigh;
    res.s_BytesExceeded = m_CurrentBytesExceeded;
    res.s_MemoryStatus = m_MemoryStatus;
    std::uint64_t assignmentMemoryBasis{
        core::CProgramCounters::counter(counter_t::E_TSADAssignmentMemoryBasis)};
    res.s_AssignmentMemoryBasis =
        static_cast<model_t::EAssignmentMemoryBasis>(assignmentMemoryBasis);
    res.s_BucketStartTime = bucketStartTime;
    for (const auto& resource : m_Resources) {
        resource.first->updateModelSizeStats(res);
    }
    res.s_AllocationFailures += m_AllocationFailures.size();
    res.s_OverallCategorizerStats.s_MemoryCategorizationFailures += m_CategorizerAllocationFailures;
    return res;
}

std::size_t CResourceMonitor::adjustedUsage(std::size_t usage) const {
    // We scale the reported memory usage by the inverse of the byte limit margin.
    // This gives the user a fairer indication of how close the job is to hitting
    // the model memory limit in a concise manner (as the limit is scaled down by
    // the margin during the beginning period of the job's existence).
    std::size_t adjustedUsage{
        static_cast<std::size_t>(static_cast<double>(usage) / m_ByteLimitMargin)};

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

void CResourceMonitor::startPruning() {
    LOG_DEBUG(<< "Pruning started. Window (buckets): " << m_PruneWindow);
    m_HasPruningStarted = true;
    if (m_MemoryStatus == model_t::E_MemoryStatusOk) {
        m_MemoryStatus = model_t::E_MemoryStatusSoftLimit;
    }
}

void CResourceMonitor::endPruning() {
    LOG_DEBUG(<< "Pruning no longer necessary.");
    m_HasPruningStarted = false;
    if (m_MemoryStatus == model_t::E_MemoryStatusSoftLimit) {
        m_MemoryStatus = model_t::E_MemoryStatusOk;
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
