/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_model_CResourceMonitor_h
#define INCLUDED_ml_model_CResourceMonitor_h

#include <core/CoreTypes.h>

#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/unordered_map.hpp>

#include <functional>

namespace CResourceMonitorTest {
struct testMonitor;
struct testPruning;
}
namespace CResourceLimitTest {
class CTestFixture;
}
namespace CAnomalyJobLimitTest {
struct testFoo;
}

namespace ml {
namespace model {

class CAnomalyDetector;
class CAnomalyDetectorModel;
class CResourcePruner;

//! \brief Assess memory used by models and decide on further memory allocations.
//!
//! DESCRIPTION:\n
//! Assess memory used by models and decide on further memory allocations.
class MODEL_EXPORT CResourceMonitor {
public:
    struct MODEL_EXPORT SResults {
        std::size_t s_Usage;
        std::size_t s_AdjustedUsage;
        std::size_t s_ByFields;
        std::size_t s_PartitionFields;
        std::size_t s_OverFields;
        std::size_t s_AllocationFailures;
        model_t::EMemoryStatus s_MemoryStatus;
        core_t::TTime s_BucketStartTime;
        std::size_t s_BytesExceeded;
        std::size_t s_BytesMemoryLimit;
    };

public:
    using TDetectorPtrSizePr = std::pair<CAnomalyDetector*, std::size_t>;
    using TDetectorPtrSizeUMap = boost::unordered_map<CAnomalyDetector*, std::size_t>;
    using TMemoryUsageReporterFunc = std::function<void(const CResourceMonitor::SResults&)>;
    using TTimeSizeMap = std::map<core_t::TTime, std::size_t>;

    //! The minimum time between prunes
    static const core_t::TTime MINIMUM_PRUNE_FREQUENCY;
    //! Default memory limit for resource monitor
    static const std::size_t DEFAULT_MEMORY_LIMIT_MB;
    //! The initial byte limit margin to use if none is supplied
    static const double DEFAULT_BYTE_LIMIT_MARGIN;
    //! The maximum value of elapsed time used to scale the byte limit margin
    static const core_t::TTime MAXIMUM_BYTE_LIMIT_MARGIN_PERIOD;

public:
    //! Default constructor
    explicit CResourceMonitor(bool persistenceInForeground = false,
                              double byteLimitMargin = DEFAULT_BYTE_LIMIT_MARGIN);

    //! Query the resource monitor to find out if the models are
    //! taking up too much memory and further allocations should be banned
    bool areAllocationsAllowed() const;

    //! Return the amount of remaining space for allocations
    std::size_t allocationLimit() const;

    //! Tell this resource monitor about a CAnomalyDetector class -
    //! these classes contain all the model memory and are used
    //! to query the current overall usage
    void registerComponent(CAnomalyDetector& detector);

    //! Tell this resource monitor that a CAnomalyDetector class is
    //! going to be deleted.
    void unRegisterComponent(CAnomalyDetector& detector);

    //! Set a callback used when the memory usage grows
    void memoryUsageReporter(const TMemoryUsageReporterFunc& reporter);

    //! Recalculate the memory usage if there is a memory limit
    void refresh(CAnomalyDetector& detector);

    //! Recalculate the memory usage regardless of whether there is a memory limit
    void forceRefresh(CAnomalyDetector& detector);

    //! Set the internal memory limit, as specified in a limits config file
    void memoryLimit(std::size_t limitMBs);

    //! Get the memory status
    model_t::EMemoryStatus getMemoryStatus();

    //! Send a memory usage report if it's changed by more than a certain percentage
    void sendMemoryUsageReportIfSignificantlyChanged(core_t::TTime bucketStartTime);

    //! Send a memory usage report
    void sendMemoryUsageReport(core_t::TTime bucketStartTime);

    //! Create a memory usage report
    SResults createMemoryUsageReport(core_t::TTime bucketStartTime);

    //! We are being told that a class has failed to allocate memory
    //! based on the resource limits, and we will report this to the
    //! user when we can
    void acceptAllocationFailureResult(core_t::TTime time);

    //! We are being told that aggressive pruning has taken place
    //! to avoid hitting the resource limit, and we should report this
    //! to the user when we can
    void acceptPruningResult();

    //! Accessor for no limit flag
    bool haveNoLimit() const;

    //! Prune models where necessary
    bool pruneIfRequired(core_t::TTime endTime);

    //! Accounts for any extra memory to the one
    //! reported by the components.
    //! Used in conjunction  with clearExtraMemory()
    //! in order to ensure enough memory remains
    //! for model's parts that have not been fully allocated yet.
    void addExtraMemory(std::size_t reserved);

    //! Clears all extra memory
    void clearExtraMemory();

    //! Decrease the margin on the memory limit.
    //!
    //! We start off applying a 'safety' margin to the memory limit because
    //! it is difficult to accurately estimate the long term memory
    //! usage at this point. This safety margin is gradually decreased over time
    //! by calling this once per bucket processed until the initially requested memory limit is reached.
    void decreaseMargin(core_t::TTime elapsedTime);

private:
    //! Updates the memory limit fields and the prune threshold
    //! to the given value.
    void updateMemoryLimitsAndPruneThreshold(std::size_t limitMBs);

    //! Update the given model and recalculate the total usage
    void memUsage(CAnomalyDetector* detector);

    //! Determine if we need to send a usage report, based on
    //! increased usage, or increased errors
    bool needToSendReport();

    //! After a change in memory usage, check whether allocations
    //! shoule be allowed or not
    void updateAllowAllocations();

    //! Get the high memory limit with margin applied.
    std::size_t highLimit() const;

    //! Get the low memory limit with margin applied.
    std::size_t lowLimit() const;

    //! Returns the sum of used memory plus any extra memory
    std::size_t totalMemory() const;

    //! Adjusts the amount of memory reported to take into
    //! account the current value of the byte limit margin and the effects
    //! of background persistence.
    std::size_t adjustedUsage(std::size_t usage) const;

    //! Returns the amount by which reported memory usage is scaled depending on the type of persistence in use
    std::size_t persistenceMemoryIncreaseFactor() const;

private:
    //! The registered collection of components
    TDetectorPtrSizeUMap m_Detectors;

    //! Is there enough free memory to allow creating new components
    bool m_AllowAllocations;

    //! The relative margin to apply to the byte limits.
    double m_ByteLimitMargin;

    //! The upper limit for memory usage, checked on increasing values
    std::size_t m_ByteLimitHigh;

    //! The lower limit for memory usage, checked on decreasing values
    std::size_t m_ByteLimitLow;

    //! Memory usage by anomaly detectors on the most recent calculation
    std::size_t m_CurrentAnomalyDetectorMemory;

    //! Extra memory to enable accounting of soon to be allocated memory
    std::size_t m_ExtraMemory;

    //! The total memory usage on the previous usage report
    std::size_t m_PreviousTotal;

    //! The highest known value for total memory usage
    std::size_t m_Peak;

    //! Callback function to fire when memory usage increases by 1%
    TMemoryUsageReporterFunc m_MemoryUsageReporter;

    //! Keep track of classes telling us about allocation failures
    TTimeSizeMap m_AllocationFailures;

    //! The time at which the last allocation failure was reported
    core_t::TTime m_LastAllocationFailureReport;

    //! Keep track of the model memory status
    model_t::EMemoryStatus m_MemoryStatus;

    //! Keep track of whether pruning has started, for efficiency in most cases
    bool m_HasPruningStarted;

    //! The threshold at which pruning should kick in and head
    //! towards for the sweet spot
    std::size_t m_PruneThreshold;

    //! The last time we did a full prune of all the models
    core_t::TTime m_LastPruneTime;

    //! Number of buckets to go back when pruning
    std::size_t m_PruneWindow;

    //! The largest that the prune window can grow to - determined from the models
    std::size_t m_PruneWindowMaximum;

    //! The smallest that the prune window can shrink to - 4 weeks
    std::size_t m_PruneWindowMinimum;

    //! Don't do any sort of memory checking if this is set
    bool m_NoLimit;

    //! The number of bytes over the high limit for memory usage at the last allocation failure
    std::size_t m_CurrentBytesExceeded;

    //! Is persistence occurring in the foreground?
    bool m_PersistenceInForeground;

    //! Test friends
    friend struct CResourceMonitorTest::testMonitor;
    friend struct CResourceMonitorTest::testPruning;
    friend class CResourceLimitTest::CTestFixture;
    friend struct CAnomalyJobLimitTest::testFoo;
};

} // model

} // ml

#endif // INCLUDED_ml_model_CResourceMonitor_h
