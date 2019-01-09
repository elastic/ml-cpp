/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStatistics_h
#define INCLUDED_ml_core_CStatistics_h

#include <core/CNonCopyable.h>
#include <core/CStat.h>
#include <core/ImportExport.h>

#include <boost/array.hpp>

#include <iosfwd>
#include <memory>
#include <vector>

#include <stdint.h>

class CStatisticsTest;

namespace ml {
namespace stat_t {

//! Changing the order of these enumeration values will corrupt persisted model
//! state, so don't.  Any new statistics should be added in the penultimate
//! position in the enum, immediately before E_LastEnumStat.
enum EStatTypes {
    //! The number of new people not created in the data gatherer
    //! because there wasn't enough free resource
    E_NumberNewPeopleNotAllowed,

    //! The number of new people created in the data gatherer
    E_NumberNewPeople,

    //! The number of new people put into a recycled slot
    E_NumberNewPeopleRecycled,

    //! The number of records successfully handled by the API
    E_NumberApiRecordsHandled,

    //! The latest value of the model memory check
    E_MemoryUsage,

    //! The number of times memory checks have been carried out
    E_NumberMemoryUsageChecks,

    //! The number of records encountered with no time field
    E_NumberRecordsNoTimeField,

    //! The number of errors converting time fields to numeric values
    E_NumberTimeFieldConversionErrors,

    //! The number of records with out-of-order time fields
    E_NumberTimeOrderErrors,

    //! The number of new attributes created in the data gatherer
    E_NumberNewAttributes,

    //! The number of new attributes not created in the data gatherer
    //! because there wasn't enough free resource
    E_NumberNewAttributesNotAllowed,

    //! The number of new people put into a recycled slot
    E_NumberNewAttributesRecycled,

    //! The number of 'over' fields created
    E_NumberOverFields,

    //! The number of 'by' fields created
    E_NumberByFields,

    //! The number of times "ExcludeFrequent" has been invoked by the model
    E_NumberExcludedFrequentInvocations,

    //! The number of samples received outside the latency window
    E_NumberSamplesOutsideLatencyWindow,

    //! The number of model creation failures from being over memory limit
    E_NumberMemoryLimitModelCreationFailures,

    //! The number of old people or attributes pruned from the models
    E_NumberPrunedItems,

    //! The number of times partial memory estimates have been carried out
    E_NumberMemoryUsageEstimates,

    // Add any new values here

    //! This MUST be last
    E_LastEnumStat
};
}

namespace core {

class CStatisticsServer;
class CStateRestoreTraverser;
class CStatePersistInserter;

//! \brief
//! A collection of runtime global statistics
//!
//! DESCRIPTION:\n
//! A static collection of runtime global statistics and counters, atomically incremented
//!
//! To add a new stat, put it into the stat_t::EStatTypes enum, and add its details and
//! description to the ostream<< operator function in the implementation file
//!
//! IMPLEMENTATION DECISIONS:\n
//! A singleton class: there should only be one collection of global stats
//!
class CORE_EXPORT CStatistics : private CNonCopyable {
private:
    class CStatsCache;

private:
    using TStatArray = boost::array<CStat, stat_t::E_LastEnumStat>;
    using TStatsCacheUPtr = std::unique_ptr<CStatsCache>;

public:
    //! Singleton pattern
    static CStatistics& instance();

    //! Provide access to the relevant stat from the collection
    static CStat& stat(int index);

    //! copy the collection of live statistics to a cache
    static void cacheStats();

    //! Transfer ownership of the cached statistics to the caller
    static TStatsCacheUPtr transferCachedStats();

    //! \name Persistence
    //@{
    //! Restore the static members of this class from persisted state
    static bool staticsAcceptRestoreTraverser(CStateRestoreTraverser& traverser);

    //! Persist the static members of this class
    static void staticsAcceptPersistInserter(CStatePersistInserter& inserter);
    //@}

private:
    //! Constructor of a Singleton is private
    CStatistics();

    //! The unique instance.
    static CStatistics ms_Instance;

    //! Collection of statistics
    TStatArray m_Stats;

    //! cache of stats for use in persistence
    TStatsCacheUPtr m_Cache;

    //! Enabling printing out the current statistics.
    friend CORE_EXPORT std::ostream& operator<<(std::ostream& o, const CStatistics& stats);

    //! Helper class for storing a snapshot of statistics prior to persisting them
    friend class CStatsCache;

    //! Befriend the test suite
    friend class ::CStatisticsTest;

private:
    //! Maintains a snapshot of statistics to be used for persistence
    //!
    //! IMPLEMENTATION DECISIONS:\n
    //! Only used by CStatistics so nested within that class definition
    //! Instances of this class exist only transitorily, ownership is transferred by move semantics only.
    class CORE_EXPORT CStatsCache : private CNonCopyable {
    public:
        CStatsCache();

        //! populate the cache with current live values
        void populate(const CStatistics::TStatArray& statArray);

        //! Provide access to the relevant stat from the collection
        uint64_t stat(int index) const;

    private:
        using TUInt64Vec = std::vector<uint64_t>;

    private:
        TUInt64Vec m_Stats;
    };
};

} // core
} // ml

#endif // INCLUDED_ml_core_CStatistics_h
