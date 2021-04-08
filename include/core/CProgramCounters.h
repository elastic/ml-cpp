/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CProgramCounters_h
#define INCLUDED_ml_core_CProgramCounters_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <iosfwd>
#include <set>
#include <string>
#include <vector>

namespace CProgramCountersTest {
class CProgramCountersTestRunner;
struct testCounters;
struct testUnknownCounter;
struct testMissingCounter;
struct testCacheCounters;
struct testPersist;
}

namespace ml {
namespace test {
class CProgramCounterClearingFixture;
}
namespace counter_t {

//! The enum values must be explicitly assigned & names should have a meaningful prefix to effectively namespace counters
//! New counters may be added anywhere before E_LastEnumCounter and must be grouped by namespace prefix.
//! The E_LastEnumCounter value must be bumped for every new enum added.
//! Don't forget to also add a description of the new enum value to m_CounterDefinitions.
enum ECounterTypes {
    // Time Series Anomaly Detection

    //! The number of new people not created in the data gatherer
    //! because there wasn't enough free resource
    E_TSADNumberNewPeopleNotAllowed = 0,

    //! The number of new people created in the data gatherer
    E_TSADNumberNewPeople = 1,

    //! The number of new people put into a recycled slot
    E_TSADNumberNewPeopleRecycled = 2,

    //! The number of records successfully handled by the API
    E_TSADNumberApiRecordsHandled = 3,

    //! The latest value of the model memory check
    E_TSADMemoryUsage = 4,

    //! The maximum value of the model memory check
    E_TSADPeakMemoryUsage = 28,

    //! The number of times memory checks have been carried out
    E_TSADNumberMemoryUsageChecks = 5,

    //! The number of records encountered with no time field
    E_TSADNumberRecordsNoTimeField = 6,

    //! The number of errors converting time fields to numeric values
    E_TSADNumberTimeFieldConversionErrors = 7,

    //! The number of records with out-of-order time fields
    E_TSADNumberTimeOrderErrors = 8,

    //! The number of new attributes created in the data gatherer
    E_TSADNumberNewAttributes = 9,

    //! The number of new attributes not created in the data gatherer
    //! because there wasn't enough free resource
    E_TSADNumberNewAttributesNotAllowed = 10,

    //! The number of new people put into a recycled slot
    E_TSADNumberNewAttributesRecycled = 11,

    //! The number of 'over' fields created
    E_TSADNumberOverFields = 12,

    //! The number of 'by' fields created
    E_TSADNumberByFields = 13,

    //! The number of times "ExcludeFrequent" has been invoked by the model
    E_TSADNumberExcludedFrequentInvocations = 14,

    //! The number of samples received outside the latency window
    E_TSADNumberSamplesOutsideLatencyWindow = 15,

    //! The number of model creation failures from being over memory limit
    E_TSADNumberMemoryLimitModelCreationFailures = 16,

    //! The number of old people or attributes pruned from the models
    E_TSADNumberPrunedItems = 17,

    //! The number of times partial memory estimates have been carried out
    E_TSADNumberMemoryUsageEstimates = 18,

    //! Which option is being used to get model memory for node assignment?
    E_TSADAssignmentMemoryBasis = 29,

    // Data Frame Outlier Detection

    //! The estimated peak memory usage for outlier detection in bytes
    E_DFOEstimatedPeakMemoryUsage = 19,

    //! The peak memory usage of outlier detection in bytes
    E_DFOPeakMemoryUsage = 20,

    //! The time in ms to create the ensemble for outlier detection
    E_DFOTimeToCreateEnsemble = 21,

    //! The time in ms to compute the outlier scores
    E_DFOTimeToComputeScores = 22,

    //! The number of partitions used
    E_DFONumberPartitions = 23,

    // Data Frame Train Predictive Model

    //! The estimated peak memory usage for training a predictive model
    E_DFTPMEstimatedPeakMemoryUsage = 24,

    //! The peak memory usage for training a predictive model in bytes
    E_DFTPMPeakMemoryUsage = 25,

    //! The time in ms to train the model
    E_DFTPMTimeToTrain = 26,

    //! The trained forest total number of trees
    E_DFTPMTrainedForestNumberTrees = 27,

    // Add any new values here

    //! This MUST be last, increment the value for every new enum added
    E_LastEnumCounter = 30
};

static constexpr std::size_t NUM_COUNTERS = static_cast<std::size_t>(E_LastEnumCounter);

using TCounterTypeSet = std::set<ECounterTypes>;
}

namespace core {

class CStateRestoreTraverser;
class CStatePersistInserter;

struct SCounterDefinition {
    counter_t::ECounterTypes s_Type;
    std::string s_Name;
    std::string s_Description;
};

//! \brief
//! A collection of runtime global counters
//!
//! DESCRIPTION:\n
//! A static collection of runtime global counters, atomically incremented
//!
//! To add a new counter, add it to the counter_t::ECounterTypes enum in
//! the penultimate position and add its details and description to the
//! m_CounterDefinitions array.
//! In addition to this, individual programs should add the new enum type to
//! the collection passed to registerProgramCounterTypes.
//!
//! Enums must not be 'reused'. If no longer needed add a comment to that effect.
//!
//! Enum names should have a meaningful prefix to effectively namespace counters
//! used in separate applications, e.g. \c TSAD is used for the Time Series Anomaly Detection
//! counters.
//!
//! IMPLEMENTATION DECISIONS:\n
//! A singleton class: there should only be one collection of global counters
//!
class CORE_EXPORT CProgramCounters {
public:
    //! \brief
    //! The cache of program counters is cleared upon destruction of an instance of this class.
    class CORE_EXPORT CCacheManager {
    public:
        ~CCacheManager();
    };

private:
    //! \brief
    //! An atomic counter object
    //!
    //! DESCRIPTION:\n
    //! A wrapper for an atomic uint64_t
    //!
    //! This provides a thread-safe way to aggregate runtime counts
    //!
    //! IMPLEMENTATION DECISIONS:\n
    //! Implicitly not copyable - because the atomic member is not copyable.
    //! Counter values are assumed to only ever increase, therefore incrementing
    //! and assigning are publicly accessible, decrementing is not.
    //! Post increment is not supported - to enforce best practice
    //! Implicit conversion to/from std::uint64_t is also allowed.
    //!
    class CORE_EXPORT CCounter {
    public:
        CCounter() : m_Counter(0) {}

        explicit CCounter(std::uint64_t counter) : m_Counter(counter) {}

        CCounter& operator=(std::uint64_t counter) {
            m_Counter = counter;
            return *this;
        }

        CCounter& operator++() {
            ++m_Counter;
            return *this;
        }

        CCounter& operator+=(std::uint64_t counter) {
            m_Counter += counter;
            return *this;
        }

        void max(std::uint64_t counter) {
            std::uint64_t previousCounter{m_Counter.load(std::memory_order_relaxed)};
            while (previousCounter < counter &&
                   m_Counter.compare_exchange_weak(previousCounter, counter) == false) {
            }
        }

        operator std::uint64_t() const { return m_Counter; }

    private:
        //! Decrement operator - for test cases only.
        CCounter& operator--() {
            --m_Counter;
            return *this;
        }

    private:
        std::atomic_uint_fast64_t m_Counter;

        //! Befriend the test suite
        friend class CProgramCountersTest::CProgramCountersTestRunner;
        friend struct CProgramCountersTest::testCounters;
    };

private:
    using TCounter = CCounter;
    using TCounterArray = std::array<TCounter, counter_t::NUM_COUNTERS>;
    using TCounterDefinitionArray = std::array<SCounterDefinition, counter_t::NUM_COUNTERS>;
    using TUInt64Vec = std::vector<std::uint64_t>;

public:
    //! Singleton pattern
    //! In practice this is not used in production code, to reduce verbosity.
    static CProgramCounters& instance();

    //! Provide access to the relevant counter from the collection
    static TCounter& counter(counter_t::ECounterTypes counterType);
    static TCounter& counter(std::size_t index);

    //! Copy the collection of live counters to a cache
    static void cacheCounters();

    //! Clear the collection of cached counters
    static void clearCachedCounters();

    //! \name Persistence
    //@{
    //! Restore the static members of this class from persisted state
    static bool staticsAcceptRestoreTraverser(CStateRestoreTraverser& traverser);

    //! Persist the static members of this class
    static void staticsAcceptPersistInserter(CStatePersistInserter& inserter);
    //@}

    //! Register interest in a subset of counters. For use in printing values.
    //! Outside of test code this should be called no more than once.
    //! If never called then the total set of counters will be used.
    static void registerProgramCounterTypes(const counter_t::TCounterTypeSet& counters);

private:
    //! Constructor of a Singleton is private
    CProgramCounters() = default;
    CProgramCounters(CProgramCounters&) = delete;
    CProgramCounters& operator=(CProgramCounters&) = delete;

    //! The unique instance.
    static CProgramCounters ms_Instance;

    //! Collection of counters
    TCounterArray m_Counters;

    //! A dummy counter used if ever an attempt is made to restore an unknown counter type
    TCounter m_DummyCounter;

    //! Cache of counters for use in persistence
    TUInt64Vec m_Cache;

    //! Set of counter types of interest to a particular program
    counter_t::TCounterTypeSet m_ProgramCounterTypes;

    //! Descriptions of the counters. For use when printing the values.
    TCounterDefinitionArray m_CounterDefinitions{
        {{counter_t::E_TSADNumberNewPeopleNotAllowed,
          "E_TSADNumberNewPeopleNotAllowed", "Number of new people not allowed"},
         {counter_t::E_TSADNumberNewPeople, "E_TSADNumberNewPeople", "Number of new people created"},
         {counter_t::E_TSADNumberNewPeopleRecycled, "E_TSADNumberNewPeopleRecycled",
          "Number of new people recycled into existing space"},
         {counter_t::E_TSADNumberApiRecordsHandled, "E_TSADNumberApiRecordsHandled",
          "Number of records successfully ingested into the engine API"},
         {counter_t::E_TSADMemoryUsage, "E_TSADMemoryUsage",
          "The estimated memory currently used by the engine and models"},
         {counter_t::E_TSADPeakMemoryUsage, "E_TSADPeakMemoryUsage",
          "The maximum memory used by the engine and models"},
         {counter_t::E_TSADNumberMemoryUsageChecks, "E_TSADNumberMemoryUsageChecks",
          "Number of times a model memory usage check has been carried out"},
         {counter_t::E_TSADNumberMemoryUsageEstimates, "E_TSADNumberMemoryUsageEstimates",
          "Number of times a partial memory usage estimate has been carried out"},
         {counter_t::E_TSADNumberRecordsNoTimeField, "E_TSADNumberRecordsNoTimeField",
          "Number of records that didn't contain a Time field"},
         {counter_t::E_TSADNumberTimeFieldConversionErrors, "E_TSADNumberTimeFieldConversionErrors",
          "Number of records where the format of the time field could not be converted"},
         {counter_t::E_TSADNumberTimeOrderErrors, "E_TSADNumberTimeOrderErrors",
          "Number of records not in ascending time order"},
         {counter_t::E_TSADNumberNewAttributesNotAllowed,
          "E_TSADNumberNewAttributesNotAllowed", "Number of new attributes not allowed"},
         {counter_t::E_TSADNumberNewAttributes, "E_TSADNumberNewAttributes", "Number of new attributes created"},
         {counter_t::E_TSADNumberNewAttributesRecycled, "E_TSADNumberNewAttributesRecycled",
          "Number of new attributes recycled into existing space"},
         {counter_t::E_TSADNumberByFields, "E_TSADNumberByFields", "Number of 'by' fields within the model"},
         {counter_t::E_TSADNumberOverFields, "E_TSADNumberOverFields",
          "Number of 'over' fields within the model"},
         {counter_t::E_TSADNumberExcludedFrequentInvocations, "E_TSADNumberExcludedFrequentInvocations",
          "The number of times 'ExcludeFrequent' has been invoked by the model"},
         {counter_t::E_TSADNumberSamplesOutsideLatencyWindow, "E_TSADNumberSamplesOutsideLatencyWindow",
          "The number of samples received outside the latency window"},
         {counter_t::E_TSADNumberMemoryLimitModelCreationFailures, "E_TSADNumberMemoryLimitModelCreationFailures",
          "The number of model creation failures from being over memory limit"},
         {counter_t::E_TSADNumberPrunedItems, "E_TSADNumberPrunedItems",
          "The number of old people or attributes pruned from the models"},
         {counter_t::E_TSADAssignmentMemoryBasis, "E_TSADAssignmentMemoryBasis",
          "Which option is being used to get model memory for node assignment?"},
         {counter_t::E_DFOEstimatedPeakMemoryUsage, "E_DFOEstimatedPeakMemoryUsage",
          "The upfront estimate of the peak memory outlier detection would use"},
         {counter_t::E_DFOPeakMemoryUsage, "E_DFOPeakMemoryUsage", "The peak memory outlier detection used"},
         {counter_t::E_DFOTimeToCreateEnsemble, "E_DFOTimeToCreateEnsemble",
          "The time it took to create the ensemble used for outlier detection"},
         {counter_t::E_DFOTimeToComputeScores, "E_DFOTimeToComputeScores",
          "The time it took to compute outlier scores"},
         {counter_t::E_DFONumberPartitions, "E_DFONumberPartitions",
          "The number of partitions outlier detection used"},
         {counter_t::E_DFTPMEstimatedPeakMemoryUsage, "E_DFTPMEstimatedPeakMemoryUsage",
          "The upfront estimate of the peak memory training the predictive model would use"},
         {counter_t::E_DFTPMPeakMemoryUsage, "E_DFTPMPeakMemoryUsage",
          "The peak memory training the predictive model used"},
         {counter_t::E_DFTPMTimeToTrain, "E_DFTPMTimeToTrain", "The time it took to train the predictive model"},
         {counter_t::E_DFTPMTrainedForestNumberTrees, "E_DFTPMTrainedForestNumberTrees",
          "The total number of trees in the trained forest"}}};

    //! Enabling printing out the current counters.
    friend CORE_EXPORT std::ostream& operator<<(std::ostream& o,
                                                const CProgramCounters& counters);

    //! Befriend the test suite
    friend class test::CProgramCounterClearingFixture;
    friend class CProgramCountersTest::CProgramCountersTestRunner;
    friend struct CProgramCountersTest::testCounters;
    friend struct CProgramCountersTest::testUnknownCounter;
    friend struct CProgramCountersTest::testMissingCounter;
    friend struct CProgramCountersTest::testCacheCounters;
    friend struct CProgramCountersTest::testPersist;
};

} // core
} // ml

#endif // INCLUDED_ml_core_CProgramCounters_h
