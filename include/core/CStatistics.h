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
#ifndef INCLUDED_ml_core_CStatistics_h
#define INCLUDED_ml_core_CStatistics_h

#include <core/ImportExport.h>
#include <core/CNonCopyable.h>
#include <core/CStat.h>

#include <boost/array.hpp>

#include <map>
#include <iosfwd>

#include <stdint.h>

namespace ml
{
namespace stat_t
{

//! Changing the order of these enumeration values will corrupt persisted model
//! state, so don't.  Any new statistics should be added in the penultimate
//! position in the enum, immediately before E_LastEnumStat.
enum EStatTypes
{
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

namespace core
{

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
class CORE_EXPORT CStatistics : private CNonCopyable
{
    public:
        //! Singleton pattern
        static CStatistics &instance(void);

        //! Provide access to the relevant stat from the collection
        static CStat &stat(int index);

        //! \name Persistence
        //@{
        //! Restore the static members of this class from persisted state
        static bool staticsAcceptRestoreTraverser(CStateRestoreTraverser &traverser);

        //! Persist the static members of this class
        static void staticsAcceptPersistInserter(CStatePersistInserter &inserter);
        //@}

    private:
        typedef boost::array<CStat, stat_t::E_LastEnumStat> TStatArray;

    private:
        //! Constructor of a Singleton is private
        CStatistics(void);

        //! The unique instance.
        static CStatistics ms_Instance;

        //! Collection of statistics
        TStatArray m_Stats;

        //! Enabling printing out the current statistics.
        friend CORE_EXPORT std::ostream &operator<<(std::ostream &o, const CStatistics &stats);
};

} // core
} // ml

#endif // INCLUDED_ml_core_CStatistics_h
