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

#ifndef INCLUDED_ml_model_CMetricPartialStatistic_h
#define INCLUDED_ml_model_CMetricPartialStatistic_h

#include <core/CLogger.h>
#include <core/CMemoryDec.h>
#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CoreTypes.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CChecksum.h>
#include <maths/common/CDoublePrecisionStorage.h>

#include <model/CMetricStatisticWrappers.h>
#include <model/ImportExport.h>

#include <vector>

namespace ml {
namespace model {

//! \brief A partial metric statistic.
//!
//! DESCRIPTION:\n
//! A partial statistic is composed by the value of the statistic
//! and the mean time of the measurements that compose the statistic.
//! The class groups the two together and provides functions in order
//! to add measurements to the statistic, combine it with others
//! and read its values.
//!
//! \tparam STATISTIC has the following requirements:
//!   -# Member function void add(double value, unsigned int count)
//!   -# Implementations for every function in CMetricStatisticsWrapper
//!   -# Member operator +=
//!   -# Supported by maths::common::CChecksum::calculate
//!   -# Supported by core::memory_debug::dynamicSize
//!   -# Supported by core::memory::dynamicSize
//!   -# Have overload of operator<<
template<class STATISTIC>
class CMetricPartialStatistic {
public:
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TMeanAccumulator =
        maths::common::CBasicStatistics::SSampleMean<maths::common::CDoublePrecisionStorage>::TAccumulator;

public:
    static const std::string VALUE_TAG;
    static const std::string TIME_TAG;

public:
    CMetricPartialStatistic(std::size_t dimension)
        : m_Value(CMetricStatisticWrappers::template make<STATISTIC>(dimension)) {}

    //! Persist to a state document.
    void persist(core::CStatePersistInserter& inserter) const {
        CMetricStatisticWrappers::persist(m_Value, VALUE_TAG, inserter);
        inserter.insertValue(TIME_TAG, m_Time.toDelimited());
    }

    //! Restore from the supplied state document traverser.
    bool restore(core::CStateRestoreTraverser& traverser) {
        do {
            const std::string& name = traverser.name();
            if (name == VALUE_TAG) {
                if (CMetricStatisticWrappers::restore(traverser, m_Value) == false) {
                    LOG_ERROR(<< "Invalid statistic in " << traverser.value());
                    return false;
                }
            } else if (name == TIME_TAG) {
                if (m_Time.fromDelimited(traverser.value()) == false) {
                    LOG_ERROR(<< "Invalid time in " << traverser.value());
                    return false;
                }
            }
        } while (traverser.next());
        return true;
    }

    //! Add a new measurement.
    //!
    //! \param[in] value The value of the statistic.
    //! \param[in] time The time of the statistic.
    //! \param[in] count The number of measurements in the statistic.
    inline void add(const TDouble1Vec& value, core_t::TTime time, unsigned int count) {
        CMetricStatisticWrappers::add(value, count, m_Value);
        m_Time.add(static_cast<double>(time), count);
    }

    //! Returns the aggregated value of all the measurements.
    inline TDouble1Vec value() const {
        return CMetricStatisticWrappers::value(m_Value);
    }

    //! Returns the combined count of all the measurements.
    inline double count() const {
        return maths::common::CBasicStatistics::count(m_Time);
    }

    //! Returns the mean time of all the measurements.
    inline core_t::TTime time() const {
        return static_cast<core_t::TTime>(maths::common::CBasicStatistics::mean(m_Time) + 0.5);
    }

    //! Combine two partial statistics.
    inline const CMetricPartialStatistic& operator+=(const CMetricPartialStatistic& rhs) {
        m_Value += rhs.m_Value;
        m_Time += rhs.m_Time;
        return *this;
    }

    //! Get the checksum of the partial statistic
    inline uint64_t checksum(uint64_t seed) const {
        seed = maths::common::CChecksum::calculate(seed, m_Value);
        return maths::common::CChecksum::calculate(seed, m_Time);
    }

    //! Debug the memory used by the statistic.
    inline void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CMetricPartialStatistic", sizeof(*this));
        core::memory_debug::dynamicSize("m_Value", m_Value, mem);
        core::memory_debug::dynamicSize("m_Time", m_Time, mem);
    }

    //! Get the memory used by the statistic.
    inline std::size_t memoryUsage() const {
        return sizeof(*this) + core::memory::dynamicSize(m_Value) +
               core::memory::dynamicSize(m_Time);
    }

    //! Print partial statistic
    inline std::string print() const {
        std::ostringstream result;
        result << m_Value << ' ' << maths::common::CBasicStatistics::mean(m_Time);
        return result.str();
    }

private:
    STATISTIC m_Value;
    TMeanAccumulator m_Time;
};

template<class STATISTIC>
const std::string CMetricPartialStatistic<STATISTIC>::VALUE_TAG("a");
template<class STATISTIC>
const std::string CMetricPartialStatistic<STATISTIC>::TIME_TAG("b");
}
}

#endif // INCLUDED_ml_model_CMetricPartialStatistic_h
