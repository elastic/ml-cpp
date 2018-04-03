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

#ifndef INCLUDED_ml_model_CMetricPartialStatistic_h
#define INCLUDED_ml_model_CMetricPartialStatistic_h

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CoreTypes.h>
#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CDoublePrecisionStorage.h>

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
//!   -# Supported by maths::CChecksum::calculate
//!   -# Supported by core::CMemoryDebug::dynamicSize
//!   -# Supported by core::CMemory::dynamicSize
//!   -# Have overload of operator<<
template<class STATISTIC>
class CMetricPartialStatistic {
    public:
        typedef core::CSmallVector<double, 1>                                                      TDouble1Vec;
        typedef maths::CBasicStatistics::SSampleMean<maths::CDoublePrecisionStorage>::TAccumulator TMeanAccumulator;

    public:
        static const std::string VALUE_TAG;
        static const std::string TIME_TAG;

    public:
        CMetricPartialStatistic(std::size_t dimension) :
            m_Value(CMetricStatisticWrappers::template make<STATISTIC>(dimension))
        {}

        //! Persist to a state document.
        void persist(core::CStatePersistInserter &inserter) const {
            CMetricStatisticWrappers::persist(m_Value, VALUE_TAG, inserter);
            inserter.insertValue(TIME_TAG, m_Time.toDelimited());
        }

        //! Restore from the supplied state document traverser.
        bool restore(core::CStateRestoreTraverser &traverser) {
            do {
                const std::string &name = traverser.name();
                if (name == VALUE_TAG) {
                    if (CMetricStatisticWrappers::restore(traverser, m_Value) == false) {
                        LOG_ERROR("Invalid statistic in " << traverser.value());
                        return false;
                    }
                } else if (name == TIME_TAG)   {
                    if (m_Time.fromDelimited(traverser.value()) == false) {
                        LOG_ERROR("Invalid time in " << traverser.value());
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
        inline void add(const TDouble1Vec &value,
                        core_t::TTime time,
                        unsigned int count) {
            CMetricStatisticWrappers::add(value, count, m_Value);
            m_Time.add(static_cast<double>(time), count);
        }

        //! Returns the aggregated value of all the measurements.
        inline TDouble1Vec value(void) const {
            return CMetricStatisticWrappers::value(m_Value);
        }

        //! Returns the combined count of all the measurements.
        inline double count(void) const {
            return maths::CBasicStatistics::count(m_Time);
        }

        //! Returns the mean time of all the measurements.
        inline core_t::TTime time(void) const {
            return static_cast<core_t::TTime>(
                maths::CBasicStatistics::mean(m_Time) + 0.5);
        }

        //! Combine two partial statistics.
        inline const CMetricPartialStatistic &operator+=(const CMetricPartialStatistic &rhs) {
            m_Value += rhs.m_Value;
            m_Time += rhs.m_Time;
            return *this;
        }

        //! Get the checksum of the partial statistic
        inline uint64_t checksum(uint64_t seed) const {
            seed = maths::CChecksum::calculate(seed, m_Value);
            return maths::CChecksum::calculate(seed, m_Time);
        }

        //! Debug the memory used by the statistic.
        inline void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
            mem->setName("CMetricPartialStatistic", sizeof(*this));
            core::CMemoryDebug::dynamicSize("m_Value", m_Value, mem);
            core::CMemoryDebug::dynamicSize("m_Time", m_Time, mem);
        }

        //! Get the memory used by the statistic.
        inline std::size_t memoryUsage(void) const {
            return sizeof(*this)
                   + core::CMemory::dynamicSize(m_Value)
                   + core::CMemory::dynamicSize(m_Time);
        }

        //! Print partial statistic
        inline std::string print(void) const {
            std::ostringstream result;
            result << m_Value << ' ' << maths::CBasicStatistics::mean(m_Time);
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
