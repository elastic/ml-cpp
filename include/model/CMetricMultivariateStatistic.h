/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CMetricMultivariateStatistic_h
#define INCLUDED_ml_model_CMetricMultivariateStatistic_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CSmallVector.h>

#include <maths/CChecksum.h>

#include <model/CMetricStatisticWrappers.h>

#include <cstddef>
#include <string>
#include <vector>

namespace ml {
namespace model {

//! \brief Wraps up one of our basic statistic objects for vector valued
//! quantities.
//!
//! DESCRIPTION:\n
//! This is a wrapper around our basic metric statistic to enable vector
//! valued statistics. In particular, it implements the contract for our
//! basic statistic object for vector valued quantities.
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
class CMetricMultivariateStatistic {
public:
    using TDouble1Vec = core::CSmallVector<double, 1>;

public:
    static const std::string VALUE_TAG;

public:
    CMetricMultivariateStatistic(std::size_t n) : m_Values(n) {}

    //! Persist to a state document.
    void persist(core::CStatePersistInserter& inserter) const {
        for (std::size_t i = 0; i < m_Values.size(); ++i) {
            CMetricStatisticWrappers::persist(m_Values[i], VALUE_TAG, inserter);
        }
    }

    //! Restore from the supplied state document traverser.
    bool restore(core::CStateRestoreTraverser& traverser) {
        std::size_t i = 0;
        do {
            const std::string& name = traverser.name();
            if (name == VALUE_TAG) {
                if (CMetricStatisticWrappers::restore(traverser, m_Values[i++]) == false) {
                    LOG_ERROR(<< "Invalid statistic in " << traverser.value());
                    return false;
                }
            }
        } while (traverser.next());
        return true;
    }

    //! Add a new measurement.
    //!
    //! \param[in] value The value of the statistic.
    //! \param[in] count The number of measurements in the statistic.
    void add(const TDouble1Vec& value, unsigned int count) {
        if (value.size() != m_Values.size()) {
            LOG_ERROR(<< "Inconsistent input data:"
                      << " # values = " << value.size() << ", expected "
                      << m_Values.size());
            return;
        }
        for (std::size_t i = 0; i < value.size(); ++i) {
            m_Values[i].add(value[i], count);
        }
    }

    //! Returns the aggregated value of all the measurements.
    TDouble1Vec value() const {
        std::size_t dimension = m_Values.size();
        TDouble1Vec result(dimension);
        for (std::size_t i = 0; i < dimension; ++i) {
            TDouble1Vec vi = CMetricStatisticWrappers::value(m_Values[i]);
            if (vi.size() > 1) {
                result.resize(vi.size() * dimension);
            }
            for (std::size_t j = 0; j < vi.size(); ++j) {
                result[i + j * dimension] = vi[j];
            }
        }
        return result;
    }

    //! Returns the aggregated value of all the measurements suitable
    //! for computing influence.
    TDouble1Vec influencerValue() const {
        std::size_t dimension = m_Values.size();
        TDouble1Vec result(dimension);
        for (std::size_t i = 0; i < dimension; ++i) {
            TDouble1Vec vi = CMetricStatisticWrappers::influencerValue(m_Values[i]);
            if (vi.size() > 1) {
                result.resize(vi.size() * dimension);
            }
            for (std::size_t j = 0; j < vi.size(); ++j) {
                result[i + j * dimension] = vi[j];
            }
        }
        return result;
    }

    //! Returns the count of all the measurements.
    double count() const {
        return CMetricStatisticWrappers::count(m_Values[0]);
    }

    //! Combine two partial statistics.
    const CMetricMultivariateStatistic& operator+=(const CMetricMultivariateStatistic& rhs) {
        for (std::size_t i = 0; i < m_Values.size(); ++i) {
            m_Values[i] += rhs.m_Values[i];
        }
        return *this;
    }

    //! Get the checksum of the partial statistic
    uint64_t checksum(uint64_t seed) const {
        return maths::CChecksum::calculate(seed, m_Values);
    }

    //! Debug the memory used by the statistic.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CMetricPartialStatistic", sizeof(*this));
        core::CMemoryDebug::dynamicSize("m_Value", m_Values, mem);
    }

    //! Get the memory used by the statistic.
    std::size_t memoryUsage() const {
        return sizeof(*this) + core::CMemory::dynamicSize(m_Values);
    }

    //! Print partial statistic
    std::string print() const {
        std::ostringstream result;
        result << core::CContainerPrinter::print(m_Values);
        return result.str();
    }

private:
    using TStatistic2Vec = core::CSmallVector<STATISTIC, 2>;

private:
    TStatistic2Vec m_Values;
};

template<class STATISTIC>
const std::string CMetricMultivariateStatistic<STATISTIC>::VALUE_TAG("a");

template<class STATISTIC>
std::ostream& operator<<(std::ostream& o,
                         const CMetricMultivariateStatistic<STATISTIC>& statistic) {
    return o << statistic.print();
}
}
}

#endif // INCLUDED_ml_model_CMetricMultivariateStatistic_h
