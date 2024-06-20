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

#ifndef INCLUDED_ml_model_CMetricStatShims_h
#define INCLUDED_ml_model_CMetricStatShims_h

#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CQuantileSketch.h>

#include <cstddef>

namespace ml {
namespace model {
template<typename STAT>
class CMetricMultivariateStatistic;

//! \brief Accumulates the sum and count of a metric.
class CSumAccumulator {
public:
    double value() const { return m_Sum; }
    unsigned int count() const { return std::max(m_Count, 1U); }

    void add(double value, unsigned int count) {
        m_Sum += value;
        m_Count += count;
    }

    std::string toDelimited() const {
        return core::CStringUtils::typeToString(m_Count) + "," +
               core::CStringUtils::typeToStringPrecise(m_Sum, core::CIEEE754::E_DoublePrecision);
    }

    bool fromDelimited(const std::string& value) {
        std::size_t delimPos{value.find(',')};
        if (delimPos == std::string::npos) {
            LOG_ERROR(<< "Invalid sum in '" << value << "'");
            return false;
        }
        if (core::CStringUtils::stringToType(value.substr(0, delimPos), m_Count) == false) {
            LOG_ERROR(<< "Invalid sum in '" << value.substr(0, delimPos) << "'");
            return false;
        }
        if (core::CStringUtils::stringToType(value.substr(delimPos + 1), m_Sum) == false) {
            LOG_ERROR(<< "Invalid sum in '" << value.substr(delimPos + 1) << "'");
            return false;
        }
        return true;
    }

    std::uint64_t checksum() const {
        return maths::common::CChecksum::calculate(m_Count, m_Sum);
    }

private:
    unsigned int m_Count{0};
    double m_Sum{0.0};
};

namespace metric_stat_shims {
using TDouble1Vec = core::CSmallVector<double, 1>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TVarianceAccumulator = maths::common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TMedianAccumulator = maths::common::CFixedQuantileSketch<30>;

template<typename STAT>
class StatFactory {
public:
    static STAT make(std::size_t) { return STAT{}; }
};
template<typename STAT>
class StatFactory<CMetricMultivariateStatistic<STAT>> {
public:
    static CMetricMultivariateStatistic<STAT> make(std::size_t dimension) {
        return CMetricMultivariateStatistic<STAT>{dimension};
    }
};
template<typename STAT>
STAT makeStat(std::size_t dimension) {
    return StatFactory<STAT>::make(dimension);
}

template<typename STAT>
std::size_t dimension(const STAT&) {
    return 1;
}
template<typename STAT>
std::size_t dimension(const CMetricMultivariateStatistic<STAT>& stat) {
    return stat.dimension();
}

template<typename STAT>
bool wouldAdd(const TDouble1Vec&, STAT&) {
    return true;
}
template<typename LESS>
bool wouldAdd(const TDouble1Vec& value,
              maths::common::CBasicStatistics::COrderStatisticsStack<double, 1, LESS>& stat) {
    return stat.wouldAdd(value[0]);
}

template<typename STAT>
void add(const TDouble1Vec& value, unsigned int count, STAT& stat) {
    stat.add(value[0], count);
}
template<typename STAT>
void add(const TDouble1Vec& value, unsigned int count, CMetricMultivariateStatistic<STAT>& stat) {
    stat.add(value, count);
}

inline TDouble1Vec value(const CSumAccumulator& stat) {
    return {stat.value()};
}
inline TDouble1Vec value(const TMeanAccumulator& stat) {
    return {maths::common::CBasicStatistics::mean(stat)};
}
inline TDouble1Vec value(const TVarianceAccumulator& stat) {
    if (maths::common::CBasicStatistics::count(stat) >= 2.0) {
        return {maths::common::CBasicStatistics::maximumLikelihoodVariance(stat),
                maths::common::CBasicStatistics::mean(stat)};
    }
    return {};
}
inline TDouble1Vec value(const TMedianAccumulator& stat) {
    double result;
    if (stat.quantile(50.0, result) == false) {
        return {};
    }
    return {result};
}
template<typename LESS>
TDouble1Vec
value(const maths::common::CBasicStatistics::COrderStatisticsStack<double, 1, LESS>& stat) {
    return {stat[0]};
}
template<typename STAT>
TDouble1Vec value(const CMetricMultivariateStatistic<STAT>& stat) {
    return stat.value();
}

template<typename STAT>
TDouble1Vec influencerValue(const STAT& stat) {
    return value(stat);
}
inline TDouble1Vec influencerValue(const TVarianceAccumulator& stat) {
    // We always return an influence value (independent of the count)
    // because this is not used to directly compute a variance only
    // to adjust the bucket variance.
    return {maths::common::CBasicStatistics::maximumLikelihoodVariance(stat),
            maths::common::CBasicStatistics::mean(stat)};
}
template<typename STAT>
TDouble1Vec influencerValue(const CMetricMultivariateStatistic<STAT>& stat) {
    return stat.influencerValue();
}

inline double count(const CSumAccumulator& stat) {
    return static_cast<double>(stat.count());
}
inline double count(const TMeanAccumulator& stat) {
    return static_cast<double>(maths::common::CBasicStatistics::count(stat));
}
inline double count(const TVarianceAccumulator& stat) {
    return static_cast<double>(maths::common::CBasicStatistics::count(stat));
}
inline double count(const TMedianAccumulator& stat) {
    return stat.count();
}
template<typename LESS>
inline double
count(const maths::common::CBasicStatistics::COrderStatisticsStack<double, 1, LESS>& /*stat*/) {
    return 1.0;
}
template<typename STAT>
double count(const CMetricMultivariateStatistic<STAT>& stat) {
    return stat.count();
}

template<typename STAT>
void persist(const STAT& stat, const std::string& tag, core::CStatePersistInserter& inserter) {
    inserter.insertValue(tag, stat.toDelimited());
}
inline void persist(const TMedianAccumulator& stat,
                    const std::string& tag,
                    core::CStatePersistInserter& inserter) {
    inserter.insertLevel(
        tag, [&](auto& inserter_) { stat.acceptPersistInserter(inserter_); });
}
template<typename STAT>
inline void persist(const CMetricMultivariateStatistic<STAT>& stat,
                    const std::string& tag,
                    core::CStatePersistInserter& inserter) {
    inserter.insertLevel(tag, [&](auto& inserter_) { stat.persist(inserter_); });
}

template<typename STAT>
bool restore(core::CStateRestoreTraverser& traverser, STAT& stat) {
    if (stat.fromDelimited(traverser.value()) == false) {
        LOG_ERROR(<< "Invalid statistic in " << traverser.value());
        return false;
    }
    return true;
}
inline bool restore(core::CStateRestoreTraverser& traverser, TMedianAccumulator& stat) {
    return traverser.traverseSubLevel([&](auto& traverser_) {
        return stat.acceptRestoreTraverser(traverser_);
    });
}
template<typename STAT>
inline bool restore(core::CStateRestoreTraverser& traverser,
                    CMetricMultivariateStatistic<STAT>& stat) {
    return traverser.traverseSubLevel(
        [&](auto& traverser_) { return stat.restore(traverser_); });
}
}
}
}

#endif // INCLUDED_ml_model_CMetricStatShims_h
