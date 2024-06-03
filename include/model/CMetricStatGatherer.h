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

#ifndef INCLUDED_ml_model_CMetricStatGatherer_h
#define INCLUDED_ml_model_CMetricStatGatherer_h

#include <core/CIEEE754.h>
#include <core/CLogger.h>
#include <core/CMemoryUsage.h>
#include <core/CPersistUtils.h>
#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>
#include <core/RestoreMacros.h>
#include <core/UnwrapRef.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CChecksum.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/COrderings.h>
#include <maths/common/CQuantileSketch.h>

#include <model/CBucketQueue.h>
#include <model/CDataClassifier.h>
#include <model/CFeatureData.h>
#include <model/CMetricMultivariateStatistic.h>
#include <model/CMetricStatShims.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>
#include <model/SModelParams.h>

#include <boost/unordered_map.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace model {
namespace metric_stat_gatherer_detail {

using TDouble1Vec = core::CSmallVector<double, 1>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMinAccumulator = maths::common::CBasicStatistics::SMin<double>::TAccumulator;
using TMaxAccumulator = maths::common::CBasicStatistics::SMax<double>::TAccumulator;
using TVarianceAccumulator = maths::common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TMedianAccumulator = maths::common::CFixedQuantileSketch<30>;
using TMultivariateMeanAccumulator = CMetricMultivariateStatistic<TMeanAccumulator>;
using TSampleVec = std::vector<CSample>;

extern const std::string COUNT_TAG;
extern const std::string STAT_TAG;
extern const std::string TIME_TAG;
extern const std::string MAP_KEY_TAG;
extern const std::string MAP_VALUE_TAG;

//! \brief Manages persistence of influence bucket statistics.
template<typename STAT>
class CStrStatUMapSerializer {
public:
    using TStrStatUMap = boost::unordered_map<std::string, STAT>;

public:
    explicit CStrStatUMapSerializer(const STAT& initial)
        : m_Initial(initial) {}

    void operator()(const TStrStatUMap& map,
                    core::CStatePersistInserter& inserter) const {
        using TStrCRef = std::reference_wrapper<const std::string>;
        using TStatCRef = std::reference_wrapper<const STAT>;
        using TStrCRefStatCRefPr = std::pair<TStrCRef, TStatCRef>;
        using TStrCRefStatCRefPrVec = std::vector<TStrCRefStatCRefPr>;
        TStrCRefStatCRefPrVec ordered;
        ordered.reserve(map.size());
        for (const auto& stat : map) {
            ordered.emplace_back(*stat.first, stat.second);
        }
        std::sort(ordered.begin(), ordered.end(), maths::common::COrderings::SFirstLess{});
        for (const auto& stat : ordered) {
            inserter.insertValue(MAP_KEY_TAG, stat.first);
            metric_stat_shims::persist(core::unwrap_ref(stat.second), MAP_VALUE_TAG, inserter);
        }
    }

    bool operator()(TStrStatUMap& map,
                    core::CStateRestoreTraverser& traverser) const {
        std::string key;
        do {
            const std::string& name{traverser.name()};
            RESTORE_NO_ERROR(MAP_KEY_TAG, key = traverser.value())
            RESTORE(MAP_VALUE_TAG,
                    metric_stat_shims::restore(
                        traverser, map.emplace(key, m_Initial).first->second))
        } while (traverser.next());
        return true;
    }

private:
    STAT m_Initial;
};

template<typename STAT>
using TStrStatUMapQueue =
    CBucketQueue<boost::unordered_map<std::string, STAT>>;
template<typename STAT>
using TStrStatUMapQueueSerializer =
    typename TStrStatUMapQueue<STAT>::template CSerializer<CStrStatUMapSerializer<STAT>>;

class CMidTime {
public:
    explicit CMidTime(core_t::TTime bucketLength) : m_Time{bucketLength / 2} {}

    core_t::TTime value() const { return m_Time; }

    void add(core_t::TTime, unsigned int) {}

    std::string toDelimited() const {
        return core::CStringUtils::typeToString(m_Time);
    }
    bool fromDelimited(const std::string& value) {
        return core::CStringUtils::stringToType(value, m_Time);
    }
    std::uint64_t checksum() const {
        return static_cast<std::uint64_t>(m_Time);
    }

private:
    core_t::TTime m_Time{0};
};

class CLastTime {
public:
    explicit CLastTime(core_t::TTime) {}

    core_t::TTime value() const { return m_Time; }

    void add(core_t::TTime time, unsigned int) { m_Time = time; }

    std::string toDelimited() const {
        return core::CStringUtils::typeToString(m_Time);
    }
    bool fromDelimited(const std::string& value) {
        return core::CStringUtils::stringToType(value, m_Time);
    }
    std::uint64_t checksum() const {
        return static_cast<std::uint64_t>(m_Time);
    }

private:
    core_t::TTime m_Time{0};
};

class CMeanTime {
public:
    explicit CMeanTime(core_t::TTime) {}

    core_t::TTime value() const {
        return static_cast<core_t::TTime>(
            std::round(maths::common::CBasicStatistics::mean(m_Time)));
    }

    void add(core_t::TTime time, unsigned int count) {
        m_Time.add(static_cast<double>(time), count);
    }

    std::string toDelimited() const { return m_Time.toDelimited(); }
    bool fromDelimited(const std::string& value) {
        return m_Time.fromDelimited(value);
    }
    std::uint64_t checksum() const { return m_Time.checksum(); }

private:
    TMeanAccumulator m_Time;
};

//! \brief Gathers a STAT statistic and it's bucket time.
template<typename STAT, typename TIME>
class CStatGatherer {
public:
    using TStat = STAT;

public:
    explicit CStatGatherer(core_t::TTime bucketLength, const STAT& initial)
        : m_Time{bucketLength}, m_Stat{initial} {}

    std::size_t dimension() const {
        return metric_stat_shims::dimension(m_Stat);
    }
    core_t::TTime bucketTime(core_t::TTime time) const {
        return time + m_Time.value();
    }
    TDouble1Vec value() const { return metric_stat_shims::value(m_Stat); }
    double count() const { return metric_stat_shims::count(m_Stat); }
    double varianceScale() const { return 1.0; }
    TSampleVec samples(core_t::TTime time) const {
        return {{this->bucketTime(time), this->value(), 1.0,
                 static_cast<double>(this->count())}};
    }

    void add(core_t::TTime bucketTime, const TDouble1Vec& value, unsigned int count) {
        if (metric_stat_shims::wouldAdd(value, m_Stat)) {
            m_Time.add(bucketTime, count);
            metric_stat_shims::add(value, count, m_Stat);
        }
    }

    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        metric_stat_shims::persist(m_Time, TIME_TAG, inserter);
        metric_stat_shims::persist(m_Stat, STAT_TAG, inserter);
    }
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        do {
            const std::string& name{traverser.name()};
            RESTORE(TIME_TAG, metric_stat_shims::restore(traverser, m_Time))
            RESTORE(STAT_TAG, metric_stat_shims::restore(traverser, m_Stat))
        } while (traverser.next());
        return true;
    }

    std::uint64_t checksum() const {
        return maths::common::CChecksum::calculate(m_Time.checksum(), m_Stat);
    }

private:
    TIME m_Time;
    STAT m_Stat;
};

using TSumGatherer = CStatGatherer<CSumAccumulator, CMidTime>;
using TMeanGatherer = CStatGatherer<TMeanAccumulator, CMeanTime>;
using TMultivariateMeanGatherer = CStatGatherer<TMultivariateMeanAccumulator, CMeanTime>;
using TMedianGatherer = CStatGatherer<TMedianAccumulator, CMeanTime>;
using TMinGatherer = CStatGatherer<TMinAccumulator, CLastTime>;
using TMaxGatherer = CStatGatherer<TMaxAccumulator, CLastTime>;
using TVarianceGatherer = CStatGatherer<TVarianceAccumulator, CMeanTime>;

} // metric_stat_gatherer_detail::

//! \brief Bucket metric statistic gatherer.
template<typename STAT, model_t::EFeature FEATURE>
class CMetricStatGatherer {
public:
    using TDouble1Vec = metric_stat_gatherer_detail::TDouble1Vec;
    using TStrVecCItr = std::vector<std::string>::const_iterator;

    static const std::string CLASSIFIER_TAG;
    static const std::string BUCKET_STATS_TAG;
    static const std::string INFLUENCER_BUCKET_STATS_TAG;

private:
    using TBaseStat = typename STAT::TStat;
    using TStatQueue = CBucketQueue<STAT>;
    using TStrBaseStatUMap =
        boost::unordered_map<std::string, TBaseStat>;
    using TStrBaseStatUMapQueue = CBucketQueue<TStrBaseStatUMap>;
    using TStrBaseStatUMapQueueVec = std::vector<TStrBaseStatUMapQueue>;
    using TStrVec = std::vector<std::string>;
    using TStrStatUMapQueueSerializer =
        metric_stat_gatherer_detail::TStrStatUMapQueueSerializer<TBaseStat>;

public:
    CMetricStatGatherer(std::size_t latencyBuckets,
                        std::size_t dimension,
                        core_t::TTime startTime,
                        core_t::TTime bucketLength,
                        TStrVecCItr beginInfluencers,
                        TStrVecCItr endInfluencers)
        : m_BaseStat{metric_stat_shims::makeStat<TBaseStat>(dimension)},
          m_BucketStats(latencyBuckets,
                        bucketLength,
                        startTime,
                        STAT{bucketLength, metric_stat_shims::makeStat<TBaseStat>(dimension)}),
          m_InfluencerBucketStats(
              std::distance(beginInfluencers, endInfluencers),
              TStoredStringPtrBaseStatUMapQueue(latencyBuckets,
                                                bucketLength,
                                                startTime,
                                                TStoredStringPtrBaseStatUMap(1))) {}

    //! Get the dimension of the underlying statistic.
    std::size_t dimension() const { return m_BaseStat.dimension(); }

    //! Get the feature data for the current bucketing interval.
    SMetricFeatureData featureData(core_t::TTime time) const {

        using namespace metric_stat_gatherer_detail;
        using TStrCRef = std::reference_wrapper<const std::string>;
        using TStrCRefDouble1VecDoublePrPrVecVec = SMetricFeatureData::TStrCRefDouble1VecDoublePrPrVecVec;

        const auto& gatherer = m_BucketStats.get(time);

        if (gatherer.value().empty() == false) {
            TStrCRefDouble1VecDoublePrPrVecVec influenceValues(
                m_InfluencerBucketStats.size());
            for (std::size_t i = 0; i < m_InfluencerBucketStats.size(); ++i) {
                const auto& influencerStats = m_InfluencerBucketStats[i].get(time);
                influenceValues[i].reserve(influencerStats.size());
                for (const auto & [ name, stat ] : influencerStats) {
                    influenceValues[i].emplace_back(
                        TStrCRef(*name),
                        std::make_pair(metric_stat_shims::influencerValue(stat),
                                       metric_stat_shims::count(stat)));
                }
            }

            time = maths::common::CIntegerTools::floor(time, m_BucketStats.bucketLength());
            return {gatherer.bucketTime(time),
                    gatherer.value(),
                    gatherer.varianceScale(),
                    gatherer.count(),
                    influenceValues,
                    m_Classifier.isInteger(),
                    m_Classifier.isNonNegative(),
                    gatherer.samples(time)};
        }

        return {m_Classifier.isInteger(), m_Classifier.isNonNegative(), {}};
    }

    //! Update the state with a new measurement.
    //!
    //! \param[in] time The time of \p value.
    //! \param[in] value The measurement value.
    //! \param[in] influences The influencing field values which label \p value.
    void add(core_t::TTime time,
             const TDouble1Vec& value,
             unsigned int count,
             const TStrVec& influences) {
        core_t::TTime bucketTime{time % m_BucketStats.bucketLength()};
        m_Classifier.add(FEATURE, value, count);
        m_BucketStats.get(time).add(bucketTime, value, count);
        for (std::size_t i = 0; i < influences.size(); ++i) {
            if (influences[i] != nullptr) {
                metric_stat_shims::add(value, count,
                                       m_InfluencerBucketStats[i]
                                           .get(time)
                                           .emplace(influences[i], m_BaseStat)
                                           .first->second);
            }
        }
    }

    //! Update the state to represent the start of a new bucket.
    void startNewBucket(core_t::TTime time) {
        m_BucketStats.push(STAT{m_BucketStats.bucketLength(), m_BaseStat}, time);
        for (auto& stat : m_InfluencerBucketStats) {
            stat.push(TStrBaseStatUMap(1), time);
        }
    }

    //! Reset bucket.
    void resetBucket(core_t::TTime bucketStart) {
        m_BucketStats.get(bucketStart) = STAT{m_BucketStats.bucketLength(), m_BaseStat};
        for (auto& stat : m_InfluencerBucketStats) {
            stat.get(bucketStart).clear();
        }
    }

    //! \name Persistence
    //@{
    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        using namespace metric_stat_gatherer_detail;
        inserter.insertLevel(CLASSIFIER_TAG, [this](core::CStatePersistInserter& inserter_) {
            m_Classifier.acceptPersistInserter(inserter_);
        });
        if (m_BucketStats.empty() == false) {
            inserter.insertLevel(BUCKET_STATS_TAG, [this](core::CStatePersistInserter& inserter_) {
                m_BucketStats.acceptPersistInserter(inserter_);
            });
        }
        TStrStatUMapQueueSerializer influencerSerializer{
            TStrBaseStatUMap(1),
            CStrStatUMapSerializer<TBaseStat>(m_BaseStat)};
        for (const auto& stats : m_InfluencerBucketStats) {
            inserter.insertLevel(INFLUENCER_BUCKET_STATS_TAG, [&](auto& inserter_) {
                influencerSerializer(stats, inserter_);
            });
        }
    }

    //! Create from part of a state document.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        using namespace metric_stat_gatherer_detail;
        TStrStatUMapQueueSerializer influencerSerializer{
            TStrBaseStatUMap(1),
            CStrStatUMapSerializer<TBaseStat>(m_BaseStat)};
        std::size_t i{0};
        do {
            const std::string& name{traverser.name()};
            RESTORE(CLASSIFIER_TAG,
                    traverser.traverseSubLevel([this](core::CStateRestoreTraverser& traverser_) {
                        return m_Classifier.acceptRestoreTraverser(traverser_);
                    }))
            RESTORE(BUCKET_STATS_TAG,
                    traverser.traverseSubLevel([this](core::CStateRestoreTraverser& traverser_) {
                        return m_BucketStats.acceptRestoreTraverser(traverser_);
                    }))
            RESTORE(INFLUENCER_BUCKET_STATS_TAG,
                    i < m_InfluencerBucketStats.size() &&
                        traverser.traverseSubLevel([&](auto& traverser_) {
                            return influencerSerializer(m_InfluencerBucketStats[i++], traverser_);
                        }));
        } while (traverser.next());
        return true;
    }
    //@}

    //! Get the checksum of this gatherer.
    std::uint64_t checksum() const {
        std::uint64_t seed{static_cast<std::uint64_t>(m_Classifier.isInteger())};
        seed = maths::common::CChecksum::calculate(seed, m_Classifier.isNonNegative());
        seed = maths::common::CChecksum::calculate(seed, m_BucketStats);
        return maths::common::CChecksum::calculate(seed, m_InfluencerBucketStats);
    }

    //! Debug the memory used by this gatherer.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CMetricStatGatherer");
        core::memory_debug::dynamicSize("m_BaseStat", m_BaseStat, mem);
        core::memory_debug::dynamicSize("m_BucketStats", m_BucketStats, mem);
        core::memory_debug::dynamicSize("m_InfluencerBucketStats",
                                        m_InfluencerBucketStats, mem);
    }

    //! Get the memory used by this gatherer.
    std::size_t memoryUsage() const {
        return core::memory::dynamicSize(m_BaseStat) +
               core::memory::dynamicSize(m_BucketStats) +
               core::memory::dynamicSize(m_InfluencerBucketStats);
    }

    //! Is the gatherer holding redundant data?
    bool isRedundant() const {
        for (const auto& bucket : m_BucketStats) {
            if (bucket.count() > 0) {
                return false;
            }
        }
        return true;
    }

private:
    //! Gathers influencer stats.
    TBaseStat m_BaseStat;

    //! Classifies the stat series.
    CDataClassifier m_Classifier;

    //! The stat for each bucket within the latency window.
    TStatQueue m_BucketStats;

    //! The stat for each influencing field value and bucket within the latency window.
    TStrBaseStatUMapQueueVec m_InfluencerBucketStats;
};

template<typename STAT, model_t::EFeature FEATURE>
const std::string CMetricStatGatherer<STAT, FEATURE>::CLASSIFIER_TAG{"a"};
template<typename STAT, model_t::EFeature FEATURE>
const std::string CMetricStatGatherer<STAT, FEATURE>::BUCKET_STATS_TAG{"b"};
template<typename STAT, model_t::EFeature FEATURE>
const std::string CMetricStatGatherer<STAT, FEATURE>::INFLUENCER_BUCKET_STATS_TAG{"c"};

//! \brief Sum statistic gatherer.
using TSumGatherer =
    CMetricStatGatherer<metric_stat_gatherer_detail::TSumGatherer, model_t::E_IndividualMeanByPerson>;

//! \brief Mean statistic gatherer.
using TMeanGatherer =
    CMetricStatGatherer<metric_stat_gatherer_detail::TMeanGatherer, model_t::E_IndividualMeanByPerson>;

//! \brief Multivariate mean statistic gatherer.
using TMultivariateMeanGatherer =
    CMetricStatGatherer<metric_stat_gatherer_detail::TMultivariateMeanGatherer, model_t::E_IndividualMeanByPerson>;

//! \brief Median statistic gatherer.
using TMedianGatherer =
    CMetricStatGatherer<metric_stat_gatherer_detail::TMedianGatherer, model_t::E_IndividualMedianByPerson>;

//! \brief Minimum statistic gatherer.
using TMinGatherer =
    CMetricStatGatherer<metric_stat_gatherer_detail::TMinGatherer, model_t::E_IndividualMinByPerson>;

//! \brief Maximum statistic gatherer.
using TMaxGatherer =
    CMetricStatGatherer<metric_stat_gatherer_detail::TMaxGatherer, model_t::E_IndividualMaxByPerson>;

//! \brief Variance statistic gatherer.
using TVarianceGatherer =
    CMetricStatGatherer<metric_stat_gatherer_detail::TVarianceGatherer, model_t::E_IndividualVarianceByPerson>;
}
}

#endif // INCLUDED_ml_model_CMetricStatGatherer_h
