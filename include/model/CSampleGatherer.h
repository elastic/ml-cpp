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

#ifndef INCLUDED_ml_model_CSampleGatherer_h
#define INCLUDED_ml_model_CSampleGatherer_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStoredStringPtr.h>
#include <core/CoreTypes.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/COrderings.h>

#include <model/CBucketQueue.h>
#include <model/CDataClassifier.h>
#include <model/CMetricPartialStatistic.h>
#include <model/CMetricStatisticWrappers.h>
#include <model/CModelParams.h>
#include <model/CSampleQueue.h>
#include <model/CStringStore.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <vector>

namespace ml {
namespace model {

//! \brief Metric statistic gatherer.
//!
//! DESCRIPTION:\n
//! Wraps up the functionality to sample a statistic of a fixed
//! number of metric values, which is supplied to the add function,
//! for a latency window.
//!
//! This also computes the statistic value of all metric values
//! and for each influencing field values for every bucketing
//! interval in the latency window.
//!
//! \tparam STATISTIC This must satisfy the requirements imposed
//! by CMetricPartialStatistic.
template<typename STATISTIC, model_t::EFeature FEATURE>
class CSampleGatherer {
public:
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TStrVec = std::vector<std::string>;
    using TStrVecCItr = TStrVec::const_iterator;
    using TStrCRef = boost::reference_wrapper<const std::string>;
    using TDouble1VecDoublePr = std::pair<TDouble1Vec, double>;
    using TStrCRefDouble1VecDoublePrPr = std::pair<TStrCRef, TDouble1VecDoublePr>;
    using TStrCRefDouble1VecDoublePrPrVec = std::vector<TStrCRefDouble1VecDoublePrPr>;
    using TStrCRefDouble1VecDoublePrPrVecVec = std::vector<TStrCRefDouble1VecDoublePrPrVec>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TSampleQueue = CSampleQueue<STATISTIC>;
    using TSampleVec = typename TSampleQueue::TSampleVec;
    using TMetricPartialStatistic = CMetricPartialStatistic<STATISTIC>;
    using TStatBucketQueue = CBucketQueue<TMetricPartialStatistic>;
    using TStoredStringPtrVec = std::vector<core::CStoredStringPtr>;
    using TStoredStringPtrStatUMap =
        boost::unordered_map<core::CStoredStringPtr, STATISTIC, core::CHashing::CMurmurHash2String>;
    using TStoredStringPtrStatUMapBucketQueue = CBucketQueue<TStoredStringPtrStatUMap>;
    using TStoredStringPtrStatUMapBucketQueueVec = std::vector<TStoredStringPtrStatUMapBucketQueue>;

public:
    static const std::string CLASSIFIER_TAG;
    static const std::string SAMPLE_STATS_TAG;
    static const std::string BUCKET_STATS_TAG;
    static const std::string INFLUENCER_BUCKET_STATS_TAG;
    static const std::string DIMENSION_TAG;

public:
    CSampleGatherer(const SModelParams& params,
                    std::size_t dimension,
                    core_t::TTime startTime,
                    core_t::TTime bucketLength,
                    TStrVecCItr beginInfluencers,
                    TStrVecCItr endInfluencers)
        : m_Dimension(dimension),
          m_SampleStats(dimension,
                        params.s_SampleCountFactor,
                        params.s_LatencyBuckets,
                        params.s_SampleQueueGrowthFactor,
                        bucketLength),
          m_BucketStats(params.s_LatencyBuckets,
                        bucketLength,
                        startTime,
                        TMetricPartialStatistic(dimension)),
          m_InfluencerBucketStats(std::distance(beginInfluencers, endInfluencers),
                                  TStoredStringPtrStatUMapBucketQueue(params.s_LatencyBuckets + 3,
                                                                      bucketLength,
                                                                      startTime,
                                                                      TStoredStringPtrStatUMap(
                                                                          1))) {}

    //! \name Persistence
    //@{
    //! Persist state by passing information to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        inserter.insertValue(DIMENSION_TAG, m_Dimension);
        inserter.insertLevel(CLASSIFIER_TAG,
                             boost::bind(&CDataClassifier::acceptPersistInserter,
                                         &m_Classifier,
                                         _1));
        if (m_SampleStats.size() > 0) {
            inserter.insertLevel(SAMPLE_STATS_TAG,
                                 boost::bind(&TSampleQueue::acceptPersistInserter,
                                             &m_SampleStats,
                                             _1));
        }
        if (m_BucketStats.size() > 0) {
            inserter.insertLevel(BUCKET_STATS_TAG,
                                 boost::bind<void>(TStatBucketQueueSerializer(
                                                       TMetricPartialStatistic(m_Dimension)),
                                                   boost::cref(m_BucketStats),
                                                   _1));
        }
        for (const auto& stats : m_InfluencerBucketStats) {
            inserter.insertLevel(INFLUENCER_BUCKET_STATS_TAG,
                                 boost::bind<void>(TStoredStringPtrStatUMapBucketQueueSerializer(
                                                       TStoredStringPtrStatUMap(1),
                                                       CStoredStringPtrStatUMapSerializer(
                                                           m_Dimension)),
                                                   boost::cref(stats),
                                                   _1));
        }
    }

    //! Create from part of a state document.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        std::size_t i = 0u;
        do {
            const std::string& name = traverser.name();
            TMetricPartialStatistic stat(m_Dimension);
            RESTORE_BUILT_IN(DIMENSION_TAG, m_Dimension)
            RESTORE(CLASSIFIER_TAG,
                    traverser.traverseSubLevel(
                        boost::bind(&CDataClassifier::acceptRestoreTraverser, &m_Classifier, _1)))
            RESTORE(SAMPLE_STATS_TAG,
                    traverser.traverseSubLevel(
                        boost::bind(&TSampleQueue::acceptRestoreTraverser, &m_SampleStats, _1)))
            RESTORE(BUCKET_STATS_TAG,
                    traverser.traverseSubLevel(
                        boost::bind<bool>(TStatBucketQueueSerializer(
                                              TMetricPartialStatistic(m_Dimension)),
                                          boost::ref(m_BucketStats),
                                          _1)))
            RESTORE(INFLUENCER_BUCKET_STATS_TAG,
                    i < m_InfluencerBucketStats.size() &&
                        traverser.traverseSubLevel(
                            boost::bind<bool>(TStoredStringPtrStatUMapBucketQueueSerializer(
                                                  TStoredStringPtrStatUMap(1),
                                                  CStoredStringPtrStatUMapSerializer(m_Dimension)),
                                              boost::ref(m_InfluencerBucketStats[i++]),
                                              _1)))
        } while (traverser.next());
        return true;
    }
    //@}

    //! Get the dimension of the underlying statistic.
    std::size_t dimension(void) const { return m_Dimension; }

    //! Get the feature data for the bucketing interval containing
    //! \p time.
    //!
    //! \param[in] time The start time of the sampled bucket.
    //! \param[in] effectiveSampleCount The effective historical
    //! number of measurements in a sample.
    SMetricFeatureData featureData(core_t::TTime time,
                                   core_t::TTime /*bucketLength*/,
                                   double effectiveSampleCount) const {
        const TMetricPartialStatistic& bucketPartial = m_BucketStats.get(time);
        double count = bucketPartial.count();
        if (count > 0.0) {
            core_t::TTime bucketTime = bucketPartial.time();
            TDouble1Vec bucketValue = bucketPartial.value();
            if (bucketValue.size() > 0) {
                TStrCRefDouble1VecDoublePrPrVecVec influenceValues(m_InfluencerBucketStats.size());
                for (std::size_t i = 0u; i < m_InfluencerBucketStats.size(); ++i) {
                    const TStoredStringPtrStatUMap& influencerStats =
                        m_InfluencerBucketStats[i].get(time);
                    influenceValues[i].reserve(influencerStats.size());
                    for (const auto& stat : influencerStats) {
                        influenceValues[i]
                            .emplace_back(boost::cref(*stat.first),
                                          std::make_pair(CMetricStatisticWrappers::influencerValue(
                                                             stat.second),
                                                         CMetricStatisticWrappers::count(
                                                             stat.second)));
                    }
                }
                return {bucketTime,
                        bucketValue,
                        model_t::varianceScale(FEATURE, effectiveSampleCount, count),
                        count,
                        influenceValues,
                        m_Classifier.isInteger(),
                        m_Classifier.isNonNegative(),
                        m_Samples};
            }
        }
        return {m_Classifier.isInteger(), m_Classifier.isNonNegative(), m_Samples};
    }

    //! Create samples if possible for the given bucket.
    //!
    //! \param[in] time The start time of the sampled bucket.
    //! \param[in] sampleCount The measurement count in a sample.
    //! \return True if there are new samples and false otherwise.
    bool sample(core_t::TTime time, unsigned int sampleCount) {
        if (sampleCount > 0 && m_SampleStats.canSample(time)) {
            TSampleVec newSamples;
            m_SampleStats.sample(time, sampleCount, FEATURE, newSamples);
            m_Samples.insert(m_Samples.end(), newSamples.begin(), newSamples.end());
            return !newSamples.empty();
        }
        return false;
    }

    //! Update the state with a new measurement.
    //!
    //! \param[in] time The time of \p value.
    //! \param[in] value The measurement value.
    //! \param[in] sampleCount The measurement count in a sample.
    //! \param[in] influences The influencing field values which
    //! label \p value.
    inline void add(core_t::TTime time,
                    const TDouble1Vec& value,
                    unsigned int sampleCount,
                    const TStoredStringPtrVec& influences) {
        this->add(time, value, 1, sampleCount, influences);
    }

    //! Update the state with a new mean statistic.
    //!
    //! \param[in] time The approximate time of \p statistic.
    //! \param[in] statistic The statistic value.
    //! \param[in] count The number of measurements in \p statistic.
    //! \param[in] sampleCount The measurement count in a sample.
    //! \param[in] influences The influencing field values which
    //! label \p value.
    void add(core_t::TTime time,
             const TDouble1Vec& statistic,
             unsigned int count,
             unsigned int sampleCount,
             const TStoredStringPtrVec& influences) {
        if (sampleCount > 0) {
            m_SampleStats.add(time, statistic, count, sampleCount);
        }
        m_BucketStats.get(time).add(statistic, time, count);
        m_Classifier.add(FEATURE, statistic, count);
        std::size_t n = std::min(influences.size(), m_InfluencerBucketStats.size());
        for (std::size_t i = 0u; i < n; ++i) {
            if (!influences[i]) {
                continue;
            }
            TStoredStringPtrStatUMap& stats = m_InfluencerBucketStats[i].get(time);
            auto j = stats
                         .emplace(influences[i],
                                  CMetricStatisticWrappers::template make<STATISTIC>(m_Dimension))
                         .first;
            CMetricStatisticWrappers::add(statistic, count, j->second);
        }
    }

    //! Update the state to represent the start of a new bucket.
    void startNewBucket(core_t::TTime time) {
        m_BucketStats.push(TMetricPartialStatistic(m_Dimension), time);
        for (auto&& stats : m_InfluencerBucketStats) {
            stats.push(TStoredStringPtrStatUMap(1), time);
        }
        m_Samples.clear();
    }

    //! Reset the bucket state for the bucket containing \p bucketStart.
    void resetBucket(core_t::TTime bucketStart) {
        m_BucketStats.get(bucketStart) = TMetricPartialStatistic(m_Dimension);
        for (auto&& stats : m_InfluencerBucketStats) {
            stats.get(bucketStart) = TStoredStringPtrStatUMap(1);
        }
        m_SampleStats.resetBucket(bucketStart);
    }

    //! Is the gatherer holding redundant data?
    bool isRedundant(core_t::TTime samplingCutoffTime) const {
        if (m_SampleStats.latestEnd() >= samplingCutoffTime) {
            return false;
        }
        for (const auto& bucket : m_BucketStats) {
            if (bucket.count() > 0.0) {
                return false;
            }
        }
        return true;
    }

    //! Get the checksum of this gatherer.
    uint64_t checksum(void) const {
        uint64_t seed = static_cast<uint64_t>(m_Classifier.isInteger());
        seed = maths::CChecksum::calculate(seed, m_Classifier.isNonNegative());
        seed = maths::CChecksum::calculate(seed, m_SampleStats);
        seed = maths::CChecksum::calculate(seed, m_BucketStats);
        return maths::CChecksum::calculate(seed, m_InfluencerBucketStats);
    }

    //! Debug the memory used by this gatherer.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
        mem->setName("CSampleGatherer", sizeof(*this));
        core::CMemoryDebug::dynamicSize("m_SampleStats", m_SampleStats, mem);
        core::CMemoryDebug::dynamicSize("m_BucketStats", m_BucketStats, mem);
        core::CMemoryDebug::dynamicSize("m_InfluencerBucketStats", m_InfluencerBucketStats, mem);
        core::CMemoryDebug::dynamicSize("m_Samples", m_Samples, mem);
    }

    //! Get the memory used by this gatherer.
    std::size_t memoryUsage(void) const {
        return sizeof(*this) + core::CMemory::dynamicSize(m_SampleStats) +
               core::CMemory::dynamicSize(m_BucketStats) +
               core::CMemory::dynamicSize(m_InfluencerBucketStats) +
               core::CMemory::dynamicSize(m_Samples);
    }

    //! Print this gatherer for debug.
    std::string print(void) const {
        std::ostringstream result;
        result << m_Classifier.isInteger() << ' ' << m_Classifier.isNonNegative() << ' '
               << m_BucketStats.print() << ' ' << m_SampleStats.print() << ' '
               << core::CContainerPrinter::print(m_Samples) << ' '
               << core::CContainerPrinter::print(m_InfluencerBucketStats);
        return result.str();
    }

private:
    static const std::string MAP_KEY_TAG;
    static const std::string MAP_VALUE_TAG;

private:
    //! \brief Manages persistence of bucket statistics.
    struct SStatSerializer {
        void operator()(const TMetricPartialStatistic& stat,
                        core::CStatePersistInserter& inserter) const {
            stat.persist(inserter);
        }

        bool operator()(TMetricPartialStatistic& stat,
                        core::CStateRestoreTraverser& traverser) const {
            return stat.restore(traverser);
        }
    };
    using TStatBucketQueueSerializer =
        typename TStatBucketQueue::template CSerializer<SStatSerializer>;

    //! \brief Manages persistence of influence bucket statistics.
    class CStoredStringPtrStatUMapSerializer {
    public:
        CStoredStringPtrStatUMapSerializer(std::size_t dimension)
            : m_Initial(CMetricStatisticWrappers::template make<STATISTIC>(dimension)) {}

        void operator()(const TStoredStringPtrStatUMap& map,
                        core::CStatePersistInserter& inserter) const {
            using TStatCRef = boost::reference_wrapper<const STATISTIC>;
            using TStrCRefStatCRefPr = std::pair<TStrCRef, TStatCRef>;
            using TStrCRefStatCRefPrVec = std::vector<TStrCRefStatCRefPr>;
            TStrCRefStatCRefPrVec ordered;
            ordered.reserve(map.size());
            for (const auto& stat : map) {
                ordered.emplace_back(TStrCRef(*stat.first), TStatCRef(stat.second));
            }
            std::sort(ordered.begin(), ordered.end(), maths::COrderings::SFirstLess());
            for (const auto& stat : ordered) {
                inserter.insertValue(MAP_KEY_TAG, stat.first);
                CMetricStatisticWrappers::persist(stat.second.get(), MAP_VALUE_TAG, inserter);
            }
        }

        bool operator()(TStoredStringPtrStatUMap& map,
                        core::CStateRestoreTraverser& traverser) const {
            std::string key;
            do {
                const std::string& name = traverser.name();
                RESTORE_NO_ERROR(MAP_KEY_TAG, key = traverser.value())
                RESTORE(MAP_VALUE_TAG,
                        CMetricStatisticWrappers::restore(traverser,
                                                          map.insert({CStringStore::influencers()
                                                                          .get(key),
                                                                      m_Initial})
                                                              .first->second))
            } while (traverser.next());
            return true;
        }

    private:
        STATISTIC m_Initial;
    };
    using TStoredStringPtrStatUMapBucketQueueSerializer =
        typename TStoredStringPtrStatUMapBucketQueue::template CSerializer<
            CStoredStringPtrStatUMapSerializer>;

private:
    //! The dimension of the statistic being gathered.
    std::size_t m_Dimension;

    //! Classifies the sampled statistics.
    CDataClassifier m_Classifier;

    //! The queue holding the partial aggregate statistics within
    //! latency window used for building samples.
    TSampleQueue m_SampleStats;

    //! The aggregation of the measurements received for each
    //! bucket within latency window.
    TStatBucketQueue m_BucketStats;

    //! The aggregation of the measurements received for each
    //! bucket and influencing field within latency window.
    TStoredStringPtrStatUMapBucketQueueVec m_InfluencerBucketStats;

    //! The samples of the aggregate statistic in the current
    //! bucketing interval.
    TSampleVec m_Samples;
};

template<typename STATISTIC, model_t::EFeature FEATURE>
const std::string CSampleGatherer<STATISTIC, FEATURE>::CLASSIFIER_TAG("a");
template<typename STATISTIC, model_t::EFeature FEATURE>
const std::string CSampleGatherer<STATISTIC, FEATURE>::SAMPLE_STATS_TAG("b");
template<typename STATISTIC, model_t::EFeature FEATURE>
const std::string CSampleGatherer<STATISTIC, FEATURE>::BUCKET_STATS_TAG("c");
template<typename STATISTIC, model_t::EFeature FEATURE>
const std::string CSampleGatherer<STATISTIC, FEATURE>::INFLUENCER_BUCKET_STATS_TAG("d");
template<typename STATISTIC, model_t::EFeature FEATURE>
const std::string CSampleGatherer<STATISTIC, FEATURE>::DIMENSION_TAG("e");
template<typename STATISTIC, model_t::EFeature FEATURE>
const std::string CSampleGatherer<STATISTIC, FEATURE>::MAP_KEY_TAG("a");
template<typename STATISTIC, model_t::EFeature FEATURE>
const std::string CSampleGatherer<STATISTIC, FEATURE>::MAP_VALUE_TAG("b");

//! Overload print operator for feature data.
MODEL_EXPORT
inline std::ostream& operator<<(std::ostream& o, const SMetricFeatureData& fd) {
    return o << fd.print();
}
}
}

#endif // INCLUDED_ml_model_CSampleGatherer_h
