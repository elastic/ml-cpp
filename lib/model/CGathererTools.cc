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

#include <model/CGathererTools.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CMultinomialConjugate.h>
#include <maths/CMultivariatePrior.h>
#include <maths/Constants.h>
#include <maths/CTools.h>

#include <model/CStringStore.h>

#include <boost/bind.hpp>

namespace ml {
namespace model {

namespace {

const std::string CLASSIFIER_TAG("a");
const std::string LAST_TIME_TAG("b");
const std::string BUCKET_SUM_QUEUE_TAG("c");
const std::string INFLUENCER_BUCKET_SUM_QUEUE_TAG("e");

// Nested tags
const std::string SUM_SAMPLE_TAG("a");
const std::string SUM_MAP_KEY_TAG("b");
const std::string SUM_MAP_VALUE_TAG("c");

//! \brief Manages persistence of bucket sums.
struct SSumSerializer {
    typedef std::vector<CSample> TSampleVec;

    void operator()(const TSampleVec &sample, core::CStatePersistInserter &inserter) const {
        inserter.insertValue(SUM_SAMPLE_TAG,
                             core::CPersistUtils::toString(sample, CSample::SToString()));
    }
    bool operator()(TSampleVec &sample, core::CStateRestoreTraverser &traverser) const {
        if (   traverser.name() != SUM_SAMPLE_TAG ||
               core::CPersistUtils::fromString(traverser.value(),
                                               CSample::SFromString(),
                                               sample) == false) {
            LOG_ERROR("Invalid sample in: " << traverser.value())
            return false;
        }
        return true;
    }
};

//! \brief Manages persistence of influence bucket sums.
struct SInfluencerSumSerializer {
    typedef boost::unordered_map<core::CStoredStringPtr, double> TStoredStringPtrDoubleUMap;
    typedef TStoredStringPtrDoubleUMap::const_iterator TStoredStringPtrDoubleUMapCItr;
    typedef boost::reference_wrapper<const std::string> TStrCRef;
    typedef std::pair<TStrCRef, double> TStrCRefDoublePr;
    typedef std::vector<TStrCRefDoublePr> TStrCRefDoublePrVec;

    void operator()(const TStoredStringPtrDoubleUMap &map, core::CStatePersistInserter &inserter) const {
        TStrCRefDoublePrVec ordered;
        ordered.reserve(map.size());
        for (TStoredStringPtrDoubleUMapCItr i = map.begin(); i != map.end(); ++i) {
            ordered.emplace_back(TStrCRef(*i->first), i->second);
        }
        std::sort(ordered.begin(), ordered.end(), maths::COrderings::SFirstLess());
        for (std::size_t i = 0u; i < ordered.size(); ++i) {
            inserter.insertValue(SUM_MAP_KEY_TAG, ordered[i].first);
            inserter.insertValue(SUM_MAP_VALUE_TAG,
                                 ordered[i].second,
                                 core::CIEEE754::E_SinglePrecision);
        }
    }

    bool operator()(TStoredStringPtrDoubleUMap &map, core::CStateRestoreTraverser &traverser) const {
        std::string key;
        do {
            const std::string &name = traverser.name();
            if (name == SUM_MAP_KEY_TAG) {
                key = traverser.value();
            } else if (name == SUM_MAP_VALUE_TAG) {
                if (core::CStringUtils::stringToType(traverser.value(),
                                                     map[CStringStore::influencers().get(key)]) == false) {
                    LOG_ERROR("Invalid sum in " << traverser.value());
                    return false;
                }
            }
        } while (traverser.next());
        return true;
    }
};

} // unnamed::

CGathererTools::CArrivalTimeGatherer::CArrivalTimeGatherer(void) : m_LastTime(FIRST_TIME) {
}

CGathererTools::TOptionalDouble CGathererTools::CArrivalTimeGatherer::featureData(void) const {
    return maths::CBasicStatistics::count(m_Value) > 0.0 ?
           TOptionalDouble(maths::CBasicStatistics::mean(m_Value)) : TOptionalDouble();
}

void CGathererTools::CArrivalTimeGatherer::startNewBucket(void) {
    m_Value = TAccumulator();
}

void CGathererTools::CArrivalTimeGatherer::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    // Because we always serialize immediately after processing a bucket
    // we will have already used the bucket value and samples so these
    // don't need to be serialized.
    inserter.insertValue(LAST_TIME_TAG, m_LastTime);
}

bool CGathererTools::CArrivalTimeGatherer::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name = traverser.name();
        if (name == LAST_TIME_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_LastTime) == false) {
                LOG_ERROR("Invalid last time in " << traverser.value());
                continue;
            }
        }
    } while (traverser.next());

    return true;
}

uint64_t CGathererTools::CArrivalTimeGatherer::checksum(void) const {
    return maths::CChecksum::calculate(static_cast<uint64_t>(m_LastTime), m_Value);
}

std::string CGathererTools::CArrivalTimeGatherer::print(void) const {
    std::ostringstream o;
    if (maths::CBasicStatistics::count(m_Value) > 0.0) {
        o << maths::CBasicStatistics::mean(m_Value);
    } else {
        o << "-";
    }
    o << " (" << m_LastTime << ")";
    return o.str();
}

const core_t::TTime CGathererTools::CArrivalTimeGatherer::FIRST_TIME(std::numeric_limits<core_t::TTime>::min());

CGathererTools::CSumGatherer::CSumGatherer(const SModelParams &params,
                                           std::size_t /*dimension*/,
                                           core_t::TTime startTime,
                                           core_t::TTime bucketLength,
                                           TStrVecCItr beginInfluencers,
                                           TStrVecCItr endInfluencers) :
    m_Classifier(),
    m_BucketSums(params.s_LatencyBuckets, bucketLength, startTime),
    m_InfluencerBucketSums(std::distance(beginInfluencers, endInfluencers),
                           TStoredStringPtrDoubleUMapQueue(params.s_LatencyBuckets + 3,
                                                           bucketLength,
                                                           startTime,
                                                           TStoredStringPtrDoubleUMap(1))) {
}

std::size_t CGathererTools::CSumGatherer::dimension(void) const {
    return 1;
}

SMetricFeatureData CGathererTools::CSumGatherer::featureData(core_t::TTime time, core_t::TTime /*bucketLength*/,
                                                             const TSampleVec &emptySample) const {
    typedef boost::reference_wrapper<const std::string> TStrCRef;
    typedef std::pair<TDouble1Vec, double> TDouble1VecDoublePr;
    typedef std::pair<TStrCRef, TDouble1VecDoublePr> TStrCRefDouble1VecDoublePrPr;
    typedef std::vector<TStrCRefDouble1VecDoublePrPr> TStrCRefDouble1VecDoublePrPrVec;
    typedef std::vector<TStrCRefDouble1VecDoublePrPrVec> TStrCRefDouble1VecDoublePrPrVecVec;

    const TSampleVec *sum = &m_BucketSums.get(time);
    if (sum->empty()) {
        sum = &emptySample;
    }
    TStrCRefDouble1VecDoublePrPrVecVec influenceValues(m_InfluencerBucketSums.size());
    for (std::size_t i = 0u; i < m_InfluencerBucketSums.size(); ++i) {
        const TStoredStringPtrDoubleUMap &influencerStats = m_InfluencerBucketSums[i].get(time);
        influenceValues[i].reserve(influencerStats.size());
        for (const auto &stat : influencerStats) {
            influenceValues[i].emplace_back(TStrCRef(*stat.first),
                                            TDouble1VecDoublePr(TDouble1Vec{stat.second}, 1.0));

        }
    }

    if (!sum->empty()) {
        return {(*sum)[0].time(),
                (*sum)[0].value(),
                (*sum)[0].varianceScale(),
                (*sum)[0].count(),
                influenceValues,
                m_Classifier.isInteger() &&
                maths::CIntegerTools::isInteger(((*sum)[0].value())[0]),
                m_Classifier.isNonNegative(), *sum};
    }
    return {m_Classifier.isInteger(), m_Classifier.isNonNegative(), *sum};
}

bool CGathererTools::CSumGatherer::sample(core_t::TTime /*time*/, unsigned int /*sampleCount*/) {
    return false;
}

void CGathererTools::CSumGatherer::startNewBucket(core_t::TTime time) {
    TSampleVec &sum = m_BucketSums.earliest();
    if (!sum.empty()) {
        m_Classifier.add(model_t::E_IndividualSumByBucketAndPerson, sum[0].value(), 1);
    }
    m_BucketSums.push(TSampleVec(), time);
    for (std::size_t i = 0u; i < m_InfluencerBucketSums.size(); ++i) {
        m_InfluencerBucketSums[i].push(TStoredStringPtrDoubleUMap(1), time);
    }
}

void CGathererTools::CSumGatherer::resetBucket(core_t::TTime bucketStart) {
    m_BucketSums.get(bucketStart).clear();
    for (std::size_t i = 0u; i < m_InfluencerBucketSums.size(); ++i) {
        m_InfluencerBucketSums[i].get(bucketStart).clear();
    }
}

void CGathererTools::CSumGatherer::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    inserter.insertLevel(CLASSIFIER_TAG,
                         boost::bind(&CDataClassifier::acceptPersistInserter,
                                     &m_Classifier,
                                     _1));
    if (m_BucketSums.size() > 0) {
        inserter.insertLevel(BUCKET_SUM_QUEUE_TAG,
                             boost::bind<void>(TSampleVecQueue::CSerializer<SSumSerializer>(),
                                               boost::cref(m_BucketSums),
                                               _1));
    }
    for (std::size_t i = 0u; i < m_InfluencerBucketSums.size(); ++i) {
        inserter.insertLevel(INFLUENCER_BUCKET_SUM_QUEUE_TAG,
                             boost::bind<void>(TStoredStringPtrDoubleUMapQueue::CSerializer<SInfluencerSumSerializer>(),
                                               boost::cref(m_InfluencerBucketSums[i]),
                                               _1));
    }
}

bool CGathererTools::CSumGatherer::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    std::size_t i = 0u;
    do {
        const std::string &name = traverser.name();
        if (name == CLASSIFIER_TAG) {
            if (traverser.traverseSubLevel(boost::bind(&CDataClassifier::acceptRestoreTraverser,
                                                       &m_Classifier,
                                                       _1)) == false) {
                LOG_ERROR("Invalid classifier in " << traverser.value());
                continue;
            }
        } else if (name == BUCKET_SUM_QUEUE_TAG) {
            if (traverser.traverseSubLevel(
                    boost::bind<bool>(TSampleVecQueue::CSerializer<SSumSerializer>(),
                                      boost::ref(m_BucketSums),
                                      _1)) == false) {
                LOG_ERROR("Invalid bucket queue in " << traverser.value());
                return false;
            }
        } else if (name == INFLUENCER_BUCKET_SUM_QUEUE_TAG) {
            if (   i < m_InfluencerBucketSums.size() &&
                   traverser.traverseSubLevel(
                       boost::bind<bool>(TStoredStringPtrDoubleUMapQueue::CSerializer<SInfluencerSumSerializer>(TStoredStringPtrDoubleUMap(1)),
                                         boost::ref(m_InfluencerBucketSums[i++]),
                                         _1)) == false) {
                LOG_ERROR("Invalid bucket queue in " << traverser.value());
                return false;
            }
        }
    } while (traverser.next());

    return true;
}

uint64_t CGathererTools::CSumGatherer::checksum(void) const {
    uint64_t seed = static_cast<uint64_t>(m_Classifier.isInteger());
    seed = maths::CChecksum::calculate(seed, m_Classifier.isNonNegative());
    seed = maths::CChecksum::calculate(seed, m_BucketSums);
    return maths::CChecksum::calculate(seed, m_InfluencerBucketSums);
}

void CGathererTools::CSumGatherer::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CSumGatherer");
    core::CMemoryDebug::dynamicSize("m_BucketSums", m_BucketSums, mem);
    core::CMemoryDebug::dynamicSize("m_InfluencerBucketSums", m_InfluencerBucketSums, mem);
}

std::size_t CGathererTools::CSumGatherer::memoryUsage(void) const {
    return  core::CMemory::dynamicSize(m_BucketSums)
            + core::CMemory::dynamicSize(m_InfluencerBucketSums);
}

std::string CGathererTools::CSumGatherer::print(void) const {
    std::ostringstream result;
    result << m_Classifier.isInteger()
           << ' ' << m_BucketSums.print()
           << ' ' << core::CContainerPrinter::print(m_InfluencerBucketSums);
    return result.str();
}

bool CGathererTools::CSumGatherer::isRedundant(core_t::TTime /*samplingCutoffTime*/) const {
    for (const auto &bucket : m_BucketSums) {
        if (bucket.empty() == false) {
            return false;
        }
    }
    return true;
}

}
}
