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

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CIntegerTools.h>
#include <maths/CTools.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CInterimBucketCorrector.h>

#include <cmath>

namespace ml
{
namespace model
{
namespace
{
const std::size_t COMPONENT_SIZE(24);
const std::string COUNT_TREND_TAG("a");
const std::string COUNT_MEAN_TAG("b");

double meanDecayRate(core_t::TTime bucketLength)
{
    return  CAnomalyDetectorModelConfig::DEFAULT_DECAY_RATE
          * CAnomalyDetectorModelConfig::bucketNormalizationFactor(bucketLength);
}

double trendDecayRate(core_t::TTime bucketLength)
{
    return CAnomalyDetectorModelConfig::trendDecayRate(meanDecayRate(bucketLength), bucketLength);
}
}

CInterimBucketCorrector::CInterimBucketCorrector(core_t::TTime bucketLength)
        : m_BucketLength(bucketLength),
          m_CountTrend(trendDecayRate(bucketLength), bucketLength, COMPONENT_SIZE)
{
}

CInterimBucketCorrector::CInterimBucketCorrector(const CInterimBucketCorrector &other)
        : m_BucketLength(other.m_BucketLength),
          m_CountTrend(other.m_CountTrend),
          m_CountMean(other.m_CountMean)
{
}

core_t::TTime CInterimBucketCorrector::calcBucketMidPoint(core_t::TTime time) const
{
    return maths::CIntegerTools::floor(time, m_BucketLength) + m_BucketLength / 2;
}

void CInterimBucketCorrector::update(core_t::TTime time, std::size_t bucketCount)
{
    core_t::TTime bucketMidPoint = this->calcBucketMidPoint(time);

    m_CountTrend.addPoint(bucketMidPoint, static_cast<double>(bucketCount));

    double alpha = std::exp(-meanDecayRate(m_BucketLength));
    m_CountMean.age(alpha);
    m_CountMean.add(bucketCount);
}

double CInterimBucketCorrector::estimateBucketCompleteness(core_t::TTime time,
                                                           std::size_t currentCount) const
{
    core_t::TTime bucketMidPoint = this->calcBucketMidPoint(time);
    double bucketCount = m_CountTrend.initialized() ?
                         maths::CBasicStatistics::mean(m_CountTrend.value(bucketMidPoint)) :
                         maths::CBasicStatistics::mean(m_CountMean);
    return bucketCount > 0.0 ?
           maths::CTools::truncate(  static_cast<double>(currentCount)
                                   / bucketCount, 0.0, 1.0) : 1.0;
}

double CInterimBucketCorrector::corrections(core_t::TTime time,
                                            std::size_t currentCount,
                                            double mode,
                                            double value) const
{
    double correction = (1.0 - this->estimateBucketCompleteness(time, currentCount)) * mode;
    return maths::CTools::truncate(mode - value,
                                   std::min(0.0, correction),
                                   std::max(0.0, correction));
}

CInterimBucketCorrector::TDouble10Vec
    CInterimBucketCorrector::corrections(core_t::TTime time,
                                         std::size_t currentCount,
                                         const TDouble10Vec &modes,
                                         const TDouble10Vec &values) const
{
    TDouble10Vec corrections(values.size(), 0.0);
    double incompleteBucketFraction = 1.0 - this->estimateBucketCompleteness(time, currentCount);
    double correction = 0.0;
    for (std::size_t i = 0; i < corrections.size(); ++i)
    {
        correction = incompleteBucketFraction * modes[i];
        corrections[i] = maths::CTools::truncate(modes[i] - values[i],
                                                 std::min(0.0, correction),
                                                 std::max(0.0, correction));
    }
    return corrections;
}

void CInterimBucketCorrector::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    mem->setName("CInterimBucketCorrector");
    core::CMemoryDebug::dynamicSize("m_CountTrend", m_CountTrend, mem);
}

std::size_t CInterimBucketCorrector::memoryUsage(void) const
{
    return core::CMemory::dynamicSize(m_CountTrend);
}

void CInterimBucketCorrector::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(COUNT_MEAN_TAG, m_CountMean.toDelimited());
    core::CPersistUtils::persist(COUNT_TREND_TAG, m_CountTrend, inserter);
}

bool CInterimBucketCorrector::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name = traverser.name();
        RESTORE_NO_ERROR(COUNT_TREND_TAG,
                         maths::CTimeSeriesDecomposition restored(trendDecayRate(m_BucketLength),
                                                                  m_BucketLength, COMPONENT_SIZE,
                                                                  traverser);
                         m_CountTrend.swap(restored))
        RESTORE(COUNT_MEAN_TAG, m_CountMean.fromDelimited(traverser.value()))
    }
    while (traverser.next());
    return true;
}
}
}
