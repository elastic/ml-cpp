/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CIntegerTools.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CInterimBucketCorrector.h>

#include <cmath>

namespace ml {
namespace model {
namespace {
const std::size_t COMPONENT_SIZE(24);
const std::string COMPLETENESS_TAG{"a"};
const std::string FINAL_COUNT_TREND_TAG{"b"};
const std::string FINAL_COUNT_MEAN_TAG{"c"};

double decayRate(core_t::TTime bucketLength) {
    return CAnomalyDetectorModelConfig::DEFAULT_DECAY_RATE *
           CAnomalyDetectorModelConfig::bucketNormalizationFactor(bucketLength);
}

double trendDecayRate(core_t::TTime bucketLength) {
    return CAnomalyDetectorModelConfig::trendDecayRate(decayRate(bucketLength), bucketLength);
}
}

CInterimBucketCorrector::CInterimBucketCorrector(core_t::TTime bucketLength)
    : m_BucketLength(bucketLength), m_Completeness{0.0},
      m_FinalCountTrend(trendDecayRate(bucketLength), bucketLength, COMPONENT_SIZE) {
}

void CInterimBucketCorrector::currentBucketCount(core_t::TTime time, uint64_t count) {
    m_Completeness = this->estimateBucketCompleteness(time, count);
}

void CInterimBucketCorrector::finalBucketCount(core_t::TTime time, uint64_t count) {
    core_t::TTime bucketMidPoint{this->calcBucketMidPoint(time)};
    m_Completeness = 1.0;
    m_FinalCountTrend.addPoint(bucketMidPoint, static_cast<double>(count));
    m_FinalCountMean.age(std::exp(-decayRate(m_BucketLength)));
    m_FinalCountMean.add(static_cast<double>(count));
}

double CInterimBucketCorrector::completeness() const {
    return m_Completeness;
}

double CInterimBucketCorrector::corrections(double mode, double value) const {
    double correction{(1.0 - m_Completeness) * mode};
    return maths::CTools::truncate(mode - value, std::min(0.0, correction),
                                   std::max(0.0, correction));
}

CInterimBucketCorrector::TDouble10Vec
CInterimBucketCorrector::corrections(const TDouble10Vec& modes,
                                     const TDouble10Vec& values) const {
    TDouble10Vec corrections(values.size(), 0.0);
    double incompleteBucketFraction{1.0 - m_Completeness};
    double correction{0.0};
    for (std::size_t i = 0; i < corrections.size(); ++i) {
        correction = incompleteBucketFraction * modes[i];
        corrections[i] = maths::CTools::truncate(
            modes[i] - values[i], std::min(0.0, correction), std::max(0.0, correction));
    }
    return corrections;
}

void CInterimBucketCorrector::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CInterimBucketCorrector");
    core::CMemoryDebug::dynamicSize("m_CountTrend", m_FinalCountTrend, mem);
}

std::size_t CInterimBucketCorrector::memoryUsage() const {
    return core::CMemory::dynamicSize(m_FinalCountTrend);
}

void CInterimBucketCorrector::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(COMPLETENESS_TAG, m_Completeness, core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(FINAL_COUNT_MEAN_TAG, m_FinalCountMean.toDelimited());
    core::CPersistUtils::persist(FINAL_COUNT_TREND_TAG, m_FinalCountTrend, inserter);
}

bool CInterimBucketCorrector::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(COMPLETENESS_TAG, m_Completeness)
        if (name == FINAL_COUNT_TREND_TAG) {
            maths::SDistributionRestoreParams changeModelParams{
                maths_t::E_ContinuousData, decayRate(m_BucketLength)};
            maths::STimeSeriesDecompositionRestoreParams params{
                trendDecayRate(m_BucketLength), m_BucketLength, COMPONENT_SIZE, changeModelParams};
            maths::CTimeSeriesDecomposition restored(params, traverser);
            m_FinalCountTrend.swap(restored);
            continue;
        }
        RESTORE(FINAL_COUNT_MEAN_TAG, m_FinalCountMean.fromDelimited(traverser.value()))
    } while (traverser.next());
    return true;
}

core_t::TTime CInterimBucketCorrector::calcBucketMidPoint(core_t::TTime time) const {
    return maths::CIntegerTools::floor(time, m_BucketLength) + m_BucketLength / 2;
}

double CInterimBucketCorrector::estimateBucketCompleteness(core_t::TTime time,
                                                           std::uint64_t count_) const {
    double count{static_cast<double>(count_)};
    core_t::TTime bucketMidPoint{this->calcBucketMidPoint(time)};
    double bucketCount{m_FinalCountTrend.initialized()
                           ? maths::CBasicStatistics::mean(m_FinalCountTrend.value(bucketMidPoint))
                           : maths::CBasicStatistics::mean(m_FinalCountMean)};
    return bucketCount > 0.0 ? maths::CTools::truncate(count / bucketCount, 0.0, 1.0) : 1.0;
}
}
}
