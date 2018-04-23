/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CExpandingWindow.h>

#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CMathsFuncs.h>

#include <algorithm>
#include <cmath>

namespace ml {
namespace maths {
namespace {
const std::string BUCKET_LENGTH_INDEX_TAG("a");
const std::string BUCKET_VALUES_TAG("b");
const std::string START_TIME_TAG("c");
}

CExpandingWindow::CExpandingWindow(core_t::TTime bucketLength,
                                   TTimeCRng bucketLengths,
                                   std::size_t size,
                                   double decayRate)
    : m_DecayRate(decayRate), m_BucketLength(bucketLength),
      m_BucketLengths(bucketLengths), m_BucketLengthIndex(0),
      m_StartTime(boost::numeric::bounds<core_t::TTime>::lowest()),
      m_BucketValues(size % 2 == 0 ? size : size + 1) {
}

bool CExpandingWindow::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    m_BucketValues.clear();
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(BUCKET_LENGTH_INDEX_TAG, m_BucketLengthIndex)
        RESTORE_BUILT_IN(START_TIME_TAG, m_StartTime)
        RESTORE(BUCKET_VALUES_TAG,
                core::CPersistUtils::restore(BUCKET_VALUES_TAG, m_BucketValues, traverser));
    } while (traverser.next());
    return true;
}

void CExpandingWindow::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(BUCKET_LENGTH_INDEX_TAG, m_BucketLengthIndex);
    inserter.insertValue(START_TIME_TAG, m_StartTime);
    core::CPersistUtils::persist(BUCKET_VALUES_TAG, m_BucketValues, inserter);
}

core_t::TTime CExpandingWindow::startTime() const {
    return m_StartTime;
}

core_t::TTime CExpandingWindow::endTime() const {
    return m_StartTime + (static_cast<core_t::TTime>(m_BucketValues.size()) *
                          m_BucketLengths[m_BucketLengthIndex]);
}

core_t::TTime CExpandingWindow::bucketLength() const {
    return m_BucketLengths[m_BucketLengthIndex];
}

const CExpandingWindow::TFloatMeanAccumulatorVec& CExpandingWindow::values() const {
    return m_BucketValues;
}

CExpandingWindow::TFloatMeanAccumulatorVec
CExpandingWindow::valuesMinusPrediction(const TPredictor& predictor) const {
    core_t::TTime start{CIntegerTools::floor(this->startTime(), m_BucketLength)};
    core_t::TTime end{CIntegerTools::ceil(this->endTime(), m_BucketLength)};
    core_t::TTime size{static_cast<core_t::TTime>(m_BucketValues.size())};
    core_t::TTime offset{
        static_cast<core_t::TTime>(CBasicStatistics::mean(m_MeanOffset) + 0.5)};

    TFloatMeanAccumulatorVec predictions(size);
    for (core_t::TTime time = start + offset; time < end; time += m_BucketLength) {
        core_t::TTime bucket{(time - start) / m_BucketLengths[m_BucketLengthIndex]};
        if (bucket >= 0 && bucket < size) {
            predictions[bucket].add(predictor(time));
        }
    }

    TFloatMeanAccumulatorVec result(m_BucketValues);
    for (core_t::TTime i = 0; i < size; ++i) {
        if (CBasicStatistics::count(result[i]) > 0.0) {
            CBasicStatistics::moment<0>(result[i]) -=
                CBasicStatistics::mean(predictions[i]);
        }
    }

    return result;
}

void CExpandingWindow::initialize(core_t::TTime time) {
    m_StartTime = CIntegerTools::floor(time, m_BucketLengths[0]);
}

void CExpandingWindow::propagateForwardsByTime(double time) {
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
        LOG_ERROR(<< "Bad propagation time " << time);
    }
    double factor = std::exp(-m_DecayRate * time);
    for (auto& value : m_BucketValues) {
        value.age(factor);
    }
}

void CExpandingWindow::add(core_t::TTime time, double value, double weight) {
    if (time >= m_StartTime) {
        while (this->needToCompress(time)) {
            m_BucketLengthIndex = (m_BucketLengthIndex + 1) % m_BucketLengths.size();
            auto end = m_BucketValues.begin();

            if (m_BucketLengthIndex == 0) {
                m_StartTime = CIntegerTools::floor(time, m_BucketLengths[0]);
            } else {
                std::size_t compression = m_BucketLengths[m_BucketLengthIndex] /
                                          m_BucketLengths[m_BucketLengthIndex - 1];
                for (std::size_t i = 0u; i < m_BucketValues.size(); i += compression, ++end) {
                    std::swap(*end, m_BucketValues[i]);
                    for (std::size_t j = 1u;
                         j < compression && i + j < m_BucketValues.size(); ++j) {
                        *end += m_BucketValues[i + j];
                    }
                }
            }
            std::fill(end, m_BucketValues.end(), TFloatMeanAccumulator());
        }

        m_BucketValues[(time - m_StartTime) / m_BucketLengths[m_BucketLengthIndex]]
            .add(value, weight);
        m_MeanOffset.add(static_cast<double>(time % m_BucketLength));
    }
}

bool CExpandingWindow::needToCompress(core_t::TTime time) const {
    return time >= this->endTime();
}

uint64_t CExpandingWindow::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_BucketLengthIndex);
    seed = CChecksum::calculate(seed, m_StartTime);
    return CChecksum::calculate(seed, m_BucketValues);
}

void CExpandingWindow::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CScanningPeriodicityTest");
    core::CMemoryDebug::dynamicSize("m_BucketValues", m_BucketValues, mem);
}

std::size_t CExpandingWindow::memoryUsage() const {
    return core::CMemory::dynamicSize(m_BucketValues);
}
}
}
