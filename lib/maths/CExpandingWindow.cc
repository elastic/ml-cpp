/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CExpandingWindow.h>

#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CompressUtils.h>
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
const core::TPersistenceTag BUCKET_LENGTH_INDEX_TAG{"a", "bucket_length_index"};
const core::TPersistenceTag BUCKET_VALUES_TAG{"b", "bucket_values"};
const core::TPersistenceTag START_TIME_TAG{"c", "start_time"};
const core::TPersistenceTag MEAN_OFFSET_TAG{"d", "mean_offset"};
const std::size_t MAX_BUFFER_SIZE{5};
}

CExpandingWindow::CExpandingWindow(core_t::TTime bucketLength,
                                   TTimeCRng bucketLengths,
                                   std::size_t size,
                                   double decayRate,
                                   bool deflate)
    : m_Deflate{deflate}, m_DecayRate{decayRate}, m_Size{size},
      m_DataBucketLength{bucketLength}, m_BucketLengths{bucketLengths},
      m_BucketLengthIndex{0}, m_StartTime{boost::numeric::bounds<core_t::TTime>::lowest()},
      m_BufferedTimeToPropagate(0.0), m_BucketValues(size % 2 == 0 ? size : size + 1) {
    m_BufferedValues.reserve(MAX_BUFFER_SIZE);
    this->deflate(true);
}

bool CExpandingWindow::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    m_BucketValues.clear();
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(BUCKET_LENGTH_INDEX_TAG, m_BucketLengthIndex)
        RESTORE_BUILT_IN(START_TIME_TAG, m_StartTime)
        RESTORE(BUCKET_VALUES_TAG,
                core::CPersistUtils::restore(BUCKET_VALUES_TAG, m_BucketValues, traverser));
        RESTORE(MEAN_OFFSET_TAG, m_MeanOffset.fromDelimited(traverser.value()))
    } while (traverser.next());
    this->deflate(true);
    return true;
}

void CExpandingWindow::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    CScopeInflate inflate(*this, false);
    inserter.insertValue(BUCKET_LENGTH_INDEX_TAG, m_BucketLengthIndex);
    inserter.insertValue(START_TIME_TAG, m_StartTime);
    core::CPersistUtils::persist(BUCKET_VALUES_TAG, m_BucketValues, inserter);
    inserter.insertValue(MEAN_OFFSET_TAG, m_MeanOffset.toDelimited());
}

core_t::TTime CExpandingWindow::bucketLength() const {
    return m_BucketLengths[m_BucketLengthIndex];
}

std::size_t CExpandingWindow::size() const {
    return m_Size;
}

core_t::TTime CExpandingWindow::dataPointsTimeOffsetInBucket() const {
    return static_cast<core_t::TTime>(CBasicStatistics::mean(m_MeanOffset) + 0.5);
}

core_t::TTime CExpandingWindow::beginValuesTime() const {
    return this->dataPointsTimeOffsetInBucket() +
           (CIntegerTools::ceil(m_StartTime, m_DataBucketLength) +
            CIntegerTools::floor(m_StartTime + this->bucketLength() - 1, m_DataBucketLength)) /
               2;
}

core_t::TTime CExpandingWindow::endValuesTime() const {
    return this->beginValuesTime() +
           static_cast<core_t::TTime>(m_Size) * this->bucketLength();
}

CExpandingWindow::TFloatMeanAccumulatorVec CExpandingWindow::values() const {
    CScopeInflate inflate(*this, false);
    return m_BucketValues;
}

CExpandingWindow::TFloatMeanAccumulatorVec
CExpandingWindow::valuesMinusPrediction(const TPredictor& predictor) const {
    CScopeInflate inflate(*this, false);
    return this->valuesMinusPrediction(m_BucketValues, predictor);
}

CExpandingWindow::TFloatMeanAccumulatorVec
CExpandingWindow::valuesMinusPrediction(TFloatMeanAccumulatorVec bucketValues,
                                        const TPredictor& predictor) const {

    core_t::TTime start{CIntegerTools::ceil(m_StartTime, m_DataBucketLength)};
    core_t::TTime size{static_cast<core_t::TTime>(bucketValues.size())};
    core_t::TTime offset{this->dataPointsTimeOffsetInBucket()};

    for (core_t::TTime i = 0; i < size; ++i) {
        if (CBasicStatistics::count(bucketValues[i]) > 0.0) {
            core_t::TTime bucketStart{start + offset + i * this->bucketLength()};
            core_t::TTime bucketEnd{bucketStart + this->bucketLength()};
            TFloatMeanAccumulator prediction;
            for (core_t::TTime time = bucketStart; time < bucketEnd; time += m_DataBucketLength) {
                prediction.add(predictor(time));
            }
            CBasicStatistics::moment<0>(bucketValues[i]) -= CBasicStatistics::mean(prediction);
        }
    }

    return bucketValues;
}

void CExpandingWindow::initialize(core_t::TTime time) {
    m_StartTime = CIntegerTools::floor(time, m_BucketLengths[0]);
}

void CExpandingWindow::shiftTime(core_t::TTime dt) {
    m_StartTime += dt;
}

void CExpandingWindow::propagateForwardsByTime(double time) {
    if (CMathsFuncs::isFinite(time) == false || time < 0.0) {
        LOG_ERROR(<< "Bad propagation time " << time);
        return;
    }
    double factor{std::exp(-m_DecayRate * time)};
    for (auto& value : m_BufferedValues) {
        value.second.age(factor);
    }
    m_BufferedTimeToPropagate += time;
}

void CExpandingWindow::add(core_t::TTime time, double value, double weight) {
    if (time >= m_StartTime) {
        if (this->needToCompress(time)) {
            CScopeInflate inflate(*this, true);
            do {
                m_BucketLengthIndex = (m_BucketLengthIndex + 1) %
                                      m_BucketLengths.size();
                auto end = m_BucketValues.begin();

                if (m_BucketLengthIndex == 0) {
                    this->initialize(time);
                } else {
                    std::size_t compression(m_BucketLengths[m_BucketLengthIndex] /
                                            m_BucketLengths[m_BucketLengthIndex - 1]);
                    for (std::size_t i = 0u; i < m_BucketValues.size();
                         i += compression, ++end) {
                        std::swap(*end, m_BucketValues[i]);
                        for (std::size_t j = 1u;
                             j < compression && i + j < m_BucketValues.size(); ++j) {
                            *end += m_BucketValues[i + j];
                        }
                    }
                }
                std::fill(end, m_BucketValues.end(), TFloatMeanAccumulator());
            } while (this->needToCompress(time));
        }

        std::size_t index((time - m_StartTime) / m_BucketLengths[m_BucketLengthIndex]);
        if (m_Deflate == false) {
            m_BucketValues[index].add(value, weight);
        } else {
            if (m_BufferedValues.empty() || index != m_BufferedValues.back().first) {
                if (m_BufferedValues.size() == MAX_BUFFER_SIZE) {
                    CScopeInflate inflate(*this, true);
                }
                m_BufferedValues.push_back({index, TFloatMeanAccumulator{}});
            }
            m_BufferedValues.back().second.add(value, weight);
        }
        m_MeanOffset.add(static_cast<double>(time % m_DataBucketLength));
    }
}

bool CExpandingWindow::needToCompress(core_t::TTime time) const {
    return time >= this->endTime();
}

uint64_t CExpandingWindow::checksum(uint64_t seed) const {
    CScopeInflate inflate(*this, false);
    seed = CChecksum::calculate(seed, m_BucketLengthIndex);
    seed = CChecksum::calculate(seed, m_StartTime);
    seed = CChecksum::calculate(seed, m_BucketValues);
    return CChecksum::calculate(seed, m_MeanOffset);
}

void CExpandingWindow::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CExpandingWindow");
    core::CMemoryDebug::dynamicSize("m_BucketValues", m_BucketValues, mem);
    core::CMemoryDebug::dynamicSize("m_DeflatedBucketValues", m_DeflatedBucketValues, mem);
}

std::size_t CExpandingWindow::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_BucketValues)};
    mem += core::CMemory::dynamicSize(m_DeflatedBucketValues);
    return mem;
}

core_t::TTime CExpandingWindow::endTime() const {
    return m_StartTime + static_cast<core_t::TTime>(m_Size) * this->bucketLength();
}

void CExpandingWindow::deflate(bool commit) const {
    if (m_Deflate && m_BucketValues.size() > 0) {
        const_cast<CExpandingWindow*>(this)->doDeflate(commit);
    }
}

void CExpandingWindow::doDeflate(bool commit) {
    if (commit) {
        bool lengthOnly{false};
        core::CDeflator compressor(lengthOnly);
        compressor.addVector(m_BucketValues);
        compressor.finishAndTakeData(m_DeflatedBucketValues);
    }
    m_BucketValues.clear();
    m_BucketValues.shrink_to_fit();
}

void CExpandingWindow::inflate(bool commit) const {
    if (m_Deflate && m_BucketValues.empty()) {
        const_cast<CExpandingWindow*>(this)->doInflate(commit);
    }
}

void CExpandingWindow::doInflate(bool commit) {
    bool lengthOnly{false};
    core::CInflator decompressor(lengthOnly);
    decompressor.addVector(m_DeflatedBucketValues);
    TByteVec inflated;
    decompressor.finishAndTakeData(inflated);
    m_BucketValues.resize(inflated.size() / sizeof(TFloatMeanAccumulator));
    std::copy(inflated.begin(), inflated.end(),
              reinterpret_cast<TByte*>(m_BucketValues.data()));
    double factor{std::exp(-m_DecayRate * m_BufferedTimeToPropagate)};
    for (auto& value : m_BucketValues) {
        value.age(factor);
    }
    for (auto& value : m_BufferedValues) {
        m_BucketValues[value.first] += value.second;
    }
    if (commit) {
        m_BufferedValues.clear();
        m_BufferedTimeToPropagate = 0.0;
    }
}

CExpandingWindow::CScopeInflate::CScopeInflate(const CExpandingWindow& window, bool commit)
    : m_Window{window}, m_Commit{commit} {
    m_Window.inflate(commit);
}

CExpandingWindow::CScopeInflate::~CScopeInflate() {
    m_Window.deflate(m_Commit);
}
}
}
