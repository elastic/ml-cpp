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

#include <model/CSampleCounts.h>

#include <core/CMemory.h>
#include <core/CStringUtils.h>

#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/Constants.h>
#include <maths/COrderings.h>

#include <model/CDataGatherer.h>

#include <map>

#include <boost/ref.hpp>

namespace ml {
namespace model {

namespace {
const std::string SAMPLE_COUNT_TAG("b");
const std::string MEAN_NON_ZERO_BUCKET_COUNT_TAG("c");
const std::string EFFECTIVE_SAMPLE_VARIANCE_TAG("d");

const double NUMBER_BUCKETS_TO_ESTIMATE_SAMPLE_COUNT(3.0);
const double NUMBER_BUCKETS_TO_REFRESH_SAMPLE_COUNT(30.0);

using TStrCRef = boost::reference_wrapper<const std::string>;
using TStrCRefUInt64Map = std::map<TStrCRef, uint64_t, maths::COrderings::SReferenceLess>;
}

CSampleCounts::CSampleCounts(unsigned int sampleCountOverride) :
    m_SampleCountOverride(sampleCountOverride) {
}

CSampleCounts::CSampleCounts(bool isForPersistence,
                             const CSampleCounts &other) :
    m_SampleCountOverride(other.m_SampleCountOverride),
    m_SampleCounts(other.m_SampleCounts),
    m_MeanNonZeroBucketCounts(other.m_MeanNonZeroBucketCounts),
    m_EffectiveSampleVariances(other.m_EffectiveSampleVariances) {
    if (!isForPersistence) {
        LOG_ABORT("This constructor only creates clones for persistence");
    }
}

CSampleCounts *CSampleCounts::cloneForPersistence(void) const {
    return new CSampleCounts(true, *this);
}

void CSampleCounts::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    // Note m_SampleCountOverride is only for unit tests at present,
    // hence not persisted or restored.

    core::CPersistUtils::persist(SAMPLE_COUNT_TAG, m_SampleCounts, inserter);
    core::CPersistUtils::persist(MEAN_NON_ZERO_BUCKET_COUNT_TAG, m_MeanNonZeroBucketCounts, inserter);
    core::CPersistUtils::persist(EFFECTIVE_SAMPLE_VARIANCE_TAG, m_EffectiveSampleVariances, inserter);
}

bool CSampleCounts::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name = traverser.name();
        if (name == SAMPLE_COUNT_TAG) {
            if (core::CPersistUtils::restore(name, m_SampleCounts, traverser) == false) {
                LOG_ERROR("Invalid sample counts");
                return false;
            }
        } else if (name == MEAN_NON_ZERO_BUCKET_COUNT_TAG) {
            if (core::CPersistUtils::restore(name, m_MeanNonZeroBucketCounts, traverser) == false) {
                LOG_ERROR("Invalid non-zero bucket count means");
                return false;
            }
        } else if (name == EFFECTIVE_SAMPLE_VARIANCE_TAG) {
            if (core::CPersistUtils::restore(name, m_EffectiveSampleVariances, traverser) == false) {
                LOG_ERROR("Invalid effective sample variances");
                return false;
            }
        }
    } while (traverser.next());
    return true;
}

unsigned int CSampleCounts::count(std::size_t id) const {
    return m_SampleCountOverride > 0 ?
           m_SampleCountOverride :
           id < m_SampleCounts.size() ? m_SampleCounts[id] : 0;
}

double CSampleCounts::effectiveSampleCount(std::size_t id) const {
    if (id < m_EffectiveSampleVariances.size()) {
        // This uses the fact that variance ~ 1 / count.
        double count = maths::CBasicStatistics::count(m_EffectiveSampleVariances[id]);
        double mean  = maths::CBasicStatistics::mean(m_EffectiveSampleVariances[id]);
        return count > 0.0 ? 1.0 / mean : this->count(id);
    }
    return 0.0;
}

void CSampleCounts::resetSampleCount(const CDataGatherer &gatherer,
                                     std::size_t id) {
    if (m_SampleCountOverride > 0) {
        return;
    }

    if (id >= m_MeanNonZeroBucketCounts.size()) {
        LOG_ERROR("Bad identifier " << id);
        return;
    }

    const TMeanAccumulator &count_ = m_MeanNonZeroBucketCounts[id];
    if (maths::CBasicStatistics::count(count_)
        >= NUMBER_BUCKETS_TO_ESTIMATE_SAMPLE_COUNT) {
        unsigned                         sampleCountThreshold = 0;
        const CDataGatherer::TFeatureVec &features = gatherer.features();
        for (CDataGatherer::TFeatureVecCItr i = features.begin(); i != features.end(); ++i) {
            sampleCountThreshold = std::max(sampleCountThreshold,
                                            model_t::minimumSampleCount(*i));
        }
        double count = maths::CBasicStatistics::mean(count_);
        m_SampleCounts[id] = std::max(sampleCountThreshold,
                                      static_cast<unsigned int>(count + 0.5));
        LOG_DEBUG("Setting sample count to " << m_SampleCounts[id]
                                             << " for " << this->name(gatherer, id));
    }
}

void CSampleCounts::refresh(const CDataGatherer &gatherer) {
    if (m_SampleCountOverride > 0) {
        return;
    }

    unsigned                         sampleCountThreshold = 0;
    const CDataGatherer::TFeatureVec &features = gatherer.features();
    for (CDataGatherer::TFeatureVecCItr i = features.begin(); i != features.end(); ++i) {
        sampleCountThreshold = std::max(sampleCountThreshold,
                                        model_t::minimumSampleCount(*i));
    }

    for (std::size_t id = 0u; id < m_MeanNonZeroBucketCounts.size(); ++id) {
        const TMeanAccumulator &count_ = m_MeanNonZeroBucketCounts[id];
        if (m_SampleCounts[id] > 0) {
            if (maths::CBasicStatistics::count(count_)
                >= NUMBER_BUCKETS_TO_REFRESH_SAMPLE_COUNT) {
                double count = maths::CBasicStatistics::mean(count_);
                double scale = count / static_cast<double>(m_SampleCounts[id]);
                if (   scale < maths::MINIMUM_ACCURATE_VARIANCE_SCALE ||
                       scale > maths::MAXIMUM_ACCURATE_VARIANCE_SCALE) {
                    unsigned int oldCount = m_SampleCounts[id];
                    unsigned int newCount = std::max(sampleCountThreshold,
                                                     static_cast<unsigned int>(count + 0.5));
                    LOG_TRACE("Sample count " << oldCount
                                              << " is too far from the bucket mean " << count
                                              << " count, resetting to " << newCount
                                              << ". This may cause temporary instability"
                                              << " for " << this->name(gatherer, id) << " (" << id
                                              << "). (Mean count " << count_ << ")");
                    m_SampleCounts[id] = newCount;
                    // Avoid compiler warning in the case of LOG_TRACE being compiled out
                    static_cast<void>(oldCount);
                }
            }
        } else if (maths::CBasicStatistics::count(count_)
                   >= NUMBER_BUCKETS_TO_ESTIMATE_SAMPLE_COUNT) {
            double count = maths::CBasicStatistics::mean(count_);
            m_SampleCounts[id] = std::max(sampleCountThreshold,
                                          static_cast<unsigned int>(count + 0.5));
            LOG_TRACE("Setting sample count to " << m_SampleCounts[id]
                                                 << " for " << this->name(gatherer, id) << " (" << id
                                                 << "). (Mean count " << count_ << ")");
        }
    }
}

void CSampleCounts::updateSampleVariance(std::size_t id) {
    m_EffectiveSampleVariances[id].add(1.0 / static_cast<double>(this->count(id)));
}

void CSampleCounts::updateMeanNonZeroBucketCount(std::size_t id,
                                                 double count,
                                                 double alpha) {
    m_MeanNonZeroBucketCounts[id].add(count);
    m_MeanNonZeroBucketCounts[id].age(alpha);
    m_EffectiveSampleVariances[id].age(alpha);
}

void CSampleCounts::recycle(const TSizeVec &idsToRemove) {
    for (std::size_t i = 0u; i < idsToRemove.size(); ++i) {
        std::size_t id = idsToRemove[i];
        if (id >= m_SampleCounts.size()) {
            continue;
        }
        m_SampleCounts[id] = 0;
        m_MeanNonZeroBucketCounts[id] = TMeanAccumulator();
        m_EffectiveSampleVariances[id] = TMeanAccumulator();
    }
    LOG_TRACE("m_SampleCounts = " << core::CContainerPrinter::print(m_SampleCounts));
    LOG_TRACE("m_MeanNonZeroBucketCounts = "
              << core::CContainerPrinter::print(m_MeanNonZeroBucketCounts));
    LOG_TRACE("m_EffectiveSampleVariances = "
              << core::CContainerPrinter::print(m_EffectiveSampleVariances));
}

void CSampleCounts::remove(std::size_t lowestIdToRemove) {
    if (lowestIdToRemove < m_SampleCounts.size()) {
        m_SampleCounts.erase(m_SampleCounts.begin() + lowestIdToRemove,
                             m_SampleCounts.end());
        m_MeanNonZeroBucketCounts.erase(  m_MeanNonZeroBucketCounts.begin()
                                          + lowestIdToRemove,
                                          m_MeanNonZeroBucketCounts.end());
        m_EffectiveSampleVariances.erase(  m_EffectiveSampleVariances.begin()
                                           + lowestIdToRemove,
                                           m_EffectiveSampleVariances.end());
        LOG_TRACE("m_SampleCounts = " << core::CContainerPrinter::print(m_SampleCounts));
        LOG_TRACE("m_MeanNonZeroBucketCounts = "
                  << core::CContainerPrinter::print(m_MeanNonZeroBucketCounts));
        LOG_TRACE("m_EffectiveSampleVariances = "
                  << core::CContainerPrinter::print(m_EffectiveSampleVariances));
    }
}

void CSampleCounts::resize(std::size_t id) {
    if (id >= m_SampleCounts.size()) {
        m_SampleCounts.resize(id + 1);
        m_MeanNonZeroBucketCounts.resize(id + 1);
        m_EffectiveSampleVariances.resize(id + 1);
    }
}

uint64_t CSampleCounts::checksum(const CDataGatherer &gatherer) const {
    TStrCRefUInt64Map hashes;
    for (std::size_t id = 0u; id < m_SampleCounts.size(); ++id) {
        if (gatherer.isPopulation() ?
            gatherer.isAttributeActive(id) :
            gatherer.isPersonActive(id)) {
            uint64_t &hash = hashes[TStrCRef(this->name(gatherer, id))];
            hash = maths::CChecksum::calculate(hash, m_SampleCounts[id]);
            hash = maths::CChecksum::calculate(hash, m_MeanNonZeroBucketCounts[id]);
            hash = maths::CChecksum::calculate(hash, m_EffectiveSampleVariances[id]);
        }
    }
    LOG_TRACE("hashes = " << core::CContainerPrinter::print(hashes));
    return maths::CChecksum::calculate(0, hashes);
}

void CSampleCounts::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CSampleCounts");
    core::CMemoryDebug::dynamicSize("m_SampleCounts", m_SampleCounts, mem);
    core::CMemoryDebug::dynamicSize("m_MeanNonZeroBucketCounts",
                                    m_MeanNonZeroBucketCounts, mem);
    core::CMemoryDebug::dynamicSize("m_EffectiveSampleVariances",
                                    m_EffectiveSampleVariances, mem);
}

std::size_t CSampleCounts::memoryUsage(void) const {
    std::size_t mem = core::CMemory::dynamicSize(m_SampleCounts);
    mem += core::CMemory::dynamicSize(m_MeanNonZeroBucketCounts);
    mem += core::CMemory::dynamicSize(m_EffectiveSampleVariances);
    return mem;
}

void CSampleCounts::clear(void) {
    m_SampleCounts.clear();
    m_MeanNonZeroBucketCounts.clear();
    m_EffectiveSampleVariances.clear();
}

const std::string &CSampleCounts::name(const CDataGatherer &gatherer,
                                       std::size_t id) const {
    return gatherer.isPopulation() ?
           gatherer.attributeName(id) :
           gatherer.personName(id);
}

} // model
} // ml
