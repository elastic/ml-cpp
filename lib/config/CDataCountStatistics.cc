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

#include <config/CDataCountStatistics.h>

#include <core/CCompressUtils.h>
#include <core/CContainerPrinter.h>
#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CMaskIterator.h>

#include <maths/CIntegerTools.h>
#include <maths/COrderings.h>
#include <maths/CPRNG.h>
#include <maths/CSampling.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CDetectorRecord.h>
#include <config/CDetectorSpecification.h>
#include <config/CTools.h>

#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <algorithm>

namespace ml {
namespace config {
namespace {

typedef std::vector<bool> TBoolVec;

//! We sample a subset of short bucket lengths buckets for runtime.
TBoolVec bucketSampleMask(core_t::TTime bucketLength) {
    double n = ::ceil(std::max(3600.0 / static_cast<double>(bucketLength), 1.0));
    TBoolVec result(static_cast<std::size_t>(::sqrt(n)), true);
    result.resize(static_cast<std::size_t>(n), false);
    return result;
}

//! Insert with the same semantics as boost::unordered_map/set::emplace.
template <typename T>
std::size_t emplace(const std::string* name, std::vector<std::pair<const std::string*, T>>& stats) {
    std::size_t i = static_cast<std::size_t>(
        std::lower_bound(stats.begin(), stats.end(), name, maths::COrderings::SFirstLess()) -
        stats.begin());
    if (i == stats.size()) {
        stats.push_back(std::make_pair(name, T()));
    } else if (*name != *stats[i].first) {
        stats.insert(stats.begin() + i, std::make_pair(name, T()));
    }
    return i;
}

const core::CHashing::CMurmurHash2String HASHER;

//! \brief A unique key identifying a collection of count statistics.
class CCountStatisticsKey {
public:
    CCountStatisticsKey(const CDetectorSpecification& spec) {
        m_Fields[0] = spec.byField();
        m_Fields[1] = spec.overField();
        m_Fields[2] = spec.partitionField();
    }

    std::size_t hash(void) const {
        std::size_t result = m_Fields[0] ? HASHER(*m_Fields[0]) : 1;
        boost::hash_combine(result, m_Fields[1] ? HASHER(*m_Fields[1]) : 1);
        boost::hash_combine(result, m_Fields[2] ? HASHER(*m_Fields[2]) : 1);
        return result;
    }

    bool operator==(const CCountStatisticsKey& rhs) const {
        return std::equal(boost::begin(m_Fields), boost::end(m_Fields), boost::begin(rhs.m_Fields));
    }

private:
    CDetectorSpecification::TOptionalStr m_Fields[3];
};

//! \brief Hashes a count statistic key.
class CCountStatisticsKeyHasher {
public:
    std::size_t operator()(const CCountStatisticsKey& key) const { return key.hash(); }
};

//! Get some partition statistics on the heap.
CDataCountStatistics* partitionCountStatistics(const CAutoconfigurerParams& params) {
    return new CPartitionDataCountStatistics(params);
}

//! Get some by and partition statistics on the heap.
CDataCountStatistics* byAndPartitionStatistics(const CAutoconfigurerParams& params) {
    return new CByAndPartitionDataCountStatistics(params);
}

//! Get some by, over and partition statistics on the heap.
CDataCountStatistics* byOverAndPartitionStatistics(const CAutoconfigurerParams& params) {
    return new CByOverAndPartitionDataCountStatistics(params);
}

const std::string EMPTY;
const std::size_t DV_NUMBER_HASHES = 7;
const std::size_t DV_MAX_SIZE = 1000;
const maths::CBjkstUniqueValues BJKST(DV_NUMBER_HASHES, DV_MAX_SIZE);
const std::size_t SKETCH_SIZE = 20u;
const std::size_t CS_SIZE = 15u;
const maths::CQuantileSketch QUANTILES(maths::CQuantileSketch::E_Linear, CS_SIZE);
}

//////// CBucketCountStatistics ////////

void CBucketCountStatistics::add(const TSizeSizeSizeTr& partition,
                                 TDetectorRecordCItr beginRecords,
                                 TDetectorRecordCItr endRecords) {
    ++m_CurrentBucketPartitionCounts[partition];
    for (TDetectorRecordCItr record = beginRecords; record != endRecords; ++record) {
        if (record->function() == config_t::E_DistinctCount) {
            if (const std::string* name = record->argumentFieldName()) {
                const std::string& value = *record->argumentFieldValue();
                std::size_t i = emplace(name, m_CurrentBucketArgumentDataPerPartition);
                SBucketArgumentData& data = m_CurrentBucketArgumentDataPerPartition[i]
                                                .second.emplace(partition, BJKST)
                                                .first->second;
                data.s_DistinctValues.add(CTools::category32(value));
                data.s_MeanStringLength.add(static_cast<double>(value.length()));
            }
        }
    }
}

void CBucketCountStatistics::capture(void) {
    typedef TSizeSizeSizeTrUInt64UMap::const_iterator TSizeSizeSizeTrUInt64UMapCItr;
    typedef TSizeSizeSizeTrArgumentDataUMap::iterator TSizeSizeSizeTrArgumentDataUMapItr;

    m_BucketPartitionCount += m_CurrentBucketPartitionCounts.size();
    for (TSizeSizeSizeTrUInt64UMapCItr i = m_CurrentBucketPartitionCounts.begin();
         i != m_CurrentBucketPartitionCounts.end();
         ++i) {
        TSizeSizePr id(i->first.first, i->first.third);
        double count = static_cast<double>(i->second);
        m_CountMomentsPerPartition[id].add(count);
        m_CountQuantiles.emplace(id, QUANTILES).first->second.add(count);
    }
    m_CurrentBucketPartitionCounts.clear();

    for (std::size_t i = 0u; i < m_CurrentBucketArgumentDataPerPartition.size(); ++i) {
        const std::string* name = m_CurrentBucketArgumentDataPerPartition[i].first;
        TSizeSizeSizeTrArgumentDataUMap& values = m_CurrentBucketArgumentDataPerPartition[i].second;
        std::size_t j = emplace(name, m_ArgumentMomentsPerPartition);
        for (TSizeSizeSizeTrArgumentDataUMapItr k = values.begin(); k != values.end(); ++k) {
            TSizeSizePr id(k->first.first, k->first.third);
            SArgumentMoments& moments = m_ArgumentMomentsPerPartition[j].second[id];
            double dc = static_cast<double>(k->second.s_DistinctValues.number());
            double info = dc * maths::CBasicStatistics::mean(k->second.s_MeanStringLength);
            moments.s_DistinctCount.add(dc);
            moments.s_InfoContent.add(info);
        }
        m_CurrentBucketArgumentDataPerPartition[i].second.clear();
    }
}

uint64_t CBucketCountStatistics::bucketPartitionCount(void) const {
    return m_BucketPartitionCount;
}

const CBucketCountStatistics::TSizeSizePrMomentsUMap&
CBucketCountStatistics::countMomentsPerPartition(void) const {
    return m_CountMomentsPerPartition;
}

const CBucketCountStatistics::TSizeSizePrQuantileUMap&
CBucketCountStatistics::countQuantilesPerPartition(void) const {
    return m_CountQuantiles;
}

const CBucketCountStatistics::TSizeSizePrArgumentMomentsUMap&
CBucketCountStatistics::argumentMomentsPerPartition(const std::string& name) const {
    typedef TStrCPtrSizeSizePrArgumentMomentsUMapPrVec::const_iterator
        TStrCPtrPartitionArgumentMomentsUMapPrVecCItr;
    static const TSizeSizePrArgumentMomentsUMap EMPTY;
    TStrCPtrPartitionArgumentMomentsUMapPrVecCItr result =
        std::lower_bound(m_ArgumentMomentsPerPartition.begin(),
                         m_ArgumentMomentsPerPartition.end(),
                         &name,
                         maths::COrderings::SFirstLess());
    return result != m_ArgumentMomentsPerPartition.end() && *result->first == name ? result->second
                                                                                   : EMPTY;
}

//////// CDataCountStatistics ////////

CDataCountStatistics::CDataCountStatistics(const CAutoconfigurerParams& params)
    : m_Params(params),
      m_RecordCount(0),
      m_ArrivalTimeDistribution(maths::CQuantileSketch::E_PiecewiseConstant, SKETCH_SIZE),
      m_BucketIndices(params.candidateBucketLengths().size(), 0),
      m_BucketCounts(params.candidateBucketLengths().size(), 0),
      m_BucketStatistics(params.candidateBucketLengths().size()) {
    const TTimeVec& candidates = params.candidateBucketLengths();
    m_BucketMasks.reserve(candidates.size());
    for (std::size_t bid = 0u; bid < candidates.size(); ++bid) {
        m_BucketMasks.push_back(bucketSampleMask(candidates[bid]));
        maths::CSampling::random_shuffle(m_Rng,
                                         m_BucketMasks[bid].begin(),
                                         m_BucketMasks[bid].end());
    }
}

CDataCountStatistics::~CDataCountStatistics(void) {
}

void CDataCountStatistics::add(TDetectorRecordCItr beginRecords, TDetectorRecordCItr endRecords) {
    ++m_RecordCount;

    core_t::TTime time = beginRecords->time();

    m_Earliest.add(time);
    m_Latest.add(time);

    if (m_LastRecordTime) {
        m_ArrivalTimeDistribution.add(static_cast<double>(time - *m_LastRecordTime));
    }
    m_LastRecordTime = time;

    this->fillLastBucketEndTimes(time);

    const TTimeVec& candidates = this->params().candidateBucketLengths();
    for (std::size_t bid = 0u; bid < m_LastBucketEndTimes.size(); ++bid) {
        if (time - m_LastBucketEndTimes[bid] >= candidates[bid]) {
            for (core_t::TTime i = 0; i < (time - m_LastBucketEndTimes[bid]) / candidates[bid];
                 ++i) {
                if (m_BucketMasks[bid][m_BucketIndices[bid]++]) {
                    ++m_BucketCounts[bid];
                    m_BucketStatistics[bid].capture();
                }
                if ((m_BucketIndices[bid] % m_BucketMasks.size()) == 0) {
                    m_BucketIndices[bid] = 0;
                    maths::CSampling::random_shuffle(m_Rng,
                                                     m_BucketMasks[bid].begin(),
                                                     m_BucketMasks[bid].end());
                }
            }
            m_LastBucketEndTimes[bid] = maths::CIntegerTools::floor(time, candidates[bid]);
        }
    }

    std::size_t partition = beginRecords->partitionFieldValueHash();
    m_Partitions.insert(partition);
    if (this->samplePartition(partition)) {
        std::size_t by = beginRecords->byFieldValueHash();
        std::size_t over = beginRecords->overFieldValueHash();
        m_SampledPartitions.insert(partition);
        m_SampledTimeSeries.insert(std::make_pair(by, partition));
        CBucketCountStatistics::TSizeSizeSizeTr id(by, over, partition);
        for (std::size_t bid = 0u; bid < m_BucketStatistics.size(); ++bid) {
            if (m_BucketMasks[bid][m_BucketIndices[bid]]) {
                m_BucketStatistics[bid].add(id, beginRecords, endRecords);
            }
        }
    }
}

uint64_t CDataCountStatistics::recordCount(void) const {
    return m_RecordCount;
}

const CDataCountStatistics::TUInt64Vec& CDataCountStatistics::bucketCounts(void) const {
    return m_BucketCounts;
}

const maths::CQuantileSketch& CDataCountStatistics::arrivalTimeDistribution(void) const {
    return m_ArrivalTimeDistribution;
}

core_t::TTime CDataCountStatistics::timeRange(void) const {
    return m_Latest[0] - m_Earliest[0];
}

std::size_t CDataCountStatistics::numberSampledTimeSeries(void) const {
    return m_SampledTimeSeries.size();
}

const CDataCountStatistics::TBucketStatisticsVec&
CDataCountStatistics::bucketStatistics(void) const {
    return m_BucketStatistics;
}

const CAutoconfigurerParams& CDataCountStatistics::params(void) const {
    return m_Params;
}

bool CDataCountStatistics::samplePartition(std::size_t partition) const {
    if (m_SampledPartitions.count(partition) > 0) {
        return true;
    }
    maths::CPRNG::CXorOShiro128Plus rng(partition);
    double n = static_cast<double>(m_Partitions.size());
    double p = 1.0 - 0.99 * std::min(::floor(0.1 * n) / 1000.0, 1.0);
    return maths::CSampling::uniformSample(rng, 0.0, 1.0) < p;
}

void CDataCountStatistics::fillLastBucketEndTimes(core_t::TTime time) {
    if (m_LastBucketEndTimes.empty()) {
        const TTimeVec& candidates = this->params().candidateBucketLengths();
        m_LastBucketEndTimes.reserve(candidates.size());
        for (std::size_t i = 0u; i < candidates.size(); ++i) {
            m_LastBucketEndTimes.push_back(maths::CIntegerTools::ceil(time, candidates[i]));
        }
    }
}

//////// CPartitionDataCountStatistics ////////

CPartitionDataCountStatistics::CPartitionDataCountStatistics(const CAutoconfigurerParams& params)
    : CDataCountStatistics(params) {
}

void CPartitionDataCountStatistics::add(TDetectorRecordCItr beginRecords,
                                        TDetectorRecordCItr endRecords) {
    if (beginRecords != endRecords) {
        this->CDataCountStatistics::add(beginRecords, endRecords);
    }
}

//////// CByAndPartitionDataCountStatistics ////////

CByAndPartitionDataCountStatistics::CByAndPartitionDataCountStatistics(
    const CAutoconfigurerParams& params)
    : CDataCountStatistics(params) {
}

void CByAndPartitionDataCountStatistics::add(TDetectorRecordCItr beginRecords,
                                             TDetectorRecordCItr endRecords) {
    if (beginRecords != endRecords) {
        this->CDataCountStatistics::add(beginRecords, endRecords);
    }
}

//////// CByOverAndPartitionDataCountStatistics ////////

CByOverAndPartitionDataCountStatistics::CByOverAndPartitionDataCountStatistics(
    const CAutoconfigurerParams& params)
    : CDataCountStatistics(params) {
}

void CByOverAndPartitionDataCountStatistics::add(TDetectorRecordCItr beginRecords,
                                                 TDetectorRecordCItr endRecords) {
    if (beginRecords == endRecords) {
        return;
    }

    typedef TSizeSizePrCBjkstUMap::iterator TSizeSizePrCBjkstUMapItr;

    this->CDataCountStatistics::add(beginRecords, endRecords);

    std::size_t partition = beginRecords->partitionFieldValueHash();
    if (this->samplePartition(partition)) {
        std::size_t by = beginRecords->byFieldValueHash();
        std::size_t over = beginRecords->overFieldValueHash();
        TSizeSizePrCBjkstUMapItr i =
            m_DistinctOverValues.emplace(std::make_pair(by, partition), BJKST).first;
        i->second.add(CTools::category32(over));
    }
}

const CByOverAndPartitionDataCountStatistics::TSizeSizePrCBjkstUMap&
CByOverAndPartitionDataCountStatistics::sampledByAndPartitionDistinctOverCounts(void) const {
    return m_DistinctOverValues;
}

//////// CDataCountStatisticsDirectAddressTable ////////

CDataCountStatisticsDirectAddressTable::CDataCountStatisticsDirectAddressTable(
    const CAutoconfigurerParams& params)
    : m_Params(params) {
}

void CDataCountStatisticsDirectAddressTable::build(const TDetectorSpecificationVec& specs) {
    typedef boost::unordered_map<CCountStatisticsKey, std::size_t, CCountStatisticsKeyHasher>
        TCountStatisticsKeySizeUMap;

    std::size_t size = 0u;
    for (std::size_t i = 0u; i < specs.size(); ++i) {
        size = std::max(size, specs[i].id() + 1);
    }

    m_DetectorSchema.resize(size);

    TCountStatisticsKeySizeUMap uniques;
    for (std::size_t i = 0u; i < specs.size(); ++i) {
        std::size_t id = specs[i].id();
        std::size_t next = uniques.size();
        std::size_t index = uniques.emplace(specs[i], next).first->second;
        if (index == next) {
            m_RecordSchema.push_back(TPtrDiffVec(1, static_cast<ptrdiff_t>(id)));
            m_DataCountStatistics.push_back(this->stats(specs[i]));
        } else {
            m_RecordSchema[index].push_back(static_cast<ptrdiff_t>(id));
        }
        m_DetectorSchema[id] = index;
    }

    LOG_DEBUG("There are " << m_DataCountStatistics.size() << " sets of count statistics");
}

void CDataCountStatisticsDirectAddressTable::pruneUnsed(const TDetectorSpecificationVec& specs) {
    typedef boost::unordered_set<std::size_t> TSizeUSet;

    TSizeUSet used;
    for (std::size_t i = 0u; i < specs.size(); ++i) {
        used.insert(m_DetectorSchema[specs[i].id()]);
    }

    std::size_t last = 0u;
    for (std::size_t i = 0u; i < m_DataCountStatistics.size(); ++i) {
        if (last != i) {
            m_DataCountStatistics[last].swap(m_DataCountStatistics[i]);
            m_RecordSchema[last].swap(m_RecordSchema[i]);
            for (std::size_t j = 0u; j < m_RecordSchema[last].size(); ++j) {
                m_DetectorSchema[m_RecordSchema[last][j]] = last;
            }
        }
        if (used.count(i) > 0) {
            ++last;
        }
    }
    m_DataCountStatistics.erase(m_DataCountStatistics.begin() + last, m_DataCountStatistics.end());
    m_RecordSchema.erase(m_RecordSchema.begin() + last, m_RecordSchema.end());
}

void CDataCountStatisticsDirectAddressTable::add(const TDetectorRecordVec& records) {
    for (std::size_t i = 0u; i < m_RecordSchema.size(); ++i) {
        m_DataCountStatistics[i]->add(core::begin_masked(records, m_RecordSchema[i]),
                                      core::end_masked(records, m_RecordSchema[i]));
    }
}

const CDataCountStatistics&
CDataCountStatisticsDirectAddressTable::statistics(const CDetectorSpecification& spec) const {
    return *m_DataCountStatistics[m_DetectorSchema[spec.id()]];
}

CDataCountStatisticsDirectAddressTable::TDataCountStatisticsPtr
CDataCountStatisticsDirectAddressTable::stats(const CDetectorSpecification& spec) const {
    typedef CDataCountStatistics* (*TStatistics)(const CAutoconfigurerParams&);
    static TStatistics STATISTICS[] = {&partitionCountStatistics,
                                       &byAndPartitionStatistics,
                                       &byOverAndPartitionStatistics};
    return TDataCountStatisticsPtr(
        (STATISTICS[spec.overField() ? 2 : (spec.byField() ? 1 : 0)])(m_Params));
}
}
}
