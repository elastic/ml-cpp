/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CDataCountStatistics_h
#define INCLUDED_ml_config_CDataCountStatistics_h

#include <core/CMaskIterator.h>
#include <core/CoreTypes.h>
#include <core/CTriple.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBjkstUniqueValues.h>
#include <maths/CPRNG.h>
#include <maths/CQuantileSketch.h>

#include <config/ImportExport.h>

#include <boost/ref.hpp>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <cstddef>
#include <vector>
#include <stdint.h>

namespace ml
{
namespace config
{
class CAutoconfigurerParams;
class CDetectorRecord;
class CDetectorSpecification;

//! \brief Statistics for the bucketed data.
class CONFIG_EXPORT CBucketCountStatistics
{
    public:
        using TSizeSizePr = std::pair<std::size_t, std::size_t>;
        using TSizeSizeSizeTr = core::CTriple<std::size_t, std::size_t, std::size_t>;
        using TSizeSizePrUInt64UMap = boost::unordered_map<TSizeSizePr, uint64_t>;
        using TSizeSizeSizeTrUInt64UMap = boost::unordered_map<TSizeSizeSizeTr, uint64_t>;
        using TDetectorRecordVec = std::vector<CDetectorRecord>;
        using TDetectorRecordCItr = core::CMaskIterator<TDetectorRecordVec::const_iterator>;
        using TMoments = maths::CBasicStatistics::SSampleMeanVarSkew<double>::TAccumulator;

        //! \brief The moments of a categorical function argument field.
        struct CONFIG_EXPORT SArgumentMoments
        {
            //! The distinct count moments.
            TMoments s_DistinctCount;
            //! The information content moments.
            TMoments s_InfoContent;
        };

        using TSizeSizePrMomentsUMap = boost::unordered_map<TSizeSizePr, TMoments>;
        using TSizeSizePrArgumentMomentsUMap = boost::unordered_map<TSizeSizePr, SArgumentMoments>;
        using TStrCPtrSizeSizePrArgumentMomentsUMapPr = std::pair<const std::string*, TSizeSizePrArgumentMomentsUMap>;
        using TStrCPtrSizeSizePrArgumentMomentsUMapPrVec = std::vector<TStrCPtrSizeSizePrArgumentMomentsUMapPr>;
        using TSizeSizePrQuantileUMap = boost::unordered_map<TSizeSizePr, maths::CQuantileSketch>;

    public:
        //! Add the record for \p partition.
        void add(const TSizeSizeSizeTr &partition,
                 TDetectorRecordCItr beginRecords,
                 TDetectorRecordCItr endRecords);

        //! Capture the current bucket statistics.
        void capture();

        //! Get the total count of distinct partitions and buckets seen to date.
        uint64_t bucketPartitionCount() const;

        //! Get the moments of the count distribution per partition.
        const TSizeSizePrMomentsUMap &countMomentsPerPartition() const;

        //! Get the quantile summary for the count distribution per partition.
        const TSizeSizePrQuantileUMap &countQuantilesPerPartition() const;

        //! Get the moments of the distribution of the distinct count of argument
        //! field values for \p name.
        const TSizeSizePrArgumentMomentsUMap &argumentMomentsPerPartition(const std::string &name) const;

    private:
        using TMean = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

        //! \brief Bucket data stored about argument field.
        struct CONFIG_EXPORT SBucketArgumentData
        {
            SBucketArgumentData(const maths::CBjkstUniqueValues distinctValues) :
                    s_DistinctValues(distinctValues)
            {}
            //! The approximate distinct values.
            maths::CBjkstUniqueValues s_DistinctValues;
            //! A sample of the unique strings in the bucket.
            TMean s_MeanStringLength;
        };

        using TSizeSizeSizeTrArgumentDataUMap = boost::unordered_map<TSizeSizeSizeTr, SBucketArgumentData>;
        using TStrCPtrSizeSizeSizeTrBjkstArgumentDataUMapPr = std::pair<const std::string*, TSizeSizeSizeTrArgumentDataUMap>;
        using TStrCPtrSizeSizeSizeTrArgumentDataUMapPrVec = std::vector<TStrCPtrSizeSizeSizeTrBjkstArgumentDataUMapPr>;

    private:
        //! The distinct partitions seen this bucket.
        TSizeSizeSizeTrUInt64UMap m_CurrentBucketPartitionCounts;

        //! The distinct counts of the argument fields for each partition
        //! for the current bucket.
        TStrCPtrSizeSizeSizeTrArgumentDataUMapPrVec m_CurrentBucketArgumentDataPerPartition;

        //! The total count of distinct partitions and buckets seen to date.
        uint64_t m_BucketPartitionCount;

        //! The moments of the distribution of partition counts.
        TSizeSizePrMomentsUMap m_CountMomentsPerPartition;

        //! The count quantiles.
        TSizeSizePrQuantileUMap m_CountQuantiles;

        //! The moments of the distribution of distinct counts of the argument
        //! fields per partition and bucket length.
        TStrCPtrSizeSizePrArgumentMomentsUMapPrVec m_ArgumentMomentsPerPartition;
};

//! \brief The root of the class hierarchy for useful count statistics.
//!
//! DESCRIPTION:\n
//! The root of the class hierarchy for all useful count statistics related
//! to combinations of fields and their various roles in detectors.
//!
//! IMPLEMENTATION:\n
//! It turns out to be useful to separate all gathering of data statistics
//! from all detector penalties because otherwise we significantly duplicate
//! information when multiple detectors share the same set of statistics.
//! Each logical set of data statistics has its own hierarchy and is managed
//! by a direct address table which enumerates unique combinations and maps
//! detectors to their appropriate statistics.
class CONFIG_EXPORT CDataCountStatistics
{
    public:
        using TUInt64Vec = std::vector<uint64_t>;
        using TDetectorRecordVec = std::vector<CDetectorRecord>;
        using TDetectorRecordCItr = core::CMaskIterator<TDetectorRecordVec::const_iterator>;
        using TBucketStatisticsVec = std::vector<CBucketCountStatistics>;

    public:
        CDataCountStatistics(const CAutoconfigurerParams &params);
        virtual ~CDataCountStatistics();

        //! Update the statistics with [\p beginRecords, \p endRecords).
        virtual void add(TDetectorRecordCItr beginRecords, TDetectorRecordCItr endRecords) = 0;

        //! Get the total count of records added.
        uint64_t recordCount() const;

        //! Get the total count of each bucket length.
        const TUInt64Vec &bucketCounts() const;

        //! Get the arrival time distribution
        const maths::CQuantileSketch &arrivalTimeDistribution() const;

        //! Get the total time range.
        core_t::TTime timeRange() const;

        //! Get the number of time series.
        std::size_t numberSampledTimeSeries() const;

        //! Get the counts of distinct (bucket, by, partition) triples
        //! per bucket length seen to date.
        const TBucketStatisticsVec &bucketStatistics() const;

        //! Extract the by field value.
        template<typename T>
        static std::size_t by(const std::pair<const std::pair<std::size_t, std::size_t>, T> &p)
        {
            return p.first.first;
        }

        //! Extract the partition field value.
        template<typename T>
        static std::size_t partition(const std::pair<const std::pair<std::size_t, std::size_t>, T> &p)
        {
            return p.first.second;
        }

    protected:
        using TTimeVec = std::vector<core_t::TTime>;
        using TSizeUSet = boost::unordered_set<std::size_t>;

    protected:
        //! Get the parameters.
        const CAutoconfigurerParams &params() const;

        //! Check if we should sample the partition.
        bool samplePartition(std::size_t partition) const;

    private:
        using TBoolVec = std::vector<bool>;
        using TBoolVecVec = std::vector<TBoolVec>;
        using TSizeVec = std::vector<std::size_t>;
        using TSizeSizePr = std::pair<std::size_t, std::size_t>;
        using TSizeSizePrUSet = boost::unordered_set<TSizeSizePr>;
        using TOptionalTime = boost::optional<core_t::TTime>;
        using TAutoconfigurerParamsCRef = boost::reference_wrapper<const CAutoconfigurerParams>;
        using TMinTimeAccumulator = maths::CBasicStatistics::COrderStatisticsStack<core_t::TTime, 1>;
        using TMaxTimeAccumulator = maths::CBasicStatistics::COrderStatisticsStack<core_t::TTime, 1, std::greater<core_t::TTime> >;

    private:
        //! Fill in the last bucket end times if they are empty.
        void fillLastBucketEndTimes(core_t::TTime time);

    private:
        //! The parameters.
        TAutoconfigurerParamsCRef m_Params;

        //! The total count of records added.
        uint64_t m_RecordCount;

        //! The last record time.
        TOptionalTime m_LastRecordTime;

        //! The approximate distribution function of arrival times.
        maths::CQuantileSketch m_ArrivalTimeDistribution;

        //! The earliest example time.
        TMinTimeAccumulator m_Earliest;

        //! The latest example time.
        TMaxTimeAccumulator m_Latest;

        //! The times of the ends of the last complete buckets.
        TTimeVec m_LastBucketEndTimes;

        //! The set of all partitions.
        TSizeUSet m_Partitions;

        //! The partitions which are being sampled.
        TSizeUSet m_SampledPartitions;

        //! The sampled distinct time series.
        TSizeSizePrUSet m_SampledTimeSeries;

        //! The pseudo r.n.g. for generating permutations of the masks.
        maths::CPRNG::CXorOShiro128Plus m_Rng;

        //! The current index into the masks.
        TSizeVec m_BucketIndices;

        //! The bucket sampling masks.
        TBoolVecVec m_BucketMasks;

        //! The total count of complete buckets seen.
        TUInt64Vec m_BucketCounts;

        //! The bucket statistics.
        TBucketStatisticsVec m_BucketStatistics;
};

//! \brief The count statistics for detectors with no "by" or "over" field.
class CONFIG_EXPORT CPartitionDataCountStatistics : public CDataCountStatistics
{
    public:
        CPartitionDataCountStatistics(const CAutoconfigurerParams &params);

        //! Update the statistics with [\p beginRecords, \p endRecords).
        virtual void add(TDetectorRecordCItr beginRecords, TDetectorRecordCItr endRecords);
};

//! \brief The count statistics for detectors with no "over" field.
class CONFIG_EXPORT CByAndPartitionDataCountStatistics : public CDataCountStatistics
{
    public:
        using TSizeSizePr = std::pair<std::size_t, std::size_t>;
        using TSizeSizePrUSet = boost::unordered_set<TSizeSizePr>;
        using TSizeSizePrUInt64UMap = boost::unordered_map<TSizeSizePr, uint64_t>;
        using TSizeSizePrUInt64UMapCItr = TSizeSizePrUInt64UMap::const_iterator;

    public:
        CByAndPartitionDataCountStatistics(const CAutoconfigurerParams &params);

        //! Update the statistics with [\p beginRecords, \p endRecords).
        virtual void add(TDetectorRecordCItr beginRecords, TDetectorRecordCItr endRecords);
};

//! \brief The count statistics for detectors with a "by" and an "over" field.
class CONFIG_EXPORT CByOverAndPartitionDataCountStatistics : public CDataCountStatistics
{
    public:
        using TSizeUInt64UMap = boost::unordered_map<std::size_t, uint64_t>;
        using TSizeSizePr = std::pair<std::size_t, std::size_t>;
        using TSizeSizePrCBjkstUMap = boost::unordered_map<TSizeSizePr, maths::CBjkstUniqueValues>;
        using TSizeSizePrCBjkstUMapCItr = TSizeSizePrCBjkstUMap::const_iterator;

    public:
        CByOverAndPartitionDataCountStatistics(const CAutoconfigurerParams &params);

        //! Update the statistics with [\p beginRecords, \p endRecords).
        virtual void add(TDetectorRecordCItr beginRecords, TDetectorRecordCItr endRecords);

        //! Get the distinct count of over field values per (by, partition) pair.
        const TSizeSizePrCBjkstUMap &sampledByAndPartitionDistinctOverCounts() const;

    private:
        //! The distinct count of over values per (by, partition) pair.
        TSizeSizePrCBjkstUMap m_DistinctOverValues;
};

//! \brief Responsible for creating unique data count statistics and providing
//! a fast scheme for looking up the detectors' statistics and updating the
//! statistics with the detectors' records.
//!
//! DESCRIPTION:\n
//! There are many more distinct detectors than data count statistics and we
//! only want to create update each distinct statistics once. To do this we
//! maintain a direct address table from every detector, built once up front
//! on the set of initial candidate detectors, to a corresponding collection
//! unique data count statistics.
class CONFIG_EXPORT CDataCountStatisticsDirectAddressTable
{
    public:
        using TDetectorRecordVec = std::vector<CDetectorRecord>;
        using TDetectorSpecificationVec = std::vector<CDetectorSpecification>;

    public:
        CDataCountStatisticsDirectAddressTable(const CAutoconfigurerParams &params);

        //! Build the table from \p specs.
        void build(const TDetectorSpecificationVec &specs);

        //! Clear the state (as a precursor to build).
        void pruneUnsed(const TDetectorSpecificationVec &specs);

        //! Update the statistics with \p records.
        void add(const TDetectorRecordVec &records);

        //! Get the detector \p spec's statistics.
        const CDataCountStatistics &statistics(const CDetectorSpecification &spec) const;

    private:
        using TSizeVec = std::vector<std::size_t>;
        using TPtrDiffVec = std::vector<ptrdiff_t>;
        using TPtrDiffVecVec = std::vector<TPtrDiffVec>;
        using TAutoconfigurerParamsCRef = boost::reference_wrapper<const CAutoconfigurerParams>;
        using TDataCountStatisticsPtr = std::shared_ptr<CDataCountStatistics>;
        using TDataCountStatisticsPtrVec = std::vector<TDataCountStatisticsPtr>;

    private:
        //! Get the statistics for \p spec.
        TDataCountStatisticsPtr stats(const CDetectorSpecification &spec) const;

    private:
        //! The parameters.
        TAutoconfigurerParamsCRef m_Params;

        //! The many-to-one map from detector to data count statistic.
        TSizeVec m_DetectorSchema;

        //! The one-to-one map from data count statistic to first detector.
        TPtrDiffVecVec m_RecordSchema;

        //! The actual count statistics.
        TDataCountStatisticsPtrVec m_DataCountStatistics;
};

}
}

#endif // INCLUDED_ml_config_CDataCountStatistics_h
