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

#ifndef INCLUDED_ml_config_CTooMuchDataPenalty_h
#define INCLUDED_ml_config_CTooMuchDataPenalty_h

#include <config/CPenalty.h>
#include <config/ImportExport.h>

#include <vector>

namespace ml
{
namespace config
{
class CAutoconfigurerParams;
class CBucketCountStatistics;
class CPartitionDataCountStatistics;
class CByAndPartitionDataCountStatistics;
class CByOverAndPartitionDataCountStatistics;

//! \brief A penalty for using detectors which have too many populated buckets
//! for a given bucket length.
//!
//! DESCRIPTION:\n
//! If the function supports both do and don't ignore empty versions there is
//! no point in using the ignore empty versions if there are too many populated
//! buckets. The important factor is the number of populated buckets for each
//! distinct (by, partition) field value pair. This applies a bucket length
//! specific penalty based on the proportion of populated buckets verses total
//! buckets.
class CONFIG_EXPORT CTooMuchDataPenalty : public CPenalty
{
    public:
        CTooMuchDataPenalty(const CAutoconfigurerParams &params);

        //! Create a copy on the heap.
        virtual CTooMuchDataPenalty *clone(void) const;

        //! Get the name of this penalty.
        virtual std::string name(void) const;

    private:
        using TUInt64Vec = std::vector<uint64_t>;
        using TBucketCountStatisticsVec = std::vector<CBucketCountStatistics>;

    private:
        //! Compute a penalty for rare detectors.
        virtual void penaltyFromMe(CDetectorSpecification &spec) const;

        //! Compute the penalty for optionally a partition.
        void penaltyFor(const CPartitionDataCountStatistics &stats,
                        CDetectorSpecification &spec) const;

        //! Compute the penalty for a by field and optionally a partition.
        void penaltyFor(const CByAndPartitionDataCountStatistics &stats,
                        CDetectorSpecification &spec) const;

        //! Compute the penalty for a by, over and optionally a partition field.
        void penaltyFor(const CByOverAndPartitionDataCountStatistics &stats,
                        CDetectorSpecification &spec) const;

        //! The common penalty calculation.
        void penaltyFor(const TUInt64Vec &bucketCounts,
                        const TBucketCountStatisticsVec &bucketDistinctTupleCounts,
                        CDetectorSpecification &spec) const;
};

}
}

#endif // INCLUDED_ml_config_CTooMuchDataPenalty_h
