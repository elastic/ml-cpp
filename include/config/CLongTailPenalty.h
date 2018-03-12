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

#ifndef INCLUDED_ml_config_CLongTailPenalty_h
#define INCLUDED_ml_config_CLongTailPenalty_h

#include <config/CPenalty.h>
#include <config/ImportExport.h>

#include <boost/ref.hpp>
#include <boost/unordered_map.hpp>

namespace ml {
namespace config {
class CAutoconfigurerParams;
class CByAndPartitionDataCountStatistics;
class CByOverAndPartitionDataCountStatistics;

//! \brief A penalty for using rare commands with by fields which have many
//! unique or infrequently occurring values.
//!
//! DESCRIPTION:\n
//! If a significant proportion of the count is in by field values which have
//! close to the minimum count, either in terms of distinct over field values
//! which generate them or distinct buckets in which they occur, then rare
//! detectors will fail to detect any anomalies. As such we penalize detectors
//! based on the proportion of count in a tail which is defined as a threshold
//! on the difference between the by field count and the minimum count.
class CONFIG_EXPORT CLongTailPenalty : public CPenalty {
    public:
        CLongTailPenalty(const CAutoconfigurerParams &params);

        //! Create a copy on the heap.
        virtual CLongTailPenalty *clone(void) const;

        //! Get the name of this penalty.
        virtual std::string name(void) const;

    private:
        typedef boost::unordered_map<std::size_t, uint64_t> TSizeUInt64UMap;

    private:
        //! Compute a penalty for rare detectors.
        virtual void penaltyFromMe(CDetectorSpecification &spec) const;

        //! Compute the penalty for a by field and optionally a partition.
        void penaltyFor(const CByAndPartitionDataCountStatistics &stats,
                        CDetectorSpecification &spec) const;

        //! Compute the penalty for a by, over and optionally a partition field.
        void penaltyFor(const CByOverAndPartitionDataCountStatistics &stats,
                        CDetectorSpecification &spec) const;

        //! Extract the tail and total counts from \p counts.
        template<typename STATS, typename MAP>
        void extractTailCounts(const MAP &counts,
                               TSizeUInt64UMap &totals,
                               TSizeUInt64UMap &tail) const;

        //! Compute the penalty for the rare counts and total counts \p rares
        //! and \p totals, respectively.
        double penaltyFor(TSizeUInt64UMap &rares, TSizeUInt64UMap &totals) const;
};

}
}

#endif // INCLUDED_ml_config_CLongTailPenalty_h
