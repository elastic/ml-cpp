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

#include <config/CNotEnoughDataPenalty.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CTools.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CDataCountStatistics.h>
#include <config/CDetectorSpecification.h>
#include <config/CTools.h>

#include <boost/range.hpp>

#include <cstddef>

namespace ml {
namespace config {
namespace {
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;

//! Get the description prefix.
std::string descriptionPrefix(const CDetectorSpecification &spec,
                              const TMeanAccumulator &meanOccupied,
                              std::size_t partitions) {
    if (spec.byField() && spec.partitionField()) {
        return "A significant proportion, "
               + CTools::prettyPrint(100.0 * maths::CBasicStatistics::count(meanOccupied)
                                     / static_cast<double>(partitions))
               + "%, of distinct partition and by fields combinations are sparse.";
    } else if (spec.byField()) {
        return "A significant proportion, "
               + CTools::prettyPrint(100.0 * maths::CBasicStatistics::count(meanOccupied)
                                     / static_cast<double>(partitions))
               + "%, of distinct by fields are sparse.";
    } else if (spec.partitionField()) {
        return "A significant proportion, "
               + CTools::prettyPrint(100.0 * maths::CBasicStatistics::count(meanOccupied)
                                     / static_cast<double>(partitions))
               + "%, of distinct partition fields are sparse.";
    }
    return "";
}

const bool IGNORE_EMPTY[] = { false, true };
}

CNotEnoughDataPenalty::CNotEnoughDataPenalty(const CAutoconfigurerParams &params) :
    CPenalty(params) {
}

CNotEnoughDataPenalty *CNotEnoughDataPenalty::clone(void) const {
    return new CNotEnoughDataPenalty(*this);
}

std::string CNotEnoughDataPenalty::name(void) const {
    return "not enough data";
}

void CNotEnoughDataPenalty::penaltyFromMe(CDetectorSpecification &spec) const {
    if (!config_t::isRare(spec.function())) {
        if (const CPartitionDataCountStatistics *partitionStats =
                dynamic_cast<const CPartitionDataCountStatistics*>(spec.countStatistics())) {
            this->penaltyFor(*partitionStats, spec);
        } else if (const CByAndPartitionDataCountStatistics *byAndPartitionStats =
                       dynamic_cast<const CByAndPartitionDataCountStatistics*>(spec.countStatistics())) {
            this->penaltyFor(*byAndPartitionStats, spec);
        } else if (const CByOverAndPartitionDataCountStatistics *byOverAndPartitionStats =
                       dynamic_cast<const CByOverAndPartitionDataCountStatistics*>(spec.countStatistics())) {
            this->penaltyFor(*byOverAndPartitionStats, spec);
        }
    }
}

void CNotEnoughDataPenalty::penaltyFor(const CPartitionDataCountStatistics &stats,
                                       CDetectorSpecification &spec) const {
    this->penaltyFor(stats.bucketCounts(), stats.bucketStatistics(), spec);
}

void CNotEnoughDataPenalty::penaltyFor(const CByAndPartitionDataCountStatistics &stats,
                                       CDetectorSpecification &spec) const {
    this->penaltyFor(stats.bucketCounts(), stats.bucketStatistics(), spec);
}

void CNotEnoughDataPenalty::penaltyFor(const CByOverAndPartitionDataCountStatistics &stats,
                                       CDetectorSpecification &spec) const {
    this->penaltyFor(stats.bucketCounts(), stats.bucketStatistics(), spec);
}

void CNotEnoughDataPenalty::penaltyFor(const TUInt64Vec &bucketCounts,
                                       const TBucketCountStatisticsVec &statistics,
                                       CDetectorSpecification &spec) const {
    typedef CBucketCountStatistics::TSizeSizePrMomentsUMap::const_iterator TSizeSizePrMomentsUMapCItr;

    const CAutoconfigurerParams::TTimeVec &candidates = this->params().candidateBucketLengths();

    LOG_TRACE("bucket counts = " << core::CContainerPrinter::print(bucketCounts));

    TSizeVec   indices;
    TDoubleVec penalties;
    TStrVec    descriptions;
    indices.reserve(2 * candidates.size());
    penalties.reserve(2 * candidates.size());
    descriptions.reserve(2 * candidates.size());

    config_t::EFunctionCategory function = spec.function();

    // Per partition occupancy.
    for (std::size_t i = 0u; i < boost::size(IGNORE_EMPTY); ++i) {
        for (std::size_t bid = 0u; bid < candidates.size(); ++bid) {
            uint64_t bc = bucketCounts[bid];
            if (bc > 0) {
                const CBucketCountStatistics                         &                        si = statistics[bid];
                const CBucketCountStatistics::TSizeSizePrMomentsUMap &mi = si.countMomentsPerPartition();

                TMeanAccumulator penalty_;
                TMeanAccumulator meanOccupied;

                for (TSizeSizePrMomentsUMapCItr j = mi.begin(); j != mi.end(); ++j) {
                    double occupied = maths::CBasicStatistics::count(j->second) / static_cast<double>(bc);
                    double penalty  = CTools::logInterpolate(
                        this->params().lowPopulatedBucketFraction(function, IGNORE_EMPTY[i]),
                        this->params().minimumPopulatedBucketFraction(function, IGNORE_EMPTY[i]),
                        1.0, 1.0 / static_cast<double>(bc), occupied);
                    penalty_.add(maths::CTools::fastLog(penalty));
                    if (penalty < 1.0) {
                        meanOccupied.add(occupied);
                    }
                }

                double      penalty = std::min(::exp(maths::CBasicStatistics::mean(penalty_)), 1.0);
                std::size_t index = this->params().penaltyIndexFor(bid, IGNORE_EMPTY[i]);
                indices.push_back(index);
                penalties.push_back(penalty);
                descriptions.push_back("");
                if (penalty < 1.0) {
                    if (spec.byField() || spec.partitionField()) {
                        descriptions.back() =  descriptionPrefix(spec, meanOccupied, si.countMomentsPerPartition().size())
                                              + " On average, only "
                                              + CTools::prettyPrint(100.0 * maths::CBasicStatistics::mean(meanOccupied))
                                              + "% of their buckets have a value";
                    } else {
                        descriptions.back() =  std::string("On average only ")
                                              + CTools::prettyPrint(100.0 * maths::CBasicStatistics::mean(meanOccupied))
                                              + "% of partition buckets have a value";
                    }
                }
            }
        }
    }
    spec.applyPenalties(indices, penalties, descriptions);
}

}
}
