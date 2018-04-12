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

#include <config/CTooMuchDataPenalty.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CTools.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CDataCountStatistics.h>
#include <config/CDetectorSpecification.h>
#include <config/CTools.h>

#include <cmath>
#include <cstddef>

namespace ml {
namespace config {
namespace {
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

//! Get the description prefix.
std::string descriptionPrefix(const CDetectorSpecification& spec,
                              const TMeanAccumulator& meanOccupied,
                              std::size_t partitions) {
    if (spec.byField() && spec.partitionField()) {
        return "A significant proportion, " +
               CTools::prettyPrint(100.0 * maths::CBasicStatistics::count(meanOccupied) /
                                   static_cast<double>(partitions)) +
               "%, of distinct partition and by fields combinations have "
               "values in many buckets.";
    } else if (spec.byField()) {
        return "A significant proportion, " +
               CTools::prettyPrint(100.0 * maths::CBasicStatistics::count(meanOccupied) /
                                   static_cast<double>(partitions)) +
               "%, of distinct by fields have values in many buckets.";
    } else if (spec.partitionField()) {
        return "A significant proportion, " +
               CTools::prettyPrint(100.0 * maths::CBasicStatistics::count(meanOccupied) /
                                   static_cast<double>(partitions)) +
               "%, of distinct partition fields have values in many buckets.";
    }
    return "";
}
}

CTooMuchDataPenalty::CTooMuchDataPenalty(const CAutoconfigurerParams& params)
    : CPenalty(params) {
}

CTooMuchDataPenalty* CTooMuchDataPenalty::clone() const {
    return new CTooMuchDataPenalty(*this);
}

std::string CTooMuchDataPenalty::name() const {
    return "too much data";
}

void CTooMuchDataPenalty::penaltyFromMe(CDetectorSpecification& spec) const {
    if (config_t::hasDoAndDontIgnoreEmptyVersions(spec.function()) && !spec.isPopulation()) {
        if (const CPartitionDataCountStatistics* partitionStats =
                dynamic_cast<const CPartitionDataCountStatistics*>(spec.countStatistics())) {
            this->penaltyFor(*partitionStats, spec);
        } else if (const CByAndPartitionDataCountStatistics* byAndPartitionStats =
                       dynamic_cast<const CByAndPartitionDataCountStatistics*>(
                           spec.countStatistics())) {
            this->penaltyFor(*byAndPartitionStats, spec);
        } else if (const CByOverAndPartitionDataCountStatistics* byOverAndPartitionStats =
                       dynamic_cast<const CByOverAndPartitionDataCountStatistics*>(
                           spec.countStatistics())) {
            this->penaltyFor(*byOverAndPartitionStats, spec);
        }
    }
}

void CTooMuchDataPenalty::penaltyFor(const CPartitionDataCountStatistics& stats,
                                     CDetectorSpecification& spec) const {
    this->penaltyFor(stats.bucketCounts(), stats.bucketStatistics(), spec);
}

void CTooMuchDataPenalty::penaltyFor(const CByAndPartitionDataCountStatistics& stats,
                                     CDetectorSpecification& spec) const {
    this->penaltyFor(stats.bucketCounts(), stats.bucketStatistics(), spec);
}

void CTooMuchDataPenalty::penaltyFor(const CByOverAndPartitionDataCountStatistics& stats,
                                     CDetectorSpecification& spec) const {
    this->penaltyFor(stats.bucketCounts(), stats.bucketStatistics(), spec);
}

void CTooMuchDataPenalty::penaltyFor(const TUInt64Vec& bucketCounts,
                                     const TBucketCountStatisticsVec& statistics,
                                     CDetectorSpecification& spec) const {
    using TSizeSizePrMomentsUMapCItr = CBucketCountStatistics::TSizeSizePrMomentsUMap::const_iterator;

    const CAutoconfigurerParams::TTimeVec& candidates =
        this->params().candidateBucketLengths();

    LOG_TRACE(<< "bucket counts = " << core::CContainerPrinter::print(bucketCounts));

    TSizeVec indices;
    TDoubleVec penalties;
    TStrVec descriptions;
    indices.reserve(2 * candidates.size());
    penalties.reserve(2 * candidates.size());
    descriptions.reserve(2 * candidates.size());

    config_t::EFunctionCategory function = spec.function();

    for (std::size_t bid = 0u; bid < candidates.size(); ++bid) {
        uint64_t bc = bucketCounts[bid];
        if (bc > 0) {
            const CBucketCountStatistics& si = statistics[bid];
            const CBucketCountStatistics::TSizeSizePrMomentsUMap& mi =
                si.countMomentsPerPartition();

            TMeanAccumulator penalty_;
            TMeanAccumulator penalizedOccupancy;

            for (TSizeSizePrMomentsUMapCItr j = mi.begin(); j != mi.end(); ++j) {
                double occupied = maths::CBasicStatistics::count(j->second) /
                                  static_cast<double>(bc);
                double penalty = CTools::logInterpolate(
                    this->params().highPopulatedBucketFraction(function, true),
                    this->params().maximumPopulatedBucketFraction(function, true),
                    1.0, 1.0 / static_cast<double>(bucketCounts[bid]), occupied);
                penalty_.add(maths::CTools::fastLog(penalty));
                if (penalty < 1.0) {
                    penalizedOccupancy.add(occupied);
                }
            }

            if (maths::CBasicStatistics::count(penalizedOccupancy) >
                0.95 * static_cast<double>(mi.size())) {
                double penalty =
                    std::min(std::exp(maths::CBasicStatistics::mean(penalty_)), 1.0);
                std::size_t index = this->params().penaltyIndexFor(bid, true);
                indices.push_back(index);
                penalties.push_back(penalty);
                descriptions.push_back("");
                if (penalty < 1.0) {
                    if (spec.byField() || spec.partitionField()) {
                        descriptions.back() =
                            descriptionPrefix(spec, penalizedOccupancy, mi.size()) +
                            " On average, " +
                            CTools::prettyPrint(100.0 * maths::CBasicStatistics::mean(
                                                            penalizedOccupancy)) +
                            "% of their buckets have a value";
                    } else {
                        descriptions.back() =
                            "A significant proportion, " +
                            CTools::prettyPrint(100.0 * maths::CBasicStatistics::mean(
                                                            penalizedOccupancy)) +
                            "%, of " + CTools::prettyPrint(candidates[bid]) +
                            " buckets have a value";
                    }
                }
            }
        }
    }
    spec.applyPenalties(indices, penalties, descriptions);
}
}
}
