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

#include <config/CLongTailPenalty.h>

#include <core/CLogger.h>
#include <core/CHashing.h>

#include <maths/CBasicStatistics.h>
#include <maths/CTools.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CDataCountStatistics.h>
#include <config/CDetectorSpecification.h>
#include <config/CTools.h>

#include <boost/unordered_map.hpp>

namespace ml
{
namespace config
{
namespace
{

//! Return \p count.
uint64_t count(const CBucketCountStatistics::TMoments &count)
{
    return static_cast<uint64_t>(maths::CBasicStatistics::count(count));
}

//! Extract the distinct count.
uint64_t count(const maths::CBjkstUniqueValues &distinct)
{
    return distinct.number();
}

}

CLongTailPenalty::CLongTailPenalty(const CAutoconfigurerParams &params) :
        CPenalty(params)
{
}

CLongTailPenalty *CLongTailPenalty::clone(void) const
{
    return new CLongTailPenalty(*this);
}

std::string CLongTailPenalty::name(void) const
{
    return "long tail";
}

void CLongTailPenalty::penaltyFromMe(CDetectorSpecification &spec) const
{
    if (config_t::isRare(spec.function()))
    {
        if (const CByAndPartitionDataCountStatistics *byAndPartitionStats =
                dynamic_cast<const CByAndPartitionDataCountStatistics*>(spec.countStatistics()))
        {
            this->penaltyFor(*byAndPartitionStats, spec);
        }
        else if (const CByOverAndPartitionDataCountStatistics *byOverAndPartitionStats =
                     dynamic_cast<const CByOverAndPartitionDataCountStatistics*>(spec.countStatistics()))
        {
            this->penaltyFor(*byOverAndPartitionStats, spec);
        }
    }
 }

void CLongTailPenalty::penaltyFor(const CByAndPartitionDataCountStatistics &stats,
                                  CDetectorSpecification &spec) const
{
    std::size_t n = stats.bucketStatistics().size();

    TSizeVec indices;
    TDoubleVec penalties;
    TStrVec descriptions;
    indices.reserve(2 * n);
    penalties.reserve(2 * n);
    descriptions.reserve(2 * n);

    for (std::size_t bid = 0u; bid < n; ++bid)
    {
        // Penalize the case that many by fields values appear in close
        // to the minimum number of buckets.
        TSizeUInt64UMap totals;
        TSizeUInt64UMap tail;
        this->extractTailCounts<CByAndPartitionDataCountStatistics>(
                                    stats.bucketStatistics()[bid].countMomentsPerPartition(), totals, tail);
        const TSizeVec &indices_ = this->params().penaltyIndicesFor(bid);
        indices.insert(indices.end(), indices_.begin(), indices_.end());
        double penalty = this->penaltyFor(tail, totals);
        std::string description =  penalty < 1.0 ?
                                   std::string("A significant proportion of categories have similar frequency at '")
                                 + CTools::prettyPrint(this->params().candidateBucketLengths()[bid])
                                 + "' resolution" : std::string();
        std::fill_n(std::back_inserter(penalties), indices_.size(), penalty);
        std::fill_n(std::back_inserter(descriptions), indices_.size(), description);
    }

    spec.applyPenalties(indices, penalties, descriptions);
}

void CLongTailPenalty::penaltyFor(const CByOverAndPartitionDataCountStatistics &stats,
                                  CDetectorSpecification &spec) const
{
    // Penalize the case that many by fields values have close to the
    // minimum number of over field values.
    TSizeUInt64UMap totals;
    TSizeUInt64UMap tail;
    this->extractTailCounts<CByOverAndPartitionDataCountStatistics>(
                                stats.sampledByAndPartitionDistinctOverCounts(), totals, tail);
    double penalty = this->penaltyFor(tail, totals);
    spec.applyPenalty(penalty, penalty < 1.0 ? "A significant proportion of categories have a similar frequency in the population" : "");
}

template<typename STATS, typename MAP>
void CLongTailPenalty::extractTailCounts(const MAP &counts,
                                         TSizeUInt64UMap &totals,
                                         TSizeUInt64UMap &tail) const
{
    typedef maths::CBasicStatistics::COrderStatisticsStack<uint64_t, 1> TMinAccumulator;
    typedef boost::unordered_map<std::size_t, TMinAccumulator> TSizeMinAccumulatorUMap;
    typedef typename MAP::const_iterator TItr;

    TSizeMinAccumulatorUMap mins;

    for (TItr i = counts.begin(); i != counts.end(); ++i)
    {
        uint64_t n = count(i->second);
        std::size_t partition = STATS::partition(*i);
        mins[partition].add(n);
        totals[partition] += n;
    }

    for (TItr i = counts.begin(); i != counts.end(); ++i)
    {
        uint64_t n = count(i->second);
        std::size_t partition = STATS::partition(*i);
        const TMinAccumulator &min = mins[partition];
        if (   n <= static_cast<uint64_t>(  this->params().highCardinalityInTailFactor()
                                          * static_cast<double>(min[0]) + 0.5)
            || n <= this->params().highCardinalityInTailIncrement() + min[0])
        {
            tail[partition] += n;
        }
    }
}

double CLongTailPenalty::penaltyFor(TSizeUInt64UMap &tail, TSizeUInt64UMap &totals) const
{
    typedef TSizeUInt64UMap::const_iterator TSizeUInt64UMapCItr;
    typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
    TMeanAccumulator result;
    for (TSizeUInt64UMapCItr i = tail.begin(); i != tail.end(); ++i)
    {
        double rare    = static_cast<double>(i->second);
        double total   = static_cast<double>(totals[i->first]);
        double penalty = CTools::logInterpolate(this->params().highCardinalityHighTailFraction(),
                                           this->params().highCardinalityMaximumTailFraction(),
                                           1.0, std::min(10.0 / total, 1.0), rare / total);
        result.add(::sqrt(-std::min(maths::CTools::fastLog(penalty), 0.0)), total);
    }
    return ::exp(-::pow(maths::CBasicStatistics::mean(result), 2.0));
}

}
}

