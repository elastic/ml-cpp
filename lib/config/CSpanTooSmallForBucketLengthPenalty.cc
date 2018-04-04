/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <config/CSpanTooSmallForBucketLengthPenalty.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CDataCountStatistics.h>
#include <config/CDetectorSpecification.h>
#include <config/CTools.h>

namespace ml
{
namespace config
{

CSpanTooSmallForBucketLengthPenalty::CSpanTooSmallForBucketLengthPenalty(const CAutoconfigurerParams &params) :
        CPenalty(params)
{}

CSpanTooSmallForBucketLengthPenalty *CSpanTooSmallForBucketLengthPenalty::clone() const
{
    return new CSpanTooSmallForBucketLengthPenalty(*this);
}

std::string CSpanTooSmallForBucketLengthPenalty::name() const
{
    return "span too small for bucket length";
}

void CSpanTooSmallForBucketLengthPenalty::penaltyFromMe(CDetectorSpecification &spec) const
{
    if (const CDataCountStatistics *stats = spec.countStatistics())
    {
        const TTimeVec &candidates = this->params().candidateBucketLengths();

        TSizeVec indices;
        TDoubleVec penalties;
        TStrVec descriptions;
        indices.reserve(2 * candidates.size());
        penalties.reserve(2 * candidates.size());
        descriptions.reserve(2 * candidates.size());

        for (std::size_t bid = 0u; bid < candidates.size(); ++bid)
        {
            const TSizeVec &indices_ = this->params().penaltyIndicesFor(bid);
            indices.insert(indices.end(), indices_.begin(), indices_.end());
            double penalty = CTools::logInterpolate(this->params().minimumNumberOfBucketsForConfig(),
                                                    this->params().lowNumberOfBucketsForConfig(),
                                                    0.0, 1.0, static_cast<double>(  stats->timeRange()
                                                                                  / candidates[bid]));
            std::string description = penalty < 1.0 ? "The data span is too short to properly assess the bucket length" : "";
            std::fill_n(std::back_inserter(penalties), indices_.size(), penalty);
            std::fill_n(std::back_inserter(descriptions), indices_.size(), description);
        }

        spec.applyPenalties(indices, penalties, descriptions);
    }
}

}
}
