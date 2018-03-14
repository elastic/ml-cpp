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

#include <config/CSparseCountPenalty.h>

#include <maths/CBasicStatistics.h>
#include <maths/CQuantileSketch.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTools.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CDetectorSpecification.h>
#include <config/CDataCountStatistics.h>

#include <boost/range.hpp>

#include <vector>

namespace ml {
namespace config {
namespace {

typedef std::vector<double> TDoubleVec;

//! Extract the \p n quantiles from \p quantiles.
void extract(const maths::CQuantileSketch &quantiles, std::size_t n, TDoubleVec &result) {
    for (std::size_t i = 1u; i <= n; ++i) {
        double x;
        quantiles.quantile(100.0 * static_cast<double>(i) / static_cast<double>(n + 1), x);
        result[i - 1] = x;
    }
}

//! Get the quantiles adjusted for empty buckets.
const maths::CQuantileSketch &correctForEmptyBuckets(bool ignoreEmpty,
                                                     uint64_t buckets,
                                                     maths::CQuantileSketch &placeholder,
                                                     const maths::CQuantileSketch &quantiles) {
    if (!ignoreEmpty) {
        double n = static_cast<double>(buckets) - quantiles.count();
        if (n > 0.0) {
            placeholder = quantiles;
            placeholder.add(0.0, static_cast<double>(buckets) - quantiles.count());
            return placeholder;
        }
    }
    return quantiles;
}

//! Get the mean adjusted for empty buckets.
double correctForEmptyBuckets(bool ignoreEmpty,
                              uint64_t buckets,
                              const CBucketCountStatistics::TMoments &moments) {
    double n = maths::CBasicStatistics::count(moments);
    double m = maths::CBasicStatistics::mean(moments);
    return ignoreEmpty ? m : n / static_cast<double>(buckets) * m;
}

const uint64_t MINIMUM_BUCKETS_TO_TEST = 20;
const bool     IGNORE_EMPTY[] = { false, true };

}

CSparseCountPenalty::CSparseCountPenalty(const CAutoconfigurerParams &params) : CPenalty(params) {
}

CSparseCountPenalty *CSparseCountPenalty::clone(void) const {
    return new CSparseCountPenalty(*this);
}

std::string CSparseCountPenalty::name(void) const {
    return "sparse count penalty";
}

void CSparseCountPenalty::penaltyFromMe(CDetectorSpecification &spec) const {
    if (spec.function() != config_t::E_Count || spec.function() == config_t::E_Sum) {
        return;
    }

    typedef std::vector<TDoubleVec>                                             TDoubleVecVec;
    typedef CBucketCountStatistics::TSizeSizePrQuantileUMap                     TSizeSizePrQuantileUMap;
    typedef TSizeSizePrQuantileUMap::const_iterator                             TSizeSizePrQuantileUMapCItr;
    typedef std::vector<const TSizeSizePrQuantileUMap *>                        TSizeSizePrQuantileUMapCPtrVec;
    typedef std::vector<const CBucketCountStatistics::TSizeSizePrMomentsUMap *> TSizeSizePrMomentsUMapCPtrVec;
    typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator          TMeanAccumulator;
    typedef std::vector<TMeanAccumulator>                                       TMeanAccumulatorVec;

    if (const CDataCountStatistics *stats = spec.countStatistics()) {
        const CAutoconfigurerParams::TTimeVec &candidates = this->params().candidateBucketLengths();

        TSizeSizePrQuantileUMapCPtrVec quantiles;
        quantiles.reserve(candidates.size());
        TSizeSizePrMomentsUMapCPtrVec moments;
        moments.reserve(candidates.size());
        core_t::TTime longest = 0;

        for (std::size_t bid = 0u; bid < candidates.size(); ++bid) {
            if (stats->bucketCounts()[bid] > MINIMUM_BUCKETS_TO_TEST) {
                quantiles.push_back(&(stats->bucketStatistics()[bid].countQuantilesPerPartition()));
                moments.push_back(  &(stats->bucketStatistics()[bid].countMomentsPerPartition()));
                longest = std::max(longest, candidates[bid]);
            }
        }

        if (quantiles.size() > 3) {
            for (std::size_t iid = 0u; iid < boost::size(IGNORE_EMPTY); ++iid) {
                std::size_t            nb = quantiles.size();
                std::size_t            nq = 19;
                TDoubleVecVec          xq(nb, TDoubleVec(nq));
                TDoubleVec             means(nb);
                TDoubleVec             counts(nb);
                TDoubleVec             significances(nb - 1);
                TMeanAccumulatorVec    penalties_(nb - 1);
                maths::CQuantileSketch placeholder(maths::CQuantileSketch::E_Linear, 1);

                for (TSizeSizePrQuantileUMapCItr q0 = quantiles[0]->begin(); q0 != quantiles[0]->end(); ++q0) {
                    const CBucketCountStatistics::TSizeSizePr &partition = q0->first;

                    uint64_t                               bc = stats->bucketCounts()[0];
                    const maths::CQuantileSketch           &          qe0 = correctForEmptyBuckets(IGNORE_EMPTY[iid], bc, placeholder, q0->second);
                    const CBucketCountStatistics::TMoments &m0 = moments[0]->find(partition)->second;
                    double                                 me0 = correctForEmptyBuckets(IGNORE_EMPTY[iid], bc, m0);
                    extract(qe0, nq, xq[0]);
                    means[0] = me0;
                    counts[0] = maths::CBasicStatistics::count(m0);

                    bool skip = false;
                    for (std::size_t bid = 1u; bid < nb; ++bid) {
                        TSizeSizePrQuantileUMapCItr qi = quantiles[bid]->find(partition);
                        if (qi == quantiles[bid]->end()) {
                            skip = true;
                            break;
                        }

                        bc = stats->bucketCounts()[bid];
                        const maths::CQuantileSketch           &          qei = correctForEmptyBuckets(IGNORE_EMPTY[iid], bc, placeholder, qi->second);
                        const CBucketCountStatistics::TMoments &mi = moments[bid]->find(partition)->second;
                        double                                 mei = correctForEmptyBuckets(IGNORE_EMPTY[iid], bc, mi);
                        extract(qei, nq, xq[bid]);
                        means[bid] = mei;
                        counts[bid] = maths::CBasicStatistics::count(mi);
                    }
                    if (skip) {
                        continue;
                    }

                    std::fill_n(significances.begin(), nb - 1, 0.0);
                    for (std::size_t i = 0u; i < 2; ++i) {
                        for (std::size_t bid = 0u; bid + 1 < nb; ++bid) {
                            significances[bid] = std::max(significances[bid],
                                                          maths::CStatisticalTests::twoSampleKS(xq[bid], xq[nb - 1]));
                        }

                        // If the rate is high w.r.t. the bucket length we expect the mean and variance
                        // to scale as the ratio of bucket lengths. In order to test if the distribution
                        // is similar under this transformation the quantile points are adjusted by the
                        // map x -> L / l * m + (L / l)^(1/2) * (x - m), where l is the source bucket
                        // length and L is the target bucket length. Note that the resulting moments
                        // of the distribution are scaled appropriately as can be verified from their
                        // definition in terms of the integral of the derivative of the distribution.

                        for (std::size_t bid = 0u; bid < nb; ++bid) {
                            if (longest == candidates[bid]) {
                                continue;
                            }
                            double scale = static_cast<double>(longest) / static_cast<double>(candidates[bid]);
                            for (std::size_t j = 0u; j < xq[bid].size(); ++j) {
                                xq[bid][j] = scale * means[bid] + ::sqrt(scale) * (xq[bid][j] - means[bid]);
                            }
                        }
                    }

                    for (std::size_t bid = 0u; bid + 1 < nb; ++bid) {
                        double pi = std::min(10.0 * significances[bid], 1.0);
                        penalties_[bid].add(std::min(maths::CTools::fastLog(pi), 0.0), counts[bid]);
                    }
                }

                TSizeVec   indices;
                TDoubleVec penalties;
                TStrVec    descriptions;
                indices.reserve(2 * (nb - 1));
                penalties.reserve(2 * (nb - 1));
                descriptions.reserve(2 * (nb - 1));

                for (std::size_t bid = 0u; bid < penalties_.size(); ++bid) {
                    std::size_t index = this->params().penaltyIndexFor(bid, IGNORE_EMPTY[iid]);
                    indices.push_back(index);
                    double      penalty = ::exp(maths::CBasicStatistics::mean(penalties_[bid]));
                    std::string description;
                    if (penalty < 1.0) {
                        description = "The bucket length does not properly capture the variation in event rate";
                    }
                    penalties.push_back(penalty);
                    descriptions.push_back(description);
                }

                spec.applyPenalties(indices, penalties, descriptions);
            }
        }
    }
}

}
}
