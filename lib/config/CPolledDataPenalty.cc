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

#include <config/CPolledDataPenalty.h>

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/COrderings.h>
#include <maths/CQuantileSketch.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CDataCountStatistics.h>
#include <config/CDetectorSpecification.h>
#include <config/CTools.h>

#include <algorithm>
#include <math.h>
#include <vector>

namespace ml {
namespace config {
namespace {
const double LOG_TENTH_NUMBER_POLLING_INTERVALS = 10.0;
}

CPolledDataPenalty::CPolledDataPenalty(const CAutoconfigurerParams &params) : CPenalty(params) {}

CPolledDataPenalty *CPolledDataPenalty::clone(void) const { return new CPolledDataPenalty(*this); }

std::string CPolledDataPenalty::name(void) const { return "polled data penalty"; }

void CPolledDataPenalty::penaltyFromMe(CDetectorSpecification &spec) const {
    if (const CDataCountStatistics *stats = spec.countStatistics()) {
        if (TOptionalTime interval = this->pollingInterval(*stats)) {
            const TTimeVec &candidates = this->params().candidateBucketLengths();

            TSizeVec indices;
            TDoubleVec penalties;
            TStrVec descriptions;
            indices.reserve(2 * candidates.size());
            penalties.reserve(2 * candidates.size());
            descriptions.reserve(2 * candidates.size());

            for (std::size_t bid = 0u; bid < candidates.size(); ++bid) {
                if (candidates[bid] < *interval) {
                    const TSizeVec &indices_ = this->params().penaltyIndicesFor(bid);
                    indices.insert(indices.end(), indices_.begin(), indices_.end());
                    std::fill_n(std::back_inserter(penalties),
                                indices_.size(),
                                ::pow(0.1,
                                      static_cast<double>(stats->timeRange()) /
                                          static_cast<double>(*interval) /
                                          LOG_TENTH_NUMBER_POLLING_INTERVALS));
                    std::fill_n(std::back_inserter(descriptions),
                                indices_.size(),
                                CTools::prettyPrint(candidates[bid]) +
                                    " is shorter than possible polling interval " +
                                    CTools::prettyPrint(*interval));
                }
            }

            spec.applyPenalties(indices, penalties, descriptions);
        }
    }
}

CPolledDataPenalty::TOptionalTime
CPolledDataPenalty::pollingInterval(const CDataCountStatistics &stats) const {
    typedef maths::CBasicStatistics::COrderStatisticsStack<maths::CQuantileSketch::TFloatFloatPr,
                                                           2,
                                                           maths::COrderings::SSecondGreater>
        TMaxAccumulator;

    const maths::CQuantileSketch &F = stats.arrivalTimeDistribution();
    const maths::CQuantileSketch::TFloatFloatPrVec &knots = F.knots();
    if (knots.size() == 1) {
        return static_cast<core_t::TTime>(knots[0].first);
    }

    // Find the two biggest steps in the c.d.f.

    TMaxAccumulator steps;
    for (std::size_t i = 0u; i < knots.size(); ++i) {
        steps.add(knots[i]);
    }

    // Check that nearly all the probability mass is in these two steps
    // and that the value of the smaller abscissa is much less than the
    // value of the larger abscissa.
    double lower = steps[0].first;
    double upper = steps[1].first;
    double mass = (steps[0].second + steps[1].second) / F.count();
    if (lower > upper) {
        std::swap(lower, upper);
    }

    double f[4];
    F.cdf(lower - 0.01 * upper, f[0]);
    F.cdf(lower + 0.01 * upper, f[1]);
    F.cdf(upper - 0.01 * upper, f[2]);
    F.cdf(upper + 0.01 * upper, f[3]);
    mass = f[1] - f[0] + f[3] - f[2];

    if (mass > this->params().polledDataMinimumMassAtInterval() &&
        lower < this->params().polledDataJitter() * upper) {
        return static_cast<core_t::TTime>(upper);
    } else {
    }

    return TOptionalTime();
}
}
}
