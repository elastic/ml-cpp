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

#include <config/CLowVariationPenalty.h>

#include <maths/CTools.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CDataCountStatistics.h>
#include <config/CDetectorSpecification.h>
#include <config/CTools.h>

#include <boost/unordered_map.hpp>

#include <vector>

namespace ml {
namespace config {
namespace {

typedef std::vector<double> TDoubleVec;
typedef std::vector<std::size_t> TSizeVec;
typedef std::vector<std::string> TStrVec;
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
typedef maths::CBasicStatistics::SSampleMeanVarSkew<double>::TAccumulator TMomentsAccumulator;
typedef boost::unordered_map<std::size_t, TMomentsAccumulator> TSizeMomentsUMap;

const double MIN = 0.9 * constants::DETECTOR_SCORE_EPSILON / constants::MAXIMUM_DETECTOR_SCORE;
const double INF = boost::numeric::bounds<double>::highest();

//! Compute the coefficient of variation from \p moments.
template <typename MOMENTS> double cov(const MOMENTS &moments) {
    double m = ::fabs(maths::CBasicStatistics::mean(moments));
    double sd = ::sqrt(maths::CBasicStatistics::maximumLikelihoodVariance(moments));
    return sd == 0.0 ? 0.0 : (m == 0.0 ? INF : sd / m);
}

//! Compute the penalty for the partition moments \p moments.
template <typename MOMENTS>
void penaltyImpl(const CAutoconfigurerParams &params,
                 const MOMENTS &moments,
                 double &penalty,
                 double &proportionWithLowVariation) {
    TMeanAccumulator penalty_;
    proportionWithLowVariation = 0.0;
    for (typename MOMENTS::const_iterator i = moments.begin(); i != moments.end(); ++i) {
        double pi = CTools::logInterpolate(params.lowCoefficientOfVariation(),
                                           params.minimumCoefficientOfVariation(),
                                           1.0,
                                           MIN,
                                           cov(i->second));
        penalty_.add(maths::CTools::fastLog(pi), maths::CBasicStatistics::count(i->second));
        if (pi < 1.0) {
            proportionWithLowVariation += 1.0;
        }
    }
    penalty = std::min(::exp(maths::CBasicStatistics::mean(penalty_)), 1.0);
    proportionWithLowVariation /= static_cast<double>(moments.size());
}

//! Compute the distinct count penalty for the partition moments \p moments.
struct SDistinctCountPenalty {
    template <typename MOMENTS>
    void operator()(const CAutoconfigurerParams &params,
                    const MOMENTS &moments,
                    double &penalty,
                    double &proportionWithLowVariation) const {
        TMeanAccumulator penalty_;
        for (typename MOMENTS::const_iterator i = moments.begin(); i != moments.end(); ++i) {
            double pi = CTools::logInterpolate(params.lowCoefficientOfVariation(),
                                               params.minimumCoefficientOfVariation(),
                                               1.0,
                                               MIN,
                                               cov(i->second.s_DistinctCount));
            penalty_.add(maths::CTools::fastLog(pi),
                         maths::CBasicStatistics::count(i->second.s_DistinctCount));
            if (pi < 1.0) {
                proportionWithLowVariation += 1.0;
            }
        }
        penalty = std::min(::exp(maths::CBasicStatistics::mean(penalty_)), 1.0);
        proportionWithLowVariation /= static_cast<double>(moments.size());
    }
};

//! Compute the info content penalty for the partition moments \p moments.
struct SInfoContentPenalty {
    template <typename MOMENTS>
    void operator()(const CAutoconfigurerParams &params,
                    const MOMENTS &moments,
                    double &penalty,
                    double &proportionWithLowVariation) const {
        TMeanAccumulator penalty_;
        for (typename MOMENTS::const_iterator i = moments.begin(); i != moments.end(); ++i) {
            double pi = CTools::logInterpolate(params.lowCoefficientOfVariation(),
                                               params.minimumCoefficientOfVariation(),
                                               1.0,
                                               MIN,
                                               cov(i->second.s_InfoContent));
            penalty_.add(maths::CTools::fastLog(pi),
                         maths::CBasicStatistics::count(i->second.s_InfoContent));
            if (pi < 1.0) {
                proportionWithLowVariation += 1.0;
            }
        }
        penalty = std::min(::exp(maths::CBasicStatistics::mean(penalty_)), 1.0);
        proportionWithLowVariation /= static_cast<double>(moments.size());
    }
};

//! Get the description prefix.
std::string descriptionPrefix(const CDetectorSpecification &spec,
                              double proportionWithLowVariation) {
    if (spec.byField() && spec.partitionField()) {
        return "A significant proportion, " +
               CTools::prettyPrint(100.0 * proportionWithLowVariation) +
               "%, of distinct partition and by fields combinations";
    }
    if (spec.byField()) {
        return "A significant proportion, " +
               CTools::prettyPrint(100.0 * proportionWithLowVariation) + "%, of distinct by fields";
    }
    if (spec.partitionField()) {
        return "A significant proportion, " +
               CTools::prettyPrint(100.0 * proportionWithLowVariation) +
               "%, of distinct partition fields";
    }
    return "";
}

//! Apply the penalties for count analysis from \p stats.
template <typename STATS>
void penaltyForCountImpl(const CAutoconfigurerParams &params,
                         const STATS &stats,
                         CDetectorSpecification &spec) {
    std::size_t n = stats.bucketStatistics().size();

    TSizeVec indices;
    TDoubleVec penalties;
    TStrVec descriptions;
    indices.reserve(2 * n);
    penalties.reserve(2 * n);
    descriptions.reserve(2 * n);

    for (std::size_t bid = 0u; bid < n; ++bid) {
        const TSizeVec &indices_ = params.penaltyIndicesFor(bid);
        double penalty;
        double proportionWithLowVariation;
        penaltyImpl(params,
                    stats.bucketStatistics()[bid].countMomentsPerPartition(),
                    penalty,
                    proportionWithLowVariation);
        indices.insert(indices.end(), indices_.begin(), indices_.end());
        std::string description;
        if (penalty < 1.0) {
            if (spec.byField() || spec.partitionField()) {
                description = descriptionPrefix(spec, proportionWithLowVariation) + " have " +
                              (penalty == MIN ? "too " : "") + "low" +
                              " variation in their bucket counts";
            } else {
                description = std::string("The variation in the bucket counts is ") +
                              (penalty == MIN ? "too " : "") + "low";
            }
        }
        std::fill_n(std::back_inserter(penalties), indices_.size(), penalty);
        std::fill_n(std::back_inserter(descriptions), indices_.size(), description);
    }

    spec.applyPenalties(indices, penalties, descriptions);
}

//! Apply the penalties for distinct count analysis from \p stats.
template <typename STATS, typename PENALTY>
void penaltyForImpl(const CAutoconfigurerParams &params,
                    const STATS &stats,
                    PENALTY computePenalty,
                    const std::string &function,
                    CDetectorSpecification &spec) {
    std::size_t n = stats.bucketStatistics().size();

    TSizeVec indices;
    TDoubleVec penalties;
    TStrVec descriptions;
    indices.reserve(2 * n);
    penalties.reserve(2 * n);
    descriptions.reserve(2 * n);

    for (std::size_t bid = 0u; bid < n; ++bid) {
        const TSizeVec &indices_ = params.penaltyIndicesFor(bid);
        indices.insert(indices.end(), indices_.begin(), indices_.end());
        const std::string &argument = *spec.argumentField();
        double penalty = 0.0;
        double proportionWithLowVariation = 0.0;
        computePenalty(params,
                       stats.bucketStatistics()[bid].argumentMomentsPerPartition(argument),
                       penalty,
                       proportionWithLowVariation);
        std::string description;
        if (penalty < 1.0) {
            if (spec.byField() || spec.partitionField()) {
                description = descriptionPrefix(spec, proportionWithLowVariation) + " have " +
                              (penalty == MIN ? "too " : "") + "low" +
                              " variation in their bucket " + function;
            } else {
                description = std::string("The variation in the bucket ") + function + " is " +
                              (penalty == MIN ? "too " : "") + "low";
            }
        }
        std::fill_n(std::back_inserter(penalties), indices_.size(), penalty);
        std::fill_n(std::back_inserter(descriptions), indices_.size(), description);
    }

    spec.applyPenalties(indices, penalties, descriptions);
}
}

CLowVariationPenalty::CLowVariationPenalty(const CAutoconfigurerParams &params)
    : CPenalty(params) {}

CLowVariationPenalty *CLowVariationPenalty::clone(void) const {
    return new CLowVariationPenalty(*this);
}

std::string CLowVariationPenalty::name(void) const { return "low variation"; }

void CLowVariationPenalty::penaltyFromMe(CDetectorSpecification &spec) const {
#define APPLY_COUNTING_PENALTY(penalty)                                                            \
    if (const CDataCountStatistics *stats_ = spec.countStatistics()) {                             \
        if (const CPartitionDataCountStatistics *partitionStats =                                  \
                dynamic_cast<const CPartitionDataCountStatistics *>(stats_)) {                     \
            this->penalty(*partitionStats, spec);                                                  \
        } else if (const CByAndPartitionDataCountStatistics *byAndPartitionStats =                 \
                       dynamic_cast<const CByAndPartitionDataCountStatistics *>(stats_)) {         \
            this->penalty(*byAndPartitionStats, spec);                                             \
        } else if (const CByOverAndPartitionDataCountStatistics *byOverAndPartitionStats =         \
                       dynamic_cast<const CByOverAndPartitionDataCountStatistics *>(stats_)) {     \
            this->penalty(*byOverAndPartitionStats, spec);                                         \
        }                                                                                          \
    }

    switch (spec.function()) {
        case config_t::E_Count:
            APPLY_COUNTING_PENALTY(penaltiesForCount) break;
        case config_t::E_Rare:
            break;
        case config_t::E_DistinctCount:
            APPLY_COUNTING_PENALTY(penaltyForDistinctCount) break;
        case config_t::E_InfoContent:
            APPLY_COUNTING_PENALTY(penaltyForInfoContent) break;
        case config_t::E_Mean:
        case config_t::E_Min:
        case config_t::E_Max:
        case config_t::E_Sum:
        case config_t::E_Varp:
        case config_t::E_Median:
            break;
    }
}

void CLowVariationPenalty::penaltiesForCount(const CPartitionDataCountStatistics &stats,
                                             CDetectorSpecification &spec) const {
    penaltyForCountImpl(this->params(), stats, spec);
}

void CLowVariationPenalty::penaltiesForCount(const CByAndPartitionDataCountStatistics &stats,
                                             CDetectorSpecification &spec) const {
    penaltyForCountImpl(this->params(), stats, spec);
}

void CLowVariationPenalty::penaltiesForCount(const CByOverAndPartitionDataCountStatistics &stats,
                                             CDetectorSpecification &spec) const {
    penaltyForCountImpl(this->params(), stats, spec);
}

void CLowVariationPenalty::penaltyForDistinctCount(const CPartitionDataCountStatistics &stats,
                                                   CDetectorSpecification &spec) const {
    penaltyForImpl(this->params(), stats, SDistinctCountPenalty(), "distinct counts", spec);
}

void CLowVariationPenalty::penaltyForDistinctCount(const CByAndPartitionDataCountStatistics &stats,
                                                   CDetectorSpecification &spec) const {
    penaltyForImpl(this->params(), stats, SDistinctCountPenalty(), "distinct counts", spec);
}

void CLowVariationPenalty::penaltyForDistinctCount(
    const CByOverAndPartitionDataCountStatistics &stats,
    CDetectorSpecification &spec) const {
    penaltyForImpl(this->params(), stats, SDistinctCountPenalty(), "distinct counts", spec);
}

void CLowVariationPenalty::penaltyForInfoContent(const CPartitionDataCountStatistics &stats,
                                                 CDetectorSpecification &spec) const {
    penaltyForImpl(this->params(), stats, SInfoContentPenalty(), "info content", spec);
}

void CLowVariationPenalty::penaltyForInfoContent(const CByAndPartitionDataCountStatistics &stats,
                                                 CDetectorSpecification &spec) const {
    penaltyForImpl(this->params(), stats, SInfoContentPenalty(), "info content", spec);
}

void CLowVariationPenalty::penaltyForInfoContent(
    const CByOverAndPartitionDataCountStatistics &stats,
    CDetectorSpecification &spec) const {
    penaltyForImpl(this->params(), stats, SInfoContentPenalty(), "info content", spec);
}
}
}
