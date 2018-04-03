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

#include <config/CLowInformationContentPenalty.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CDataSummaryStatistics.h>
#include <config/CDetectorSpecification.h>
#include <config/CFieldStatistics.h>
#include <config/CTools.h>

#include <math.h>

namespace ml {
namespace config {
namespace {
const double LOG_MIN = 0.5 * ::log(0.9 * constants::DETECTOR_SCORE_EPSILON / constants::MAXIMUM_DETECTOR_SCORE);
}

CLowInformationContentPenalty::CLowInformationContentPenalty(const CAutoconfigurerParams& params) : CPenalty(params) {
}

CLowInformationContentPenalty* CLowInformationContentPenalty::clone(void) const {
    return new CLowInformationContentPenalty(*this);
}

std::string CLowInformationContentPenalty::name(void) const {
    return "low information content";
}

void CLowInformationContentPenalty::penaltyFromMe(CDetectorSpecification& spec) const {
    if (config_t::isInfoContent(spec.function())) {
        if (const CFieldStatistics* stats = spec.argumentFieldStatistics()) {
            if (const CCategoricalDataSummaryStatistics* summary = stats->categoricalSummary()) {
                // Note that the empirical entropy is maximized when
                double minimumLength = static_cast<double>(summary->minimumLength());
                double maximumLength = static_cast<double>(summary->maximumLength());
                double cardinality = static_cast<double>(summary->distinctCount());
                double entropy = summary->entropy();
                double penalty = cardinality == 1.0 ? 0.0
                                                    : ::exp(CTools::interpolate(this->params().lowLengthRangeForInfoContent(),
                                                                                this->params().minimumLengthRangeForInfoContent(),
                                                                                0.0,
                                                                                LOG_MIN,
                                                                                maximumLength - minimumLength)) *
                                                          ::exp(CTools::interpolate(this->params().lowMaximumLengthForInfoContent(),
                                                                                    this->params().minimumMaximumLengthForInfoContent(),
                                                                                    0.0,
                                                                                    LOG_MIN,
                                                                                    maximumLength)) *
                                                          ::exp(CTools::logInterpolate(this->params().lowEntropyForInfoContent(),
                                                                                       this->params().minimumEntropyForInfoContent(),
                                                                                       0.0,
                                                                                       LOG_MIN,
                                                                                       entropy / ::log(cardinality))) *
                                                          ::exp(CTools::logInterpolate(this->params().lowDistinctCountForInfoContent(),
                                                                                       this->params().minimumDistinctCountForInfoContent(),
                                                                                       LOG_MIN,
                                                                                       0.0,
                                                                                       cardinality));
                std::string description;
                if (penalty < 1.0) {
                    description = "There is weak evidence that '" + *spec.argumentField() + "' carries information";
                }
                spec.applyPenalty(penalty, description);
            }
        }
    }
}
}
}
