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

#include <config/CFieldRolePenalty.h>

#include <maths/CTools.h>

#include <config/CDataSummaryStatistics.h>
#include <config/CFieldStatistics.h>
#include <config/CTools.h>
#include <config/Constants.h>

#include <cstddef>

namespace ml {
namespace config {
namespace {

const std::string EMPTY;
const std::string SPACE(" ");

//! Get the penalty description prefix.
std::string prefix(const std::string& description) {
    return description.empty() ? EMPTY : SPACE;
}
}

//////// CCantBeNumeric ////////

CCantBeNumeric::CCantBeNumeric(const CAutoconfigurerParams& params) : CPenalty(params) {
}

CCantBeNumeric* CCantBeNumeric::clone(void) const {
    return new CCantBeNumeric(*this);
}

std::string CCantBeNumeric::name(void) const {
    return "can't be numeric";
}

void CCantBeNumeric::penaltyFromMe(const CFieldStatistics& stats, double& penalty, std::string& description) const {
    if (config_t::isNumeric(stats.type())) {
        penalty = 0.0;
        description += prefix(description) + "Can't use numeric";
    }
}

//////// CCantBeCategorical ////////

CCantBeCategorical::CCantBeCategorical(const CAutoconfigurerParams& params) : CPenalty(params) {
}

CCantBeCategorical* CCantBeCategorical::clone(void) const {
    return new CCantBeCategorical(*this);
}

std::string CCantBeCategorical::name(void) const {
    return "Can't be categorical";
}

void CCantBeCategorical::penaltyFromMe(const CFieldStatistics& stats, double& penalty, std::string& description) const {
    if (config_t::isCategorical(stats.type())) {
        penalty = 0.0;
        description += prefix(description) + "Can't use categorical";
    }
}

//////// CDontUseUnaryField ////////

CDontUseUnaryField::CDontUseUnaryField(const CAutoconfigurerParams& params) : CPenalty(params) {
}

CDontUseUnaryField* CDontUseUnaryField::clone(void) const {
    return new CDontUseUnaryField(*this);
}

std::string CDontUseUnaryField::name(void) const {
    return "don't use unary field";
}

void CDontUseUnaryField::penaltyFromMe(const CFieldStatistics& stats, double& penalty, std::string& description) const {
    if (const CCategoricalDataSummaryStatistics* summary = stats.categoricalSummary()) {
        if (summary->distinctCount() == 1) {
            penalty = 0.0;
            description += prefix(description) + "There's no point using a unary field";
        }
    }
}

//////// CDistinctCountThreshold ////////

CDistinctCountThresholdPenalty::CDistinctCountThresholdPenalty(const CAutoconfigurerParams& params,
                                                               std::size_t distinctCountForPenaltyOfOne,
                                                               std::size_t distinctCountForPenaltyOfZero)
    : CPenalty(params),
      m_DistinctCountForPenaltyOfOne(static_cast<double>(distinctCountForPenaltyOfOne)),
      m_DistinctCountForPenaltyOfZero(static_cast<double>(distinctCountForPenaltyOfZero)) {
}

CDistinctCountThresholdPenalty* CDistinctCountThresholdPenalty::clone(void) const {
    return new CDistinctCountThresholdPenalty(*this);
}

std::string CDistinctCountThresholdPenalty::name(void) const {
    return "distinct count thresholds " + core::CStringUtils::typeToString(m_DistinctCountForPenaltyOfZero) + " and " +
           core::CStringUtils::typeToString(m_DistinctCountForPenaltyOfOne);
}

void CDistinctCountThresholdPenalty::penaltyFromMe(const CFieldStatistics& stats, double& penalty, std::string& description) const {
    if (const CCategoricalDataSummaryStatistics* summary = stats.categoricalSummary()) {
        double penalty_ = CTools::interpolate(
            m_DistinctCountForPenaltyOfZero, m_DistinctCountForPenaltyOfOne, 0.0, 1.0, static_cast<double>(summary->distinctCount()));
        if (penalty_ < 1.0) {
            penalty *= penalty_;
            description += prefix(description) + "A distinct count of " + core::CStringUtils::typeToString(summary->distinctCount()) +
                           " is" + (penalty_ == 0.0 ? " too " : " ") +
                           (m_DistinctCountForPenaltyOfZero > m_DistinctCountForPenaltyOfOne ? "high" : "low");
        }
    }
}
}
}
