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

#include <config/CPenalty.h>

#include <core/CLogger.h>

#include <maths/CTools.h>

#include <config/CDetectorSpecification.h>
#include <config/Constants.h>

namespace ml {
namespace config {
namespace {
const std::string PENALTY_NAME("CPenalty");
}

CPenalty::CPenalty(const CAutoconfigurerParams& params) : m_Params(params) {
}

CPenalty::CPenalty(const CPenalty& other) : m_Params(other.m_Params) {
    m_Penalties.reserve(other.m_Penalties.size());
    for (std::size_t i = 0u; i < other.m_Penalties.size(); ++i) {
        m_Penalties.push_back(TPenaltyCPtr(other.m_Penalties[i]->clone()));
    }
}

CPenalty::CPenalty(CClosure closure) : m_Params(closure.penalties()[0]->params()) {
    m_Penalties.swap(closure.penalties());
}

CPenalty::~CPenalty(void) {
}

std::string CPenalty::name(void) const {
    std::string result;
    for (std::size_t i = 0u; i < m_Penalties.size(); ++i) {
        result += (result.empty() ? "'" : " x '") + m_Penalties[i]->name() + "'";
    }
    return result;
}

CPenalty* CPenalty::clone(void) const {
    return new CPenalty(*this);
}

const CPenalty& CPenalty::operator*=(const CPenalty& rhs) {
    m_Penalties.push_back(TPenaltyCPtr(rhs.clone()));
    return *this;
}

const CPenalty& CPenalty::operator*=(CClosure rhs) {
    m_Penalties.insert(m_Penalties.end(), rhs.penalties().begin(), rhs.penalties().end());
    return *this;
}

void CPenalty::penalty(const CFieldStatistics& stats,
                       double& penalty,
                       std::string& description) const {
    this->penaltyFromMe(stats, penalty, description);
    if (scoreIsZeroFor(penalty)) {
        return;
    }
    for (std::size_t i = 0u; i < m_Penalties.size(); ++i) {
        m_Penalties[i]->penalty(stats, penalty, description);
        if (scoreIsZeroFor(penalty)) {
            break;
        }
    }
}

void CPenalty::penalize(CDetectorSpecification& spec) const {
    this->penaltyFromMe(spec);
    if (spec.score() == 0.0) {
        return;
    }
    for (std::size_t i = 0u; i < m_Penalties.size(); ++i) {
        LOG_TRACE("Applying '" << m_Penalties[i]->name() << "' to " << spec.description());
        m_Penalties[i]->penalize(spec);
        if (spec.score() == 0.0) {
            break;
        }
    }
}

double CPenalty::score(double penalty) {
    return constants::DETECTOR_SCORE_EPSILON *
           ::floor(constants::MAXIMUM_DETECTOR_SCORE * penalty / constants::DETECTOR_SCORE_EPSILON);
}

bool CPenalty::scoreIsZeroFor(double penalty) {
    return penalty * constants::MAXIMUM_DETECTOR_SCORE < constants::DETECTOR_SCORE_EPSILON;
}

const CAutoconfigurerParams& CPenalty::params(void) const {
    return m_Params;
}

void CPenalty::penaltyFromMe(const CFieldStatistics& /*stats*/,
                             double& /*penalty*/,
                             std::string& /*description*/) const {
}

void CPenalty::penaltyFromMe(CDetectorSpecification& /*spec*/) const {
}

CPenalty::CClosure::CClosure(const CPenalty& penalty) {
    this->add(penalty);
}

CPenalty* CPenalty::CClosure::clone(void) const {
    return new CPenalty(*this);
}

CPenalty::CClosure& CPenalty::CClosure::add(const CPenalty& penalty) {
    m_Penalties.push_back(TPenaltyCPtr(penalty.clone()));
    return *this;
}

CPenalty::TPenaltyCPtrVec& CPenalty::CClosure::penalties(void) {
    return m_Penalties;
}

CPenalty::CClosure operator*(const CPenalty& lhs, const CPenalty& rhs) {
    return CPenalty::CClosure(lhs).add(rhs);
}
CPenalty::CClosure operator*(CPenalty::CClosure lhs, const CPenalty& rhs) {
    return lhs.add(rhs);
}

CPenalty::CClosure operator*(const CPenalty& lhs, CPenalty::CClosure rhs) {
    return rhs.add(lhs);
}
}
}
