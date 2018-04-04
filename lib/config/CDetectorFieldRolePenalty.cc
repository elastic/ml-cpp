/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <config/CDetectorFieldRolePenalty.h>

#include <core/CLogger.h>

#include <config/CDetectorSpecification.h>

#include <algorithm>
#include <cstddef>
#include <string>

namespace ml {
namespace config {
namespace {

using TGetStatistics = const CFieldStatistics* (CDetectorSpecification::*)() const;
const TGetStatistics STATISTIC[] = {
    &CDetectorSpecification::argumentFieldStatistics,
    &CDetectorSpecification::byFieldStatistics,
    &CDetectorSpecification::overFieldStatistics,
    &CDetectorSpecification::partitionFieldStatistics,
};
}

CDetectorFieldRolePenalty::CDetectorFieldRolePenalty(const CAutoconfigurerParams& params) : CPenalty(params) {
    std::fill_n(m_FieldRolePenalties, constants::NUMBER_FIELD_INDICES, static_cast<const CPenalty*>(0));
}

CDetectorFieldRolePenalty* CDetectorFieldRolePenalty::clone() const {
    return new CDetectorFieldRolePenalty(*this);
}

std::string CDetectorFieldRolePenalty::name() const {
    std::string arguments;
    for (std::size_t i = 0u; i < constants::NUMBER_FIELD_INDICES; ++i) {
        if (m_FieldRolePenalties[i]) {
            arguments += (arguments.empty() ? "'" : ", '") + constants::name(i) + ' ' + m_FieldRolePenalties[i]->name() + "'";
        }
    }
    return "field role penalty(" + arguments + ")";
}

void CDetectorFieldRolePenalty::addPenalty(std::size_t index, const CPenalty& penalty) {
    m_FieldRolePenalties[index] = &penalty;
}

void CDetectorFieldRolePenalty::penaltyFromMe(CDetectorSpecification& spec) const {
    double penalty = 1.0;
    for (std::size_t i = 0u; i < constants::NUMBER_FIELD_INDICES; ++i) {
        if (const CFieldStatistics* stats = (spec.*STATISTIC[i])()) {
            std::string description;
            m_FieldRolePenalties[i]->penalty(*stats, penalty, description);
            if (!description.empty()) {
                description += " for the '" + constants::name(i) + "' field";
            }
            spec.applyPenalty(penalty, description);
        }
    }
}
}
}
