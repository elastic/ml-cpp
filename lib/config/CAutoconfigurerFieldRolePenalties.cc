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

#include <config/CAutoconfigurerFieldRolePenalties.h>

#include <core/CLogger.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CFieldRolePenalty.h>
#include <config/Constants.h>

#include <boost/range.hpp>

namespace ml {
namespace config {
namespace {
const std::size_t CATEGORICAL_ARGUMENT_INDEX = 0u;
const std::size_t METRIC_ARGUMENT_INDEX = 1u;
const std::size_t BY_INDEX = 2u;
const std::size_t RARE_BY_INDEX = 3u;
const std::size_t OVER_INDEX = 4u;
const std::size_t PARTITION_INDEX = 5u;

using TCountThreshold = std::size_t (CAutoconfigurerParams::*)() const;

const std::size_t PENALTY_INDICES[] = {BY_INDEX, RARE_BY_INDEX, OVER_INDEX, PARTITION_INDEX};
const TCountThreshold PENALTY_THRESHOLD[] = {&CAutoconfigurerParams::highNumberByFieldValues,
                                             &CAutoconfigurerParams::highNumberRareByFieldValues,
                                             &CAutoconfigurerParams::lowNumberOverFieldValues,
                                             &CAutoconfigurerParams::highNumberPartitionFieldValues};
const TCountThreshold HARD_CUTOFF[] = {&CAutoconfigurerParams::maximumNumberByFieldValues,
                                       &CAutoconfigurerParams::maximumNumberRareByFieldValues,
                                       &CAutoconfigurerParams::minimumNumberOverFieldValues,
                                       &CAutoconfigurerParams::maximumNumberPartitionFieldValues};
}

CAutoconfigurerFieldRolePenalties::CAutoconfigurerFieldRolePenalties(const CAutoconfigurerParams& params) {
    m_Penalties[CATEGORICAL_ARGUMENT_INDEX].reset((CCantBeNumeric(params) * CDontUseUnaryField(params)).clone());
    m_Penalties[METRIC_ARGUMENT_INDEX].reset(new CCantBeCategorical(params));
    for (std::size_t i = 0u; i < boost::size(PENALTY_INDICES); ++i) {
        m_Penalties[PENALTY_INDICES[i]].reset(
            (CCantBeNumeric(params) * CDistinctCountThresholdPenalty(params, (params.*PENALTY_THRESHOLD[i])(), (params.*HARD_CUTOFF[i])()) *
             CDontUseUnaryField(params))
                .clone());
    }
}

const CPenalty& CAutoconfigurerFieldRolePenalties::categoricalFunctionArgumentPenalty() const {
    return *m_Penalties[CATEGORICAL_ARGUMENT_INDEX];
}

const CPenalty& CAutoconfigurerFieldRolePenalties::metricFunctionArgumentPenalty() const {
    return *m_Penalties[METRIC_ARGUMENT_INDEX];
}

const CPenalty& CAutoconfigurerFieldRolePenalties::byPenalty() const {
    return *m_Penalties[BY_INDEX];
}

const CPenalty& CAutoconfigurerFieldRolePenalties::rareByPenalty() const {
    return *m_Penalties[RARE_BY_INDEX];
}

const CPenalty& CAutoconfigurerFieldRolePenalties::overPenalty() const {
    return *m_Penalties[OVER_INDEX];
}

const CPenalty& CAutoconfigurerFieldRolePenalties::partitionPenalty() const {
    return *m_Penalties[PARTITION_INDEX];
}
}
}
