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

#include <config/CAutoconfigurerDetectorPenalties.h>

#include <config/CAutoconfigurerFieldRolePenalties.h>
#include <config/CDetectorSpecification.h>
#include <config/ConfigTypes.h>

#include <config/CDetectorFieldRolePenalty.h>
#include <config/CLongTailPenalty.h>
#include <config/CLowInformationContentPenalty.h>
#include <config/CLowVariationPenalty.h>
#include <config/CNotEnoughDataPenalty.h>
#include <config/CPolledDataPenalty.h>
#include <config/CSpanTooSmallForBucketLengthPenalty.h>
#include <config/CSparseCountPenalty.h>
#include <config/CTooMuchDataPenalty.h>

#include <cstddef>

namespace ml {
namespace config {
namespace {

//! Get the index of the detector \p spec's field role penalty.
std::size_t fieldRolePenaltyIndex(const CDetectorSpecification& spec) {
    static const std::size_t SKIPS[] = {1, 2, 3, 6, 9, 18};
    return (spec.argumentField() ? SKIPS[0 + config_t::isMetric(spec.function())] : 0) +
           (spec.byField() ? SKIPS[2 + config_t::isRare(spec.function())] : 0) + (spec.overField() ? SKIPS[4] : 0) +
           (spec.partitionField() ? SKIPS[5] : 0);
}
}

CAutoconfigurerDetectorPenalties::CAutoconfigurerDetectorPenalties(const CAutoconfigurerParams& params,
                                                                   const CAutoconfigurerFieldRolePenalties& fieldRolePenalties)
    : m_Params(params), m_FieldRolePenalties(fieldRolePenalties) {
}

CAutoconfigurerDetectorPenalties::TPenaltyPtr CAutoconfigurerDetectorPenalties::penaltyFor(const CDetectorSpecification& spec) {
    return TPenaltyPtr((this->fieldRolePenalty(spec) * CSpanTooSmallForBucketLengthPenalty(m_Params) * CPolledDataPenalty(m_Params) *
                        CLongTailPenalty(m_Params) * CLowInformationContentPenalty(m_Params) * CNotEnoughDataPenalty(m_Params) *
                        CTooMuchDataPenalty(m_Params) * CLowVariationPenalty(m_Params) * CSparseCountPenalty(m_Params))
                           .clone());
}

const CPenalty& CAutoconfigurerDetectorPenalties::fieldRolePenalty(const CDetectorSpecification& spec) {
    m_DetectorFieldRolePenalties.resize(36);
    TPenaltyPtr& result = m_DetectorFieldRolePenalties[fieldRolePenaltyIndex(spec)];
    if (!result) {
        CDetectorFieldRolePenalty penalty(m_Params);
        const CAutoconfigurerFieldRolePenalties& penalties = m_FieldRolePenalties;
        if (spec.argumentField()) {
            penalty.addPenalty(constants::ARGUMENT_INDEX,
                               config_t::isMetric(spec.function()) ? penalties.metricFunctionArgumentPenalty()
                                                                   : penalties.categoricalFunctionArgumentPenalty());
        }
        if (spec.byField()) {
            penalty.addPenalty(constants::BY_INDEX, config_t::isRare(spec.function()) ? penalties.rareByPenalty() : penalties.byPenalty());
        }
        if (spec.overField()) {
            penalty.addPenalty(constants::OVER_INDEX, penalties.overPenalty());
        }
        if (spec.partitionField()) {
            penalty.addPenalty(constants::PARTITION_INDEX, penalties.partitionPenalty());
        }
        result.reset(penalty.clone());
    }
    return *result;
}
}
}
