/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <model/CAnnotatedProbabilityBuilder.h>

#include <maths/common/CMultinomialConjugate.h>
#include <maths/common/Constants.h>

#include <utility>

namespace ml {
namespace model {

CAnnotatedProbabilityBuilder::CAnnotatedProbabilityBuilder(SAnnotatedProbability& annotatedProbability)
    : m_Result(annotatedProbability), m_NumberAttributeProbabilities(1),
      m_AttributeProbabilityPrior(nullptr),
      m_PersonAttributeProbabilityPrior(nullptr), m_MinAttributeProbabilities(1),
      m_DistinctTotalAttributes(0), 
      m_IsPopulation(false), m_IsRare(false), m_IsFreqRare(false) {
    m_Result.s_AttributeProbabilities.clear();
    m_Result.s_Influences.clear();
}

CAnnotatedProbabilityBuilder::CAnnotatedProbabilityBuilder(SAnnotatedProbability& annotatedProbability,
                                                           std::size_t numberAttributeProbabilities,
                                                           function_t::EFunction function)
    : m_Result(annotatedProbability),
      m_NumberAttributeProbabilities(numberAttributeProbabilities),
      m_AttributeProbabilityPrior(nullptr),
      m_PersonAttributeProbabilityPrior(nullptr),
      m_MinAttributeProbabilities(numberAttributeProbabilities),
      m_DistinctTotalAttributes(0), m_IsPopulation(function_t::isPopulation(function)),
      m_IsRare(false), m_IsFreqRare(false) {
    m_Result.s_AttributeProbabilities.clear();
    m_Result.s_Influences.clear();

    if (function == function_t::E_IndividualRare || function == function_t::E_PopulationRare) {
        m_IsRare = true;
    } else if (function == function_t::E_PopulationFreqRare) {
        m_IsFreqRare = true;
    }
}

void CAnnotatedProbabilityBuilder::attributeProbabilityPrior(const maths::common::CMultinomialConjugate* prior) {
    m_AttributeProbabilityPrior = prior;
}

void CAnnotatedProbabilityBuilder::personAttributeProbabilityPrior(
    const maths::common::CMultinomialConjugate* prior) {
    m_PersonAttributeProbabilityPrior = prior;
}

void CAnnotatedProbabilityBuilder::probability(double p) {
    m_Result.s_Probability = p;
}

void CAnnotatedProbabilityBuilder::anomalyScoreExplanation(SAnnotatedProbability::TAnomalyScoreExplanation explanation) {
    m_Result.s_AnomalyScoreExplanation = explanation;
}

SAnnotatedProbability::TAnomalyScoreExplanation&
CAnnotatedProbabilityBuilder::anomalyScoreExplanation() {
    return m_Result.s_AnomalyScoreExplanation;
}

void CAnnotatedProbabilityBuilder::multiBucketImpact(double multiBucketImpact) {
    m_Result.s_MultiBucketImpact = multiBucketImpact;
}

void CAnnotatedProbabilityBuilder::addAttributeProbability(
    std::size_t cid,
    const core::CStoredStringPtr& attribute,
    double pGivenAttribute_,
    model_t::CResultType type,
    model_t::EFeature feature,
    const TStoredStringPtr1Vec& correlatedAttributes,
    const TSizeDoublePr1Vec& correlated) {
    type.set(m_Result.s_ResultType.asInterimOrFinal());
    SAttributeProbability pGivenAttribute(cid, attribute, pGivenAttribute_, type,
                                          feature, correlatedAttributes, correlated);
    m_MinAttributeProbabilities.add(pGivenAttribute);
    ++m_DistinctTotalAttributes;
}

void CAnnotatedProbabilityBuilder::build() {

    if (m_NumberAttributeProbabilities > 0 && m_MinAttributeProbabilities.count() > 0) {
        m_MinAttributeProbabilities.sort();
        m_Result.s_AttributeProbabilities.reserve(m_MinAttributeProbabilities.count());
        double cutoff = std::max(1.1 * m_MinAttributeProbabilities[0].s_Probability,
                                 maths::common::LARGEST_SIGNIFICANT_PROBABILITY);

        for (std::size_t i = 0; i < m_MinAttributeProbabilities.count() &&
                                m_MinAttributeProbabilities[i].s_Probability <= cutoff;
             ++i) {
            m_Result.s_AttributeProbabilities.push_back(m_MinAttributeProbabilities[i]);
        }
    }
}
}
}
