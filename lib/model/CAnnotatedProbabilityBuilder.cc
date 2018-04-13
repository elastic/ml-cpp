/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CAnnotatedProbabilityBuilder.h>

#include <maths/CMultinomialConjugate.h>
#include <maths/Constants.h>

#include <utility>

namespace ml
{
namespace model
{

CAnnotatedProbabilityBuilder::CAnnotatedProbabilityBuilder(SAnnotatedProbability &annotatedProbability)
        : m_Result(annotatedProbability),
          m_NumberAttributeProbabilities(1),
          m_NumberOfPeople(0),
          m_AttributeProbabilityPrior(0),
          m_PersonAttributeProbabilityPrior(0),
          m_MinAttributeProbabilities(1),
          m_DistinctTotalAttributes(0),
          m_DistinctRareAttributes(0),
          m_RareAttributes(0),
          m_IsPopulation(false),
          m_IsRare(false),
          m_IsFreqRare(false)
{
    m_Result.s_AttributeProbabilities.clear();
    m_Result.s_Influences.clear();
}

CAnnotatedProbabilityBuilder::CAnnotatedProbabilityBuilder(SAnnotatedProbability &annotatedProbability,
                                                           std::size_t numberAttributeProbabilities,
                                                           function_t::EFunction function,
                                                           std::size_t numberOfPeople)
        : m_Result(annotatedProbability),
          m_NumberAttributeProbabilities(numberAttributeProbabilities),
          m_NumberOfPeople(numberOfPeople),
          m_AttributeProbabilityPrior(0),
          m_PersonAttributeProbabilityPrior(0),
          m_MinAttributeProbabilities(numberAttributeProbabilities),
          m_DistinctTotalAttributes(0),
          m_DistinctRareAttributes(0),
          m_RareAttributes(0),
          m_IsPopulation(function_t::isPopulation(function)),
          m_IsRare(false),
          m_IsFreqRare(false)
{
    m_Result.s_AttributeProbabilities.clear();
    m_Result.s_Influences.clear();

    if (function == function_t::E_IndividualRare || function == function_t::E_PopulationRare)
    {
        m_IsRare = true;
    }
    else if (function == function_t::E_PopulationFreqRare)
    {
        m_IsFreqRare = true;
    }
}

void CAnnotatedProbabilityBuilder::personFrequency(double frequency, bool everSeenBefore)
{
    if (m_IsRare && m_IsPopulation == false)
    {
        if (everSeenBefore)
        {
            double period = (frequency == 0.0) ? 0.0 : 1 / frequency;
            m_Result.addDescriptiveData(annotated_probability::E_PERSON_PERIOD, period);
        }
        else
        {
            m_Result.addDescriptiveData(annotated_probability::E_PERSON_NEVER_SEEN_BEFORE, 1.0);
        }
    }
}

void CAnnotatedProbabilityBuilder::attributeProbabilityPrior(const maths::CMultinomialConjugate *prior)
{
    m_AttributeProbabilityPrior = prior;
}

void CAnnotatedProbabilityBuilder::personAttributeProbabilityPrior(const maths::CMultinomialConjugate *prior)
{
    m_PersonAttributeProbabilityPrior = prior;
}

void CAnnotatedProbabilityBuilder::probability(double p)
{
    m_Result.s_Probability = p;
}

void CAnnotatedProbabilityBuilder::addAttributeProbability(std::size_t cid,
                                                           const core::CStoredStringPtr &attribute,
                                                           double pAttribute,
                                                           double pGivenAttribute_,
                                                           model_t::CResultType type,
                                                           model_t::EFeature feature,
                                                           const TStoredStringPtr1Vec &correlatedAttributes,
                                                           const TSizeDoublePr1Vec &correlated)
{
    type.set(m_Result.s_ResultType.asInterimOrFinal());
    SAttributeProbability pGivenAttribute(cid, attribute, pGivenAttribute_, type, feature, correlatedAttributes, correlated);
    this->addAttributeDescriptiveData(cid, pAttribute, pGivenAttribute);
    m_MinAttributeProbabilities.add(pGivenAttribute);
    ++m_DistinctTotalAttributes;
}

void CAnnotatedProbabilityBuilder::addAttributeDescriptiveData(std::size_t cid,
                                                               double pAttribute,
                                                               SAttributeProbability &attributeProbability)
{
    if (m_IsPopulation && (m_IsRare || m_IsFreqRare))
    {
        double concentration;
        m_AttributeProbabilityPrior->concentration(static_cast<double>(cid), concentration);
        attributeProbability.addDescriptiveData(annotated_probability::E_ATTRIBUTE_CONCENTRATION, concentration);

        double activityConcentration;
        m_PersonAttributeProbabilityPrior->concentration(static_cast<double>(cid), activityConcentration);
        attributeProbability.addDescriptiveData(annotated_probability::E_ACTIVITY_CONCENTRATION, activityConcentration);

        if (pAttribute < maths::LARGEST_SIGNIFICANT_PROBABILITY)
        {
            m_DistinctRareAttributes++;
            m_RareAttributes += activityConcentration;
        }
    }
}

void CAnnotatedProbabilityBuilder::build()
{
    this->addDescriptiveData();

    if (m_NumberAttributeProbabilities > 0 && m_MinAttributeProbabilities.count() > 0)
    {
        m_MinAttributeProbabilities.sort();
        m_Result.s_AttributeProbabilities.reserve(m_MinAttributeProbabilities.count());
        double cutoff = std::max(1.1 * m_MinAttributeProbabilities[0].s_Probability,
                                 maths::LARGEST_SIGNIFICANT_PROBABILITY);

        for (std::size_t i = 0u;
             i < m_MinAttributeProbabilities.count() && m_MinAttributeProbabilities[i].s_Probability <= cutoff;
             ++i)
        {
            m_Result.s_AttributeProbabilities.push_back(m_MinAttributeProbabilities[i]);
        }
    }
}

void CAnnotatedProbabilityBuilder::addDescriptiveData()
{
    if (m_IsPopulation && (m_IsRare || m_IsFreqRare))
    {
        m_Result.addDescriptiveData(annotated_probability::E_PERSON_COUNT,
                                    static_cast<double>(m_NumberOfPeople));
        if (m_IsRare)
        {
            m_Result.addDescriptiveData(annotated_probability::E_DISTINCT_RARE_ATTRIBUTES_COUNT,
                                        static_cast<double>(m_DistinctRareAttributes));
            m_Result.addDescriptiveData(annotated_probability::E_DISTINCT_TOTAL_ATTRIBUTES_COUNT,
                                        static_cast<double>(m_DistinctTotalAttributes));
        }
        else if (m_IsFreqRare)
        {
            double totalConcentration = m_PersonAttributeProbabilityPrior->totalConcentration();
            m_Result.addDescriptiveData(annotated_probability::E_RARE_ATTRIBUTES_COUNT, m_RareAttributes);
            m_Result.addDescriptiveData(annotated_probability::E_TOTAL_ATTRIBUTES_COUNT, totalConcentration);
        }
    }
}

}
}
