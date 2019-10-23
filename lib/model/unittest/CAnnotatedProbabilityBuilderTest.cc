/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CMultinomialConjugate.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CStringStore.h>
#include <model/FunctionTypes.h>
#include <model/ModelTypes.h>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CAnnotatedProbabilityBuilderTest)

using namespace ml;
using namespace model;

namespace {
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble4Vec = core::CSmallVector<double, 4>;
using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoublePr1Vec = core::CSmallVector<TSizeDoublePr, 1>;
using TStoredStringPtr1Vec = core::CSmallVector<core::CStoredStringPtr, 1>;

const std::string EMPTY_STRING;
const core::CStoredStringPtr EMPTY_STRING_PTR(CStringStore::names().getEmpty());
const std::string C1("c1");
const core::CStoredStringPtr C1_PTR(CStringStore::names().get(C1));
const std::string C2("c2");
const core::CStoredStringPtr C2_PTR(CStringStore::names().get(C2));
const std::string C3("c3");
const core::CStoredStringPtr C3_PTR(CStringStore::names().get(C3));
const std::string C4("c4");
const core::CStoredStringPtr C4_PTR(CStringStore::names().get(C4));
const TStoredStringPtr1Vec NO_CORRELATED_ATTRIBUTES;
const TSizeDoublePr1Vec NO_CORRELATES;

class CAnnotatedProbabilityBuilderForTest : public CAnnotatedProbabilityBuilder {
public:
    CAnnotatedProbabilityBuilderForTest(SAnnotatedProbability& annotatedProbability)
        : CAnnotatedProbabilityBuilder(annotatedProbability) {}

    CAnnotatedProbabilityBuilderForTest(SAnnotatedProbability& annotatedProbability,
                                        std::size_t numberAttributeProbabilities,
                                        function_t::EFunction function,
                                        std::size_t numberOfPeople)
        : CAnnotatedProbabilityBuilder(annotatedProbability,
                                       numberAttributeProbabilities,
                                       function,
                                       numberOfPeople) {}
};
}

BOOST_AUTO_TEST_CASE(testProbability) {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result);

    builder.probability(0.42);
    BOOST_REQUIRE_EQUAL(0.42, result.s_Probability);

    builder.probability(0.99);
    BOOST_REQUIRE_EQUAL(0.99, result.s_Probability);
}

BOOST_AUTO_TEST_CASE(testAddAttributeProbabilityGivenIndividualCount) {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 1, function_t::E_IndividualCount, 42);

    builder.addAttributeProbability(0, EMPTY_STRING_PTR, 1.0, 0.68,
                                    model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualCountByBucketAndPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    BOOST_REQUIRE_EQUAL(std::size_t(1), result.s_AttributeProbabilities.size());
    BOOST_REQUIRE_EQUAL(EMPTY_STRING, *result.s_AttributeProbabilities[0].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.68, result.s_AttributeProbabilities[0].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_IndividualCountByBucketAndPerson,
                      result.s_AttributeProbabilities[0].s_Feature);
    BOOST_TEST_REQUIRE(result.s_AttributeProbabilities[0].s_DescriptiveData.empty());
}

BOOST_AUTO_TEST_CASE(testAddAttributeProbabilityGivenPopulationCount) {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 3, function_t::E_PopulationCount, 42);

    builder.addAttributeProbability(0, EMPTY_STRING_PTR, 1.0, 0.09,
                                    model_t::CResultType::E_Unconditional,
                                    model_t::E_PopulationCountByBucketPersonAndAttribute,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(1, C1_PTR, 1.0, 0.05, model_t::CResultType::E_Unconditional,
                                    model_t::E_PopulationCountByBucketPersonAndAttribute,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(2, C2_PTR, 1.0, 0.04, model_t::CResultType::E_Unconditional,
                                    model_t::E_PopulationCountByBucketPersonAndAttribute,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(3, C3_PTR, 1.0, 0.06, model_t::CResultType::E_Unconditional,
                                    model_t::E_PopulationCountByBucketPersonAndAttribute,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    BOOST_REQUIRE_EQUAL(std::size_t(2), result.s_AttributeProbabilities.size());

    BOOST_REQUIRE_EQUAL(C2, *result.s_AttributeProbabilities[0].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.04, result.s_AttributeProbabilities[0].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_PopulationCountByBucketPersonAndAttribute,
                      result.s_AttributeProbabilities[0].s_Feature);
    BOOST_TEST_REQUIRE(result.s_AttributeProbabilities[0].s_DescriptiveData.empty());

    BOOST_REQUIRE_EQUAL(C1, *result.s_AttributeProbabilities[1].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.05, result.s_AttributeProbabilities[1].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_PopulationCountByBucketPersonAndAttribute,
                      result.s_AttributeProbabilities[1].s_Feature);
    BOOST_TEST_REQUIRE(result.s_AttributeProbabilities[1].s_DescriptiveData.empty());
}

BOOST_AUTO_TEST_CASE(testAddAttributeProbabilityGivenIndividualRare) {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 1, function_t::E_IndividualRare, 42);

    builder.addAttributeProbability(0, EMPTY_STRING_PTR, 1.0, 0.68,
                                    model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    BOOST_REQUIRE_EQUAL(std::size_t(1), result.s_AttributeProbabilities.size());
    BOOST_TEST_REQUIRE(result.s_AttributeProbabilities[0].s_DescriptiveData.empty());
}

BOOST_AUTO_TEST_CASE(testAddAttributeProbabilityGivenPopulationRare) {
    maths::CMultinomialConjugate attributePrior(
        maths::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1u; i <= 4u; ++i) {
        TDouble1Vec samples(i, static_cast<double>(i));
        maths_t::TDoubleWeightsAry1Vec weights(i, maths_t::CUnitWeights::UNIT);
        attributePrior.addSamples(samples, weights);
    }

    maths::CMultinomialConjugate personAttributePrior(
        maths::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1u; i <= 4u; ++i) {
        TDouble1Vec samples(2 * i, static_cast<double>(i));
        maths_t::TDoubleWeightsAry1Vec weights(2 * i, maths_t::CUnitWeights::UNIT);
        personAttributePrior.addSamples(samples, weights);
    }

    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 2, function_t::E_PopulationRare, 42);
    builder.attributeProbabilityPrior(&attributePrior);
    builder.personAttributeProbabilityPrior(&personAttributePrior);

    builder.addAttributeProbability(1, C1_PTR, 0.051, 0.02, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(2, C2_PTR, 0.06, 0.06, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(3, C3_PTR, 0.07, 0.01, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(4, C4_PTR, 0.03, 0.03, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    BOOST_REQUIRE_EQUAL(std::size_t(2), result.s_AttributeProbabilities.size());

    BOOST_REQUIRE_EQUAL(std::size_t(3), result.s_DescriptiveData.size());
    BOOST_REQUIRE_EQUAL(annotated_probability::E_PERSON_COUNT,
                      result.s_DescriptiveData[0].first);
    BOOST_REQUIRE_EQUAL(42.0, result.s_DescriptiveData[0].second);
    BOOST_REQUIRE_EQUAL(annotated_probability::E_DISTINCT_RARE_ATTRIBUTES_COUNT,
                      result.s_DescriptiveData[1].first);
    BOOST_REQUIRE_EQUAL(1.0, result.s_DescriptiveData[1].second);
    BOOST_REQUIRE_EQUAL(annotated_probability::E_DISTINCT_TOTAL_ATTRIBUTES_COUNT,
                      result.s_DescriptiveData[2].first);
    BOOST_REQUIRE_EQUAL(4.0, result.s_DescriptiveData[2].second);

    BOOST_REQUIRE_EQUAL(C3, *result.s_AttributeProbabilities[0].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.01, result.s_AttributeProbabilities[0].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                      result.s_AttributeProbabilities[0].s_Feature);
    BOOST_REQUIRE_EQUAL(std::size_t(2),
                      result.s_AttributeProbabilities[0].s_DescriptiveData.size());
    BOOST_REQUIRE_EQUAL(annotated_probability::E_ATTRIBUTE_CONCENTRATION,
                      result.s_AttributeProbabilities[0].s_DescriptiveData[0].first);
    BOOST_REQUIRE_EQUAL(
        3.0, result.s_AttributeProbabilities[0].s_DescriptiveData[0].second);
    BOOST_REQUIRE_EQUAL(annotated_probability::E_ACTIVITY_CONCENTRATION,
                      result.s_AttributeProbabilities[0].s_DescriptiveData[1].first);
    BOOST_REQUIRE_EQUAL(
        6.0, result.s_AttributeProbabilities[0].s_DescriptiveData[1].second);

    BOOST_REQUIRE_EQUAL(C1, *result.s_AttributeProbabilities[1].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.02, result.s_AttributeProbabilities[1].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                      result.s_AttributeProbabilities[1].s_Feature);
    BOOST_REQUIRE_EQUAL(std::size_t(2),
                      result.s_AttributeProbabilities[1].s_DescriptiveData.size());
    BOOST_REQUIRE_EQUAL(annotated_probability::E_ATTRIBUTE_CONCENTRATION,
                      result.s_AttributeProbabilities[1].s_DescriptiveData[0].first);
    BOOST_REQUIRE_EQUAL(
        1.0, result.s_AttributeProbabilities[1].s_DescriptiveData[0].second);
    BOOST_REQUIRE_EQUAL(annotated_probability::E_ACTIVITY_CONCENTRATION,
                      result.s_AttributeProbabilities[1].s_DescriptiveData[1].first);
    BOOST_REQUIRE_EQUAL(
        2.0, result.s_AttributeProbabilities[1].s_DescriptiveData[1].second);
}

BOOST_AUTO_TEST_CASE(testAddAttributeProbabilityGivenPopulationFreqRare) {
    maths::CMultinomialConjugate attributePrior(
        maths::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1u; i <= 4u; ++i) {
        TDouble1Vec samples(i, static_cast<double>(i));
        maths_t::TDoubleWeightsAry1Vec weights(i, maths_t::CUnitWeights::UNIT);
        attributePrior.addSamples(samples, weights);
    }

    maths::CMultinomialConjugate personAttributePrior(
        maths::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1u; i <= 4u; ++i) {
        TDouble1Vec samples(2 * i, static_cast<double>(i));
        maths_t::TDoubleWeightsAry1Vec weights(2 * i, maths_t::CUnitWeights::UNIT);
        personAttributePrior.addSamples(samples, weights);
    }

    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(
        result, 2, function_t::E_PopulationFreqRare, 70);
    builder.attributeProbabilityPrior(&attributePrior);
    builder.personAttributeProbabilityPrior(&personAttributePrior);

    builder.addAttributeProbability(1, C1_PTR, 0.051, 0.02, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(2, C2_PTR, 0.06, 0.06, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(3, C3_PTR, 0.07, 0.01, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(4, C4_PTR, 0.03, 0.03, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    BOOST_REQUIRE_EQUAL(std::size_t(2), result.s_AttributeProbabilities.size());

    BOOST_REQUIRE_EQUAL(std::size_t(3), result.s_DescriptiveData.size());
    BOOST_REQUIRE_EQUAL(annotated_probability::E_PERSON_COUNT,
                      result.s_DescriptiveData[0].first);
    BOOST_REQUIRE_EQUAL(70.0, result.s_DescriptiveData[0].second);
    BOOST_REQUIRE_EQUAL(annotated_probability::E_RARE_ATTRIBUTES_COUNT,
                      result.s_DescriptiveData[1].first);
    BOOST_REQUIRE_EQUAL(8.0, result.s_DescriptiveData[1].second);
    BOOST_REQUIRE_EQUAL(annotated_probability::E_TOTAL_ATTRIBUTES_COUNT,
                      result.s_DescriptiveData[2].first);
    BOOST_REQUIRE_EQUAL(20.0, result.s_DescriptiveData[2].second);

    BOOST_REQUIRE_EQUAL(C3, *result.s_AttributeProbabilities[0].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.01, result.s_AttributeProbabilities[0].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                      result.s_AttributeProbabilities[0].s_Feature);
    BOOST_REQUIRE_EQUAL(std::size_t(2),
                      result.s_AttributeProbabilities[0].s_DescriptiveData.size());
    BOOST_REQUIRE_EQUAL(annotated_probability::E_ATTRIBUTE_CONCENTRATION,
                      result.s_AttributeProbabilities[0].s_DescriptiveData[0].first);
    BOOST_REQUIRE_EQUAL(
        3.0, result.s_AttributeProbabilities[0].s_DescriptiveData[0].second);
    BOOST_REQUIRE_EQUAL(annotated_probability::E_ACTIVITY_CONCENTRATION,
                      result.s_AttributeProbabilities[0].s_DescriptiveData[1].first);
    BOOST_REQUIRE_EQUAL(
        6.0, result.s_AttributeProbabilities[0].s_DescriptiveData[1].second);

    BOOST_REQUIRE_EQUAL(C1, *result.s_AttributeProbabilities[1].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.02, result.s_AttributeProbabilities[1].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                      result.s_AttributeProbabilities[1].s_Feature);
    BOOST_REQUIRE_EQUAL(std::size_t(2),
                      result.s_AttributeProbabilities[1].s_DescriptiveData.size());
    BOOST_REQUIRE_EQUAL(annotated_probability::E_ATTRIBUTE_CONCENTRATION,
                      result.s_AttributeProbabilities[1].s_DescriptiveData[0].first);
    BOOST_REQUIRE_EQUAL(
        1.0, result.s_AttributeProbabilities[1].s_DescriptiveData[0].second);
    BOOST_REQUIRE_EQUAL(annotated_probability::E_ACTIVITY_CONCENTRATION,
                      result.s_AttributeProbabilities[1].s_DescriptiveData[1].first);
    BOOST_REQUIRE_EQUAL(
        2.0, result.s_AttributeProbabilities[1].s_DescriptiveData[1].second);
}

BOOST_AUTO_TEST_CASE(testPersonFrequencyGivenIndividualCount) {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 1, function_t::E_IndividualCount, 42);

    builder.personFrequency(0.3, false);

    BOOST_TEST_REQUIRE(result.s_DescriptiveData.empty());
}

BOOST_AUTO_TEST_CASE(testPersonFrequencyGivenIndividualRare) {
    {
        SAnnotatedProbability result;
        CAnnotatedProbabilityBuilderForTest builder(
            result, 1, function_t::E_IndividualRare, 42);

        builder.personFrequency(0.3, false);

        BOOST_REQUIRE_EQUAL(std::size_t(1), result.s_DescriptiveData.size());
        BOOST_REQUIRE_EQUAL(annotated_probability::E_PERSON_NEVER_SEEN_BEFORE,
                          result.s_DescriptiveData[0].first);
        BOOST_REQUIRE_EQUAL(1.0, result.s_DescriptiveData[0].second);
    }
    {
        SAnnotatedProbability result;
        CAnnotatedProbabilityBuilderForTest builder(
            result, 1, function_t::E_IndividualRare, 42);

        builder.personFrequency(0.2, true);

        BOOST_REQUIRE_EQUAL(std::size_t(1), result.s_DescriptiveData.size());
        BOOST_REQUIRE_EQUAL(annotated_probability::E_PERSON_PERIOD,
                          result.s_DescriptiveData[0].first);
        BOOST_REQUIRE_EQUAL(5.0, result.s_DescriptiveData[0].second);
    }
}

BOOST_AUTO_TEST_CASE(testPersonFrequencyGivenPopulationRare) {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 3, function_t::E_PopulationRare, 42);

    builder.personFrequency(0.3, false);

    BOOST_TEST_REQUIRE(result.s_DescriptiveData.empty());
}

BOOST_AUTO_TEST_SUITE_END()
