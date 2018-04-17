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

#include "CAnnotatedProbabilityBuilderTest.h"

#include <maths/CMultinomialConjugate.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CStringStore.h>
#include <model/FunctionTypes.h>
#include <model/ModelTypes.h>

#include <vector>

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

void CAnnotatedProbabilityBuilderTest::testProbability() {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result);

    builder.probability(0.42);
    CPPUNIT_ASSERT_EQUAL(0.42, result.s_Probability);

    builder.probability(0.99);
    CPPUNIT_ASSERT_EQUAL(0.99, result.s_Probability);
}

void CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenIndividualCount() {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 1, function_t::E_IndividualCount, 42);

    builder.addAttributeProbability(0, EMPTY_STRING_PTR, 1.0, 0.68,
                                    model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualCountByBucketAndPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), result.s_AttributeProbabilities.size());
    CPPUNIT_ASSERT_EQUAL(EMPTY_STRING, *result.s_AttributeProbabilities[0].s_Attribute);
    CPPUNIT_ASSERT_EQUAL(0.68, result.s_AttributeProbabilities[0].s_Probability);
    CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson,
                         result.s_AttributeProbabilities[0].s_Feature);
    CPPUNIT_ASSERT(result.s_AttributeProbabilities[0].s_DescriptiveData.empty());
}

void CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenPopulationCount() {
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

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), result.s_AttributeProbabilities.size());

    CPPUNIT_ASSERT_EQUAL(C2, *result.s_AttributeProbabilities[0].s_Attribute);
    CPPUNIT_ASSERT_EQUAL(0.04, result.s_AttributeProbabilities[0].s_Probability);
    CPPUNIT_ASSERT_EQUAL(model_t::E_PopulationCountByBucketPersonAndAttribute,
                         result.s_AttributeProbabilities[0].s_Feature);
    CPPUNIT_ASSERT(result.s_AttributeProbabilities[0].s_DescriptiveData.empty());

    CPPUNIT_ASSERT_EQUAL(C1, *result.s_AttributeProbabilities[1].s_Attribute);
    CPPUNIT_ASSERT_EQUAL(0.05, result.s_AttributeProbabilities[1].s_Probability);
    CPPUNIT_ASSERT_EQUAL(model_t::E_PopulationCountByBucketPersonAndAttribute,
                         result.s_AttributeProbabilities[1].s_Feature);
    CPPUNIT_ASSERT(result.s_AttributeProbabilities[1].s_DescriptiveData.empty());
}

void CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenIndividualRare() {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 1, function_t::E_IndividualRare, 42);

    builder.addAttributeProbability(0, EMPTY_STRING_PTR, 1.0, 0.68,
                                    model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), result.s_AttributeProbabilities.size());
    CPPUNIT_ASSERT(result.s_AttributeProbabilities[0].s_DescriptiveData.empty());
}

void CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenPopulationRare() {
    maths::CMultinomialConjugate attributePrior(
        maths::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1u; i <= 4u; ++i) {
        TDouble1Vec samples(i, static_cast<double>(i));
        TDouble4Vec1Vec weights(i, maths::CConstantWeights::UNIT);
        attributePrior.addSamples(maths::CConstantWeights::COUNT, samples, weights);
    }

    maths::CMultinomialConjugate personAttributePrior(
        maths::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1u; i <= 4u; ++i) {
        TDouble1Vec samples(2 * i, static_cast<double>(i));
        TDouble4Vec1Vec weights(2 * i, maths::CConstantWeights::UNIT);
        personAttributePrior.addSamples(maths::CConstantWeights::COUNT, samples, weights);
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

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), result.s_AttributeProbabilities.size());

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), result.s_DescriptiveData.size());
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_PERSON_COUNT,
                         result.s_DescriptiveData[0].first);
    CPPUNIT_ASSERT_EQUAL(42.0, result.s_DescriptiveData[0].second);
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_DISTINCT_RARE_ATTRIBUTES_COUNT,
                         result.s_DescriptiveData[1].first);
    CPPUNIT_ASSERT_EQUAL(1.0, result.s_DescriptiveData[1].second);
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_DISTINCT_TOTAL_ATTRIBUTES_COUNT,
                         result.s_DescriptiveData[2].first);
    CPPUNIT_ASSERT_EQUAL(4.0, result.s_DescriptiveData[2].second);

    CPPUNIT_ASSERT_EQUAL(C3, *result.s_AttributeProbabilities[0].s_Attribute);
    CPPUNIT_ASSERT_EQUAL(0.01, result.s_AttributeProbabilities[0].s_Probability);
    CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                         result.s_AttributeProbabilities[0].s_Feature);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2),
                         result.s_AttributeProbabilities[0].s_DescriptiveData.size());
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_ATTRIBUTE_CONCENTRATION,
                         result.s_AttributeProbabilities[0].s_DescriptiveData[0].first);
    CPPUNIT_ASSERT_EQUAL(
        3.0, result.s_AttributeProbabilities[0].s_DescriptiveData[0].second);
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_ACTIVITY_CONCENTRATION,
                         result.s_AttributeProbabilities[0].s_DescriptiveData[1].first);
    CPPUNIT_ASSERT_EQUAL(
        6.0, result.s_AttributeProbabilities[0].s_DescriptiveData[1].second);

    CPPUNIT_ASSERT_EQUAL(C1, *result.s_AttributeProbabilities[1].s_Attribute);
    CPPUNIT_ASSERT_EQUAL(0.02, result.s_AttributeProbabilities[1].s_Probability);
    CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                         result.s_AttributeProbabilities[1].s_Feature);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2),
                         result.s_AttributeProbabilities[1].s_DescriptiveData.size());
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_ATTRIBUTE_CONCENTRATION,
                         result.s_AttributeProbabilities[1].s_DescriptiveData[0].first);
    CPPUNIT_ASSERT_EQUAL(
        1.0, result.s_AttributeProbabilities[1].s_DescriptiveData[0].second);
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_ACTIVITY_CONCENTRATION,
                         result.s_AttributeProbabilities[1].s_DescriptiveData[1].first);
    CPPUNIT_ASSERT_EQUAL(
        2.0, result.s_AttributeProbabilities[1].s_DescriptiveData[1].second);
}

void CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenPopulationFreqRare() {
    maths::CMultinomialConjugate attributePrior(
        maths::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1u; i <= 4u; ++i) {
        TDouble1Vec samples(i, static_cast<double>(i));
        TDouble4Vec1Vec weights(i, maths::CConstantWeights::UNIT);
        attributePrior.addSamples(maths::CConstantWeights::COUNT, samples, weights);
    }

    maths::CMultinomialConjugate personAttributePrior(
        maths::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1u; i <= 4u; ++i) {
        TDouble1Vec samples(2 * i, static_cast<double>(i));
        TDouble4Vec1Vec weights(2 * i, maths::CConstantWeights::UNIT);
        personAttributePrior.addSamples(maths::CConstantWeights::COUNT, samples, weights);
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

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), result.s_AttributeProbabilities.size());

    CPPUNIT_ASSERT_EQUAL(std::size_t(3), result.s_DescriptiveData.size());
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_PERSON_COUNT,
                         result.s_DescriptiveData[0].first);
    CPPUNIT_ASSERT_EQUAL(70.0, result.s_DescriptiveData[0].second);
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_RARE_ATTRIBUTES_COUNT,
                         result.s_DescriptiveData[1].first);
    CPPUNIT_ASSERT_EQUAL(8.0, result.s_DescriptiveData[1].second);
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_TOTAL_ATTRIBUTES_COUNT,
                         result.s_DescriptiveData[2].first);
    CPPUNIT_ASSERT_EQUAL(20.0, result.s_DescriptiveData[2].second);

    CPPUNIT_ASSERT_EQUAL(C3, *result.s_AttributeProbabilities[0].s_Attribute);
    CPPUNIT_ASSERT_EQUAL(0.01, result.s_AttributeProbabilities[0].s_Probability);
    CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                         result.s_AttributeProbabilities[0].s_Feature);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2),
                         result.s_AttributeProbabilities[0].s_DescriptiveData.size());
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_ATTRIBUTE_CONCENTRATION,
                         result.s_AttributeProbabilities[0].s_DescriptiveData[0].first);
    CPPUNIT_ASSERT_EQUAL(
        3.0, result.s_AttributeProbabilities[0].s_DescriptiveData[0].second);
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_ACTIVITY_CONCENTRATION,
                         result.s_AttributeProbabilities[0].s_DescriptiveData[1].first);
    CPPUNIT_ASSERT_EQUAL(
        6.0, result.s_AttributeProbabilities[0].s_DescriptiveData[1].second);

    CPPUNIT_ASSERT_EQUAL(C1, *result.s_AttributeProbabilities[1].s_Attribute);
    CPPUNIT_ASSERT_EQUAL(0.02, result.s_AttributeProbabilities[1].s_Probability);
    CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                         result.s_AttributeProbabilities[1].s_Feature);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2),
                         result.s_AttributeProbabilities[1].s_DescriptiveData.size());
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_ATTRIBUTE_CONCENTRATION,
                         result.s_AttributeProbabilities[1].s_DescriptiveData[0].first);
    CPPUNIT_ASSERT_EQUAL(
        1.0, result.s_AttributeProbabilities[1].s_DescriptiveData[0].second);
    CPPUNIT_ASSERT_EQUAL(annotated_probability::E_ACTIVITY_CONCENTRATION,
                         result.s_AttributeProbabilities[1].s_DescriptiveData[1].first);
    CPPUNIT_ASSERT_EQUAL(
        2.0, result.s_AttributeProbabilities[1].s_DescriptiveData[1].second);
}

void CAnnotatedProbabilityBuilderTest::testPersonFrequencyGivenIndividualCount() {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 1, function_t::E_IndividualCount, 42);

    builder.personFrequency(0.3, false);

    CPPUNIT_ASSERT(result.s_DescriptiveData.empty());
}

void CAnnotatedProbabilityBuilderTest::testPersonFrequencyGivenIndividualRare() {
    {
        SAnnotatedProbability result;
        CAnnotatedProbabilityBuilderForTest builder(
            result, 1, function_t::E_IndividualRare, 42);

        builder.personFrequency(0.3, false);

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), result.s_DescriptiveData.size());
        CPPUNIT_ASSERT_EQUAL(annotated_probability::E_PERSON_NEVER_SEEN_BEFORE,
                             result.s_DescriptiveData[0].first);
        CPPUNIT_ASSERT_EQUAL(1.0, result.s_DescriptiveData[0].second);
    }
    {
        SAnnotatedProbability result;
        CAnnotatedProbabilityBuilderForTest builder(
            result, 1, function_t::E_IndividualRare, 42);

        builder.personFrequency(0.2, true);

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), result.s_DescriptiveData.size());
        CPPUNIT_ASSERT_EQUAL(annotated_probability::E_PERSON_PERIOD,
                             result.s_DescriptiveData[0].first);
        CPPUNIT_ASSERT_EQUAL(5.0, result.s_DescriptiveData[0].second);
    }
}

void CAnnotatedProbabilityBuilderTest::testPersonFrequencyGivenPopulationRare() {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 3, function_t::E_PopulationRare, 42);

    builder.personFrequency(0.3, false);

    CPPUNIT_ASSERT(result.s_DescriptiveData.empty());
}

CppUnit::Test* CAnnotatedProbabilityBuilderTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CAnnotatedProbabilityBuilderTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CAnnotatedProbabilityBuilderTest>(
        "CAnnotatedProbabilityBuilderTest::testProbability",
        &CAnnotatedProbabilityBuilderTest::testProbability));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnnotatedProbabilityBuilderTest>(
        "CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenIndividualCount",
        &CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenIndividualCount));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnnotatedProbabilityBuilderTest>(
        "CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenPopulationCount",
        &CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenPopulationCount));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnnotatedProbabilityBuilderTest>(
        "CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenIndividualRare",
        &CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenIndividualRare));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnnotatedProbabilityBuilderTest>(
        "CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenPopulationRare",
        &CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenPopulationRare));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnnotatedProbabilityBuilderTest>(
        "CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenPopulationFreqRare",
        &CAnnotatedProbabilityBuilderTest::testAddAttributeProbabilityGivenPopulationFreqRare));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnnotatedProbabilityBuilderTest>(
        "CAnnotatedProbabilityBuilderTest::testPersonFrequencyGivenIndividualCount",
        &CAnnotatedProbabilityBuilderTest::testPersonFrequencyGivenIndividualCount));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnnotatedProbabilityBuilderTest>(
        "CAnnotatedProbabilityBuilderTest::testPersonFrequencyGivenIndividualRare",
        &CAnnotatedProbabilityBuilderTest::testPersonFrequencyGivenIndividualRare));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnnotatedProbabilityBuilderTest>(
        "CAnnotatedProbabilityBuilderTest::testPersonFrequencyGivenPopulationRare",
        &CAnnotatedProbabilityBuilderTest::testPersonFrequencyGivenPopulationRare));
    return suiteOfTests;
}
