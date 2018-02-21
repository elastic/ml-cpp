/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CDetectionRuleTest.h"

#include <core/CLogger.h>
#include <core/CPatternSet.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CModelParams.h>
#include <model/CRuleCondition.h>
#include <model/CSearchKey.h>
#include <model/ModelTypes.h>

#include "Mocks.h"

#include <string>
#include <vector>

using namespace ml;
using namespace model;

namespace
{

typedef std::vector<model_t::EFeature> TFeatureVec;
typedef std::vector<std::string> TStrVec;

const std::string EMPTY_STRING;
}

CppUnit::Test *CDetectionRuleTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CDetectionRuleTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenCategoricalCondition",
           &CDetectionRuleTest::testApplyGivenCategoricalCondition));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenNumericalActualCondition",
           &CDetectionRuleTest::testApplyGivenNumericalActualCondition));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenNumericalTypicalCondition",
           &CDetectionRuleTest::testApplyGivenNumericalTypicalCondition));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenNumericalDiffAbsCondition",
           &CDetectionRuleTest::testApplyGivenNumericalDiffAbsCondition));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenSingleSeriesModelAndConditionWithField",
           &CDetectionRuleTest::testApplyGivenSingleSeriesModelAndConditionWithField));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenNoActualValueAvailable",
           &CDetectionRuleTest::testApplyGivenNoActualValueAvailable));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenDifferentSeriesAndIndividualModel",
           &CDetectionRuleTest::testApplyGivenDifferentSeriesAndIndividualModel));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenDifferentSeriesAndPopulationModel",
           &CDetectionRuleTest::testApplyGivenDifferentSeriesAndPopulationModel));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenMultipleConditionsWithOr",
           &CDetectionRuleTest::testApplyGivenMultipleConditionsWithOr));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenMultipleConditionsWithAnd",
           &CDetectionRuleTest::testApplyGivenMultipleConditionsWithAnd));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenTargetFieldIsPartitionAndIndividualModel",
           &CDetectionRuleTest::testApplyGivenTargetFieldIsPartitionAndIndividualModel));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testApplyGivenTimeCondition",
           &CDetectionRuleTest::testApplyGivenTimeCondition));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
           "CDetectionRuleTest::testRuleActions",
           &CDetectionRuleTest::testRuleActions));

    return suiteOfTests;
}

void CDetectionRuleTest::testApplyGivenCategoricalCondition(void)
{
    LOG_DEBUG("*** testApplyGivenCategoricalCondition ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    std::string partitionFieldName("partition");
    std::string partitionFieldValue("par_1");
    std::string personFieldName("over");
    std::string attributeFieldName("by");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_PopulationMetric, model_t::E_None, params,
                              EMPTY_STRING, partitionFieldName, partitionFieldValue,
                              personFieldName, attributeFieldName, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool added = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, added);
    std::string person2("p2");
    gathererPtr->addPerson(person2, m_ResourceMonitor, added);
    std::string attr11("a1_1");
    std::string attr12("a1_2");
    std::string attr21("a2_1");
    std::string attr22("a2_2");
    gathererPtr->addAttribute(attr11, m_ResourceMonitor, added);
    gathererPtr->addAttribute(attr12, m_ResourceMonitor, added);
    gathererPtr->addAttribute(attr21, m_ResourceMonitor, added);
    gathererPtr->addAttribute(attr22, m_ResourceMonitor, added);

    CMockModel model(params, gathererPtr, influenceCalculators);
    model.mockPopulation(true);
    CAnomalyDetectorModel::TDouble1Vec actual(1, 4.99);
    model.mockAddBucketValue(model_t::E_PopulationMeanByPersonAndAttribute, 0, 0, 100, actual);
    model.mockAddBucketValue(model_t::E_PopulationMeanByPersonAndAttribute, 0, 1, 100, actual);
    model.mockAddBucketValue(model_t::E_PopulationMeanByPersonAndAttribute, 1, 2, 100, actual);
    model.mockAddBucketValue(model_t::E_PopulationMeanByPersonAndAttribute, 1, 3, 100, actual);

    {
        std::string filterJson("[\"a1_1\",\"a2_2\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CRuleCondition condition;
        condition.type(CRuleCondition::E_Categorical);
        condition.fieldName(attributeFieldName);
        condition.valueFilter(valueFilter);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 1, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 2, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 3, 100));
    }
    {
        std::string filterJson("[\"a1*\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CRuleCondition condition;
        condition.type(CRuleCondition::E_Categorical);
        condition.fieldName(attributeFieldName);
        condition.valueFilter(valueFilter);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 1, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 2, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 3, 100) == false);
    }
    {
        std::string filterJson("[\"*2\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CRuleCondition condition;
        condition.type(CRuleCondition::E_Categorical);
        condition.fieldName(attributeFieldName);
        condition.valueFilter(valueFilter);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 1, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 2, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 3, 100));
    }
    {
        std::string filterJson("[\"*1*\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CRuleCondition condition;
        condition.type(CRuleCondition::E_Categorical);
        condition.fieldName(attributeFieldName);
        condition.valueFilter(valueFilter);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 1, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 2, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 3, 100) == false);
    }
    {
        std::string filterJson("[\"p2\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CRuleCondition condition;
        condition.type(CRuleCondition::E_Categorical);
        condition.fieldName(personFieldName);
        condition.valueFilter(valueFilter);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 1, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 2, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 3, 100));
    }
    {
        std::string filterJson("[\"par_1\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CRuleCondition condition;
        condition.type(CRuleCondition::E_Categorical);
        condition.fieldName(partitionFieldName);
        condition.valueFilter(valueFilter);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 1, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 2, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 3, 100));
    }
    {
        std::string filterJson("[\"par_2\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CRuleCondition condition;
        condition.type(CRuleCondition::E_Categorical);
        condition.fieldName(partitionFieldName);
        condition.valueFilter(valueFilter);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 1, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 2, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 3, 100) == false);
    }
}

void CDetectionRuleTest::testApplyGivenNumericalActualCondition(void)
{
    LOG_DEBUG("*** testApplyGivenNumericalActionCondition ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool addedPerson = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, addedPerson);

    CMockModel model(params, gathererPtr, influenceCalculators);
    CAnomalyDetectorModel::TDouble1Vec actual100(1, 4.99);
    CAnomalyDetectorModel::TDouble1Vec actual200(1, 5.00);
    CAnomalyDetectorModel::TDouble1Vec actual300(1, 5.01);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 100, actual100);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 200, actual200);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 300, actual300);

    {
        // Test rule with condition with operator LT

        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalActual);
        condition.condition().s_Op = CRuleCondition::E_LT;
        condition.condition().s_Threshold = 5.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 300) == false);
    }

    {
        // Test rule with condition with operator LTE

        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalActual);
        condition.condition().s_Op = CRuleCondition::E_LTE;
        condition.condition().s_Threshold = 5.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 200));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 300) == false);
    }
    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalActual);
        condition.condition().s_Op = CRuleCondition::E_GT;
        condition.condition().s_Threshold = 5.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 300));
    }
    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalActual);
        condition.condition().s_Op = CRuleCondition::E_GTE;
        condition.condition().s_Threshold = 5.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 200));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 300));
    }
}

void CDetectionRuleTest::testApplyGivenNumericalTypicalCondition(void)
{
    LOG_DEBUG("*** testApplyGivenNumericalTypicalCondition ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool addedPerson = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, addedPerson);

    CMockModel model(params, gathererPtr, influenceCalculators);
    CAnomalyDetectorModel::TDouble1Vec actual100(1, 4.99);
    CAnomalyDetectorModel::TDouble1Vec actual200(1, 5.00);
    CAnomalyDetectorModel::TDouble1Vec actual300(1, 5.01);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 100, actual100);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 200, actual200);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 300, actual300);
    CAnomalyDetectorModel::TDouble1Vec typical100(1, 44.99);
    CAnomalyDetectorModel::TDouble1Vec typical200(1, 45.00);
    CAnomalyDetectorModel::TDouble1Vec typical300(1, 45.01);
    model.mockAddBucketBaselineMean(model_t::E_IndividualMeanByPerson, 0, 0, 100, typical100);
    model.mockAddBucketBaselineMean(model_t::E_IndividualMeanByPerson, 0, 0, 200, typical200);
    model.mockAddBucketBaselineMean(model_t::E_IndividualMeanByPerson, 0, 0, 300, typical300);

    {
        // Test rule with condition with operator LT

        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalTypical);
        condition.condition().s_Op = CRuleCondition::E_LT;
        condition.condition().s_Threshold = 45.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 300) == false);
    }

    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalTypical);
        condition.condition().s_Op = CRuleCondition::E_GT;
        condition.condition().s_Threshold = 45.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 300));
    }
}

void CDetectionRuleTest::testApplyGivenNumericalDiffAbsCondition(void)
{
    LOG_DEBUG("*** testApplyGivenNumericalDiffAbsCondition ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool addedPerson = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, addedPerson);

    CMockModel model(params, gathererPtr, influenceCalculators);
    CAnomalyDetectorModel::TDouble1Vec actual100(1, 8.9);
    CAnomalyDetectorModel::TDouble1Vec actual200(1, 9.0);
    CAnomalyDetectorModel::TDouble1Vec actual300(1, 9.1);
    CAnomalyDetectorModel::TDouble1Vec actual400(1, 10.9);
    CAnomalyDetectorModel::TDouble1Vec actual500(1, 11.0);
    CAnomalyDetectorModel::TDouble1Vec actual600(1, 11.1);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 100, actual100);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 200, actual200);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 300, actual300);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 400, actual400);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 500, actual500);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 600, actual600);
    CAnomalyDetectorModel::TDouble1Vec typical100(1, 10.0);
    CAnomalyDetectorModel::TDouble1Vec typical200(1, 10.0);
    CAnomalyDetectorModel::TDouble1Vec typical300(1, 10.0);
    CAnomalyDetectorModel::TDouble1Vec typical400(1, 10.0);
    CAnomalyDetectorModel::TDouble1Vec typical500(1, 10.0);
    CAnomalyDetectorModel::TDouble1Vec typical600(1, 10.0);
    model.mockAddBucketBaselineMean(model_t::E_IndividualMeanByPerson, 0, 0, 100, typical100);
    model.mockAddBucketBaselineMean(model_t::E_IndividualMeanByPerson, 0, 0, 200, typical200);
    model.mockAddBucketBaselineMean(model_t::E_IndividualMeanByPerson, 0, 0, 300, typical300);
    model.mockAddBucketBaselineMean(model_t::E_IndividualMeanByPerson, 0, 0, 400, typical400);
    model.mockAddBucketBaselineMean(model_t::E_IndividualMeanByPerson, 0, 0, 500, typical500);
    model.mockAddBucketBaselineMean(model_t::E_IndividualMeanByPerson, 0, 0, 600, typical600);

    {
        // Test rule with condition with operator LT

        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalDiffAbs);
        condition.condition().s_Op = CRuleCondition::E_LT;
        condition.condition().s_Threshold = 1.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 300));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 400));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 500) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 600) == false);
    }

    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalDiffAbs);
        condition.condition().s_Op = CRuleCondition::E_GT;
        condition.condition().s_Threshold = 1.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 300) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 400) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 500) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 600));
    }
}

void CDetectionRuleTest::testApplyGivenSingleSeriesModelAndConditionWithField(void)
{
    LOG_DEBUG("*** testApplyGivenSingleSeriesModelAndConditionWithField ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool addedPerson = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, addedPerson);

    CMockModel model(params, gathererPtr, influenceCalculators);
    CAnomalyDetectorModel::TDouble1Vec actual100(1, 4.99);
    CAnomalyDetectorModel::TDouble1Vec actual200(1, 5.00);
    CAnomalyDetectorModel::TDouble1Vec actual300(1, 5.01);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 100, actual100);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 200, actual200);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 300, actual300);

    CRuleCondition condition;
    condition.type(CRuleCondition::E_NumericalActual);
    std::string fieldName("unknownField");
    std::string fieldValue("unknownValue");
    condition.fieldName(fieldName);
    condition.fieldValue(fieldValue);
    condition.condition().s_Op = CRuleCondition::E_LT;
    condition.condition().s_Threshold = 5.0;
    CDetectionRule rule;
    rule.addCondition(condition);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
            model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
            model_t::E_IndividualMeanByPerson, resultType, 0, 0, 200) == false);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
            model_t::E_IndividualMeanByPerson, resultType, 0, 0, 300) == false);
}

void CDetectionRuleTest::testApplyGivenNoActualValueAvailable(void)
{
    LOG_DEBUG("*** testApplyGivenNoActualValueAvailable ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool addedPerson = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, addedPerson);

    CMockModel model(params, gathererPtr, influenceCalculators);
    CAnomalyDetectorModel::TDouble1Vec actual100(1, 4.99);
    CAnomalyDetectorModel::TDouble1Vec actual200(1, 5.00);
    CAnomalyDetectorModel::TDouble1Vec actual300(1, 5.01);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 100, actual100);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 200, actual200);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 300, actual300);

    CRuleCondition condition;
    condition.type(CRuleCondition::E_NumericalActual);
    condition.condition().s_Op = CRuleCondition::E_LT;
    condition.condition().s_Threshold = 5.0;
    CDetectionRule rule;
    rule.addCondition(condition);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
            model_t::E_IndividualMeanByPerson, resultType, 0, 0, 400) == false);
}

void CDetectionRuleTest::testApplyGivenDifferentSeriesAndIndividualModel(void)
{
    LOG_DEBUG("*** testApplyGivenDifferentSeriesAndIndividualModel ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, personFieldName, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool addedPerson = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, addedPerson);
    std::string person2("p2");
    gathererPtr->addPerson(person2, m_ResourceMonitor, addedPerson);

    CMockModel model(params, gathererPtr, influenceCalculators);
    CAnomalyDetectorModel::TDouble1Vec p1Actual(1, 4.99);
    CAnomalyDetectorModel::TDouble1Vec p2Actual(1, 4.99);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 100, p1Actual);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 1, 0, 100, p2Actual);

    CRuleCondition condition;
    condition.type(CRuleCondition::E_NumericalActual);
    condition.fieldName(personFieldName);
    condition.fieldValue(person1);
    condition.condition().s_Op = CRuleCondition::E_LT;
    condition.condition().s_Threshold = 5.0;
    CDetectionRule rule;
    rule.addCondition(condition);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
            model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
            model_t::E_IndividualMeanByPerson, resultType, 1, 0, 100) == false);
}

void CDetectionRuleTest::testApplyGivenDifferentSeriesAndPopulationModel(void)
{
    LOG_DEBUG("*** testApplyGivenDifferentSeriesAndPopulationModel ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    std::string personFieldName("over");
    std::string attributeFieldName("by");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_PopulationMetric, model_t::E_None, params,
                              EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, personFieldName, attributeFieldName, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool added = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, added);
    std::string person2("p2");
    gathererPtr->addPerson(person2, m_ResourceMonitor, added);
    std::string attr11("a1_1");
    std::string attr12("a1_2");
    std::string attr21("a2_1");
    std::string attr22("a2_2");
    gathererPtr->addAttribute(attr11, m_ResourceMonitor, added);
    gathererPtr->addAttribute(attr12, m_ResourceMonitor, added);
    gathererPtr->addAttribute(attr21, m_ResourceMonitor, added);
    gathererPtr->addAttribute(attr22, m_ResourceMonitor, added);

    CMockModel model(params, gathererPtr, influenceCalculators);
    model.mockPopulation(true);
    CAnomalyDetectorModel::TDouble1Vec actual(1, 4.99);
    model.mockAddBucketValue(model_t::E_PopulationMeanByPersonAndAttribute, 0, 0, 100, actual);
    model.mockAddBucketValue(model_t::E_PopulationMeanByPersonAndAttribute, 0, 1, 100, actual);
    model.mockAddBucketValue(model_t::E_PopulationMeanByPersonAndAttribute, 1, 2, 100, actual);
    model.mockAddBucketValue(model_t::E_PopulationMeanByPersonAndAttribute, 1, 3, 100, actual);

    CRuleCondition condition;
    condition.type(CRuleCondition::E_NumericalActual);
    condition.fieldName(attributeFieldName);
    condition.fieldValue(attr12);
    condition.condition().s_Op = CRuleCondition::E_LT;
    condition.condition().s_Threshold = 5.0;
    CDetectionRule rule;
    rule.addCondition(condition);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
            model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 0, 100) == false);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
            model_t::E_PopulationMeanByPersonAndAttribute, resultType, 0, 1, 100));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
            model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 2, 100) == false);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
            model_t::E_PopulationMeanByPersonAndAttribute, resultType, 1, 3, 100) == false);
}

void CDetectionRuleTest::testApplyGivenMultipleConditionsWithOr(void)
{
    LOG_DEBUG("*** testApplyGivenMultipleConditionsWithOr ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, personFieldName, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool addedPerson = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, addedPerson);

    CMockModel model(params, gathererPtr, influenceCalculators);
    CAnomalyDetectorModel::TDouble1Vec p1Actual(1, 10.0);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 100, p1Actual);

    {
        // None applies
        CRuleCondition condition1;
        condition1.type(CRuleCondition::E_NumericalActual);
        condition1.fieldName(personFieldName);
        condition1.fieldValue(person1);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 9.0;
        CRuleCondition condition2;
        condition2.type(CRuleCondition::E_NumericalActual);
        condition2.fieldName(personFieldName);
        condition2.fieldValue(person1);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 9.5;
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);
    }
    {
        // First applies only
        CRuleCondition condition1;
        condition1.type(CRuleCondition::E_NumericalActual);
        condition1.fieldName(personFieldName);
        condition1.fieldValue(person1);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 11.0;
        CRuleCondition condition2;
        condition2.type(CRuleCondition::E_NumericalActual);
        condition2.fieldName(personFieldName);
        condition2.fieldValue(person1);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 9.5;
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    }
    {
        // Second applies only
        CRuleCondition condition1;
        condition1.type(CRuleCondition::E_NumericalActual);
        condition1.fieldName(personFieldName);
        condition1.fieldValue(person1);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 9.0;
        CRuleCondition condition2;
        condition2.type(CRuleCondition::E_NumericalActual);
        condition2.fieldName(personFieldName);
        condition2.fieldValue(person1);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 10.5;
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    }
    {
        // Both apply
        CRuleCondition condition1;
        condition1.type(CRuleCondition::E_NumericalActual);
        condition1.fieldName(personFieldName);
        condition1.fieldValue(person1);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 12.0;
        CRuleCondition condition2;
        condition2.type(CRuleCondition::E_NumericalActual);
        condition2.fieldName(personFieldName);
        condition2.fieldValue(person1);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 10.5;
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    }
}

void CDetectionRuleTest::testApplyGivenMultipleConditionsWithAnd(void)
{
    LOG_DEBUG("*** testApplyGivenMultipleConditionsWithAnd ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, personFieldName, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool addedPerson = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, addedPerson);

    CMockModel model(params, gathererPtr, influenceCalculators);
    CAnomalyDetectorModel::TDouble1Vec p1Actual(1, 10.0);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 100, p1Actual);

    {
        // None applies
        CRuleCondition condition1;
        condition1.type(CRuleCondition::E_NumericalActual);
        condition1.fieldName(personFieldName);
        condition1.fieldValue(person1);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 9.0;
        CRuleCondition condition2;
        condition2.type(CRuleCondition::E_NumericalActual);
        condition2.fieldName(personFieldName);
        condition2.fieldValue(person1);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 9.5;
        CDetectionRule rule;
        rule.conditionsConnective(CDetectionRule::E_And);
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);
    }
    {
        // First applies only
        CRuleCondition condition1;
        condition1.type(CRuleCondition::E_NumericalActual);
        condition1.fieldName(personFieldName);
        condition1.fieldValue(person1);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 11.0;
        CRuleCondition condition2;
        condition2.type(CRuleCondition::E_NumericalActual);
        condition2.fieldName(personFieldName);
        condition2.fieldValue(person1);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 9.5;
        CDetectionRule rule;
        rule.conditionsConnective(CDetectionRule::E_And);
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);
    }
    {
        // Second applies only
        CRuleCondition condition1;
        condition1.type(CRuleCondition::E_NumericalActual);
        condition1.fieldName(personFieldName);
        condition1.fieldValue(person1);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 9.0;
        CRuleCondition condition2;
        condition2.type(CRuleCondition::E_NumericalActual);
        condition2.fieldName(personFieldName);
        condition2.fieldValue(person1);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 10.5;
        CDetectionRule rule;
        rule.conditionsConnective(CDetectionRule::E_And);
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);
    }
    {
        // Both apply
        CRuleCondition condition1;
        condition1.type(CRuleCondition::E_NumericalActual);
        condition1.fieldName(personFieldName);
        condition1.fieldValue(person1);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 12.0;
        CRuleCondition condition2;
        condition2.type(CRuleCondition::E_NumericalActual);
        condition2.fieldName(personFieldName);
        condition2.fieldValue(person1);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 10.5;
        CDetectionRule rule;
        rule.conditionsConnective(CDetectionRule::E_And);
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    }
}

void CDetectionRuleTest::testApplyGivenTargetFieldIsPartitionAndIndividualModel(void)
{
    LOG_DEBUG("*** testApplyGivenTargetFieldIsPartitionAndIndividualModel ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string partitionFieldName("partition");
    std::string partitionFieldValue("partition_1");
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, partitionFieldName, partitionFieldValue, personFieldName, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    std::string person1("p1");
    bool addedPerson = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, addedPerson);
    std::string person2("p2");
    gathererPtr->addPerson(person2, m_ResourceMonitor, addedPerson);

    CMockModel model(params, gathererPtr, influenceCalculators);
    CAnomalyDetectorModel::TDouble1Vec p1Actual(1, 10.0);
    CAnomalyDetectorModel::TDouble1Vec p2Actual(1, 20.0);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 100, p1Actual);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 1, 0, 100, p2Actual);

    {
        // No targetFieldValue
        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalActual);
        condition.fieldName(personFieldName);
        condition.fieldValue(person1);
        condition.condition().s_Op = CRuleCondition::E_LT;
        condition.condition().s_Threshold = 15.0;
        CDetectionRule rule;
        rule.targetFieldName(partitionFieldName);
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 1, 0, 100));
    }
    {
        // Matching targetFieldValue
        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalActual);
        condition.fieldName(personFieldName);
        condition.fieldValue(person1);
        condition.condition().s_Op = CRuleCondition::E_LT;
        condition.condition().s_Threshold = 15.0;
        CDetectionRule rule;
        rule.targetFieldName(partitionFieldName);
        rule.targetFieldValue(partitionFieldValue);
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 1, 0, 100));
    }
    {
        // Non-matching targetFieldValue
        std::string partitionValue2("partition_2");
        CRuleCondition condition;
        condition.type(CRuleCondition::E_NumericalActual);
        condition.fieldName(personFieldName);
        condition.fieldValue(person1);
        condition.condition().s_Op = CRuleCondition::E_LT;
        condition.condition().s_Threshold = 15.0;
        CDetectionRule rule;
        rule.targetFieldName(partitionFieldName);
        rule.targetFieldValue(partitionValue2);
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 1, 0, 100) == false);
    }
}

void CDetectionRuleTest::testApplyGivenTimeCondition(void)
{
    LOG_DEBUG("*** testApplyGivenTimeCondition ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string partitionFieldName("partition");
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, partitionFieldName, EMPTY_STRING, personFieldName, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features,
                          startTime, 0));

    CMockModel model(params, gathererPtr, influenceCalculators);
    CRuleCondition conditionGte;
    conditionGte.type(CRuleCondition::E_Time);
    conditionGte.condition().s_Op = CRuleCondition::E_GTE;
    conditionGte.condition().s_Threshold = 100;
    CRuleCondition conditionLt;
    conditionLt.type(CRuleCondition::E_Time);
    conditionLt.condition().s_Op = CRuleCondition::E_LT;
    conditionLt.condition().s_Threshold = 200;

    CDetectionRule rule;
    rule.conditionsConnective(CDetectionRule::E_And);
    rule.addCondition(conditionGte);
    rule.addCondition(conditionLt);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 99) == false);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 150));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 200) == false);
}

void CDetectionRuleTest::testRuleActions(void)
{
    LOG_DEBUG("*** testRuleActions ***");

    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string partitionFieldName("partition");
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(
            new CDataGatherer(model_t::E_Metric, model_t::E_None, params,
                              EMPTY_STRING, partitionFieldName, EMPTY_STRING, personFieldName, EMPTY_STRING, EMPTY_STRING,
                              TStrVec(), false, key, features, startTime, 0));

    CMockModel model(params, gathererPtr, influenceCalculators);
    CRuleCondition conditionGte;
    conditionGte.type(CRuleCondition::E_Time);
    conditionGte.condition().s_Op = CRuleCondition::E_GTE;
    conditionGte.condition().s_Threshold = 100;

    CDetectionRule rule;
    rule.conditionsConnective(CDetectionRule::E_And);
    rule.addCondition(conditionGte);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipSampling, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100) == false);

    rule.action(CDetectionRule::E_SkipSampling);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100)  == false);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipSampling, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));

    rule.action(static_cast<CDetectionRule::ERuleAction>(3));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_FilterResults, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipSampling, model,
                model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
}
