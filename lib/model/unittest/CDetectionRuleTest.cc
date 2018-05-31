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

namespace {

using TFeatureVec = std::vector<model_t::EFeature>;
using TStrVec = std::vector<std::string>;

const std::string EMPTY_STRING;
}

CppUnit::Test* CDetectionRuleTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDetectionRuleTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
        "CDetectionRuleTest::testApplyGivenScope", &CDetectionRuleTest::testApplyGivenScope));
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
        "CDetectionRuleTest::testApplyGivenNoActualValueAvailable",
        &CDetectionRuleTest::testApplyGivenNoActualValueAvailable));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
        "CDetectionRuleTest::testApplyGivenDifferentSeriesAndIndividualModel",
        &CDetectionRuleTest::testApplyGivenDifferentSeriesAndIndividualModel));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
        "CDetectionRuleTest::testApplyGivenDifferentSeriesAndPopulationModel",
        &CDetectionRuleTest::testApplyGivenDifferentSeriesAndPopulationModel));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
        "CDetectionRuleTest::testApplyGivenMultipleConditions",
        &CDetectionRuleTest::testApplyGivenMultipleConditions));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
        "CDetectionRuleTest::testApplyGivenTimeCondition",
        &CDetectionRuleTest::testApplyGivenTimeCondition));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectionRuleTest>(
        "CDetectionRuleTest::testRuleActions", &CDetectionRuleTest::testRuleActions));

    return suiteOfTests;
}

void CDetectionRuleTest::testApplyGivenScope() {
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
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_PopulationMetric, model_t::E_None, params, EMPTY_STRING,
        partitionFieldName, partitionFieldValue, personFieldName, attributeFieldName,
        EMPTY_STRING, TStrVec(), false, key, features, startTime, 0));

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

    for (auto filterType : {CRuleScope::E_Include, CRuleScope::E_Exclude}) {
        std::string filterJson("[\"a1_1\",\"a2_2\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CDetectionRule rule;
        if (filterType == CRuleScope::E_Include) {
            rule.includeScope(attributeFieldName, valueFilter);
        } else {
            rule.excludeScope(attributeFieldName, valueFilter);
        }

        bool isInclude = filterType == CRuleScope::E_Include;
        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 0, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 1, 100) != isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 2, 100) != isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 3, 100) == isInclude);
    }

    for (auto filterType : {CRuleScope::E_Include, CRuleScope::E_Exclude}) {
        std::string filterJson("[\"a1*\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CDetectionRule rule;
        if (filterType == CRuleScope::E_Include) {
            rule.includeScope(attributeFieldName, valueFilter);
        } else {
            rule.excludeScope(attributeFieldName, valueFilter);
        }

        bool isInclude = filterType == CRuleScope::E_Include;
        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 0, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 1, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 2, 100) != isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 3, 100) != isInclude);
    }

    for (auto filterType : {CRuleScope::E_Include, CRuleScope::E_Exclude}) {
        std::string filterJson("[\"*2\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CDetectionRule rule;
        if (filterType == CRuleScope::E_Include) {
            rule.includeScope(attributeFieldName, valueFilter);
        } else {
            rule.excludeScope(attributeFieldName, valueFilter);
        }

        bool isInclude = filterType == CRuleScope::E_Include;
        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 0, 100) != isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 1, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 2, 100) != isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 3, 100) == isInclude);
    }

    for (auto filterType : {CRuleScope::E_Include, CRuleScope::E_Exclude}) {
        std::string filterJson("[\"*1*\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CDetectionRule rule;
        if (filterType == CRuleScope::E_Include) {
            rule.includeScope(attributeFieldName, valueFilter);
        } else {
            rule.excludeScope(attributeFieldName, valueFilter);
        }

        bool isInclude = filterType == CRuleScope::E_Include;
        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 0, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 1, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 2, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 3, 100) != isInclude);
    }

    for (auto filterType : {CRuleScope::E_Include, CRuleScope::E_Exclude}) {
        std::string filterJson("[\"p2\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CDetectionRule rule;
        if (filterType == CRuleScope::E_Include) {
            rule.includeScope(personFieldName, valueFilter);
        } else {
            rule.excludeScope(personFieldName, valueFilter);
        }

        bool isInclude = filterType == CRuleScope::E_Include;
        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 0, 100) != isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 1, 100) != isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 2, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 3, 100) == isInclude);
    }

    for (auto filterType : {CRuleScope::E_Include, CRuleScope::E_Exclude}) {
        std::string filterJson("[\"par_1\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CDetectionRule rule;
        if (filterType == CRuleScope::E_Include) {
            rule.includeScope(partitionFieldName, valueFilter);
        } else {
            rule.excludeScope(partitionFieldName, valueFilter);
        }

        bool isInclude = filterType == CRuleScope::E_Include;
        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 0, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 1, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 2, 100) == isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 3, 100) == isInclude);
    }

    for (auto filterType : {CRuleScope::E_Include, CRuleScope::E_Exclude}) {
        std::string filterJson("[\"par_2\"]");
        core::CPatternSet valueFilter;
        valueFilter.initFromJson(filterJson);

        CDetectionRule rule;
        if (filterType == CRuleScope::E_Include) {
            rule.includeScope(partitionFieldName, valueFilter);
        } else {
            rule.excludeScope(partitionFieldName, valueFilter);
        }

        bool isInclude = filterType == CRuleScope::E_Include;
        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 0, 100) != isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 1, 100) != isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 2, 100) != isInclude);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 3, 100) != isInclude);
    }
}

void CDetectionRuleTest::testApplyGivenNumericalActualCondition() {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, TStrVec(),
        false, key, features, startTime, 0));

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
        condition.appliesTo(CRuleCondition::E_Actual);
        condition.condition().s_Op = CRuleCondition::E_LT;
        condition.condition().s_Threshold = 5.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 300) == false);
    }

    {
        // Test rule with condition with operator LTE

        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Actual);
        condition.condition().s_Op = CRuleCondition::E_LTE;
        condition.condition().s_Threshold = 5.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 200));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 300) == false);
    }
    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Actual);
        condition.condition().s_Op = CRuleCondition::E_GT;
        condition.condition().s_Threshold = 5.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 300));
    }
    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Actual);
        condition.condition().s_Op = CRuleCondition::E_GTE;
        condition.condition().s_Threshold = 5.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 200));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 300));
    }
}

void CDetectionRuleTest::testApplyGivenNumericalTypicalCondition() {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, TStrVec(),
        false, key, features, startTime, 0));

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
        condition.appliesTo(CRuleCondition::E_Typical);
        condition.condition().s_Op = CRuleCondition::E_LT;
        condition.condition().s_Threshold = 45.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 300) == false);
    }

    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Typical);
        condition.condition().s_Op = CRuleCondition::E_GT;
        condition.condition().s_Threshold = 45.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 300));
    }
}

void CDetectionRuleTest::testApplyGivenNumericalDiffAbsCondition() {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, TStrVec(),
        false, key, features, startTime, 0));

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
        condition.appliesTo(CRuleCondition::E_DiffFromTypical);
        condition.condition().s_Op = CRuleCondition::E_LT;
        condition.condition().s_Threshold = 1.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 300));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 400));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 500) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 600) == false);
    }

    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_DiffFromTypical);
        condition.condition().s_Op = CRuleCondition::E_GT;
        condition.condition().s_Threshold = 1.0;
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100));
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 200) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 300) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 400) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 500) == false);
        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 600));
    }
}

void CDetectionRuleTest::testApplyGivenNoActualValueAvailable() {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, TStrVec(),
        false, key, features, startTime, 0));

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
    condition.appliesTo(CRuleCondition::E_Actual);
    condition.condition().s_Op = CRuleCondition::E_LT;
    condition.condition().s_Threshold = 5.0;
    CDetectionRule rule;
    rule.addCondition(condition);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                              resultType, 0, 0, 400) == false);
}

void CDetectionRuleTest::testApplyGivenDifferentSeriesAndIndividualModel() {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, personFieldName, EMPTY_STRING, EMPTY_STRING, TStrVec(),
        false, key, features, startTime, 0));

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

    CDetectionRule rule;

    std::string filterJson("[\"p1\"]");
    core::CPatternSet valueFilter;
    valueFilter.initFromJson(filterJson);
    rule.includeScope(personFieldName, valueFilter);

    CRuleCondition condition;
    condition.appliesTo(CRuleCondition::E_Actual);
    condition.condition().s_Op = CRuleCondition::E_LT;
    condition.condition().s_Threshold = 5.0;

    rule.addCondition(condition);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                              model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                              resultType, 1, 0, 100) == false);
}

void CDetectionRuleTest::testApplyGivenDifferentSeriesAndPopulationModel() {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    std::string personFieldName("over");
    std::string attributeFieldName("by");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_PopulationMetric, model_t::E_None, params, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, personFieldName, attributeFieldName,
        EMPTY_STRING, TStrVec(), false, key, features, startTime, 0));

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

    CDetectionRule rule;

    std::string filterJson("[\"" + attr12 + "\"]");
    core::CPatternSet valueFilter;
    valueFilter.initFromJson(filterJson);
    rule.includeScope(attributeFieldName, valueFilter);

    CRuleCondition condition;
    condition.appliesTo(CRuleCondition::E_Actual);
    condition.condition().s_Op = CRuleCondition::E_LT;
    condition.condition().s_Threshold = 5.0;
    rule.addCondition(condition);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                              model_t::E_PopulationMeanByPersonAndAttribute,
                              resultType, 0, 0, 100) == false);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                              model_t::E_PopulationMeanByPersonAndAttribute,
                              resultType, 0, 1, 100));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                              model_t::E_PopulationMeanByPersonAndAttribute,
                              resultType, 1, 2, 100) == false);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                              model_t::E_PopulationMeanByPersonAndAttribute,
                              resultType, 1, 3, 100) == false);
}

void CDetectionRuleTest::testApplyGivenMultipleConditions() {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, personFieldName, EMPTY_STRING, EMPTY_STRING, TStrVec(),
        false, key, features, startTime, 0));

    std::string person1("p1");
    bool addedPerson = false;
    gathererPtr->addPerson(person1, m_ResourceMonitor, addedPerson);

    CMockModel model(params, gathererPtr, influenceCalculators);
    CAnomalyDetectorModel::TDouble1Vec p1Actual(1, 10.0);
    model.mockAddBucketValue(model_t::E_IndividualMeanByPerson, 0, 0, 100, p1Actual);

    {
        // None applies
        CRuleCondition condition1;
        condition1.appliesTo(CRuleCondition::E_Actual);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 9.0;
        CRuleCondition condition2;
        condition2.appliesTo(CRuleCondition::E_Actual);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 9.5;
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100) == false);
    }
    {
        // First applies only
        CRuleCondition condition1;
        condition1.appliesTo(CRuleCondition::E_Actual);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 11.0;
        CRuleCondition condition2;
        condition2.appliesTo(CRuleCondition::E_Actual);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 9.5;
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100) == false);
    }
    {
        // Second applies only
        CRuleCondition condition1;
        condition1.appliesTo(CRuleCondition::E_Actual);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 9.0;
        CRuleCondition condition2;
        condition2.appliesTo(CRuleCondition::E_Actual);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 10.5;
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100) == false);
    }
    {
        // Both apply
        CRuleCondition condition1;
        condition1.appliesTo(CRuleCondition::E_Actual);
        condition1.condition().s_Op = CRuleCondition::E_LT;
        condition1.condition().s_Threshold = 12.0;
        CRuleCondition condition2;
        condition2.appliesTo(CRuleCondition::E_Actual);
        condition2.condition().s_Op = CRuleCondition::E_LT;
        condition2.condition().s_Threshold = 10.5;
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100));
    }
}

void CDetectionRuleTest::testApplyGivenTimeCondition() {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string partitionFieldName("partition");
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING,
        partitionFieldName, EMPTY_STRING, personFieldName, EMPTY_STRING,
        EMPTY_STRING, TStrVec(), false, key, features, startTime, 0));

    CMockModel model(params, gathererPtr, influenceCalculators);
    CRuleCondition conditionGte;
    conditionGte.appliesTo(CRuleCondition::E_Time);
    conditionGte.condition().s_Op = CRuleCondition::E_GTE;
    conditionGte.condition().s_Threshold = 100;
    CRuleCondition conditionLt;
    conditionLt.appliesTo(CRuleCondition::E_Time);
    conditionLt.condition().s_Op = CRuleCondition::E_LT;
    conditionLt.condition().s_Threshold = 200;

    CDetectionRule rule;
    rule.addCondition(conditionGte);
    rule.addCondition(conditionLt);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                              resultType, 0, 0, 99) == false);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                              model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                              model_t::E_IndividualMeanByPerson, resultType, 0, 0, 150));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                              resultType, 0, 0, 200) == false);
}

void CDetectionRuleTest::testRuleActions() {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string partitionFieldName("partition");
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING,
        partitionFieldName, EMPTY_STRING, personFieldName, EMPTY_STRING,
        EMPTY_STRING, TStrVec(), false, key, features, startTime, 0));

    CMockModel model(params, gathererPtr, influenceCalculators);
    CRuleCondition conditionGte;
    conditionGte.appliesTo(CRuleCondition::E_Time);
    conditionGte.condition().s_Op = CRuleCondition::E_GTE;
    conditionGte.condition().s_Threshold = 100;

    CDetectionRule rule;
    rule.addCondition(conditionGte);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                              model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipModelUpdate, model, model_t::E_IndividualMeanByPerson,
                              resultType, 0, 0, 100) == false);

    rule.action(CDetectionRule::E_SkipModelUpdate);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                              resultType, 0, 0, 100) == false);
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipModelUpdate, model,
                              model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));

    rule.action(static_cast<CDetectionRule::ERuleAction>(3));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipResult, model,
                              model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
    CPPUNIT_ASSERT(rule.apply(CDetectionRule::E_SkipModelUpdate, model,
                              model_t::E_IndividualMeanByPerson, resultType, 0, 0, 100));
}
