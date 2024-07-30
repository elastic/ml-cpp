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

#include <core/CLogger.h>
#include <core/CPatternSet.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CResourceMonitor.h>
#include <model/CRuleCondition.h>
#include <model/CSearchKey.h>
#include <model/ModelTypes.h>
#include <model/SModelParams.h>

#include <maths/common/CNormalMeanPrecConjugate.h>
#include <maths/common/MathsTypes.h>
#include <maths/time_series/CTimeSeriesDecomposition.h>
#include <maths/time_series/CTimeSeriesModel.h>

#include <test/CRandomNumbers.h>

#include "Mocks.h"

#include <boost/test/tools/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <memory>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDetectionRuleTest)

using namespace ml;
using namespace model;

namespace {

using TFeatureVec = std::vector<model_t::EFeature>;
using TStrVec = std::vector<std::string>;
using TMockModelPtr = std::unique_ptr<model::CMockModel>;

const std::string EMPTY_STRING;
}

class CTestFixture {
protected:
    CResourceMonitor m_ResourceMonitor;
};

BOOST_FIXTURE_TEST_CASE(testApplyGivenScope, CTestFixture) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    std::string partitionFieldName("partition");
    std::string partitionFieldValue("par_1");
    std::string personFieldName("over");
    std::string attributeFieldName("by");
    CSearchKey key(0, function_t::E_PopulationMetricMean, false, model_t::E_XF_None,
                   "", attributeFieldName, personFieldName, partitionFieldName);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_PopulationMetric, model_t::E_None, params, EMPTY_STRING,
        partitionFieldValue, personFieldName, attributeFieldName, EMPTY_STRING,
        TStrVec{}, key, features, startTime, 0));

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

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 0, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 1, 100) != isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 1, 2, 100) != isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
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

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 0, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 1, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 1, 2, 100) != isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
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

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 0, 100) != isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 1, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 1, 2, 100) != isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
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

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 0, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 1, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 1, 2, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
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

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 0, 100) != isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 1, 100) != isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 1, 2, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
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

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 0, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 1, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 1, 2, 100) == isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
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

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 0, 100) != isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 0, 1, 100) != isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 1, 2, 100) != isInclude);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_PopulationMeanByPersonAndAttribute,
                                      resultType, 1, 3, 100) != isInclude);
    }
}

BOOST_FIXTURE_TEST_CASE(testApplyGivenNumericalActualCondition, CTestFixture) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, TStrVec{}, key, features, startTime, 0));

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
        condition.op(CRuleCondition::E_LT);
        condition.value(5.0);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100));
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 200) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 300) == false);
    }

    {
        // Test rule with condition with operator LTE

        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Actual);
        condition.op(CRuleCondition::E_LTE);
        condition.value(5.0);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100));
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 200));
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 300) == false);
    }
    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Actual);
        condition.op(CRuleCondition::E_GT);
        condition.value(5.0);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 200) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 300));
    }
    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Actual);
        condition.op(CRuleCondition::E_GTE);
        condition.value(5.0);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 200));
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 300));
    }
}

BOOST_FIXTURE_TEST_CASE(testApplyGivenNumericalTypicalCondition, CTestFixture) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, TStrVec{}, key, features, startTime, 0));

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
        condition.op(CRuleCondition::E_LT);
        condition.value(45.0);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100));
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 200) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 300) == false);
    }

    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Typical);
        condition.op(CRuleCondition::E_GT);
        condition.value(45.0);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 200) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 300));
    }
}

BOOST_FIXTURE_TEST_CASE(testApplyGivenNumericalDiffAbsCondition, CTestFixture) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, TStrVec{}, key, features, startTime, 0));

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
        condition.op(CRuleCondition::E_LT);
        condition.value(1.0);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 200) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 300));
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 400));
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 500) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 600) == false);
    }

    {
        // Test rule with condition with operator GT

        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_DiffFromTypical);
        condition.op(CRuleCondition::E_GT);
        condition.value(1.0);
        CDetectionRule rule;
        rule.addCondition(condition);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100));
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 200) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 300) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 400) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 500) == false);
        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 600));
    }
}

BOOST_FIXTURE_TEST_CASE(testApplyGivenNoActualValueAvailable, CTestFixture) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, TStrVec{}, key, features, startTime, 0));

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
    condition.op(CRuleCondition::E_LT);
    condition.value(5.0);
    CDetectionRule rule;
    rule.addCondition(condition);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 400) == false);
}

BOOST_FIXTURE_TEST_CASE(testApplyGivenDifferentSeriesAndIndividualModel, CTestFixture) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING, personFieldName,
        EMPTY_STRING, EMPTY_STRING, TStrVec{}, key, features, startTime, 0));

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
    condition.op(CRuleCondition::E_LT);
    condition.value(5.0);

    rule.addCondition(condition);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100));
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 1, 0, 100) == false);
}

BOOST_FIXTURE_TEST_CASE(testApplyGivenDifferentSeriesAndPopulationModel, CTestFixture) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    std::string personFieldName("over");
    std::string attributeFieldName("by");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_PopulationMetric, model_t::E_None, params, EMPTY_STRING,
        EMPTY_STRING, personFieldName, attributeFieldName, EMPTY_STRING,
        TStrVec{}, key, features, startTime, 0));

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
    condition.op(CRuleCondition::E_LT);
    condition.value(5.0);
    rule.addCondition(condition);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 0, 100) == false);
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 0, 1, 100));
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 2, 100) == false);
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                  model_t::E_PopulationMeanByPersonAndAttribute,
                                  resultType, 1, 3, 100) == false);
}

BOOST_FIXTURE_TEST_CASE(testApplyGivenMultipleConditions, CTestFixture) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string personFieldName("series");
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING, personFieldName,
        EMPTY_STRING, EMPTY_STRING, TStrVec{}, key, features, startTime, 0));

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
        condition1.op(CRuleCondition::E_LT);
        condition1.value(9.0);
        CRuleCondition condition2;
        condition2.appliesTo(CRuleCondition::E_Actual);
        condition2.op(CRuleCondition::E_LT);
        condition2.value(9.5);
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100) == false);
    }
    {
        // First applies only
        CRuleCondition condition1;
        condition1.appliesTo(CRuleCondition::E_Actual);
        condition1.op(CRuleCondition::E_LT);
        condition1.value(11.0);
        CRuleCondition condition2;
        condition2.appliesTo(CRuleCondition::E_Actual);
        condition2.op(CRuleCondition::E_LT);
        condition2.value(9.5);
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100) == false);
    }
    {
        // Second applies only
        CRuleCondition condition1;
        condition1.appliesTo(CRuleCondition::E_Actual);
        condition1.op(CRuleCondition::E_LT);
        condition1.value(9.0);
        CRuleCondition condition2;
        condition2.appliesTo(CRuleCondition::E_Actual);
        condition2.op(CRuleCondition::E_LT);
        condition2.value(10.5);
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100) == false);
    }
    {
        // Both apply
        CRuleCondition condition1;
        condition1.appliesTo(CRuleCondition::E_Actual);
        condition1.op(CRuleCondition::E_LT);
        condition1.value(12.0);
        CRuleCondition condition2;
        condition2.appliesTo(CRuleCondition::E_Actual);
        condition2.op(CRuleCondition::E_LT);
        condition2.value(10.5);
        CDetectionRule rule;
        rule.addCondition(condition1);
        rule.addCondition(condition2);

        model_t::CResultType resultType(model_t::CResultType::E_Final);

        BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model,
                                      model_t::E_IndividualMeanByPerson,
                                      resultType, 0, 0, 100));
    }
}

BOOST_FIXTURE_TEST_CASE(testApplyGivenTimeCondition, CTestFixture) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string partitionFieldName("partition");
    std::string personFieldName("series");
    CSearchKey key(0, function_t::E_IndividualMetricMean, false, model_t::E_XF_None,
                   "", personFieldName, EMPTY_STRING, partitionFieldName);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING, personFieldName,
        EMPTY_STRING, EMPTY_STRING, TStrVec{}, key, features, startTime, 0));

    CMockModel model(params, gathererPtr, influenceCalculators);
    CRuleCondition conditionGte;
    conditionGte.appliesTo(CRuleCondition::E_Time);
    conditionGte.op(CRuleCondition::E_GTE);
    conditionGte.value(100);
    CRuleCondition conditionLt;
    conditionLt.appliesTo(CRuleCondition::E_Time);
    conditionLt.op(CRuleCondition::E_LT);
    conditionLt.value(200);

    CDetectionRule rule;
    rule.addCondition(conditionGte);
    rule.addCondition(conditionLt);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 99) == false);
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100));
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 150));
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 200) == false);
}

BOOST_FIXTURE_TEST_CASE(testRuleActions, CTestFixture) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    std::string partitionFieldName("partition");
    std::string personFieldName("series");
    CSearchKey key(0, function_t::E_IndividualMetricMean, false, model_t::E_XF_None,
                   "", personFieldName, EMPTY_STRING, partitionFieldName);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING, personFieldName,
        EMPTY_STRING, EMPTY_STRING, TStrVec{}, key, features, startTime, 0));

    CMockModel model(params, gathererPtr, influenceCalculators);
    CRuleCondition conditionGte;
    conditionGte.appliesTo(CRuleCondition::E_Time);
    conditionGte.op(CRuleCondition::E_GTE);
    conditionGte.value(100);

    CDetectionRule rule;
    rule.addCondition(conditionGte);

    model_t::CResultType resultType(model_t::CResultType::E_Final);

    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100));
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipModelUpdate, model,
                                  model_t::E_IndividualMeanByPerson, resultType,
                                  0, 0, 100) == false);

    rule.action(CDetectionRule::E_SkipModelUpdate);
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100) == false);
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipModelUpdate, model,
                                  model_t::E_IndividualMeanByPerson, resultType,
                                  0, 0, 100));

    rule.action(static_cast<CDetectionRule::ERuleAction>(3));
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipResult, model, model_t::E_IndividualMeanByPerson,
                                  resultType, 0, 0, 100));
    BOOST_TEST_REQUIRE(rule.apply(CDetectionRule::E_SkipModelUpdate, model,
                                  model_t::E_IndividualMeanByPerson, resultType,
                                  0, 0, 100));
}

TMockModelPtr initializeModel(ml::model::CResourceMonitor& resourceMonitor) {
    core_t::TTime bucketLength{600};
    model::SModelParams params{bucketLength};
    model::CSearchKey key;
    model_t::TFeatureVec features;
    // Initialize mock model
    model::CAnomalyDetectorModel::TDataGathererPtr gatherer;

    features.assign(1, model_t::E_IndividualSumByBucketAndPerson);

    gatherer = std::make_shared<model::CDataGatherer>(
        model_t::analysisCategory(features[0]), model_t::E_None, params, EMPTY_STRING,
        EMPTY_STRING, "p", EMPTY_STRING, EMPTY_STRING, TStrVec{}, key, features, 0, 0);

    std::string person("p1");
    bool addedPerson{false};
    gatherer->addPerson(person, resourceMonitor, addedPerson);

    TMockModelPtr model{new model::CMockModel(
        params, gatherer, {/* we don't care about influence */})};

    maths::time_series::CTimeSeriesDecomposition trend;
    maths::common::CNormalMeanPrecConjugate prior{
        maths::common::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData)};
    maths::common::CModelParams timeSeriesModelParams{
        bucketLength, 1.0, 0.001, 0.2, 6 * core::constants::HOUR, 24 * core::constants::HOUR};
    std::unique_ptr<maths::time_series::CUnivariateTimeSeriesModel> timeSeriesModel =
        std::make_unique<maths::time_series::CUnivariateTimeSeriesModel>(
            timeSeriesModelParams, 0, trend, prior);
    model::CMockModel::TMathsModelUPtrVec models;
    models.emplace_back(std::move(timeSeriesModel));
    model->mockTimeSeriesModels(std::move(models));
    return model;
}

BOOST_FIXTURE_TEST_CASE(testRuleTimeShiftShouldShiftTimeSeriesModelState, CTestFixture) {

    test::CRandomNumbers rng;
    test::CRandomNumbers::TDoubleVec timeShifts;
    rng.generateUniformSamples(-3600, 3600, 10, timeShifts);

    for (auto timeShift : timeShifts) {
        core_t::TTime timeShiftInSecs{static_cast<core_t::TTime>(timeShift)};
        TMockModelPtr model{initializeModel(m_ResourceMonitor)};
        // Capture state before the rule is applied
        const auto& trendModel =
            static_cast<const maths::time_series::CTimeSeriesDecomposition&>(
                static_cast<const maths::time_series::CUnivariateTimeSeriesModel*>(
                    model->model(0))
                    ->trendModel());
        core_t::TTime lastValueTime = trendModel.lastValueTime();

        core_t::TTime timestamp{100};
        CRuleCondition conditionGte;
        conditionGte.appliesTo(CRuleCondition::E_Time);
        conditionGte.op(CRuleCondition::E_GTE);
        conditionGte.value(static_cast<double>(timestamp));

        // When time shift rule is applied
        CDetectionRule rule;
        rule.addCondition(conditionGte);
        rule.addTimeShift(timeShiftInSecs);
        rule.executeCallback(*model, timestamp);

        // the time series model should have been shifted by specified amount.
        BOOST_TEST_REQUIRE(trendModel.lastValueTime() == lastValueTime + timeShiftInSecs);
        BOOST_TEST_REQUIRE(trendModel.timeShift() == timeShiftInSecs);
    }
}

BOOST_FIXTURE_TEST_CASE(testRuleTimeShiftShouldNotApplyTwice, CTestFixture) {
    // Test that if a rule has already been applied, it should not be applied again.
    core_t::TTime timeShift{3600};

    TMockModelPtr model{initializeModel(m_ResourceMonitor)};
    const auto& trendModel = static_cast<const maths::time_series::CTimeSeriesDecomposition&>(
        static_cast<const maths::time_series::CUnivariateTimeSeriesModel*>(model->model(0))
            ->trendModel());

    core_t::TTime timestamp{100};
    CRuleCondition conditionGte;
    conditionGte.appliesTo(CRuleCondition::E_Time);
    conditionGte.op(CRuleCondition::E_GTE);
    conditionGte.value(static_cast<double>(timestamp));

    // When time shift rule is applied twice
    CDetectionRule rule;
    rule.addCondition(conditionGte);
    rule.addTimeShift(timeShift);
    rule.executeCallback(*model, timestamp);

    core_t::TTime lastValueTimeAfterFirstShift = trendModel.lastValueTime();
    core_t::TTime timeShiftAfterFirstShift = trendModel.timeShift();

    // the values after the second time should be the same as the values after the first time shift.
    timestamp += timeShift; // simulate the time has moved forward by the time shift
    rule.executeCallback(*model, timestamp);
    BOOST_TEST_REQUIRE(trendModel.lastValueTime() == lastValueTimeAfterFirstShift);
    BOOST_TEST_REQUIRE(trendModel.timeShift() == timeShiftAfterFirstShift);
}

BOOST_FIXTURE_TEST_CASE(testTwoTimeShiftRuleShouldShiftTwice, CTestFixture) {
    // Test that if two rules are applied, the time series model should be shifted twice.
    core_t::TTime timeShift1{3600};
    core_t::TTime timeShift2{7200};

    TMockModelPtr model{initializeModel(m_ResourceMonitor)};
    const auto& trendModel = static_cast<const maths::time_series::CTimeSeriesDecomposition&>(
        static_cast<const maths::time_series::CUnivariateTimeSeriesModel*>(model->model(0))
            ->trendModel());

    core_t::TTime timestamp{100};
    CRuleCondition conditionGte;
    conditionGte.appliesTo(CRuleCondition::E_Time);
    conditionGte.op(CRuleCondition::E_GTE);
    conditionGte.value(static_cast<double>(timestamp));

    // When time shift rule is applied twice
    CDetectionRule rule1;
    rule1.addCondition(conditionGte);
    rule1.addTimeShift(timeShift1);
    rule1.executeCallback(*model, timestamp);

    core_t::TTime lastValueTimeAfterFirstShift = trendModel.lastValueTime();

    CDetectionRule rule2;
    rule2.addCondition(conditionGte);
    rule2.addTimeShift(timeShift2);
    rule2.executeCallback(*model, timestamp);

    // the values after the second time should be the sum of two rules.
    timestamp += timeShift1; // simulate the time has moved forward by the time shift
    rule2.executeCallback(*model, timestamp);
    BOOST_TEST_REQUIRE(trendModel.lastValueTime() == lastValueTimeAfterFirstShift + timeShift2);
    BOOST_TEST_REQUIRE(trendModel.timeShift() == timeShift1 + timeShift2);
}

BOOST_AUTO_TEST_SUITE_END()
