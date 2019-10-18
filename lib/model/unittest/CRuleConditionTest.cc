/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CModelParams.h>
#include <model/CRuleCondition.h>
#include <model/CSearchKey.h>
#include <model/ModelTypes.h>

#include "Mocks.h"

#include <boost/test/unit_test.hpp>

#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CRuleConditionTest)

using namespace ml;
using namespace model;

namespace {

using TStrVec = std::vector<std::string>;

const std::string EMPTY_STRING;
}

BOOST_AUTO_TEST_CASE(testTimeContition) {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    model_t::TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(std::make_shared<CDataGatherer>(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, TStrVec{}, key, features, startTime, 0));

    CMockModel model(params, gathererPtr, influenceCalculators);

    {
        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Time);
        condition.op(CRuleCondition::E_GTE);
        condition.value(500);

        model_t::CResultType resultType(model_t::CResultType::E_Final);
        BOOST_TEST(condition.test(model, model_t::E_IndividualCountByBucketAndPerson,
                                  resultType, std::size_t(0), std::size_t(1),
                                  core_t::TTime(450)) == false);
        BOOST_TEST(condition.test(model, model_t::E_IndividualCountByBucketAndPerson, resultType,
                                  std::size_t(0), std::size_t(1), core_t::TTime(550)));
    }

    {
        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Time);
        condition.op(CRuleCondition::E_LT);
        condition.value(600);

        model_t::CResultType resultType(model_t::CResultType::E_Final);
        BOOST_TEST(condition.test(model, model_t::E_IndividualCountByBucketAndPerson,
                                  resultType, std::size_t(0), std::size_t(1),
                                  core_t::TTime(600)) == false);
        BOOST_TEST(condition.test(model, model_t::E_IndividualCountByBucketAndPerson, resultType,
                                  std::size_t(0), std::size_t(1), core_t::TTime(599)));
    }
}

BOOST_AUTO_TEST_SUITE_END()
