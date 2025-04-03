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

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>
#include <model/CRuleCondition.h>
#include <model/CSearchKey.h>
#include <model/ModelTypes.h>
#include <model/SModelParams.h>

#include "Mocks.h"
#include "ModelTestHelpers.h"

#include <boost/test/unit_test.hpp>

#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CRuleConditionTest)

using namespace ml;
using namespace model;

BOOST_AUTO_TEST_CASE(testTimeContition) {
    constexpr core_t::TTime bucketLength = 100;
    constexpr core_t::TTime startTime = 100;
    CSearchKey const key;
    SModelParams const params(bucketLength);
    const CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    model_t::TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    auto gathererPtr = CDataGathererBuilder(model_t::E_Metric, features, params, key, startTime)
                           .buildSharedPtr();

    CMockModel const model(params, gathererPtr, influenceCalculators);

    {
        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Time);
        condition.op(CRuleCondition::E_GTE);
        condition.value(500);

        model_t::CResultType const resultType(model_t::CResultType::E_Final);
        BOOST_TEST_REQUIRE(condition.test(model, model_t::E_IndividualCountByBucketAndPerson,
                                          resultType, 0, 1, 450) == false);
        BOOST_TEST_REQUIRE(condition.test(model, model_t::E_IndividualCountByBucketAndPerson,
                                          resultType, 0, 1, 550));
    }

    {
        CRuleCondition condition;
        condition.appliesTo(CRuleCondition::E_Time);
        condition.op(CRuleCondition::E_LT);
        condition.value(600);

        model_t::CResultType const resultType(model_t::CResultType::E_Final);
        BOOST_TEST_REQUIRE(condition.test(model, model_t::E_IndividualCountByBucketAndPerson,
                                          resultType, 0, 1, 600) == false);
        BOOST_TEST_REQUIRE(condition.test(model, model_t::E_IndividualCountByBucketAndPerson,
                                          resultType, 0, 1, 599));
    }
}

BOOST_AUTO_TEST_SUITE_END()
