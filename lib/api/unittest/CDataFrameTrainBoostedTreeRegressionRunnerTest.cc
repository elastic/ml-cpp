/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameTrainBoostedTreeRegressionRunner.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>

#include <boost/test/unit_test.hpp>

#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDataFrameTrainBoostedTreeRegressionRunnerTest)

using namespace ml;
namespace {
using TStrVec = std::vector<std::string>;
}

BOOST_AUTO_TEST_CASE(testPredictionFieldNameClash) {
    TStrVec errors;
    auto errorHandler = [&errors](std::string error) { errors.push_back(error); };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    const auto spec{test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
        test::CDataFrameAnalysisSpecificationFactory::regression(), "dep_var",
        5, 6, 13000000, 0, 0)};
    rapidjson::Document jsonParameters;
    jsonParameters.Parse("{"
                         "  \"dependent_variable\": \"dep_var\","
                         "  \"prediction_field_name\": \"is_training\""
                         "}");
    const auto parameters{
        api::CDataFrameTrainBoostedTreeRegressionRunner::parameterReader().read(jsonParameters)};
    api::CDataFrameTrainBoostedTreeRegressionRunner runner(*spec, parameters);

    BOOST_TEST_REQUIRE(errors.size() == 1);
    BOOST_TEST_REQUIRE(errors[0] == "Input error: prediction_field_name must not be equal to any of [is_training].");
}

BOOST_AUTO_TEST_SUITE_END()
