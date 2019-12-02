/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameTrainBoostedTreeClassifierRunner.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>

#include <boost/test/unit_test.hpp>

#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDataFrameTrainBoostedTreeClassifierRunnerTest)

using namespace ml;
namespace {
using TRowItr = core::CDataFrame::TRowItr;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
}

BOOST_AUTO_TEST_CASE(testPredictionFieldNameClash) {
    TStrVec errors;
    auto errorHandler = [&errors](std::string error) { errors.push_back(error); };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    const auto spec{test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
        "classification", "dep_var", 5, 6, 13000000, 0, 0, {"dep_var"})};
    rapidjson::Document jsonParameters;
    jsonParameters.Parse("{"
                         "  \"dependent_variable\": \"dep_var\","
                         "  \"prediction_field_name\": \"is_training\""
                         "}");
    const auto parameters{
        api::CDataFrameTrainBoostedTreeClassifierRunner::parameterReader().read(jsonParameters)};
    api::CDataFrameTrainBoostedTreeClassifierRunner runner(*spec, parameters);

    BOOST_TEST_REQUIRE(errors.size() == 1);
    BOOST_TEST_REQUIRE(errors[0] == "Input error: prediction_field_name must not be equal to any of [is_training, prediction_probability, top_classes].");
}

BOOST_AUTO_TEST_CASE(testWriteOneRow) {
    // Prepare input data frame
    const TStrVec columnNames{"x1", "x2", "x3", "x4", "x5", "x5_prediction"};
    const TStrVec categoricalColumns{"x1", "x2", "x5"};
    const TStrVecVec rows{{"a", "b", "1.0", "1.0", "cat", "-1.0"},
                          {"a", "b", "2.0", "2.0", "cat", "-0.5"},
                          {"a", "b", "5.0", "5.0", "dog", "-0.1"},
                          {"c", "d", "5.0", "5.0", "dog", "1.0"},
                          {"e", "f", "5.0", "5.0", "dog", "1.5"}};
    std::unique_ptr<core::CDataFrame> frame =
        core::makeMainStorageDataFrame(columnNames.size()).first;
    frame->columnNames(columnNames);
    frame->categoricalColumns(categoricalColumns);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        frame->parseAndWriteRow(
            core::CVectorRange<const TStrVec>(rows[i], 0, rows[i].size()));
    }
    frame->finishWritingRows();
    BOOST_TEST_REQUIRE(frame->numberRows() == rows.size());

    // Create classification analysis runner object
    const auto spec{test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
        "classification", "x5", rows.size(), columnNames.size(), 13000000, 0, 0,
        categoricalColumns)};
    rapidjson::Document jsonParameters;
    jsonParameters.Parse("{\"dependent_variable\": \"x5\"}");
    const auto parameters{
        api::CDataFrameTrainBoostedTreeClassifierRunner::parameterReader().read(jsonParameters)};
    api::CDataFrameTrainBoostedTreeClassifierRunner runner(*spec, parameters);

    // Write results to the output stream
    std::stringstream output;
    {
        core::CJsonOutputStreamWrapper outputStreamWrapper(output);
        core::CRapidJsonConcurrentLineWriter writer(outputStreamWrapper);

        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            const auto columnHoldingDependentVariable{
                std::find(columnNames.begin(), columnNames.end(), "x5") -
                columnNames.begin()};
            const auto columnHoldingPrediction{
                std::find(columnNames.begin(), columnNames.end(), "x5_prediction") -
                columnNames.begin()};
            for (auto row = beginRows; row != endRows; ++row) {
                runner.writeOneRow(*frame, columnHoldingDependentVariable,
                                   columnHoldingPrediction, *row, writer);
            }
        });
    }
    // Verify results
    const TStrVec expectedPredictions{"cat", "cat", "cat", "dog", "dog"};
    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(output.str().c_str());
    BOOST_TEST_REQUIRE(arrayDoc.IsArray());
    BOOST_TEST_REQUIRE(arrayDoc.Size() == rows.size());
    for (std::size_t i = 0; i < arrayDoc.Size(); ++i) {
        BOOST_TEST_CONTEXT("Result for row " << i) {
            const rapidjson::Value& object = arrayDoc[rapidjson::SizeType(i)];
            BOOST_TEST_REQUIRE(object.IsObject());
            BOOST_TEST_REQUIRE(object.HasMember("x5_prediction"));
            BOOST_TEST_REQUIRE(object["x5_prediction"].GetString() ==
                               expectedPredictions[i]);
            BOOST_TEST_REQUIRE(object.HasMember("prediction_probability"));
            BOOST_TEST_REQUIRE(object["prediction_probability"].GetDouble() > 0.5);
            BOOST_TEST_REQUIRE(object.HasMember("is_training"));
            BOOST_TEST_REQUIRE(object["is_training"].GetBool());
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
