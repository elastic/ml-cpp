/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>
#include <core/CRegex.h>
#include <core/CSmallVector.h>

#include <maths/CTools.h>

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameTrainBoostedTreeClassifierRunner.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDataFrameTrainBoostedTreeClassifierRunnerTest)

using namespace ml;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;

BOOST_AUTO_TEST_CASE(testPredictionFieldNameClash) {
    TStrVec errors;
    auto errorHandler = [&errors](std::string error) { errors.push_back(error); };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    test::CDataFrameAnalysisSpecificationFactory specFactory;
    auto spec = specFactory.rows(5)
                    .columns(6)
                    .memoryLimit(13000000)
                    .predictionCategoricalFieldNames({"dep_var"})
                    .predictionFieldName("is_training")
                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
                                    "dep_var");

    BOOST_TEST_REQUIRE(errors.size() == 1);

    core::CRegex regex;
    regex.init("Input error: prediction_field_name must not be equal to.*");
    BOOST_TEST_REQUIRE(regex.matches(errors[0]));
}

namespace {
template<typename T>
void testWriteOneRow(const std::string& dependentVariableField,
                     const std::string& predictionFieldType,
                     T (rapidjson::Value::*extract)() const,
                     const std::vector<T>& expectedPredictions) {
    // Prepare input data frame
    const std::string predictionField{dependentVariableField + "_prediction"};
    const TStrVec columnNames{"x1", "x2", "x3", "x4", "x5", predictionField};
    const TStrVec categoricalColumns{"x1", "x2", "x3", "x4", "x5"};
    const TStrVecVec rows{{"a", "b", "1.0", "1.0", "cat", "-1.0"},
                          {"a", "b", "1.0", "1.0", "cat", "-0.5"},
                          {"a", "b", "5.0", "0.0", "dog", "-0.1"},
                          {"c", "d", "5.0", "0.0", "dog", "1.0"},
                          {"e", "f", "5.0", "0.0", "dog", "1.5"}};
    auto frame = core::makeMainStorageDataFrame(columnNames.size()).first;
    frame->columnNames(columnNames);
    frame->categoricalColumns(categoricalColumns);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        frame->parseAndWriteRow(
            core::CVectorRange<const TStrVec>(rows[i], 0, rows[i].size()));
    }
    frame->finishWritingRows();
    BOOST_TEST_REQUIRE(frame->numberRows() == rows.size());

    // Create classification analysis runner object
    test::CDataFrameAnalysisSpecificationFactory specFactory;
    auto spec = specFactory.rows(rows.size())
                    .columns(columnNames.size())
                    .memoryLimit(13000000)
                    .predictionCategoricalFieldNames(categoricalColumns)
                    .predictionFieldType(predictionFieldType)
                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
                                    dependentVariableField);

    rapidjson::Document jsonParameters;
    jsonParameters.Parse(specFactory.predictionParams(
        test::CDataFrameAnalysisSpecificationFactory::classification(), dependentVariableField));
    auto parameters =
        api::CDataFrameTrainBoostedTreeClassifierRunner::parameterReader().read(jsonParameters);
    api::CDataFrameTrainBoostedTreeClassifierRunner runner{*spec, parameters};

    // Write results to the output stream
    std::stringstream output;
    {
        core::CJsonOutputStreamWrapper outputStreamWrapper(output);
        core::CRapidJsonConcurrentLineWriter writer(outputStreamWrapper);

        frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
            auto columnHoldingDependentVariable =
                std::find(columnNames.begin(), columnNames.end(), dependentVariableField) -
                columnNames.begin();
            auto columnHoldingPrediction =
                std::find(columnNames.begin(), columnNames.end(), predictionField) -
                columnNames.begin();
            auto readProbability = [&](const TRowRef& row) {
                TDouble2Vec result(2);
                double p{maths::CTools::logisticFunction(row[columnHoldingPrediction])};
                result[0] = 1 - p;
                result[1] = p;
                return result;
            };
            for (auto row = beginRows; row != endRows; ++row) {
                runner.writeOneRow(*frame, columnHoldingDependentVariable,
                                   readProbability, readProbability, *row, writer);
            }
        });
    }
    // Verify results
    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(output.str().c_str());
    BOOST_TEST_REQUIRE(arrayDoc.IsArray());
    BOOST_TEST_REQUIRE(arrayDoc.Size() == rows.size());
    BOOST_TEST_REQUIRE(arrayDoc.Size() == expectedPredictions.size());
    for (std::size_t i = 0; i < arrayDoc.Size(); ++i) {
        BOOST_TEST_CONTEXT("Result for row " << i) {
            const rapidjson::Value& object = arrayDoc[rapidjson::SizeType(i)];
            BOOST_TEST_REQUIRE(object.IsObject());
            BOOST_TEST_REQUIRE(object.HasMember(predictionField));
            BOOST_TEST_REQUIRE((object[predictionField].*extract)() ==
                               expectedPredictions[i]);
            BOOST_TEST_REQUIRE(object.HasMember("prediction_probability"));
            BOOST_TEST_REQUIRE(object["prediction_probability"].GetDouble() > 0.5);
            BOOST_TEST_REQUIRE(object.HasMember("is_training"));
            BOOST_TEST_REQUIRE(object["is_training"].GetBool());
        }
    }
}
}

BOOST_AUTO_TEST_CASE(testWriteOneRowPredictionFieldTypeIsInt) {
    testWriteOneRow("x3", "int", &rapidjson::Value::GetInt, {1, 1, 1, 5, 5});
}

BOOST_AUTO_TEST_CASE(testWriteOneRowPredictionFieldTypeIsBool) {
    testWriteOneRow("x4", "bool", &rapidjson::Value::GetBool,
                    {true, true, true, false, false});
}

BOOST_AUTO_TEST_CASE(testWriteOneRowPredictionFieldTypeIsString) {
    testWriteOneRow("x5", "string", &rapidjson::Value::GetString,
                    {"cat", "cat", "cat", "dog", "dog"});
}

BOOST_AUTO_TEST_CASE(testWriteOneRowPredictionFieldTypeIsMissing) {
    testWriteOneRow("x5", "", &rapidjson::Value::GetString,
                    {"cat", "cat", "cat", "dog", "dog"});
}

BOOST_AUTO_TEST_SUITE_END()
