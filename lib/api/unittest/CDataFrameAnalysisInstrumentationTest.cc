/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CTimeUtils.h>

#include <api/CDataFrameAnalysisInstrumentation.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CDataFrameAnalyzerTrainingFactory.h>

#include <rapidjson/schema.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <memory>
#include <string>

BOOST_AUTO_TEST_SUITE(CDataFrameAnalysisInstrumentationTest)

using namespace ml;

namespace {
using TStrVec = std::vector<std::string>;
using TDoubleVec = std::vector<double>;
}

BOOST_AUTO_TEST_CASE(testMemoryState) {
    std::string jobId{"testJob"};
    std::int64_t memoryUsage{1000};
    std::int64_t timeBefore{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    std::stringstream outpustStream;
    {
        core::CJsonOutputStreamWrapper streamWrapper(outpustStream);
        core::CRapidJsonConcurrentLineWriter writer(streamWrapper);
        api::CDataFrameTrainBoostedTreeInstrumentation instrumentation(jobId);
        instrumentation.updateMemoryUsage(memoryUsage);
        instrumentation.writer(&writer);
        instrumentation.nextStep(0);
        outpustStream.flush();
    }
    std::int64_t timeAfter{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(outpustStream.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    BOOST_TEST_REQUIRE(results.IsArray() == true);

    const auto& result{results[0]};
    BOOST_TEST_REQUIRE(result["job_id"].GetString() == jobId);
    BOOST_TEST_REQUIRE(result["type"].GetString() == "analytics_memory_usage");
    BOOST_TEST_REQUIRE(result["peak_usage_bytes"].GetInt64() == memoryUsage);
    BOOST_TEST_REQUIRE(result["timestamp"].GetInt64() >= timeBefore);
    BOOST_TEST_REQUIRE(result["timestamp"].GetInt64() <= timeAfter);
}

BOOST_AUTO_TEST_CASE(testAnalysisTrainState) {
    std::string jobId{"testJob"};
    std::int64_t timeBefore{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    std::stringstream outputStream;
    {
        core::CJsonOutputStreamWrapper streamWrapper(outputStream);
        core::CRapidJsonConcurrentLineWriter writer(streamWrapper);
        api::CDataFrameTrainBoostedTreeInstrumentation instrumentation(jobId);
        instrumentation.writer(&writer);
        instrumentation.nextStep(0);
        outputStream.flush();
    }
    std::int64_t timeAfter{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    LOG_DEBUG(<< outputStream.str());

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(outputStream.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    BOOST_TEST_REQUIRE(results.IsArray() == true);

    const auto& result{results[0]};
    BOOST_TEST_REQUIRE(result["job_id"].GetString() == jobId);
    BOOST_TEST_REQUIRE(result["type"].GetString() == "analytics_memory_usage");
    BOOST_TEST_REQUIRE(result["timestamp"].GetInt64() >= timeBefore);
    BOOST_TEST_REQUIRE(result["timestamp"].GetInt64() <= timeAfter);
}

BOOST_AUTO_TEST_CASE(testTrainingRegression) {
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::regression(), "target"),
        outputWriterFactory};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        test::CDataFrameAnalyzerTrainingFactory::E_Regression, fieldNames,
        fieldValues, analyzer, expectedPredictions);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    std::ifstream schemaFileStream("testfiles/instrumentation/supervised_learning_stats.schema.json");
    BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
    std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                           std::istreambuf_iterator<char>());
    rapidjson::Document schemaDocument;
    BOOST_REQUIRE_MESSAGE(schemaDocument.Parse(schemaJson).HasParseError() == false,
                          "Cannot parse JSON schema!");
    rapidjson::SchemaDocument schema(schemaDocument);
    rapidjson::SchemaValidator validator(schema);

    for (const auto& result : results.GetArray()) {
        if (result.HasMember("analysis_stats")) {
            BOOST_TEST_REQUIRE(result["analysis_stats"].HasMember("classification_stats"));
            if (result["analysis_stats"]["classification_stats"].Accept(validator) == false) {
                rapidjson::StringBuffer sb;
                validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid schema: " << sb.GetString());
                LOG_ERROR(<< "Invalid keyword: " << validator.GetInvalidSchemaKeyword());
                sb.Clear();
                validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid document: " << sb.GetString());
                BOOST_FAIL("Schema validation failed");
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testTrainingClassification) {
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::classification(),
            "target", 100, 5, 6000000, 0, 0, {"target"}),
        outputWriterFactory};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        test::CDataFrameAnalyzerTrainingFactory::E_BinaryClassification,
        fieldNames, fieldValues, analyzer, expectedPredictions);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    LOG_DEBUG(<< output.str());

    std::ifstream schemaFileStream("testfiles/instrumentation/supervised_learning_stats.schema.json");
    BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
    std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                           std::istreambuf_iterator<char>());
    rapidjson::Document schemaDocument;
    BOOST_REQUIRE_MESSAGE(schemaDocument.Parse(schemaJson).HasParseError() == false,
                          "Cannot parse JSON schema!");
    rapidjson::SchemaDocument schema(schemaDocument);
    rapidjson::SchemaValidator validator(schema);

    for (const auto& result : results.GetArray()) {
        if (result.HasMember("analysis_stats")) {
            BOOST_TEST_REQUIRE(result["analysis_stats"].HasMember("regression_stats"));
            if (result["analysis_stats"]["regression_stats"].Accept(validator) == false) {
                rapidjson::StringBuffer sb;
                validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid schema: " << sb.GetString());
                LOG_ERROR(<< "Invalid keyword: " << validator.GetInvalidSchemaKeyword());
                sb.Clear();
                validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid document: " << sb.GetString());
                BOOST_FAIL("Schema validation failed");
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
