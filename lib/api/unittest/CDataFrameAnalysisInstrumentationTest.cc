/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
// #include <core/CTimeUtils.h>

// #include <api/CDataFrameAnalysisInstrumentation.h>

// #include <test/BoostTestCloseAbsolute.h>
// #include <test/CDataFrameAnalysisSpecificationFactory.h>
// #include <test/CDataFrameAnalyzerTrainingFactory.h>

// #include <rapidjson/schema.h>

// #include <boost/test/unit_test.hpp>

// #include <fstream>
// #include <memory>
// #include <string>

// BOOST_AUTO_TEST_SUITE(CDataFrameAnalysisInstrumentationTest)

// using namespace ml;

// namespace {
// using TStrVec = std::vector<std::string>;
// using TDoubleVec = std::vector<double>;
// }

// BOOST_AUTO_TEST_CASE(testMemoryState) {
//     std::string jobId{"testJob"};
//     std::int64_t memoryUsage{1000};
//     std::int64_t timeBefore{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
//     std::stringstream outputStream;
//     {
//         core::CJsonOutputStreamWrapper streamWrapper(outputStream);
//         api::CDataFrameTrainBoostedTreeInstrumentation instrumentation(jobId);
//         api::CDataFrameTrainBoostedTreeInstrumentation::CScopeSetOutputStream setStream{
//             instrumentation, streamWrapper};
//         instrumentation.updateMemoryUsage(memoryUsage);
//         instrumentation.nextStep();
//         outputStream.flush();
//     }
//     std::int64_t timeAfter{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};

//     rapidjson::Document results;
//     rapidjson::ParseResult ok(results.Parse(outputStream.str()));
//     BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
//     BOOST_TEST_REQUIRE(results.IsArray() == true);

//     bool hasMemoryUsage{false};
//     for (const auto& result : results.GetArray()) {
//         if (result.HasMember("analytics_memory_usage")) {
//             BOOST_TEST_REQUIRE(result["analytics_memory_usage"].IsObject() == true);
//             BOOST_TEST_REQUIRE(result["analytics_memory_usage"]["job_id"].GetString() == jobId);
//             BOOST_TEST_REQUIRE(
//                 result["analytics_memory_usage"]["peak_usage_bytes"].GetInt64() == memoryUsage);
//             BOOST_TEST_REQUIRE(result["analytics_memory_usage"]["timestamp"].GetInt64() >=
//                                timeBefore);
//             BOOST_TEST_REQUIRE(result["analytics_memory_usage"]["timestamp"].GetInt64() <= timeAfter);
//             hasMemoryUsage = true;
//         }
//     }
//     BOOST_TEST_REQUIRE(hasMemoryUsage);
// }

// BOOST_AUTO_TEST_CASE(testTrainingRegression) {
//     std::stringstream output;
//     auto outputWriterFactory = [&output]() {
//         return std::make_unique<core::CJsonOutputStreamWrapper>(output);
//     };

//     TDoubleVec expectedPredictions;

//     TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
//     TStrVec fieldValues{"", "", "", "", "", "0", ""};
//     test::CDataFrameAnalysisSpecificationFactory specFactory;
//     api::CDataFrameAnalyzer analyzer{
//         specFactory.predictionSpec(
//             test::CDataFrameAnalysisSpecificationFactory::regression(), "target"),
//         outputWriterFactory};
//     test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
//         test::CDataFrameAnalyzerTrainingFactory::E_Regression, fieldNames,
//         fieldValues, analyzer, expectedPredictions);

//     analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

//     rapidjson::Document results;
//     rapidjson::ParseResult ok(results.Parse(output.str()));
//     BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

//     std::ifstream regressionSchemaFileStream(
//         "testfiles/instrumentation/supervised_learning_stats.schema.json");
//     BOOST_REQUIRE_MESSAGE(regressionSchemaFileStream.is_open(), "Cannot open test file!");
//     std::string regressionSchemaJson((std::istreambuf_iterator<char>(regressionSchemaFileStream)),
//                                      std::istreambuf_iterator<char>());
//     rapidjson::Document regressionSchemaDocument;
//     BOOST_REQUIRE_MESSAGE(
//         regressionSchemaDocument.Parse(regressionSchemaJson).HasParseError() == false,
//         "Cannot parse JSON schema!");
//     rapidjson::SchemaDocument regressionSchema(regressionSchemaDocument);
//     rapidjson::SchemaValidator regressionValidator(regressionSchema);

//     for (const auto& result : results.GetArray()) {
//         if (result.HasMember("analysis_stats")) {
//             BOOST_TEST_REQUIRE(result["analysis_stats"].HasMember("regression_stats"));
//             if (result["analysis_stats"]["regression_stats"].Accept(regressionValidator) == false) {
//                 rapidjson::StringBuffer sb;
//                 regressionValidator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
//                 LOG_ERROR(<< "Invalid schema: " << sb.GetString());
//                 LOG_ERROR(<< "Invalid keyword: "
//                           << regressionValidator.GetInvalidSchemaKeyword());
//                 sb.Clear();
//                 regressionValidator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
//                 LOG_ERROR(<< "Invalid document: " << sb.GetString());
//                 BOOST_FAIL("Schema validation failed");
//             }
//         }
//     }

//     std::ifstream memorySchemaFileStream("testfiles/instrumentation/memory_usage.schema.json");
//     BOOST_REQUIRE_MESSAGE(memorySchemaFileStream.is_open(), "Cannot open test file!");
//     std::string memorySchemaJson((std::istreambuf_iterator<char>(memorySchemaFileStream)),
//                                  std::istreambuf_iterator<char>());
//     rapidjson::Document memorySchemaDocument;
//     BOOST_REQUIRE_MESSAGE(memorySchemaDocument.Parse(memorySchemaJson).HasParseError() == false,
//                           "Cannot parse JSON schema!");
//     rapidjson::SchemaDocument memorySchema(memorySchemaDocument);
//     rapidjson::SchemaValidator memoryValidator(memorySchema);

//     for (const auto& result : results.GetArray()) {
//         if (result.HasMember("analytics_memory_usage")) {
//             BOOST_TEST_REQUIRE(result["analytics_memory_usage"].IsObject() == true);
//             if (result["analytics_memory_usage"].Accept(memoryValidator) == false) {
//                 rapidjson::StringBuffer sb;
//                 memoryValidator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
//                 LOG_ERROR(<< "Invalid schema: " << sb.GetString());
//                 LOG_ERROR(<< "Invalid keyword: "
//                           << memoryValidator.GetInvalidSchemaKeyword());
//                 sb.Clear();
//                 memoryValidator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
//                 LOG_ERROR(<< "Invalid document: " << sb.GetString());
//                 BOOST_FAIL("Schema validation failed");
//             }
//         }
//     }
// }

// BOOST_AUTO_TEST_CASE(testTrainingClassification) {
//     std::stringstream output;
//     auto outputWriterFactory = [&output]() {
//         return std::make_unique<core::CJsonOutputStreamWrapper>(output);
//     };

//     TDoubleVec expectedPredictions;

//     TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
//     TStrVec fieldValues{"", "", "", "", "", "0", ""};
//     test::CDataFrameAnalysisSpecificationFactory specFactory;
//     api::CDataFrameAnalyzer analyzer{
//         specFactory.rows(100)
//             .memoryLimit(6000000)
//             .columns(5)
//             .predictionCategoricalFieldNames({"target"})
//             .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(), "target"),
//         outputWriterFactory};
//     test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
//         test::CDataFrameAnalyzerTrainingFactory::E_BinaryClassification,
//         fieldNames, fieldValues, analyzer, expectedPredictions);

//     analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

//     rapidjson::Document results;
//     rapidjson::ParseResult ok(results.Parse(output.str()));
//     BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

//     std::ifstream schemaFileStream("testfiles/instrumentation/supervised_learning_stats.schema.json");
//     BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
//     std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
//                            std::istreambuf_iterator<char>());
//     rapidjson::Document schemaDocument;
//     BOOST_REQUIRE_MESSAGE(schemaDocument.Parse(schemaJson).HasParseError() == false,
//                           "Cannot parse JSON schema!");
//     rapidjson::SchemaDocument schema(schemaDocument);
//     rapidjson::SchemaValidator validator(schema);

//     for (const auto& result : results.GetArray()) {
//         if (result.HasMember("analysis_stats")) {
//             BOOST_TEST_REQUIRE(result["analysis_stats"].HasMember("classification_stats"));
//             if (result["analysis_stats"]["classification_stats"].Accept(validator) == false) {
//                 rapidjson::StringBuffer sb;
//                 validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
//                 LOG_ERROR(<< "Invalid schema: " << sb.GetString());
//                 LOG_ERROR(<< "Invalid keyword: " << validator.GetInvalidSchemaKeyword());
//                 sb.Clear();
//                 validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
//                 LOG_ERROR(<< "Invalid document: " << sb.GetString());
//                 BOOST_FAIL("Schema validation failed");
//             }
//         }
//     }
// }

// BOOST_AUTO_TEST_SUITE_END()
