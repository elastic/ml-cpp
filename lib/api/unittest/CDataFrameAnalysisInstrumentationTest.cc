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

#include <core/CTimeUtils.h>
#include <core/CVectorRange.h>

#include <api/CDataFrameAnalysisInstrumentation.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CDataFrameAnalyzerTrainingFactory.h>
#include <test/CProgramCounterClearingFixture.h>

#include <boost/test/unit_test.hpp>

#include <valijson/adapters/boost_json_adapter.hpp>
#include <valijson/schema.hpp>
#include <valijson/schema_parser.hpp>
#include <valijson/utils/boost_json_utils.hpp>
#include <valijson/validator.hpp>

#include <chrono>
#include <fstream>
#include <memory>
#include <string>

BOOST_AUTO_TEST_SUITE(CDataFrameAnalysisInstrumentationTest)

using namespace ml;

namespace {
using TStrVec = std::vector<std::string>;
using TRowItr = core::CDataFrame::TRowItr;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TLossFunctionType = maths::analytics::boosted_tree::ELossType;
using TDataFrameUPtrTemporaryDirectoryPtrPr =
    test::CDataFrameAnalysisSpecificationFactory::TDataFrameUPtrTemporaryDirectoryPtrPr;

void addOutlierTestData(TStrVec fieldNames,
                        TStrVec fieldValues,
                        api::CDataFrameAnalyzer& analyzer,
                        TDoubleVec& expectedScores,
                        TDoubleVecVec& expectedFeatureInfluences,
                        std::size_t numberInliers = 100,
                        std::size_t numberOutliers = 10,
                        maths::analytics::COutliers::EMethod method = maths::analytics::COutliers::E_Ensemble,
                        std::size_t numberNeighbours = 0,
                        bool computeFeatureInfluence = false) {

    test::CRandomNumbers rng;

    TDoubleVec mean{1.0, 10.0, 4.0, 8.0, 3.0};
    TDoubleVecVec covariance{{1.0, 0.1, -0.1, 0.3, 0.2},
                             {0.1, 1.3, -0.3, 0.1, 0.1},
                             {-0.1, -0.3, 2.1, 0.1, 0.2},
                             {0.3, 0.1, 0.1, 0.8, 0.2},
                             {0.2, 0.1, 0.2, 0.2, 2.2}};

    TDoubleVecVec inliers;
    rng.generateMultivariateNormalSamples(mean, covariance, numberInliers, inliers);

    TDoubleVec outliers;
    rng.generateUniformSamples(0.0, 10.0, numberOutliers * 5, outliers);

    auto frame = core::makeMainStorageDataFrame(5).first;

    for (std::size_t i = 0; i < inliers.size(); ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                inliers[i][j], core::CIEEE754::E_DoublePrecision);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(core::make_const_range(fieldValues, 0, 5));
    }
    for (std::size_t i = 0; i < outliers.size(); i += 5) {
        for (std::size_t j = 0; j < 5; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                outliers[i + j], core::CIEEE754::E_DoublePrecision);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(core::make_const_range(fieldValues, 0, 5));
    }

    frame->finishWritingRows();
    maths::analytics::CDataFrameOutliersInstrumentationStub instrumentation;
    maths::analytics::COutliers::compute(
        {1, 1, true, method, numberNeighbours, computeFeatureInfluence, 0.05},
        *frame, instrumentation);

    expectedScores.resize(numberInliers + numberOutliers);
    expectedFeatureInfluences.resize(numberInliers + numberOutliers, TDoubleVec(5));

    frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            expectedScores[row->index()] = (*row)[5];
            if (computeFeatureInfluence) {
                for (std::size_t i = 6; i < 11; ++i) {
                    expectedFeatureInfluences[row->index()][i - 6] = (*row)[i];
                }
            }
        }
    });
}
}

BOOST_FIXTURE_TEST_CASE(testMemoryState, ml::test::CProgramCounterClearingFixture) {
    std::string jobId{"testJob"};
    std::size_t memoryLimit{core::constants::BYTES_IN_GIGABYTES};
    std::int64_t memoryUsage{500000};
    std::int64_t timeBefore{std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count()};
    std::stringstream outputStream;
    {
        core::CJsonOutputStreamWrapper streamWrapper{outputStream};
        api::CDataFrameTrainBoostedTreeInstrumentation instrumentation{jobId, memoryLimit};
        api::CDataFrameTrainBoostedTreeInstrumentation::CScopeSetOutputStream setStream{
            instrumentation, streamWrapper};
        instrumentation.updateMemoryUsage(memoryUsage);
        instrumentation.flush();
        outputStream.flush();
    }
    std::int64_t timeAfter{std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count()};

    json::value results;
    json::error_code ec;
    json::parser p;
    std::size_t written = p.write(outputStream.str(), ec);
    BOOST_TEST_REQUIRE(outputStream.str().size() == written);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    results = p.release();
    BOOST_TEST_REQUIRE(results.is_array() == true);

    bool hasMemoryUsage{false};
    for (const auto& result_ : results.as_array()) {
        BOOST_TEST_REQUIRE(result_.is_object() == true);
        json::object result = result_.as_object();
        if (result.contains("analytics_memory_usage")) {
            BOOST_TEST_REQUIRE(result["analytics_memory_usage"].is_object() == true);
            BOOST_TEST_REQUIRE(
                result["analytics_memory_usage"].as_object()["job_id"].as_string() == jobId);
            BOOST_TEST_REQUIRE(result["analytics_memory_usage"]
                                   .as_object()["peak_usage_bytes"]
                                   .to_number<std::int64_t>() == memoryUsage);
            BOOST_TEST_REQUIRE(
                result["analytics_memory_usage"].as_object()["timestamp"].to_number<std::int64_t>() >=
                timeBefore);
            BOOST_TEST_REQUIRE(
                result["analytics_memory_usage"].as_object()["timestamp"].to_number<std::int64_t>() <=
                timeAfter);
            hasMemoryUsage = true;
        }
    }
    BOOST_TEST_REQUIRE(hasMemoryUsage);
}

BOOST_FIXTURE_TEST_CASE(testTrainingRegression, ml::test::CProgramCounterClearingFixture) {
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}.predictionSpec(
        test::CDataFrameAnalysisSpecificationFactory::regression(), "target", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        TLossFunctionType::E_MseRegression, fieldNames, fieldValues, analyzer,
        expectedPredictions);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    json::error_code ec;
    json::value results = json::parse(output.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);

    std::ifstream regressionSchemaFileStream("testfiles/instrumentation/regression_stats.schema.json");
    BOOST_REQUIRE_MESSAGE(regressionSchemaFileStream.is_open(), "Cannot open test file!");
    std::string regressionSchemaJson((std::istreambuf_iterator<char>(regressionSchemaFileStream)),
                                     std::istreambuf_iterator<char>());

    json::value regressionSchemaDocument = json::parse(regressionSchemaJson, ec);
    BOOST_REQUIRE_MESSAGE(ec.failed() == false, "Cannot parse JSON schema!");

    // Parse Boost JSON schema content using valijson
    valijson::Schema schema;
    valijson::SchemaParser parser;
    valijson::adapters::BoostJsonAdapter schemaAdapter(regressionSchemaDocument);
    parser.populateSchema(schemaAdapter, schema);

    bool hasRegressionStats{false};
    bool initialMemoryReport{false};
    for (const auto& result : results.as_array()) {
        if (result.as_object().contains("analytics_memory_usage") == true &&
            result.as_object().contains("regression_stats") == false) {
            initialMemoryReport = true;
        }
        if (result.as_object().contains("regression_stats")) {
            hasRegressionStats = true;
            BOOST_TEST_REQUIRE(result.as_object().at("regression_stats").is_object() == true);
            // validate "regression_stats" against schema
            valijson::Validator validator;
            valijson::ValidationResults validatorResults;
            valijson::adapters::BoostJsonAdapter targetAdapter(
                result.as_object().at("regression_stats"));
            BOOST_REQUIRE_MESSAGE(validator.validate(schema, targetAdapter, &validatorResults),
                                  "Validation failed.");

            valijson::ValidationResults::Error error;
            unsigned int errorNum = 1;
            while (validatorResults.popError(error)) {
                LOG_ERROR(<< "Error #" << errorNum);
                LOG_ERROR(<< "  ");
                for (const std::string& contextElement : error.context) {
                    LOG_ERROR(<< contextElement << " ");
                }
                LOG_ERROR(<< "    - " << error.description);
                ++errorNum;
            }
        }
    }
    BOOST_TEST_REQUIRE(hasRegressionStats);
    BOOST_TEST_REQUIRE(initialMemoryReport);

    std::ifstream memorySchemaFileStream("testfiles/instrumentation/memory_usage.schema.json");
    BOOST_REQUIRE_MESSAGE(memorySchemaFileStream.is_open(), "Cannot open test file!");
    std::string memorySchemaJson((std::istreambuf_iterator<char>(memorySchemaFileStream)),
                                 std::istreambuf_iterator<char>());

    json::value memorySchemaDocument = json::parse(memorySchemaJson, ec);
    BOOST_REQUIRE_MESSAGE(ec.failed() == false, "Cannot parse JSON schema!");

    // Parse Boost JSON schema content using valijson
    valijson::Schema memorySchema;
    valijson::SchemaParser memoryParser;
    valijson::adapters::BoostJsonAdapter memorySchemaAdapter(memorySchemaDocument);
    parser.populateSchema(memorySchemaAdapter, memorySchema);

    bool hasMemoryUsage{false};
    for (const auto& result : results.as_array()) {
        if (result.as_object().contains("analytics_memory_usage")) {
            hasMemoryUsage = true;
            BOOST_TEST_REQUIRE(
                result.as_object().at("analytics_memory_usage").is_object() == true);
            // validate "analytics_memory_usage" against schema
            valijson::Validator validator;
            valijson::ValidationResults validationResults;
            valijson::adapters::BoostJsonAdapter targetAdapter(
                result.as_object().at("analytics_memory_usage"));
            BOOST_REQUIRE_MESSAGE(validator.validate(memorySchema, targetAdapter, &validationResults),
                                  "Validation failed.");
            valijson::ValidationResults::Error error;
            unsigned int errorNum = 1;
            while (validationResults.popError(error)) {
                LOG_ERROR(<< "Error #" << errorNum);
                LOG_ERROR(<< "  ");
                for (const std::string& contextElement : error.context) {
                    LOG_ERROR(<< contextElement << " ");
                }
                LOG_ERROR(<< "    - " << error.description);
                ++errorNum;
            }
        }
    }
    BOOST_TEST_REQUIRE(hasMemoryUsage);
}

BOOST_FIXTURE_TEST_CASE(testTrainingClassification, ml::test::CProgramCounterClearingFixture) {
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
                    .rows(100)
                    .memoryLimit(6000000)
                    .columns(5)
                    .predictionCategoricalFieldNames({"target"})
                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
                                    "target", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        TLossFunctionType::E_BinaryClassification, fieldNames, fieldValues,
        analyzer, expectedPredictions);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    json::error_code ec;
    json::value results = json::parse(output.str(), ec);

    BOOST_TEST_REQUIRE(ec.failed() == false);

    std::ifstream schemaFileStream("testfiles/instrumentation/classification_stats.schema.json");
    BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
    std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                           std::istreambuf_iterator<char>());

    json::value classificationSchemaDocument = json::parse(schemaJson, ec);
    BOOST_REQUIRE_MESSAGE(ec.failed() == false, "Cannot parse JSON schema!");

    // Parse Boost JSON schema content using valijson
    valijson::Schema schema;
    valijson::SchemaParser parser;
    valijson::adapters::BoostJsonAdapter schemaAdapter(classificationSchemaDocument);
    parser.populateSchema(schemaAdapter, schema);

    bool hasClassificationStats{false};
    bool initialMemoryReport{false};
    for (const auto& result : results.as_array()) {
        if (result.as_object().contains("analytics_memory_usage") == true &&
            result.as_object().contains("classification_stats") == false) {
            initialMemoryReport = true;
        }
        if (result.as_object().contains("classification_stats")) {
            hasClassificationStats = true;
            BOOST_TEST_REQUIRE(
                result.as_object().at("classification_stats").is_object() == true);

            // validate "classification_stats" against schema
            valijson::Validator validator;
            valijson::ValidationResults validationResults;
            valijson::adapters::BoostJsonAdapter targetAdapter(
                result.as_object().at("classification_stats"));
            BOOST_REQUIRE_MESSAGE(validator.validate(schema, targetAdapter, &validationResults),
                                  "Validation failed.");
            valijson::ValidationResults::Error error;
            unsigned int errorNum = 1;
            while (validationResults.popError(error)) {
                LOG_ERROR(<< "Error #" << errorNum);
                LOG_ERROR(<< "  ");
                for (const std::string& contextElement : error.context) {
                    LOG_ERROR(<< contextElement << " ");
                }
                LOG_ERROR(<< "    - " << error.description);
                ++errorNum;
            }
        }
    }
    BOOST_TEST_REQUIRE(hasClassificationStats);
    BOOST_TEST_REQUIRE(initialMemoryReport);
}

BOOST_FIXTURE_TEST_CASE(testOutlierDetection, ml::test::CProgramCounterClearingFixture) {
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}.outlierSpec(&frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};

    TDoubleVec expectedScores;
    TDoubleVecVec expectedFeatureInfluences;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addOutlierTestData(fieldNames, fieldValues, analyzer, expectedScores,
                       expectedFeatureInfluences);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    json::error_code ec;
    json::value results = json::parse(output.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);

    std::ifstream schemaFileStream("testfiles/instrumentation/outlier_detection_stats.schema.json");
    BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
    std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                           std::istreambuf_iterator<char>());

    json::value outlierDetectionSchemaDocument = json::parse(schemaJson, ec);
    BOOST_REQUIRE_MESSAGE(ec.failed() == false, "Cannot parse JSON schema!");

    // Parse Boost JSON schema content using valijson
    valijson::Schema schema;
    valijson::SchemaParser parser;
    valijson::adapters::BoostJsonAdapter schemaAdapter(outlierDetectionSchemaDocument);
    parser.populateSchema(schemaAdapter, schema);

    bool hasOutlierDetectionStats{false};
    bool initialMemoryReport{false};
    for (const auto& result : results.as_array()) {
        if (result.as_object().contains("analytics_memory_usage") == true &&
            result.as_object().contains("outlier_detection_stats") == false) {
            initialMemoryReport = true;
        }
        if (result.as_object().contains("outlier_detection_stats")) {
            hasOutlierDetectionStats = true;
            BOOST_TEST_REQUIRE(
                result.as_object().at("outlier_detection_stats").is_object() == true);

            // validate "outlier_detection_stats" against schema
            valijson::Validator validator;
            valijson::ValidationResults validationResults;
            valijson::adapters::BoostJsonAdapter targetAdapter(
                result.as_object().at("outlier_detection_stats"));
            BOOST_REQUIRE_MESSAGE(validator.validate(schema, targetAdapter, &validationResults),
                                  "Validation failed.");
            valijson::ValidationResults::Error error;
            unsigned int errorNum = 1;
            while (validationResults.popError(error)) {
                LOG_ERROR(<< "Error #" << errorNum);
                LOG_ERROR(<< "  ");
                for (const std::string& contextElement : error.context) {
                    LOG_ERROR(<< contextElement << " ");
                }
                LOG_ERROR(<< "    - " << error.description);
                ++errorNum;
            }
        }
    }
    BOOST_TEST_REQUIRE(hasOutlierDetectionStats);
    BOOST_TEST_REQUIRE(initialMemoryReport);
}

BOOST_AUTO_TEST_SUITE_END()
