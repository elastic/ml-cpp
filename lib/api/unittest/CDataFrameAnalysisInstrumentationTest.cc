/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <boost/test/tools/interface.hpp>
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
using TRowItr = core::CDataFrame::TRowItr;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;

void addOutlierTestData(TStrVec fieldNames,
                        TStrVec fieldValues,
                        api::CDataFrameAnalyzer& analyzer,
                        TDoubleVec& expectedScores,
                        TDoubleVecVec& expectedFeatureInfluences,
                        std::size_t numberInliers = 100,
                        std::size_t numberOutliers = 10,
                        maths::COutliers::EMethod method = maths::COutliers::E_Ensemble,
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
        frame->parseAndWriteRow(core::CVectorRange<const TStrVec>(fieldValues, 0, 5));
    }
    for (std::size_t i = 0; i < outliers.size(); i += 5) {
        for (std::size_t j = 0; j < 5; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                outliers[i + j], core::CIEEE754::E_DoublePrecision);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(core::CVectorRange<const TStrVec>(fieldValues, 0, 5));
    }

    frame->finishWritingRows();
    maths::CDataFrameOutliersInstrumentationStub instrumentation;
    maths::COutliers::compute(
        {1, 1, true, method, numberNeighbours, computeFeatureInfluence, 0.05},
        *frame, instrumentation);

    expectedScores.resize(numberInliers + numberOutliers);
    expectedFeatureInfluences.resize(numberInliers + numberOutliers, TDoubleVec(5));

    frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
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

BOOST_AUTO_TEST_CASE(testMemoryState) {
    std::string jobId{"testJob"};
    std::int64_t memoryUsage{1000};
    std::int64_t timeBefore{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};
    std::stringstream outputStream;
    {
        core::CJsonOutputStreamWrapper streamWrapper(outputStream);
        api::CDataFrameTrainBoostedTreeInstrumentation instrumentation{jobId};
        api::CDataFrameTrainBoostedTreeInstrumentation::CScopeSetOutputStream setStream{
            instrumentation, streamWrapper};
        instrumentation.updateMemoryUsage(memoryUsage);
        instrumentation.nextStep();
        outputStream.flush();
    }
    std::int64_t timeAfter{core::CTimeUtils::toEpochMs(core::CTimeUtils::now())};

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(outputStream.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    BOOST_TEST_REQUIRE(results.IsArray() == true);

    bool hasMemoryUsage{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("analytics_memory_usage")) {
            BOOST_TEST_REQUIRE(result["analytics_memory_usage"].IsObject() == true);
            BOOST_TEST_REQUIRE(result["analytics_memory_usage"]["job_id"].GetString() == jobId);
            BOOST_TEST_REQUIRE(
                result["analytics_memory_usage"]["peak_usage_bytes"].GetInt64() == memoryUsage);
            BOOST_TEST_REQUIRE(result["analytics_memory_usage"]["timestamp"].GetInt64() >=
                               timeBefore);
            BOOST_TEST_REQUIRE(result["analytics_memory_usage"]["timestamp"].GetInt64() <= timeAfter);
            hasMemoryUsage = true;
        }
    }
    BOOST_TEST_REQUIRE(hasMemoryUsage);
}

BOOST_AUTO_TEST_CASE(testTrainingRegression) {
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    test::CDataFrameAnalysisSpecificationFactory specFactory;
    api::CDataFrameAnalyzer analyzer{
        specFactory.predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::regression(), "target"),
        outputWriterFactory};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        test::CDataFrameAnalyzerTrainingFactory::E_Regression, fieldNames,
        fieldValues, analyzer, expectedPredictions);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    std::ifstream regressionSchemaFileStream("testfiles/instrumentation/regression_stats.schema.json");
    BOOST_REQUIRE_MESSAGE(regressionSchemaFileStream.is_open(), "Cannot open test file!");
    std::string regressionSchemaJson((std::istreambuf_iterator<char>(regressionSchemaFileStream)),
                                     std::istreambuf_iterator<char>());
    rapidjson::Document regressionSchemaDocument;
    BOOST_REQUIRE_MESSAGE(
        regressionSchemaDocument.Parse(regressionSchemaJson).HasParseError() == false,
        "Cannot parse JSON schema!");
    rapidjson::SchemaDocument regressionSchema(regressionSchemaDocument);
    rapidjson::SchemaValidator regressionValidator(regressionSchema);

    bool hasRegressionStats{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("regression_stats")) {
            hasRegressionStats = true;
            BOOST_TEST_REQUIRE(result["regression_stats"].IsObject() == true);
            if (result["regression_stats"].Accept(regressionValidator) == false) {
                rapidjson::StringBuffer sb;
                regressionValidator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid schema: " << sb.GetString());
                LOG_ERROR(<< "Invalid keyword: "
                          << regressionValidator.GetInvalidSchemaKeyword());
                sb.Clear();
                regressionValidator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid document: " << sb.GetString());
                BOOST_FAIL("Schema validation failed");
            }
        }
    }
    BOOST_TEST_REQUIRE(hasRegressionStats);

    std::ifstream memorySchemaFileStream("testfiles/instrumentation/memory_usage.schema.json");
    BOOST_REQUIRE_MESSAGE(memorySchemaFileStream.is_open(), "Cannot open test file!");
    std::string memorySchemaJson((std::istreambuf_iterator<char>(memorySchemaFileStream)),
                                 std::istreambuf_iterator<char>());
    rapidjson::Document memorySchemaDocument;
    BOOST_REQUIRE_MESSAGE(memorySchemaDocument.Parse(memorySchemaJson).HasParseError() == false,
                          "Cannot parse JSON schema!");
    rapidjson::SchemaDocument memorySchema(memorySchemaDocument);
    rapidjson::SchemaValidator memoryValidator(memorySchema);

    bool hasMemoryUsage{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("analytics_memory_usage")) {
            hasMemoryUsage = true;
            BOOST_TEST_REQUIRE(result["analytics_memory_usage"].IsObject() == true);
            if (result["analytics_memory_usage"].Accept(memoryValidator) == false) {
                rapidjson::StringBuffer sb;
                memoryValidator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid schema: " << sb.GetString());
                LOG_ERROR(<< "Invalid keyword: "
                          << memoryValidator.GetInvalidSchemaKeyword());
                sb.Clear();
                memoryValidator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid document: " << sb.GetString());
                BOOST_FAIL("Schema validation failed");
            }
        }
    }
    BOOST_TEST_REQUIRE(hasMemoryUsage);
}

BOOST_AUTO_TEST_CASE(testTrainingClassification) {
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    test::CDataFrameAnalysisSpecificationFactory specFactory;
    api::CDataFrameAnalyzer analyzer{
        specFactory.rows(100)
            .memoryLimit(6000000)
            .columns(5)
            .predictionCategoricalFieldNames({"target"})
            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(), "target"),
        outputWriterFactory};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        test::CDataFrameAnalyzerTrainingFactory::E_BinaryClassification,
        fieldNames, fieldValues, analyzer, expectedPredictions);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    std::ifstream schemaFileStream("testfiles/instrumentation/classification_stats.schema.json");
    BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
    std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                           std::istreambuf_iterator<char>());
    rapidjson::Document schemaDocument;
    BOOST_REQUIRE_MESSAGE(schemaDocument.Parse(schemaJson).HasParseError() == false,
                          "Cannot parse JSON schema!");
    rapidjson::SchemaDocument schema(schemaDocument);
    rapidjson::SchemaValidator validator(schema);

    bool hasClassificationStats{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("classification_stats")) {
            hasClassificationStats = true;
            BOOST_TEST_REQUIRE(result["classification_stats"].IsObject() == true);
            if (result["classification_stats"].Accept(validator) == false) {
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
    BOOST_TEST_REQUIRE(hasClassificationStats);
}

BOOST_AUTO_TEST_CASE(testOutlierDetection) {
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory{}.outlierSpec(), outputWriterFactory};

    TDoubleVec expectedScores;
    TDoubleVecVec expectedFeatureInfluences;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    addOutlierTestData(fieldNames, fieldValues, analyzer, expectedScores,
                       expectedFeatureInfluences);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    std::ifstream schemaFileStream("testfiles/instrumentation/outlier_detection_stats.schema.json");
    BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
    std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                           std::istreambuf_iterator<char>());
    rapidjson::Document schemaDocument;
    BOOST_REQUIRE_MESSAGE(schemaDocument.Parse(schemaJson).HasParseError() == false,
                          "Cannot parse JSON schema!");
    rapidjson::SchemaDocument schema(schemaDocument);
    rapidjson::SchemaValidator validator(schema);

    bool hasOutlierDetectionStats{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("outlier_detection_stats")) {
            hasOutlierDetectionStats = true;
            BOOST_TEST_REQUIRE(result["outlier_detection_stats"].IsObject() == true);
            if (result["outlier_detection_stats"].Accept(validator) == false) {
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
    BOOST_TEST_REQUIRE(hasOutlierDetectionStats);
}

BOOST_AUTO_TEST_SUITE_END()
