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
#include <core/CJsonOutputStreamWrapper.h>

#include <maths/analytics/CBoostedTreeLoss.h>

#include <api/CDataFrameAnalyzer.h>
#include <api/CDataFrameTrainBoostedTreeRunner.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CDataFrameAnalyzerTrainingFactory.h>

#include <boost/test/unit_test.hpp>

#include <valijson/adapters/boost_json_adapter.hpp>
#include <valijson/schema.hpp>
#include <valijson/schema_parser.hpp>
#include <valijson/utils/boost_json_utils.hpp>
#include <valijson/validator.hpp>

#include <fstream>
#include <string>

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(CInferenceModelMetadataTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
using TSizeVec = std::vector<std::size_t>;
using TLossFunctionType = maths::analytics::boosted_tree::ELossType;
using TDataFrameUPtrTemporaryDirectoryPtrPr =
    test::CDataFrameAnalysisSpecificationFactory::TDataFrameUPtrTemporaryDirectoryPtrPr;
}

BOOST_AUTO_TEST_CASE(testJsonSchema) {
    // Test the results the analyzer produces match running the regression directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
                    .predictionLambda(0.5)
                    .predictionEta(0.5)
                    .predictionGamma(0.5)
                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                                    "target", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        TLossFunctionType::E_MseRegression, fieldNames, fieldValues, analyzer,
        expectedPredictions);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    json::error_code ec;
    json::value results = json::parse(output.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);

    std::ifstream modelMetaDataSchemaFileStream("testfiles/model_meta_data/model_meta_data.schema.json");
    BOOST_REQUIRE_MESSAGE(modelMetaDataSchemaFileStream.is_open(), "Cannot open test file!");
    std::string modelMetaDataSchemaJson(
        (std::istreambuf_iterator<char>(modelMetaDataSchemaFileStream)),
        std::istreambuf_iterator<char>());

    json::value modelMetaDataSchemaDocument = json::parse(modelMetaDataSchemaJson, ec);
    BOOST_REQUIRE_MESSAGE(ec.failed() == false, "Cannot parse JSON schema!");

    valijson::Schema schema;
    valijson::SchemaParser parser;
    valijson::adapters::BoostJsonAdapter schemaAdapter(modelMetaDataSchemaDocument);
    parser.populateSchema(schemaAdapter, schema);

    bool hasModelMetadata{false};
    for (const auto& result : results.as_array()) {
        if (result.as_object().contains("model_metadata")) {
            hasModelMetadata = true;
            BOOST_TEST_REQUIRE(result.at_pointer("/model_metadata").is_object() = true);

            valijson::Validator validator;
            valijson::ValidationResults validationResults;
            valijson::adapters::BoostJsonAdapter targetAdapter(
                result.at_pointer("/model_metadata"));
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

    BOOST_TEST_REQUIRE(hasModelMetadata);
}

BOOST_AUTO_TEST_CASE(testHyperparameterReproducibility, *utf::tolerance(0.000001)) {
    // ensure that training leads to the same results if all identified hyperparameters
    // are explicitly specified
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    std::size_t numberSamples{100};
    test::CRandomNumbers rng;
    TSizeVec seed{0};
    rng.generateUniformSamples(0, 1000, 1, seed);

    TDoubleVec expectedPredictions;
    expectedPredictions.reserve(numberSamples);
    TDoubleVec actualPredictions;
    actualPredictions.reserve(numberSamples);

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    test::CDataFrameAnalysisSpecificationFactory specFactory;
    {
        TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
        auto spec = test::CDataFrameAnalysisSpecificationFactory{}.predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::regression(),
            "target", &frameAndDirectory);
        api::CDataFrameAnalyzer analyzer{
            std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
        test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
            TLossFunctionType::E_MseRegression, fieldNames, fieldValues,
            analyzer, numberSamples, seed[0]);

        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        json::error_code ec;
        json::value results = json::parse(output.str(), ec);
        BOOST_TEST_REQUIRE(ec.failed() == false);
        BOOST_TEST_REQUIRE(results.is_array());

        // Read hyperparameter into the new spec and the expected predictions.
        for (const auto& result_ : results.as_array()) {
            const json::object& result = result_.as_object();

            if (result.contains("model_metadata")) {
                for (const auto& hyperparameter :
                     result_.at_pointer("/model_metadata/hyperparameters").as_array()) {
                    std::string hyperparameterName{hyperparameter.at("name").as_string()};
                    if (hyperparameterName == api::CDataFrameTrainBoostedTreeRunner::ALPHA) {
                        specFactory.predictionAlpha(
                            hyperparameter.at("value").to_number<double>());
                    } else if (hyperparameterName ==
                               api::CDataFrameTrainBoostedTreeRunner::DOWNSAMPLE_FACTOR) {
                        specFactory.predictionDownsampleFactor(
                            hyperparameter.at("value").to_number<double>());
                    } else if (hyperparameterName ==
                               api::CDataFrameTrainBoostedTreeRunner::ETA) {
                        specFactory.predictionEta(
                            hyperparameter.at("value").to_number<double>());
                    } else if (hyperparameterName ==
                               api::CDataFrameTrainBoostedTreeRunner::ETA_GROWTH_RATE_PER_TREE) {
                        specFactory.predictionEtaGrowthRatePerTree(
                            hyperparameter.at("value").to_number<double>());
                    } else if (hyperparameterName ==
                               api::CDataFrameTrainBoostedTreeRunner::FEATURE_BAG_FRACTION) {
                        specFactory.predictionFeatureBagFraction(
                            hyperparameter.at("value").to_number<double>());
                    } else if (hyperparameterName ==
                               api::CDataFrameTrainBoostedTreeRunner::GAMMA) {
                        specFactory.predictionGamma(
                            hyperparameter.at("value").to_number<double>());
                    } else if (hyperparameterName ==
                               api::CDataFrameTrainBoostedTreeRunner::LAMBDA) {
                        specFactory.predictionLambda(
                            hyperparameter.at("value").to_number<double>());
                    } else if (hyperparameterName ==
                               api::CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_LIMIT) {
                        specFactory.predictionSoftTreeDepthLimit(
                            hyperparameter.at("value").to_number<double>());
                    } else if (hyperparameterName ==
                               api::CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_TOLERANCE) {
                        specFactory.predictionSoftTreeDepthTolerance(
                            hyperparameter.at("value").to_number<double>());
                    } else if (hyperparameterName ==
                               api::CDataFrameTrainBoostedTreeRunner::MAX_TREES) {
                        specFactory.predictionMaximumNumberTrees(
                            hyperparameter.at("value").to_number<std::int64_t>());
                    }
                }

            } else if (result.contains("row_results")) {
                expectedPredictions.emplace_back(result_
                                                     .at_pointer("/row_results/results/ml/target_prediction")
                                                     .to_number<double>());
            }
        }
    }

    BOOST_REQUIRE_EQUAL(expectedPredictions.size(), numberSamples);
    output.str("");
    {
        TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
        auto spec = specFactory.predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::regression(),
            "target", &frameAndDirectory);
        api::CDataFrameAnalyzer analyzer{
            std::move(spec), std::move(frameAndDirectory), outputWriterFactory};

        test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
            TLossFunctionType::E_MseRegression, fieldNames, fieldValues,
            analyzer, numberSamples, seed[0]);

        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        json::error_code ec;
        json::value results = json::parse(output.str(), ec);
        BOOST_TEST_REQUIRE(ec.failed() == false);
        BOOST_TEST_REQUIRE(results.is_array());
        for (const auto& result_ : results.as_array()) {
            const json::object& result = result_.as_object();
            if (result.contains("row_results")) {
                actualPredictions.emplace_back(result_
                                                   .at_pointer("/row_results/results/ml/target_prediction")
                                                   .to_number<double>());
            }
        }
    }
    BOOST_REQUIRE_EQUAL(actualPredictions.size(), numberSamples);
    BOOST_TEST(actualPredictions == expectedPredictions, tt::per_element());
}

BOOST_AUTO_TEST_CASE(testDataSummarization) {
    std::size_t numRows{50};
    double summarizationFraction{0.2};
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec fieldNames{"f1", "target", ".", "."};
    TStrVec fieldValues{"", "", "0", ""};
    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
                    .rows(numRows)
                    .columns(2)
                    .dataSummarizationFraction(summarizationFraction)
                    .predictionLambda(0.5)
                    .predictionEta(.5)
                    .predictionGamma(0.5)
                    .predictionAlpha(0.5)
                    .predictionSoftTreeDepthLimit(10.0)
                    .predictionSoftTreeDepthTolerance(1.0)
                    .predictionMaximumNumberTrees(3)
                    .predictionDownsampleFactor(1.0)
                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                                    "target", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        TLossFunctionType::E_MseRegression, fieldNames, fieldValues, analyzer, numRows);

    analyzer.handleRecord(fieldNames, {"", "", "", "$"});

    json::error_code ec;
    json::value results = json::parse(output.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(results.is_array());

    // Read number rows of data summarization.
    std::size_t dataSummarizationNumRows{0};
    for (const auto& result_ : results.as_array()) {
        const json::object& result = result_.as_object();
        if (result.contains("model_metadata")) {
            if (result_.at_pointer("/model_metadata").as_object().contains("data_summarization") &&
                result_.at_pointer("/model_metadata/data_summarization")
                    .as_object()
                    .contains("num_rows")) {
                dataSummarizationNumRows = result_
                                               .at_pointer("/model_metadata/data_summarization/num_rows")
                                               .to_number<std::size_t>();
            }
        }
    }

    // check correct number of rows up to a rounding error
    BOOST_REQUIRE_CLOSE_ABSOLUTE(static_cast<double>(dataSummarizationNumRows),
                                 numRows * summarizationFraction, 1.0);
}

BOOST_AUTO_TEST_SUITE_END()
