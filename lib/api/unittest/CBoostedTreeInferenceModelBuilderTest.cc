/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataAdder.h>
#include <core/CDataFrame.h>
#include <core/CDataSearcher.h>
#include <core/CFloatStorage.h>
#include <core/CProgramCounters.h>
#include <core/CStringUtils.h>

#include <maths/CLinearAlgebraEigen.h>

#include <api/CBoostedTreeInferenceModelBuilder.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameAnalyzer.h>
#include <api/CInferenceModelDefinition.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CDataFrameTestUtils.h>
#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <rapidjson/schema.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <string>

BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::api::CFrequencyEncoding::TStringDoubleUMap::const_iterator)
BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::api::COneHotEncoding::TStringStringUMap::const_iterator)

BOOST_AUTO_TEST_SUITE(CBoostedTreeInferenceModelBuilderTest)

using namespace ml;

namespace {

using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;

auto generateCategoricalData(test::CRandomNumbers& rng,
                             std::size_t rows,
                             const TDoubleVec& expectedFrequencies) {

    TDoubleVecVec frequencies;
    rng.generateDirichletSamples(expectedFrequencies, 1, frequencies);

    TDoubleVec values(1);
    for (std::size_t j = 0; j < frequencies[0].size(); ++j) {
        std::size_t target{static_cast<std::size_t>(
            static_cast<double>(rows) * frequencies[0][j] + 0.5)};
        values.resize(values.size() + target, static_cast<double>(j));
    }
    values.resize(rows, values.back());
    rng.random_shuffle(values.begin(), values.end());
    rng.discard(1000000); // Make sure the categories are not correlated

    return std::make_pair(frequencies[0], values);
}
}

BOOST_AUTO_TEST_CASE(testIntegrationRegression) {
    std::size_t numberExamples = 1000;
    std::size_t cols = 3;
    test::CRandomNumbers rng;
    TDoubleVec weights{0.1, 100.0};

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec fieldNames{"numeric_col", "categorical_col", "target_col", ".", "."};
    TStrVec expectedFieldNames{"numeric_col", "categorical_col"};

    TStrVec fieldValues{"", "", "0", "", ""};

    TDoubleVecVec frequencies;
    TDoubleVecVec values(cols);
    rng.generateUniformSamples(-10.0, 10.0, numberExamples, values[0]);
    values[1] = generateCategoricalData(rng, numberExamples, {100., 5.0, 5.0}).second;

    for (std::size_t i = 0; i < numberExamples; ++i) {
        values[2].push_back(values[0][i] * weights[0] + values[1][i] * weights[1]);
    }

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::regression(), "target_col",
            numberExamples, cols, 30000000, 0, 0, {"categorical_col"}),
        outputWriterFactory};

    TDataFrameUPtr frame =
        core::makeMainStorageDataFrame(cols + 2, numberExamples).first;
    for (std::size_t i = 0; i < numberExamples; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                values[j][i], core::CIEEE754::E_DoublePrecision);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
    }
    analyzer.handleRecord(fieldNames, {"", "", "", "", "$"});
    auto analysisRunner = analyzer.runner();
    TStrVecVec categoryMappingVector{{}, {"cat1", "cat2", "cat3"}, {}};
    auto definition = analysisRunner->inferenceModelDefinition(fieldNames, categoryMappingVector);

    // test pre-processing
    BOOST_REQUIRE_EQUAL(std::size_t(3), definition->preprocessors().size());
    bool frequency = false;
    bool target = false;
    bool oneHot = false;

    for (const auto& encoding : definition->preprocessors()) {
        if (encoding->typeString() == "frequency_encoding") {
            auto enc = static_cast<ml::api::CFrequencyEncoding*>(encoding.get());
            BOOST_REQUIRE_EQUAL(std::size_t(3), enc->frequencyMap().size());
            BOOST_TEST_REQUIRE("categorical_col_frequency" == enc->featureName());
            frequency = true;
        } else if (encoding->typeString() == "target_mean_encoding") {
            auto enc = static_cast<ml::api::CTargetMeanEncoding*>(encoding.get());
            BOOST_REQUIRE_EQUAL(std::size_t(3), enc->targetMap().size());
            BOOST_TEST_REQUIRE("categorical_col_targetmean" == enc->featureName());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(100.0177288, enc->defaultValue(), 1e-6);
            target = true;
        } else if (encoding->typeString() == "one_hot_encoding") {
            auto enc = static_cast<ml::api::COneHotEncoding*>(encoding.get());
            BOOST_REQUIRE_EQUAL(std::size_t(3), enc->hotMap().size());
            BOOST_TEST_REQUIRE("categorical_col_cat1" == enc->hotMap()["cat1"]);
            BOOST_TEST_REQUIRE("categorical_col_cat2" == enc->hotMap()["cat2"]);
            BOOST_TEST_REQUIRE("categorical_col_cat3" == enc->hotMap()["cat3"]);
            oneHot = true;
        }
    }

    BOOST_TEST_REQUIRE(oneHot);
    BOOST_TEST_REQUIRE(target);
    BOOST_TEST_REQUIRE(frequency);

    // assert trained model
    auto* trainedModel =
        dynamic_cast<api::CEnsemble*>(definition->trainedModel().get());
    BOOST_REQUIRE_EQUAL(api::CTrainedModel::E_Regression, trainedModel->targetType());
    std::size_t expectedSize{core::CProgramCounters::counter(
        ml::counter_t::E_DFTPMTrainedForestNumberTrees)};
    BOOST_REQUIRE_EQUAL(expectedSize, trainedModel->size());
    BOOST_TEST_REQUIRE("weighted_sum" == trainedModel->aggregateOutput()->stringType());
}

BOOST_AUTO_TEST_CASE(testIntegrationClassification) {
    std::size_t numberExamples = 200;
    std::size_t cols = 3;
    test::CRandomNumbers rng;
    TDoubleVec weights{0.1, 100.0};

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec fieldNames{"numeric_col", "categorical_col", "target_col", ".", "."};
    TStrVec expectedFieldNames{"numeric_col", "categorical_col"};

    TStrVec fieldValues{"", "", "0", "", ""};

    TDoubleVecVec frequencies;
    TDoubleVecVec values(cols);
    rng.generateUniformSamples(-10.0, 10.0, numberExamples, values[0]);
    values[1] = generateCategoricalData(rng, numberExamples, {100., 5.0, 5.0}).second;
    values[2] = generateCategoricalData(rng, numberExamples, {5.0, 5.0}).second;

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::classification(), "target_col",
            numberExamples, cols, 30000000, 0, 0, {"categorical_col", "target_col"}),
        outputWriterFactory};

    TDataFrameUPtr frame =
        core::makeMainStorageDataFrame(cols + 2, numberExamples).first;
    for (std::size_t i = 0; i < numberExamples; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                values[j][i], core::CIEEE754::E_DoublePrecision);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
    }
    analyzer.handleRecord(fieldNames, {"", "", "", "", "$"});
    auto analysisRunner = analyzer.runner();
    TStrVec expectedClassificationLabels{"true", "false"};
    TStrVecVec categoryMappingVector{{}, {"cat1", "cat2", "cat3"}, expectedClassificationLabels};
    auto definition = analysisRunner->inferenceModelDefinition(fieldNames, categoryMappingVector);

    LOG_DEBUG(<< "Inference model definition: " << definition->jsonString());

    // assert trained model
    auto trainedModel = dynamic_cast<api::CEnsemble*>(definition->trainedModel().get());
    BOOST_REQUIRE_EQUAL(api::CTrainedModel::E_Classification, trainedModel->targetType());
    std::size_t expectedSize{
        core::CProgramCounters::counter(counter_t::E_DFTPMTrainedForestNumberTrees)};
    BOOST_REQUIRE_EQUAL(expectedSize, trainedModel->size());
    BOOST_TEST_REQUIRE("logistic_regression" ==
                       trainedModel->aggregateOutput()->stringType());
    const auto& classificationLabels = trainedModel->classificationLabels();
    BOOST_TEST_REQUIRE(classificationLabels.is_initialized());
    BOOST_REQUIRE_EQUAL_COLLECTIONS(
        classificationLabels->begin(), classificationLabels->end(),
        expectedClassificationLabels.begin(), expectedClassificationLabels.end());

    const auto& classificationWeights = trainedModel->classificationWeights();
    BOOST_TEST_REQUIRE(classificationWeights.is_initialized());

    // Check that predicted score matches the value calculated from the inference
    // classification weights.
    std::map<bool, std::size_t> classLookup;
    for (std::size_t i = 0; i < classificationLabels->size(); ++i) {
        bool labelAsBool;
        core::CStringUtils::stringToType((*classificationLabels)[i], labelAsBool);
        classLookup[labelAsBool] = i;
    }
    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            std::string prediction{
                result["row_results"]["results"]["ml"]["target_col_prediction"].GetString()};
            double probability{
                result["row_results"]["results"]["ml"]["prediction_probability"].GetDouble()};
            double score{
                result["row_results"]["results"]["ml"]["prediction_score"].GetDouble()};
            bool predictionAsBool;
            core::CStringUtils::stringToType(prediction, predictionAsBool);
            std::size_t weight{classLookup[predictionAsBool]};
            BOOST_REQUIRE_CLOSE((*classificationWeights)[weight] * probability,
                                score, 1e-3); // 0.001%
        }
    }
}

BOOST_AUTO_TEST_CASE(testJsonSchema) {
    std::size_t numberExamples = 200;
    std::size_t cols = 3;
    test::CRandomNumbers rng;
    TDoubleVec weights{0.1, 100.0};

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec fieldNames{"numeric_col", "categorical_col", "target_col", ".", "."};

    TStrVec fieldValues{"", "", "0", "", ""};

    TDoubleVecVec frequencies;
    TDoubleVecVec values(cols);
    rng.generateUniformSamples(-10.0, 10.0, numberExamples, values[0]);
    values[1] = generateCategoricalData(rng, numberExamples, {100., 5.0, 5.0}).second;

    for (std::size_t i = 0; i < numberExamples; ++i) {
        values[2].push_back(values[0][i] * weights[0] + values[1][i] * weights[1]);
    }

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::regression(), "target_col",
            numberExamples, cols, 30000000, 0, 0, {"categorical_col"}),
        outputWriterFactory};

    TDataFrameUPtr frame =
        core::makeMainStorageDataFrame(cols + 2, numberExamples).first;
    for (std::size_t i = 0; i < numberExamples; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                values[j][i], core::CIEEE754::E_DoublePrecision);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
    }
    analyzer.handleRecord(fieldNames, {"", "", "", "", "$"});
    auto analysisRunner = analyzer.runner();
    TStrVecVec categoryMappingVector{{}, {"cat1", "cat2", "cat3"}, {}};
    auto definition = analysisRunner->inferenceModelDefinition(fieldNames, categoryMappingVector);

    std::ifstream schemaFileStream("testfiles/inference_json_schema/model_definition.schema.json");
    BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
    std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                           std::istreambuf_iterator<char>());
    rapidjson::Document schemaDocument;
    BOOST_REQUIRE_MESSAGE(schemaDocument.Parse(schemaJson).HasParseError() == false,
                          "Cannot parse JSON schema!");
    rapidjson::SchemaDocument schema(schemaDocument);

    rapidjson::Document doc;
    BOOST_REQUIRE_MESSAGE(doc.Parse(definition->jsonString()).HasParseError() == false,
                          "Error parsing JSON definition!");

    rapidjson::SchemaValidator validator(schema);
    if (doc.Accept(validator) == false) {
        rapidjson::StringBuffer sb;
        validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
        LOG_ERROR(<< "Invalid schema: " << sb.GetString());
        LOG_ERROR(<< "Invalid keyword: " << validator.GetInvalidSchemaKeyword());
        sb.Clear();
        validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
        LOG_ERROR(<< "Invalid document: " << sb.GetString());
        LOG_DEBUG(<< "Document: " << definition->jsonString());
        BOOST_FAIL("Schema validation failed");
    }

    // TODO add multivalued leaf test.
}

BOOST_AUTO_TEST_CASE(testEncoders) {
    {
        TStrVec fieldNames{"col1", "target", "col2", "col3"};
        std::size_t dependentVariableColumnIndex{1};
        TStrVecVec categoryNames{{},
                                 {"targetcat1", "targetcat2"},
                                 {"col2cat1", "col2cat2", "col2cat3"},
                                 {"col3cat1", "col3cat2"}};
        api::CClassificationInferenceModelBuilder builder(
            fieldNames, dependentVariableColumnIndex, categoryNames);
        builder.addIdentityEncoding(0);
        builder.addOneHotEncoding(2, 0);
        builder.addOneHotEncoding(2, 1);
        builder.addFrequencyEncoding(2, {1.0, 1.0, 1.0});
        builder.addOneHotEncoding(3, 0);
        builder.addFrequencyEncoding(3, {1.0, 1.0});
        auto definition{builder.build()};
        const auto& preprocessors{definition.preprocessors()};
        BOOST_REQUIRE_EQUAL(std::size_t(4), preprocessors.size());
        for (const auto& encoding : preprocessors) {
            if (encoding->typeString() == "frequency_encoding") {
                const auto& frequencyEncoding{
                    static_cast<api::CFrequencyEncoding*>(encoding.get())};
                const auto& map{frequencyEncoding->frequencyMap()};
                if (frequencyEncoding->featureName() == "col2_frequency") {
                    BOOST_REQUIRE_EQUAL(std::size_t(3), map.size());
                    BOOST_TEST_REQUIRE(map.find("col2cat1") != map.end());
                    BOOST_TEST_REQUIRE(map.find("col2cat2") != map.end());
                    BOOST_TEST_REQUIRE(map.find("col2cat3") != map.end());
                } else if (frequencyEncoding->featureName() == "col3_frequency") {
                    BOOST_REQUIRE_EQUAL(std::size_t(2), map.size());
                    BOOST_TEST_REQUIRE(map.find("col3cat1") != map.end());
                    BOOST_TEST_REQUIRE(map.find("col3cat2") != map.end());
                }
            } else if (encoding->typeString() == "one_hot_encoding") {
                const auto& oneHotEncoding{
                    static_cast<api::COneHotEncoding*>(encoding.get())};
                const auto& map{oneHotEncoding->hotMap()};

                if (oneHotEncoding->field() == "col2") {
                    BOOST_REQUIRE_EQUAL(std::size_t(2), map.size());
                    BOOST_TEST_REQUIRE(map.find("col2cat1") != map.end());
                    BOOST_TEST_REQUIRE(map.find("col2cat2") != map.end());
                } else if (oneHotEncoding->field() == "col3") {
                    BOOST_REQUIRE_EQUAL(std::size_t(1), map.size());
                    BOOST_TEST_REQUIRE(map.find("col3cat1") != map.end());
                }
            } else {
                BOOST_FAIL("Unexpected encoding type");
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
