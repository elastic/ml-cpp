/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CBoostedTreeInferenceModelBuilderTest.h"

#include <core/CDataAdder.h>
#include <core/CDataFrame.h>
#include <core/CDataSearcher.h>
#include <core/CFloatStorage.h>

#include <maths/CLinearAlgebraEigen.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameAnalyzer.h>
#include <api/CInferenceModelDefinition.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CDataFrameTestUtils.h>
#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <rapidjson/schema.h>

#include <fstream>
#include <string>

using namespace ml;

namespace {

using TStrVec = std::vector<std::string>;
using TDataAdderUPtr = std::unique_ptr<ml::core::CDataAdder>;
using TPersisterSupplier = std::function<TDataAdderUPtr()>;
using TDataSearcherUPtr = std::unique_ptr<ml::core::CDataSearcher>;
using TRestoreSearcherSupplier = std::function<TDataSearcherUPtr()>;
using TDataFrameUPtr = std::unique_ptr<ml::core::CDataFrame>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TPoint = maths::CDenseVector<maths::CFloatStorage>;
using TPointVec = std::vector<TPoint>;
using TRowItr = core::CDataFrame::TRowItr;
using TStrVecVec = std::vector<TStrVec>;

auto generateCategoricalData(test::CRandomNumbers& rng, std::size_t rows, TDoubleVec expectedFrequencies) {

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

void CBoostedTreeInferenceModelBuilderTest::testIntegrationRegression() {
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

    api::CDataFrameAnalyzer analyzer{test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
                                         "regression", "target_col", numberExamples,
                                         cols, 30000000, 0, 0, {"categorical_col"}),
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

    LOG_DEBUG(<< "Inference model definition: " << definition->jsonString());

    // assert input
    CPPUNIT_ASSERT(expectedFieldNames == definition->input().fieldNames());

    // test pre-processing
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), definition->preprocessors().size());
    bool frequency = false;
    bool target = false;
    bool oneHot = false;

    for (const auto& encoding : definition->preprocessors()) {
        if (encoding->typeString() == "frequency_encoding") {
            auto enc = static_cast<ml::api::CFrequencyEncoding*>(encoding.get());
            CPPUNIT_ASSERT_EQUAL(std::size_t(3), enc->frequencyMap().size());
            CPPUNIT_ASSERT("categorical_col_frequency" == enc->featureName());
            frequency = true;
        } else if (encoding->typeString() == "target_mean_encoding") {
            auto enc = static_cast<ml::api::CTargetMeanEncoding*>(encoding.get());
            CPPUNIT_ASSERT_EQUAL(std::size_t(3), enc->targetMap().size());
            CPPUNIT_ASSERT("categorical_col_targetmean" == enc->featureName());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(100.0177288, enc->defaultValue(), 1e-6);
            target = true;
        } else if (encoding->typeString() == "one_hot_encoding") {
            auto enc = static_cast<ml::api::COneHotEncoding*>(encoding.get());
            CPPUNIT_ASSERT_EQUAL(std::size_t(3), enc->hotMap().size());
            CPPUNIT_ASSERT("categorical_col_cat1" == enc->hotMap()["cat1"]);
            CPPUNIT_ASSERT("categorical_col_cat2" == enc->hotMap()["cat2"]);
            CPPUNIT_ASSERT("categorical_col_cat3" == enc->hotMap()["cat3"]);
            oneHot = true;
        }
    }

    CPPUNIT_ASSERT(oneHot && target && frequency);

    // assert trained model
    auto trainedModel = dynamic_cast<api::CEnsemble*>(definition->trainedModel().get());
    CPPUNIT_ASSERT_EQUAL(api::CTrainedModel::E_Regression, trainedModel->targetType());
    CPPUNIT_ASSERT_EQUAL(std::size_t(22), trainedModel->size());
    CPPUNIT_ASSERT("weighted_sum" == trainedModel->aggregateOutput()->stringType());
    CPPUNIT_ASSERT(trainedModel->featureNames() ==
                   std::vector<std::string>({"numeric_col", "categorical_col_cat1", "categorical_col_cat2",
                                             "categorical_col_cat3", "categorical_col_frequency",
                                             "categorical_col_targetmean"}));
}

void CBoostedTreeInferenceModelBuilderTest::testIntegrationClassification() {
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
    values[2] = generateCategoricalData(rng, numberExamples, {5.0, 5.0}).second;

    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
            "classification", "target_col", numberExamples, cols, 30000000, 0,
            0, {"categorical_col", "target_col"}),
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
    TStrVecVec categoryMappingVector{{}, {"cat1", "cat2", "cat3"}, {"true", "false"}};
    auto definition = analysisRunner->inferenceModelDefinition(fieldNames, categoryMappingVector);

    LOG_DEBUG(<< "Inference model definition: " << definition->jsonString());

    // assert trained model
    auto trainedModel = dynamic_cast<api::CEnsemble*>(definition->trainedModel().get());
    CPPUNIT_ASSERT_EQUAL(api::CTrainedModel::E_Classification, trainedModel->targetType());
    CPPUNIT_ASSERT_EQUAL(std::size_t(5), trainedModel->size());
    CPPUNIT_ASSERT("logistic_regression" == trainedModel->aggregateOutput()->stringType());
}

void CBoostedTreeInferenceModelBuilderTest::testJsonSchema() {
    std::size_t numberExamples = 1000;
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

    api::CDataFrameAnalyzer analyzer{test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
                                         "regression", "target_col", numberExamples,
                                         cols, 30000000, 0, 0, {"categorical_col"}),
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

    std::ifstream schemaFileStream("testfiles/inference_json_schema/definition.schema.json");
    CPPUNIT_ASSERT_MESSAGE("Cannot open test file!", schemaFileStream);
    std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                           std::istreambuf_iterator<char>());
    rapidjson::Document schemaDocument;
    CPPUNIT_ASSERT_MESSAGE("Cannot parse JSON schema!",
                           schemaDocument.Parse(schemaJson).HasParseError() == false);
    rapidjson::SchemaDocument schema(schemaDocument);

    rapidjson::Document doc;
    CPPUNIT_ASSERT_MESSAGE("Error parsing JSON definition!",
                           doc.Parse(definition->jsonString()).HasParseError() == false);

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
        CPPUNIT_ASSERT_MESSAGE("Schema validation failed", false);
    }
}

CppUnit::Test* CBoostedTreeInferenceModelBuilderTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CBoostedTreeInferenceModelBuilderTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeInferenceModelBuilderTest>(
        "CBoostedTreeInferenceModelBuilderTest::testIntegrationRegression",
        &CBoostedTreeInferenceModelBuilderTest::testIntegrationRegression));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeInferenceModelBuilderTest>(
        "CBoostedTreeInferenceModelBuilderTest::testIntegrationClassification",
        &CBoostedTreeInferenceModelBuilderTest::testIntegrationClassification));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeInferenceModelBuilderTest>(
        "CBoostedTreeInferenceModelBuilderTest::testJsonSchema",
        &CBoostedTreeInferenceModelBuilderTest::testJsonSchema));

    return suiteOfTests;
}
