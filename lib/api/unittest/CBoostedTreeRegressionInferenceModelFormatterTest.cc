/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CBoostedTreeRegressionInferenceModelFormatterTest.h"

#include <core/CDataAdder.h>
#include <core/CDataFrame.h>
#include <core/CDataSearcher.h>
#include <core/CFloatStorage.h>

#include <maths/CLinearAlgebraEigen.h>

#include <api/CBoostedTreeRegressionInferenceModelFormatter.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameAnalyzer.h>
#include <api/CInferenceModelDefinition.h>

#include <test/CDataFrameTestUtils.h>
#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <fstream>
#include <streambuf>
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
using TStrSizeUMap = std::unordered_map<std::string, std::size_t>;
using TStrSizeUMapVec = std::vector<TStrSizeUMap>;

auto regressionSpec(std::string dependentVariable,
                    std::size_t rows = 100,
                    std::size_t cols = 5,
                    std::size_t memoryLimit = 3000000,
                    std::size_t numberRoundsPerHyperparameter = 0,
                    std::size_t bayesianOptimisationRestarts = 0,
                    const TStrVec& categoricalFieldNames = TStrVec{},
                    double lambda = -1.0,
                    double gamma = -1.0,
                    double eta = -1.0,
                    std::size_t maximumNumberTrees = 0,
                    double featureBagFraction = -1.0) {

    std::string parameters = "{\n\"dependent_variable\": \"" + dependentVariable + "\"";
    if (lambda >= 0.0) {
        parameters += ",\n\"lambda\": " + core::CStringUtils::typeToString(lambda);
    }
    if (gamma >= 0.0) {
        parameters += ",\n\"gamma\": " + core::CStringUtils::typeToString(gamma);
    }
    if (eta > 0.0) {
        parameters += ",\n\"eta\": " + core::CStringUtils::typeToString(eta);
    }
    if (maximumNumberTrees > 0) {
        parameters += ",\n\"maximum_number_trees\": " +
                      core::CStringUtils::typeToString(maximumNumberTrees);
    }
    if (featureBagFraction > 0.0) {
        parameters += ",\n\"feature_bag_fraction\": " +
                      core::CStringUtils::typeToString(featureBagFraction);
    }
    if (numberRoundsPerHyperparameter > 0) {
        parameters += ",\n\"number_rounds_per_hyperparameter\": " +
                      core::CStringUtils::typeToString(numberRoundsPerHyperparameter);
    }
    if (bayesianOptimisationRestarts > 0) {
        parameters += ",\n\"bayesian_optimisation_restarts\": " +
                      core::CStringUtils::typeToString(bayesianOptimisationRestarts);
    }
    parameters += "\n}";

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, cols, memoryLimit, 1, categoricalFieldNames, true,
        test::CTestTmpDir::tmpDir(), "ml", "regression", parameters)};

    LOG_TRACE(<< "spec =\n" << spec);

    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
}

TDataFrameUPtr passDataToAnalyzer(const TStrVec& fieldNames,
                                  TStrVec& fieldValues,
                                  api::CDataFrameAnalyzer& analyzer,
                                  const TDoubleVec& weights,
                                  const TDoubleVec& values) {

    auto f = [](const TDoubleVec& weights_, const TPoint& regressors) {
        double result{0.0};
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            result += weights_[i] * regressors(i);
        }
        return result;
    };

    TPointVec rows;

    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TPoint row{weights.size() + 1};
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row(j) = values[i + j];
        }
        row(weights.size()) = f(weights, row);

        for (int j = 0; j < row.size(); ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row(j), core::CIEEE754::E_DoublePrecision);
            double xj;
            core::CStringUtils::stringToType(fieldValues[j], xj);
            row(j) = xj;
        }
        analyzer.handleRecord(fieldNames, fieldValues);

        rows.push_back(std::move(row));
    }

    return test::CDataFrameTestUtils::toMainMemoryDataFrame(rows);
}

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

CppUnit::Test* CBoostedTreeRegressionInferenceModelFormatterTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CBoostedTreeRegressionInferenceModelFormatterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeRegressionInferenceModelFormatterTest>(
        "CBoostedTreeRegressionInferenceModelFormatterTest::testIntegration",
        &CBoostedTreeRegressionInferenceModelFormatterTest::testIntegration));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBoostedTreeRegressionInferenceModelFormatterTest>(
        "CBoostedTreeRegressionInferenceModelFormatterTest::testDefinitionGeneration",
        &CBoostedTreeRegressionInferenceModelFormatterTest::testDefinitionGeneration));

    return suiteOfTests;
}

void CBoostedTreeRegressionInferenceModelFormatterTest::testDefinitionGeneration() {
    //    std::string inputFileName("testfiles/regression_runner_with_categories.json");
    //    std::string str;
    //    TStrVec fieldNames{"numeric_col", "categorical_col", "target_col"};
    //    TStrSizeUMapVec categoryMappingVector{{}, {{"cat1", 0}, {"cat2", 1}, {"cat3", 2}}, {}};
    //    {
    //        // Open the input and output files
    //        std::ifstream inputStrm(inputFileName.c_str());
    //        CPPUNIT_ASSERT(inputStrm.is_open());
    //
    //        // read file
    //        str.assign((std::istreambuf_iterator<char>(inputStrm)),
    //                   std::istreambuf_iterator<char>());
    //    }
    //    ml::api::CBoostedTreeRegressionInferenceModelFormatter formatter{
    //        str, fieldNames, categoryMappingVector};
    //    LOG_DEBUG(<< "Test output " << formatter.toString());
    //
    //    // assert input
    //    CPPUNIT_ASSERT(fieldNames == formatter.definition().input().columns());
    //
    //    // test pre-processing
    //    CPPUNIT_ASSERT_EQUAL(3ul, formatter.definition().preprocessing().size());
    //    bool frequency = false;
    //    bool target = false;
    //    bool oneHot = false;
    //
    //    for (const auto& encoding : formatter.definition().preprocessing()) {
    //        if (encoding->typeString() == "frequency_encoding") {
    //            auto enc = static_cast<ml::api::CFrequencyEncoding*>(encoding.get());
    //            CPPUNIT_ASSERT_EQUAL(3ul, enc->frequencyMap().size());
    //            CPPUNIT_ASSERT("categorical_col_frequency" == enc->featureName());
    //            frequency = true;
    //        } else if (encoding->typeString() == "target_mean_encoding") {
    //            auto enc = static_cast<ml::api::CTargetMeanEncoding*>(encoding.get());
    //            CPPUNIT_ASSERT_EQUAL(3ul, enc->targetMap().size());
    //            CPPUNIT_ASSERT("categorical_col_targetmean" == enc->featureName());
    //            CPPUNIT_ASSERT_EQUAL(100.0177, enc->defaultValue());
    //            target = true;
    //        } else if (encoding->typeString() == "one_hot_encoding") {
    //            auto enc = static_cast<ml::api::COneHotEncoding*>(encoding.get());
    //            CPPUNIT_ASSERT_EQUAL(1ul, enc->hotMap().size());
    //            CPPUNIT_ASSERT("categorical_col_cat1" == enc->hotMap()["cat1"]);
    //            oneHot = true;
    //        }
    //    }
    //
    //    CPPUNIT_ASSERT(oneHot && target && frequency);
    //
    //    // assert trained model
    //    auto trainedModel =
    //        dynamic_cast<api::CEnsemble*>(formatter.definition().trainedModel().get());
    //    CPPUNIT_ASSERT_EQUAL(api::CTrainedModel::E_Regression, trainedModel->targetType());
    //    CPPUNIT_ASSERT_EQUAL(3ul, trainedModel->size());
    //    CPPUNIT_ASSERT_EQUAL(1ul, trainedModel->trainedModels()[0].size());
    //    CPPUNIT_ASSERT_EQUAL(3ul, trainedModel->trainedModels()[1].size());
    //    CPPUNIT_ASSERT_EQUAL(3ul, trainedModel->trainedModels()[2].size());
    //    CPPUNIT_ASSERT("weighted_sum" == trainedModel->aggregateOutput()->stringType());
    //    // TODO feature names test is missing
}

void CBoostedTreeRegressionInferenceModelFormatterTest::testIntegration() {
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

    api::CDataFrameAnalyzer analyzer{regressionSpec("target_col", numberExamples, cols,
                                                    30000000, 0, 0, {"categorical_col"}),
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
    const auto& analysisRunner = analyzer.analysisRunner();
    TStrSizeUMapVec categoryMappingVector{{}, {{"cat1", 0}, {"cat2", 1}, {"cat3", 2}}, {}};
    auto definition = analysisRunner->inferenceModelDefinition(fieldNames, categoryMappingVector);
    // assert input
    CPPUNIT_ASSERT(fieldNames == definition->input().columns());

    // test pre-processing
    CPPUNIT_ASSERT_EQUAL(3ul, definition->preprocessing().size());
    bool frequency = false;
    bool target = false;
    bool oneHot = false;

    for (const auto& encoding : definition->preprocessing()) {
        if (encoding->typeString() == "frequency_encoding") {
            auto enc = static_cast<ml::api::CFrequencyEncoding*>(encoding.get());
            CPPUNIT_ASSERT_EQUAL(3ul, enc->frequencyMap().size());
            CPPUNIT_ASSERT("categorical_col_frequency" == enc->featureName());
            frequency = true;
        } else if (encoding->typeString() == "target_mean_encoding") {
            auto enc = static_cast<ml::api::CTargetMeanEncoding*>(encoding.get());
            CPPUNIT_ASSERT_EQUAL(3ul, enc->targetMap().size());
            CPPUNIT_ASSERT("categorical_col_targetmean" == enc->featureName());
            CPPUNIT_ASSERT_EQUAL(100.0177, enc->defaultValue());
            target = true;
        } else if (encoding->typeString() == "one_hot_encoding") {
            auto enc = static_cast<ml::api::COneHotEncoding*>(encoding.get());
            CPPUNIT_ASSERT_EQUAL(1ul, enc->hotMap().size());
            CPPUNIT_ASSERT("categorical_col_cat1" == enc->hotMap()["cat1"]);
            oneHot = true;
        }
    }

    CPPUNIT_ASSERT(oneHot && target && frequency);

    // assert trained model
    auto trainedModel = dynamic_cast<api::CEnsemble*>(definition->trainedModel().get());
    CPPUNIT_ASSERT_EQUAL(api::CTrainedModel::E_Regression, trainedModel->targetType());
    CPPUNIT_ASSERT_EQUAL(3ul, trainedModel->size());
    CPPUNIT_ASSERT_EQUAL(1ul, trainedModel->trainedModels()[0].size());
    CPPUNIT_ASSERT_EQUAL(3ul, trainedModel->trainedModels()[1].size());
    CPPUNIT_ASSERT_EQUAL(3ul, trainedModel->trainedModels()[2].size());
    CPPUNIT_ASSERT("weighted_sum" == trainedModel->aggregateOutput()->stringType());
    // TODO feature names test is missing
}
