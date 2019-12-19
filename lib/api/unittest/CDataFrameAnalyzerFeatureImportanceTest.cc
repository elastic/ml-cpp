/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <maths/CBasicStatistics.h>
#include <maths/CDataFrameRegressionModel.h>
#include <maths/CTools.h>

#include <api/CDataFrameAnalyzer.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <memory>
#include <random>

BOOST_AUTO_TEST_SUITE(CDataFrameAnalyzerFeatureImportanceTest)

using namespace ml;

namespace {
using TBoolVec = std::vector<bool>;
using TSizeVec = std::vector<std::size_t>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;
using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
using TMeanVarAccumulator = ml::maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

const std::string SHAP_PREFIX{"feature_importance."};

void setupLinearRegressionData(const TStrVec& fieldNames,
                               TStrVec& fieldValues,
                               api::CDataFrameAnalyzer& analyzer,
                               const TDoubleVec& weights,
                               const TDoubleVec& values,
                               double noiseVar = 0.0) {
    test::CRandomNumbers rng;
    auto target = [&weights, &rng, noiseVar](const TDoubleVec& regressors) {
        TDoubleVec result(1);
        rng.generateNormalSamples(0, noiseVar, 1, result);
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result[0] += weights[i] * regressors[i];
        }
        return core::CStringUtils::typeToStringPrecise(result[0], core::CIEEE754::E_DoublePrecision);
    };

    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = values[i + j];
        }

        for (std::size_t j = 0; j < row.size(); ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }
        fieldValues[weights.size()] = target(row);

        analyzer.handleRecord(fieldNames, fieldValues);
    }
}

void setupBinaryClassificationData(const TStrVec& fieldNames,
                                   TStrVec& fieldValues,
                                   api::CDataFrameAnalyzer& analyzer,
                                   const TDoubleVec& weights,
                                   const TDoubleVec& values) {
    TStrVec classes{"foo", "bar"};
    maths::CPRNG::CXorOShiro128Plus rng;
    std::uniform_real_distribution<double> u01;
    auto target = [&](const TDoubleVec& regressors) {
        double logOddsBar{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            logOddsBar += weights[i] * regressors[i];
        }
        return classes[u01(rng) < maths::CTools::logisticFunction(logOddsBar)];
    };

    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = values[i + j];
        }

        for (std::size_t j = 0; j < row.size(); ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }
        fieldValues[weights.size()] = target(row);

        analyzer.handleRecord(fieldNames, fieldValues);
    }
}

struct SFixture {
    rapidjson::Document
    runRegression(std::size_t shapValues, TDoubleVec weights, double noiseVar = 0.0) {
        auto outputWriterFactory = [&]() {
            return std::make_unique<core::CJsonOutputStreamWrapper>(s_Output);
        };
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
                "regression", "c5", s_Rows, 5, 8000000, 0, 0, {"c1"}, s_Alpha,
                s_Lambda, s_Gamma, s_SoftTreeDepthLimit, s_SoftTreeDepthTolerance,
                s_Eta, s_MaximumNumberTrees, s_FeatureBagFraction, shapValues),
            outputWriterFactory};
        TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
        TStrVec fieldValues{"", "", "", "", "", "0", ""};
        test::CRandomNumbers rng;

        TDoubleVec values;
        rng.generateUniformSamples(-10.0, 10.0, weights.size() * s_Rows, values);

        // make the first column categorical
        for (auto it = values.begin(); it < values.end(); it += 4) {
            *it = (*it < 0) ? -10.0 : 10.0;
        }

        setupLinearRegressionData(fieldNames, fieldValues, analyzer, weights, values, noiseVar);

        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        rapidjson::Document results;
        rapidjson::ParseResult ok(results.Parse(s_Output.str()));
        BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
        return results;
    }

    rapidjson::Document runClassification(std::size_t shapValues, TDoubleVec&& weights) {
        auto outputWriterFactory = [&]() {
            return std::make_unique<core::CJsonOutputStreamWrapper>(s_Output);
        };
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
                "classification", "c5", s_Rows, 5, 8000000, 0, 0, {"c5"}, s_Alpha,
                s_Lambda, s_Gamma, s_SoftTreeDepthLimit, s_SoftTreeDepthTolerance,
                s_Eta, s_MaximumNumberTrees, s_FeatureBagFraction, shapValues),
            outputWriterFactory};
        TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
        TStrVec fieldValues{"", "", "", "", "", "0", ""};
        test::CRandomNumbers rng;

        TDoubleVec values;
        rng.generateUniformSamples(-10.0, 10.0, weights.size() * s_Rows, values);

        setupBinaryClassificationData(fieldNames, fieldValues, analyzer, weights, values);

        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        rapidjson::Document results;
        rapidjson::ParseResult ok(results.Parse(s_Output.str()));
        BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
        return results;
    }

    double s_Alpha{2.0};
    double s_Lambda{1.0};
    double s_Gamma{10.0};
    double s_SoftTreeDepthLimit{5.0};
    double s_SoftTreeDepthTolerance{0.1};
    double s_Eta{0.9};
    std::size_t s_MaximumNumberTrees{1};
    double s_FeatureBagFraction{1.0};

    int s_Rows{2000};
    std::stringstream s_Output;
};
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceAllShap, SFixture) {
    // Test that feature importance statistically correctly recognize the impact of regressors
    // in a linear model. In particular, that the ordering is as expected and for IID features
    // the significance is proportional to the multiplier. Also make sure that the SHAP values
    // are indeed a local approximation of the prediction up to the constant bias term.

    std::size_t topShapValues{4};
    TDoubleVec weights{50, 150, 50, -50};
    auto results{runRegression(topShapValues, weights)};

    TMeanVarAccumulator bias;
    double c1Sum{0.0}, c2Sum{0.0}, c3Sum{0.0}, c4Sum{0.0};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            double c1{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c1"]
                          .GetDouble()};
            double c2{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c2"]
                          .GetDouble()};
            double c3{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c3"]
                          .GetDouble()};
            double c4{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c4"]
                          .GetDouble()};
            double prediction{
                result["row_results"]["results"]["ml"]["c5_prediction"].GetDouble()};
            // the difference between the prediction and the sum of all SHAP values constitutes bias
            bias.add(prediction - (c1 + c2 + c3 + c4));
            c1Sum += std::fabs(c1);
            c2Sum += std::fabs(c2);
            c3Sum += std::fabs(c3);
            c4Sum += std::fabs(c4);
        }
    }

    // since target is generated using the linear model
    // 50 c1 + 150 c2 + 50 c3 - 50 c4, with c1 categorical {-10,10}
    // we expect c2 > c1 > c3 \approx c4
    BOOST_TEST_REQUIRE(c2Sum > c1Sum);
    // since c1 is categorical -10 or 10, it's influence is generally higher than that of c3 and c4 which are sampled
    // randomly on [-10, 10].
    BOOST_TEST_REQUIRE(c1Sum > c3Sum);
    BOOST_TEST_REQUIRE(c1Sum > c4Sum);
    BOOST_REQUIRE_CLOSE(weights[1] / weights[2], c2Sum / c3Sum, 5.0); // ratio within 5% of ratio of coefficients
    BOOST_REQUIRE_CLOSE(c3Sum, c4Sum, 5.0); // c3 and c4 within 5% of each other
    // make sure the local approximation differs from the prediction always by the same bias (up to a numeric error)
    BOOST_REQUIRE_SMALL(ml::maths::CBasicStatistics::variance(bias), 1e-6);
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceNoImportance, SFixture) {
    // Test that feature importance calculates low SHAP values if regressors have no weight.
    // We also add high noise variance.
    std::size_t topShapValues{4};
    auto results{runRegression(topShapValues, {10.0, 0.0, 0.0, 0.0}, 10.0)};

    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            double c1{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c1"]
                          .GetDouble()};
            double c2{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c2"]
                          .GetDouble()};
            double c3{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c3"]
                          .GetDouble()};
            double c4{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c4"]
                          .GetDouble()};
            double prediction{
                result["row_results"]["results"]["ml"]["c5_prediction"].GetDouble()};
            // c1 explain 97% of the prediction value, i.e. the difference from the prediction is less than 1%.
            BOOST_REQUIRE_CLOSE(c1, prediction, 3.0);
            BOOST_REQUIRE_SMALL(c2, 0.25);
            BOOST_REQUIRE_SMALL(c3, 0.25);
            BOOST_REQUIRE_SMALL(c4, 0.25);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeClassificationFeatureImportanceAllShap, SFixture) {
    // Test that feature importance works correctly for classification. In particular, test that
    // feature importance statistically correctly recognizes the impact of regressors if the
    // log-odds of the classes are generated by a linear model. Also make sure that the SHAP
    // values are indeed a local approximation of the predicted log-odds up to the constant
    // bias term.

    std::size_t topShapValues{4};
    TMeanVarAccumulator bias;
    auto results{runClassification(topShapValues, {0.5, -0.7, 0.25, -0.25})};

    double c1Sum{0.0}, c2Sum{0.0}, c3Sum{0.0}, c4Sum{0.0};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            double c1{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c1"]
                          .GetDouble()};
            double c2{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c2"]
                          .GetDouble()};
            double c3{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c3"]
                          .GetDouble()};
            double c4{result["row_results"]["results"]["ml"][maths::CDataFrameRegressionModel::SHAP_PREFIX + "c4"]
                          .GetDouble()};
            double predictionProbability{
                result["row_results"]["results"]["ml"]["prediction_probability"].GetDouble()};
            std::string c5Prediction{
                result["row_results"]["results"]["ml"]["c5_prediction"].GetString()};
            double logOdds{0.0};
            if (c5Prediction == "bar") {
                logOdds = std::log(predictionProbability /
                                   (1.0 - predictionProbability + 1e-10));
            } else if (c5Prediction == "foo") {
                logOdds = std::log((1.0 - predictionProbability) /
                                   (predictionProbability + 1e-10));
            } else {
                BOOST_TEST_FAIL("Unknown predicted class " + c5Prediction);
            }
            // the difference between the prediction and the sum of all SHAP values constitutes bias
            bias.add(logOdds - (c1 + c2 + c3 + c4));
            c1Sum += std::fabs(c1);
            c2Sum += std::fabs(c2);
            c3Sum += std::fabs(c3);
            c4Sum += std::fabs(c4);
        }
    }

    // since the target using a linear model
    // 0.5 c1 + 0.7 c2 + 0.25 c3 - 0.25 c4
    // to generate the log odds we expect c2 > c1 > c3 \approx c4
    BOOST_TEST_REQUIRE(c2Sum > c1Sum);
    BOOST_TEST_REQUIRE(c1Sum > c3Sum);
    BOOST_TEST_REQUIRE(c1Sum > c4Sum);
    BOOST_REQUIRE_CLOSE(c3Sum, c4Sum, 40.0); // c3 and c4 within 40% of each other
    // make sure the local approximation differs from the prediction always by the same bias (up to a numeric error)
    BOOST_REQUIRE_SMALL(ml::maths::CBasicStatistics::variance(bias), 1e-6);
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceNoShap, SFixture) {
    // Test that if topShapValue is set to 0, no feature importance values are returned.
    std::size_t topShapValues{0};
    auto results{runRegression(topShapValues, {50.0, 150.0, 50.0, -50.0})};

    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(result["row_results"]["results"]["ml"].HasMember(
                                   maths::CDataFrameRegressionModel::SHAP_PREFIX + "c1") == false);
            BOOST_TEST_REQUIRE(result["row_results"]["results"]["ml"].HasMember(
                                   maths::CDataFrameRegressionModel::SHAP_PREFIX + "c2") == false);
            BOOST_TEST_REQUIRE(result["row_results"]["results"]["ml"].HasMember(
                                   maths::CDataFrameRegressionModel::SHAP_PREFIX + "c3") == false);
            BOOST_TEST_REQUIRE(result["row_results"]["results"]["ml"].HasMember(
                                   maths::CDataFrameRegressionModel::SHAP_PREFIX + "c4") == false);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
