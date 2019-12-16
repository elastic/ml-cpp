/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <maths/CBasicStatistics.h>

#include <api/CDataFrameAnalyzer.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <memory>

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

TDataFrameUPtr setupLinearRegressionData(const TStrVec& fieldNames,
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

    auto frame = core::makeMainStorageDataFrame(weights.size() + 1).first;

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
        frame->parseAndWriteRow(
            core::CVectorRange<const TStrVec>(fieldValues, 0, weights.size() + 1));
    }

    frame->finishWritingRows();

    return frame;
}

TDataFrameUPtr setupBinaryClassificationData(const TStrVec& fieldNames,
                                             TStrVec& fieldValues,
                                             api::CDataFrameAnalyzer& analyzer,
                                             const TDoubleVec& weights,
                                             const TDoubleVec& values) {
    TStrVec classes{"foo", "bar"};
    auto target = [&weights, &classes](const TDoubleVec& regressors) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * regressors[i];
        }
        return classes[result < 0.0 ? 0 : 1];
    };

    auto frame = core::makeMainStorageDataFrame(weights.size() + 1).first;
    TBoolVec categoricalFields(weights.size(), false);
    categoricalFields.push_back(true);
    frame->categoricalColumns(std::move(categoricalFields));

    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = values[i + j];
        }

        for (std::size_t j = 0; j < row.size() - 1; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }
        fieldValues[weights.size()] = target(row);

        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(
            core::CVectorRange<const TStrVec>(fieldValues, 0, weights.size() + 1));
    }
    frame->finishWritingRows();
    return frame;
}

struct SFixture {
    rapidjson::Document
    runRegression(std::size_t shapValues, TDoubleVec&& weights, double noiseVar = 0.0) {
        auto outputWriterFactory = [&]() {
            return std::make_unique<core::CJsonOutputStreamWrapper>(s_Output);
        };
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
                "regression", "c5", s_Rows, 5, 4000000, 0, 0, {"c1"}, s_Alpha,
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
            *it = (*it < 0) ? -10 : 10;
        }

        auto frame = setupLinearRegressionData(fieldNames, fieldValues, analyzer,
                                               weights, values, noiseVar);

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
                "classification", "c5", s_Rows, 5, 4000000, 0, 0, {"c5"}, s_Alpha,
                s_Lambda, s_Gamma, s_SoftTreeDepthLimit, s_SoftTreeDepthTolerance,
                s_Eta, s_MaximumNumberTrees, s_FeatureBagFraction, shapValues),
            outputWriterFactory};
        TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
        TStrVec fieldValues{"", "", "", "", "", "0", ""};
        test::CRandomNumbers rng;

        TDoubleVec values;
        rng.generateUniformSamples(-10.0, 10.0, weights.size() * s_Rows, values);

        auto frame = setupBinaryClassificationData(fieldNames, fieldValues,
                                                   analyzer, weights, values);

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

    int s_Rows{200};
    std::stringstream s_Output;
};
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceAllShap, SFixture) {
    // Test that feature importance statistically correctly recognize the impact of regressors
    // in a linear model. Test for all regressors in the model. We also make sure that the SHAP values are
    // indeed a local approximation of the prediction up to the constant bias term.
    using TMeanVarAccumulator = ml::maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    std::size_t topShapValues{4};
    TMeanVarAccumulator bias;
    auto results{runRegression(topShapValues, {50, 150, 50, -50})};

    double c1, c2, c3, c4;
    double c1Sum{0.0}, c2Sum{0.0}, c3Sum{0.0}, c4Sum{0.0};
    double prediction;
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            c1 = result["row_results"]["results"]["ml"]["shap.c1"].GetDouble();
            c2 = result["row_results"]["results"]["ml"]["shap.c2"].GetDouble();
            c3 = result["row_results"]["results"]["ml"]["shap.c3"].GetDouble();
            c4 = result["row_results"]["results"]["ml"]["shap.c4"].GetDouble();
            prediction = result["row_results"]["results"]["ml"]["c5_prediction"].GetDouble();
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
    BOOST_REQUIRE_CLOSE(c3Sum, c4Sum, 80); // c3 and c4 within 80% of each other
    // make sure the local approximation differs from the prediction always by the same bias (up to a numeric error)
    BOOST_REQUIRE_SMALL(ml::maths::CBasicStatistics::variance(bias), 1e-7);
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceNoImportance, SFixture) {
    // Test that feature importance calculates low SHAP values if regressors have no weight.
    // We also add high noise variance.
    std::size_t topShapValues{4};
    auto results{runRegression(topShapValues, {10.0, 0.0, 0.0, 0.0}, 10.0)};

    double c1, c2, c3, c4;
    double prediction;
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            c1 = result["row_results"]["results"]["ml"]["shap.c1"].GetDouble();
            c2 = result["row_results"]["results"]["ml"]["shap.c2"].GetDouble();
            c3 = result["row_results"]["results"]["ml"]["shap.c3"].GetDouble();
            c4 = result["row_results"]["results"]["ml"]["shap.c4"].GetDouble();
            prediction = result["row_results"]["results"]["ml"]["c5_prediction"].GetDouble();
            BOOST_REQUIRE_CLOSE(c1, prediction, 99); //c1 explain 99% of the prediction value
            BOOST_REQUIRE_SMALL(c2, 1e-8);
            BOOST_REQUIRE_SMALL(c3, 1e-8);
            BOOST_REQUIRE_SMALL(c4, 1e-8);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeClassificationFeatureImportanceAllShap, SFixture) {
    // Test that feature importance works correctly for classification. We make sure that the SHAP values are
    // indeed a local approximation of the log-odds up to the constant bias term.
    using TMeanVarAccumulator = ml::maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    std::size_t topShapValues{4};
    TMeanVarAccumulator bias;
    auto results{runClassification(topShapValues, {50, 70, 50, -50})};

    double c1, c2, c3, c4;
    double c1Sum{0.0}, c2Sum{0.0}, c3Sum{0.0}, c4Sum{0.0};
    double prediction_probability, logOdds{0.0};
    std::string c5_prediction;
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            c1 = result["row_results"]["results"]["ml"]["shap.c1"].GetDouble();
            c2 = result["row_results"]["results"]["ml"]["shap.c2"].GetDouble();
            c3 = result["row_results"]["results"]["ml"]["shap.c3"].GetDouble();
            c4 = result["row_results"]["results"]["ml"]["shap.c4"].GetDouble();
            prediction_probability =
                result["row_results"]["results"]["ml"]["prediction_probability"].GetDouble();
            c5_prediction =
                result["row_results"]["results"]["ml"]["c5_prediction"].GetString();
            if (c5_prediction == "bar") {
                logOdds = std::log(prediction_probability /
                                   (1 - prediction_probability + 1e-10));
            } else if (c5_prediction == "foo") {
                logOdds = std::log((1 - prediction_probability) /
                                   (prediction_probability + 1e-10));
            } else {
                BOOST_TEST_FAIL("Unknown predicted class " + c5_prediction);
            }
            // the difference between the prediction and the sum of all SHAP values constitutes bias
            bias.add(logOdds - (c1 + c2 + c3 + c4));
            c1Sum += std::fabs(c1);
            c2Sum += std::fabs(c2);
            c3Sum += std::fabs(c3);
            c4Sum += std::fabs(c4);
        }
    }

    // make sure the local approximation differs from the prediction always by the same bias (up to a numeric error)
    BOOST_REQUIRE_SMALL(ml::maths::CBasicStatistics::variance(bias), 1e-7);
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceNoShap, SFixture) {
    // Test that if topShapValue is set to 0, no feature importance values are returned.
    std::size_t topShapValues{0};
    auto results{runRegression(topShapValues, {50, 150, 50, -50})};

    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(
                result["row_results"]["results"]["ml"].HasMember("shap.c1") == false);
            BOOST_TEST_REQUIRE(
                result["row_results"]["results"]["ml"].HasMember("shap.c2") == false);
            BOOST_TEST_REQUIRE(
                result["row_results"]["results"]["ml"].HasMember("shap.c3") == false);
            BOOST_TEST_REQUIRE(
                result["row_results"]["results"]["ml"].HasMember("shap.c4") == false);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
