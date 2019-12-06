/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <maths/CBasicStatistics.h>

#include <api/CDataFrameAnalyzer.h>

#include <test/BoostTestCloseAbsolute.h>
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
using TDataAdderUPtr = std::unique_ptr<core::CDataAdder>;
using TPersisterSupplier = std::function<TDataAdderUPtr()>;
using TDataSearcherUPtr = std::unique_ptr<core::CDataSearcher>;
using TRestoreSearcherSupplier = std::function<TDataSearcherUPtr()>;
using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;

TDataFrameUPtr setupLinearRegressionData(const TStrVec& fieldNames,
                                         TStrVec& fieldValues,
                                         api::CDataFrameAnalyzer& analyzer,
                                         const TDoubleVec& weights,
                                         const TDoubleVec& values) {

    auto target = [&weights](const TDoubleVec& regressors) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * regressors[i];
        }
        return core::CStringUtils::typeToStringPrecise(result, core::CIEEE754::E_DoublePrecision);
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
    rapidjson::Document runRegression(std::size_t shapValues) {
        auto outputWriterFactory = [&]() {
            return std::make_unique<core::CJsonOutputStreamWrapper>(this->output);
        };
        api::CDataFrameAnalyzer analyzer{
            test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
                "regression", "c5", rows, 5, 4000000, 0, 0, {"c1"}, alpha,
                lambda, gamma, softTreeDepthLimit, softTreeDepthTolerance, eta,
                maximumNumberTrees, featureBagFraction, shapValues),
            outputWriterFactory};
        TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", ".", "."};
        TStrVec fieldValues{"", "", "", "", "", "0", ""};
        test::CRandomNumbers rng;
        TDoubleVec weights{50, 150, 50, -50};

        TDoubleVec values;
        rng.generateUniformSamples(-10.0, 10.0, weights.size() * rows, values);

        // make the first column categorical
        for (auto it = values.begin(); it < values.end(); it += 4) {
            *it = (*it < 0) ? -10 : 10;
        }

        auto frame = setupLinearRegressionData(fieldNames, fieldValues,
                                               analyzer, weights, values);

        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        rapidjson::Document results;
        rapidjson::ParseResult ok(results.Parse(output.str()));
        BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
        return results;
    }

    double alpha{2.0};
    double lambda{1.0};
    double gamma{10.0};
    double softTreeDepthLimit{5.0};
    double softTreeDepthTolerance{0.1};
    double eta{0.9};
    std::size_t maximumNumberTrees{1};
    double featureBagFraction{1.0};

    int rows{200};
    std::stringstream output;
};
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceAllShap, SFixture) {
    // Test that feature importance statistically correctly recognize the impact of regressors
    // in a linear model. Test for all regressors in the model. We also make sure that the SHAP values are
    // indeed a local approximation of the prediction up to the constant bias term.
    using TMeanVarAccumulator = ml::maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    std::size_t topShapValues{4};
    TMeanVarAccumulator bias;
    auto results{runRegression(topShapValues)};

    double c1, c2, c3, c4;
    double c1Sum, c2Sum, c3Sum, c4Sum;
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
    BOOST_TEST_REQUIRE(c2Sum > c1Sum);
    // since c1 is categorical -10 or 10, it's influence is generally higher than that of c3 and c4 which are sampled
    // randomly on [-10, 10].
    BOOST_TEST_REQUIRE(c1Sum > c3Sum);
    BOOST_TEST_REQUIRE(c1Sum > c4Sum);
    BOOST_REQUIRE_CLOSE(c3Sum, c4Sum, 80); // c3 and c4 within 80% of each other
    // make sure the local approximation differs from the prediction always by the same bias (up to a numeric error)
    BOOST_REQUIRE_SMALL(ml::maths::CBasicStatistics::variance(bias), 1e-7);
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceNoShap, SFixture) {
    // Test that if topShapValue is set to 0, no feature importance values are returned.
    std::size_t topShapValues{0};
    auto results{runRegression(topShapValues)};

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
