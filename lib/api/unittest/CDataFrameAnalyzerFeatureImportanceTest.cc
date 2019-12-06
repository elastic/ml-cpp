/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>
#include <core/CDataSearcher.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CProgramCounters.h>
#include <core/CStateDecompressor.h>
#include <core/CStopWatch.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeFactory.h>
#include <maths/CTools.h>

#include <api/CDataFrameAnalyzer.h>
#include <api/CDataFrameTrainBoostedTreeRegressionRunner.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/ElasticsearchStateIndex.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CRandomNumbers.h>

#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>

#include <memory>

using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDoubleVec::iterator)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TStrVec::iterator)

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
    SFixture() {}

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

        frame->readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                bias += (*row)[4];
            }
        });
        bias /= frame->numberRows();

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
    double bias{0};
};
}

BOOST_FIXTURE_TEST_CASE(testRunBoostedTreeRegressionFeatureImportanceAllShap, SFixture) {

    // Test that feature importance statistically correctly recognize the impact of regressors
    // in a linear model. Test for all regressors in the model.
    std::size_t topShapValues{4};
    auto results{runRegression(topShapValues)};

    std::ostringstream stream;
    {
        core::CJsonOutputStreamWrapper wrapper{stream};
        api::CSerializableToJson::TRapidJsonWriter writer{wrapper};
        writer.write(results);
        stream.flush();
    }
    // string writer puts the json object in an array, so we strip the external brackets
    LOG_DEBUG(<< stream.str());

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
            //            BOOST_REQUIRE_CLOSE(prediction-bias, c1+c2+c3+c4, 95);
            c1Sum += std::fabs(c1);
            c2Sum += std::fabs(c2);
            c3Sum += std::fabs(c3);
            c4Sum += std::fabs(c4);
        }
    }
    BOOST_TEST_REQUIRE(c2Sum > c1Sum);
    BOOST_TEST_REQUIRE(c1Sum > c3Sum);
    BOOST_TEST_REQUIRE(c1Sum > c4Sum);
    BOOST_REQUIRE_CLOSE(c3Sum, c4Sum, 80); // c3 and c4 within 80% of each other
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
